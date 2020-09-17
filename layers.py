#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:51:20 2019

@author: zqwu
"""

import tensorflow as tf
import numpy as np
import scipy
import keras
from keras import backend as K
from keras.models import Model, load_model
from keras import layers
from keras.layers import Dense, Layer, Input
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def Conv2dBn(filters,
             kernel_size,
             strides=(1, 1),
             padding='valid',
             activation='linear',
             kernel_initializer='glorot_uniform',
             bias_initializer='zeros',
             use_batchnorm=False,
             name=None,
             **kwargs):

    if name is None:
        block_name = ''
    else:
        block_name = name

    conv_name = block_name + '_conv'
    act_name = block_name + '_' + activation
    bn_name = block_name + '_bn'
    bn_axis = 3

    def wrapper(input_tensor):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=conv_name,
        )(input_tensor)
        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)
        if activation:
            x = layers.Activation(activation, name=act_name)(x)
        return x

    return wrapper

class weighted_binary_cross_entropy(object):
  def __init__(self, n_classes=2):
    self.n_classes = n_classes
    self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    self.__name__ = "weighted_binary_cross_entropy"
    
  def __call__(self, y_true, y_pred):
    w = y_true[..., -1]
    y_true = y_true[..., :-1]
    loss = self.loss_fn(y_true, y_pred, sample_weight=w)
    return loss


class l2_loss(object):
  def __init__(self):
    self.loss_fn = tf.keras.losses.MeanSquaredError()
    self.__name__ = "l2_loss"
    
  def __call__(self, y_true, y_pred):
    loss = self.loss_fn(y_true, y_pred)
    return loss


class GradualDefreeze(keras.callbacks.Callback):
  def __init__(self, order={0: 'none', 3: 'last', 10: 'full'}):
    self.order = order
    self.key_points = sorted(order.keys())

  def on_epoch_begin(self, epoch, logs={}):
    if not epoch in self.key_points:
      return
    level = self.order[epoch]
    print("Change de-freeze level to %s" % level)
    change_model_level(self.model, level) 
    return


class ValidMetrics(keras.callbacks.Callback):
  def __init__(self, eval_fn, valid_data=None, test_data=None):
    self.eval_fn = eval_fn
    self.valid_data = valid_data
    self.test_data = test_data

  def on_epoch_end(self, epoch, logs={}):
    if self.valid_data is not None:
      _ = self.eval_fn(self.valid_data, self)
    if self.test_data is not None:
      pass
    return


def evaluate_classification(data, model):
  """
  data: ClassificationGenerator
  model: Classify
  """
  y_preds = []
  y_trues = []
  for batch in data:
    y_pred = model.model.predict(batch[0])
    y_pred = scipy.special.softmax(y_pred, -1)
    y_true = batch[1]
    valid_inds = np.nonzero(y_true[:, -1])
    y_trues.append(y_true[valid_inds][:, :2])
    y_preds.append(y_pred[valid_inds])
  y_trues = np.concatenate(y_trues, 0)
  y_preds = np.concatenate(y_preds, 0)

  auc = roc_auc_score(y_trues, y_preds)
  prec = precision_score(y_trues[:, 1], y_preds[:, 1] > 0.5)
  recall = recall_score(y_trues[:, 1], y_preds[:, 1] > 0.5)
  f1 = f1_score(y_trues[:, 1], y_preds[:, 1] > 0.5)

  print("Precision: %.3f\tRecall: %.3f\tF1: %.3f\tAUC: %.3f" %
      (prec, recall, f1, auc))
  return prec, recall, f1, auc


def evaluate_segmentation(data, model):
  """
  data: CustomGenerator, PairGenerator, etc.
  model: Segment
  """
  y_preds = []
  y_trues = []
  tp = 0
  fp = 0
  fn = 0
  total_ct = 0
  err_ct1 = 0 # Overall false positives
  err_ct2 = 0 # Overall false negatives
  thr = 0.01 * (288 * 384)
  for batch in data:
    y_pred = model.model.predict(batch[0])
    y_pred = scipy.special.softmax(y_pred, -1)
    y_true = batch[1][:, :, :, :-1]
    w = batch[1][:, :, :, -1]

    assert y_pred.shape[0] == y_true.shape[0] == w.shape[0]
    for _y_pred, _y_true, _w in zip(y_pred, y_true, w):
      _y_pred = _y_pred[np.nonzero(_w)].reshape((-1, 2))
      _y_true = _y_true[np.nonzero(_w)].reshape((-1, 2))
      _tp = ((_y_pred[:, 1] > 0.5) * _y_true[:, 1]).sum()
      _fp = ((_y_pred[:, 1] > 0.5) * _y_true[:, 0]).sum()
      _fn = ((_y_pred[:, 1] <= 0.5) * _y_true[:, 1]).sum()

      tp += _tp
      fp += _fp
      fn += _fn
      total_ct += 1
      if _y_pred.shape[0] > (0.99*288*384) and (_tp + _fn) < thr and _fp > thr:
        err_ct1 += 1
      if _fn > thr and (_tp + _fp) < thr:
        err_ct2 += 1

  iou = tp/(tp + fp + fn)
  prec = tp/(tp + fp)
  recall = tp/(tp + fn)
  f1 = 2/(1/(prec + 1e-5) + 1/(recall + 1e-5))
  print("Precision: %.3f\tRecall: %.3f\tF1: %.3f\tIOU: %.3f\tFP: %d/%d\tFN: %d/%d" %
        (prec, recall, f1, iou, err_ct1, total_ct, err_ct2, total_ct))
  return prec, recall, f1, iou, err_ct1, err_ct2


def evaluate_segmentation_and_classification(data, model):
  """
  data: CustomGenerator, PairGenerator, etc.
  model: ClassifyOnSegment
  """
  classify_y_preds = []
  classify_y_trues = []
  tp = 0
  fp = 0
  fn = 0
  total_ct = 0
  err_ct1 = 0 # Overall false positives
  err_ct2 = 0 # Overall false negatives
  thr = 0.01 * (288 * 384)
  for batch in data:
    y_pred, y_pred_classify = model.model.predict(batch[0])
    yw_true, yw_true_classify = batch[1]

    y_pred = scipy.special.softmax(y_pred, -1)
    y_pred_classify = scipy.special.softmax(y_pred_classify, -1)
    
    y_true = yw_true[..., :-1]
    w = yw_true[..., -1]
    y_true_classify = yw_true_classify[..., :-1]
    w_true_classify = yw_true_classify[..., -1]

    classify_valid_inds = np.nonzero(w_true_classify)
    classify_y_trues.append(y_true_classify[classify_valid_inds])
    classify_y_preds.append(y_pred_classify[classify_valid_inds])

    assert y_pred.shape[0] == y_true.shape[0] == w.shape[0]
    for _y_pred, _y_true, _w in zip(y_pred, y_true, w):
      _y_pred = _y_pred[np.nonzero(_w)].reshape((-1, 2))
      _y_true = _y_true[np.nonzero(_w)].reshape((-1, 2))
      _tp = ((_y_pred[:, 1] > 0.5) * _y_true[:, 1]).sum()
      _fp = ((_y_pred[:, 1] > 0.5) * _y_true[:, 0]).sum()
      _fn = ((_y_pred[:, 1] <= 0.5) * _y_true[:, 1]).sum()

      tp += _tp
      fp += _fp
      fn += _fn
      total_ct += 1
      if _y_pred.shape[0] > (0.99*288*384) and (_tp + _fn) < thr and _fp > thr:
        err_ct1 += 1
      if _fn > thr and (_tp + _fp) < thr:
        err_ct2 += 1

  iou = tp/(tp + fp + fn)
  prec = tp/(tp + fp)
  recall = tp/(tp + fn)
  f1 = 2/(1/(prec + 1e-5) + 1/(recall + 1e-5))
  print("Precision: %.3f\tRecall: %.3f\tF1: %.3f\tIOU: %.3f\tFP: %d/%d\tFN: %d/%d" %
        (prec, recall, f1, iou, err_ct1, total_ct, err_ct2, total_ct))

  classify_y_trues = np.concatenate(classify_y_trues, 0)
  classify_y_preds = np.concatenate(classify_y_preds, 0)
  auc = roc_auc_score(classify_y_trues, classify_y_preds)
  prec = precision_score(classify_y_trues[:, 1], classify_y_preds[:, 1] > 0.5)
  recall = recall_score(classify_y_trues[:, 1], classify_y_preds[:, 1] > 0.5)
  f1 = f1_score(classify_y_trues[:, 1], classify_y_preds[:, 1] > 0.5)
  print("Precision: %.3f\tRecall: %.3f\tF1: %.3f\tAUC: %.3f" %
      (prec, recall, f1, auc))
  return


def evaluate_confusion_mat(data, model):
  """
  data: CustomGenerator, PairGenerator, etc.
  model: ClassifyOnSegment
  """
  conf_mat_seg = np.zeros((10, 10))
  conf_mat_classify = np.zeros((10, 10))
  for batch in data:
    y_pred, y_pred_classify = model.model.predict(batch[0])
    yw_true, yw_true_classify = batch[1]

    y_pred = scipy.special.softmax(y_pred, -1)
    y_pred_classify = scipy.special.softmax(y_pred_classify, -1)
    
    y_true = yw_true[..., :-1]
    w = yw_true[..., -1]
    y_true_classify = yw_true_classify[..., :-1]
    w_true_classify = yw_true_classify[..., -1]

    for c_y_pred, c_y_true, c_w in zip(y_pred_classify, y_true_classify, w_true_classify):
      if c_w == 0:
        continue
      conf_mat_classify[np.argmax(c_y_true), np.argmax(c_y_pred)] += 1

    assert y_pred.shape[0] == y_true.shape[0] == w.shape[0]
    for _y_pred, _y_true, _w in zip(y_pred, y_true, w):
      _y_pred = _y_pred[np.nonzero(_w)]
      _y_true = _y_true[np.nonzero(_w)]
      for i, j in zip(np.argmax(_y_true, 1), np.argmax(_y_pred, 1)):
        conf_mat_seg[i, j] += 1


  n_seg_classes = np.where(conf_mat_seg > 0)[0].max() + 1
  conf_mat_seg = conf_mat_seg[:n_seg_classes, :n_seg_classes]

  n_classify_classes = np.where(conf_mat_classify > 0)[0].max() + 1
  conf_mat_classify = conf_mat_classify[:n_seg_classes, :n_seg_classes]  

  conf_mat_seg = conf_mat_seg/conf_mat_seg.sum(1, keepdims=True)
  conf_mat_classify = conf_mat_classify/conf_mat_classify.sum(1, keepdims=True)
  print("Segmentation")
  print(conf_mat_seg)
  print("Classification")
  print(conf_mat_classify)
  return

  
def evaluate_classification_with_segment_model(data, model, thr=800):
  """
  data: CustomGenerator, PairGenerator, etc.
  model: Segment
  """
  y_preds = model.predict(data)
  y_trues = []
  ws = []
  for batch in data:
    y = batch[1]
    y_ct = np.sign(y[:, :, :, 1]).sum(2).sum(1)
    w_ct = (1 - np.sign(y[:, :, :, -1])).sum(2).sum(1)
    for _y, _w in zip(y_ct, w_ct):
      if _y > 500:
          sample_y = 1
          sample_w = 1
      elif _y == 0 and _w < 600:
          sample_y = 0
          sample_w = 1
      else:
          sample_y = 0
          sample_w = 0
      y_trues.append(sample_y)
      ws.append(sample_w)

  if thr is None:
    thrs = [0, 50, 100, 200, 400, 800, 1600]
  else:
    thrs = [thr]

  for t in thrs:
    _y_preds = ((y_preds[..., 1] > 0.5).sum(2).sum(1) > t)
    y_trues = np.array(y_trues)
    ws = np.array(ws)
    valid_y_trues = y_trues[np.nonzero(ws)]
    valid_y_preds = _y_preds[np.nonzero(ws)]
    prec = precision_score(valid_y_trues, valid_y_preds)
    recall = recall_score(valid_y_trues, valid_y_preds)
    f1 = f1_score(valid_y_trues, valid_y_preds)
    print(t)
    print("Precision: %.3f\tRecall: %.3f\tF1: %.3f" %
          (prec, recall, f1))
  if thr is not None:
    return prec, recall, f1


def load_partial_weights(model, model2):
    main_module_ind = 2
    main_module = model.model.layers[main_module_ind]
    main_module2 = model2.model.layers[2]

    layer_names = [l.name for l in main_module.layers]
    layer_names2 = [l.name for l in main_module2.layers]

    unmatched = []
    for l_ind, l_name in enumerate(layer_names):
        if len(main_module.layers[l_ind].weights) == 0:
            continue
        if l_name in layer_names2:
            l_ind2 = layer_names2.index(l_name)
            l_weights = main_module.layers[l_ind].get_weights()
            l_weights2 = main_module2.layers[l_ind2].get_weights()
            if not len(l_weights) == len(l_weights2):
                unmatched.append((l_ind, l_name))
                continue
            try:
                model.model.layers[main_module_ind].layers[l_ind].set_weights(l_weights2)
            except Exception as e:
                print(e)
                unmatched.append((l_ind, l_name))

    layer_names = [l.name for l in model.model.layers]
    layer_names2 = [l.name for l in model2.model.layers]
    for l_ind, l_name in enumerate(layer_names):
        if l_ind == main_module_ind:
            continue
        if len(model.model.layers[l_ind].weights) == 0:
            continue
        if l_name in layer_names2:
            l_ind2 = layer_names2.index(l_name)
            l_weights = model.model.layers[l_ind].get_weights()
            l_weights2 = model2.model.layers[l_ind2].get_weights()
            if not len(l_weights) == len(l_weights2):
                unmatched.append((l_ind, l_name))
                continue
            try:
                model.model.layers[l_ind].set_weights(l_weights2)
            except Exception as e:
                print(e)
                unmatched.append((l_ind, l_name))

    print("Unmatched: %s" % str(unmatched))
    return model


def fill_first_layer(model, model2):
    """
    Hardcoded function that copies pre conv weight from model2 (3*2) to model (3*3)
    """
    pre_conv_ind = [i for i, l in enumerate(model.model.layers) if 'pre_conv' in l.name][0]
    pre_conv_weight = model.model.layers[pre_conv_ind].get_weights()

    pre_conv_ind2 = [i for i, l in enumerate(model2.model.layers) if 'pre_conv' in l.name][0]
    pre_conv_weight2 = model2.model.layers[pre_conv_ind2].get_weights()
    w, bias = pre_conv_weight2

    assembled_w = np.stack([w[0], w[1], np.zeros_like(w[1])], 0)
    model.model.layers[pre_conv_ind].set_weights([assembled_w, bias])
    return model


def change_model_level(model, level):
  main_module = model.layers[2]
  if level == 'none':
    # All core module layers not trainable
    for i in range(len(main_module.layers)):
      main_module.layers[i].trainable = False
  elif level == 'last':
    # Only the last conv layer trainable
    last_layer_ind = None
    for i in range(len(main_module.layers)):
      main_module.layers[i].trainable = False
      if 'conv' in main_module.layers[i].name:
          last_layer_ind = i
    for i in range(last_layer_ind, len(main_module.layers)):
      main_module.layers[i].trainable = True
  elif level == 'full':
    # All core module layers trainable
    for i in range(len(main_module.layers)):
      main_module.layers[i].trainable = True
  else:
    raise ValueError("Defree level not understood: %s" % level)
  model.compile(optimizer='Adam',
                loss=model.loss,
                metrics=[])
  return
