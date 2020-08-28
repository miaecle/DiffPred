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
from keras.layers import Dense, Layer, Input
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


class weighted_binary_cross_entropy(object):
  def __init__(self, n_classes=2):
    self.n_classes = n_classes
    self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    self.__name__ = "weighted_binary_cross_entropy"
    
  def __call__(self, y_true, y_pred):
    w = y_true[:, :, :, -1]
    y_true = y_true[:, :, :, :-1]
    loss = self.loss_fn(y_true, y_pred, sample_weight=w)
    return loss


class classification_binary_cross_entropy(object):
  def __init__(self, n_classes=2):
    self.n_classes = n_classes
    self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    self.__name__ = "classification_binary_cross_entropy"

  def __call__(self, y_true, y_pred):
    w = y_true[:, -1]
    y_true = y_true[:, :-1]
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


class ClassificationValidMetrics(keras.callbacks.Callback):
  def __init__(self, valid_data=None, test_data=None):
    self.valid_data = valid_data
    self.test_data = test_data

  def on_epoch_end(self, epoch, logs={}):
    if self.valid_data is not None:
      _ = evaluate_classification(self.valid_data, self)
    if self.test_data is not None:
      pass
    return


class ValidMetrics(keras.callbacks.Callback):
  def __init__(self, valid_data=None, test_data=None):
    self.valid_data = valid_data
    self.test_data = test_data

  def on_epoch_end(self, epoch, logs={}):
    if self.valid_data is not None:
      _ = evaluate_segmentation(self.valid_data, self)
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
    # Only applicable to Segment/Classify models
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
            model.model.layers[main_module_ind].layers[l_ind].set_weights(l_weights2)
    print("Unmatched: %s" % str(unmatched))
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
