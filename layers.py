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
from sklearn.metrics import roc_auc_score, f1_score

class weighted_binary_cross_entropy(object):
  def __init__(self, n_classes=2):
    self.n_classes = n_classes
    self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    self.__name__ = "weighted_binary_cross_entropy"
    
  def __call__(self, y_true, y_pred):
    w = y_true[:, :, :, -1]
    y_true = y_true[:, :, :, :-1]
    loss = self.loss_fn(y_true, y_pred, sample_weight=w)
    print(loss)
    return loss

class l2_loss(object):
  def __init__(self):
    self.loss_fn = tf.keras.losses.MeanSquaredError()
    self.__name__ = "l2_loss"
    
  def __call__(self, y_true, y_pred):
    loss = self.loss_fn(y_true, y_pred)
    return loss


class ValidMetrics(keras.callbacks.Callback):
  def __init__(self, valid_data=None, test_data=None):
    self.valid_data = valid_data
    self.test_data = test_data

  def on_epoch_end(self, epoch, logs={}):
    if epoch % 5 != 4:
      # Run evaluation on epoch 4, 9, ...
      return
    if self.valid_data is not None:
      self.valid_data.clean_cache(force=True)
      y_preds = []
      y_trues = []
      for batch in self.valid_data:
        y_pred = self.model.predict(batch[0])
        y_pred = scipy.special.softmax(y_pred, -1)
        y_true = batch[1][:, :, :, :-1]
        w = batch[1][:, :, :, -1]
        y_preds.append(y_pred[np.nonzero(w)])
        y_trues.append(y_true[np.nonzero(w)])
        
      _y_preds = np.concatenate(y_preds, 0).reshape((-1, 2))
      _y_trues = np.concatenate(y_trues, 0).reshape((-1, 2))

      all_intersection = ((_y_preds[:, 1] > 0.5) * _y_trues[:, 1]).sum()
      all_union = (np.sign((_y_preds[:, 1] > 0.5) + _y_trues[:, 1])).sum()
      iou = all_intersection/all_union
      auc = roc_auc_score(_y_trues, _y_preds)
      f1 = f1_score(_y_trues[:, 1], _y_preds[:, 1] > 0.5)
      print('\r valid-roc-auc: %f  valid-f1: %f  valid-iou: %f\n' % (auc, f1, iou))
    if self.test_data is not None:
#       y_pred = self.model.predict(self.test_data[0])[:, :, :, 1]
#       y_true = self.test_data[1][:, :, :, 1] > 0.5
#       roc = roc_auc_score(y_true.flatten(), y_pred.flatten())
#       f1 = f1_score(y_true.flatten(), y_pred.flatten()>0.5)
#       print('\r test-roc-auc: %f  test-f1: %f\n' % (roc, f1))
      pass
    return
 
