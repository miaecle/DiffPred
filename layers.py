#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:51:20 2019

@author: zqwu
"""

import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Layer, Input
from sklearn.metrics import roc_auc_score, f1_score

class weighted_binary_cross_entropy(object):
  def __init__(self, n_classes=2):
    self.n_classes = n_classes
    self.__name__ = "weighted_binary_cross_entropy"
    
  def __call__(self, y_true, y_pred):
    w = y_true[:, :, :, -1]
    y_true = y_true[:, :, :, :-1]
    loss = K.categorical_crossentropy(y_true, y_pred, from_logits=True) * w
    return loss

class l2_loss(object):
  def __init__(self):
    self.__name__ = "l2_loss"
    
  def __call__(self, y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true))
    return loss


class ValidMetrics(keras.callbacks.Callback):
  def __init__(self, valid_data=None, test_data=None):
    self.valid_data = valid_data
    self.test_data = test_data

  def on_epoch_end(self, epoch, logs={}):
    return
    if self.valid_data is not None:
#       y_pred = self.model.predict(self.valid_data[0])
#       y_true = self.valid_data[1]
#       roc = roc_auc_score(y_true.flatten(), y_pred.flatten())
#       f1 = f1_score(y_true.flatten(), y_pred.flatten()>0.5)
#       print('\r valid-roc-auc: %f  valid-f1: %f\n' % (roc, f1))
       pass
    if self.test_data is not None:
#       y_pred = self.model.predict(self.test_data[0])[:, :, :, 1]
#       y_true = self.test_data[1][:, :, :, 1] > 0.5
#       roc = roc_auc_score(y_true.flatten(), y_pred.flatten())
#       f1 = f1_score(y_true.flatten(), y_pred.flatten()>0.5)
#       print('\r test-roc-auc: %f  test-f1: %f\n' % (roc, f1))
      pass
    return
 
