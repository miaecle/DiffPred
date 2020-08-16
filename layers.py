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
    #if epoch % 5 != 4:
      # Run evaluation on epoch 4, 9, ...
      #return
    if self.valid_data is not None:
      y_preds = []
      y_trues = []
      tp = 0
      fp = 0
      fn = 0
      for batch in self.valid_data:
        y_pred = self.model.predict(batch[0])
        y_pred = scipy.special.softmax(y_pred, -1)
        y_true = batch[1][:, :, :, :-1]
        w = batch[1][:, :, :, -1]
        y_pred = y_pred[np.nonzero(w)].reshape((-1, 2))
        y_true = y_true[np.nonzero(w)].reshape((-1, 2))
        tp += ((y_pred[:, 1] > 0.5) * y_true[:, 1]).sum()
        fp += ((y_pred[:, 1] > 0.5) * y_true[:, 0]).sum()
        fn += ((y_pred[:, 1] < 0.5) * y_true[:, 1]).sum()

      iou = tp/(tp + fp + fn)
      prec = tp/(tp + fp)
      recall = tp/(tp + fn)
      f1 = 2/(1/(prec + 1e-5) + 1/(recall + 1e-5))
      print('\r valid-prec: %.3f  valid-recall: %.3f  valid-f1: %.3f  valid-iou: %.3f\n' % (prec, recall, f1, iou))
    if self.test_data is not None:
      pass
    return
 
