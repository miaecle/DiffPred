#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:22:55 2019

@author: zqwu
"""

import segmentation_models
import tensorflow as tf
import numpy as np
import keras
import tempfile
import os
import scipy
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Layer, Input, BatchNormalization, Conv2D, Lambda
from layers import weighted_binary_cross_entropy, ValidMetrics, l2_loss
from segment_support import preprocess

class Segment(object):
  def __init__(self,
               input_shape=(288, 384, 1),
               unet_feat=32,
               fc_layers=[64, 32],
               n_classes=2,
               class_weights=[1, 10],
               freeze_encoder=False,
               model_path=None,
               **kwargs):
    self.input_shape = input_shape
    self.unet_feat = unet_feat
    self.fc_layers = fc_layers
    self.n_classes = n_classes
    self.class_weights = class_weights
    assert len(self.class_weights) == self.n_classes

    self.freeze_encoder = freeze_encoder
    if model_path is None:
      self.model_path = tempfile.mkdtemp()
    else:
      self.model_path = model_path
    self.call_backs = [keras.callbacks.TerminateOnNaN(),
                 keras.callbacks.ReduceLROnPlateau(patience=5, min_lr=1e-7)]
                       #keras.callbacks.ModelCheckpoint(self.model_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5')]
    self.valid_score_callback = ValidMetrics()

    self.loss_func = weighted_binary_cross_entropy(n_classes=n_classes)
    self.build_model()
  
  def build_model(self):
    self.input = Input(shape=self.input_shape, dtype='float32')
    self.pre_conv = Dense(3, activation=None, name='pre_conv')(self.input)
    
    self.unet = segmentation_models.models.unet.Unet(
        backbone_name='resnet34',
        input_shape=list(self.input_shape[:2]) + [3],
        classes=self.n_classes,
        activation='linear',
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils
    )
    output = self.unet(self.pre_conv)
    
    self.model = Model(self.input, output)
    self.model.compile(optimizer='Adam', 
                       loss=self.loss_func,
                       metrics=[])

  def prepare_inputs(self, X, y=None, w=None):
    if w is None:
      w = np.ones(list(X.shape[:-1]) + [1])
    if not y is None:
      _y = np.zeros(list(X.shape[:-1]) + [self.n_classes+1])
    
      _w = np.zeros_like(w)
      for i in range(self.n_classes):
        _y[..., i] = (y == i)
        _w += w * (y == i) * self.class_weights[i]
      _y[..., -1] = _w
    else:
      _y = None
    return X, _y
    
  def fit(self, 
          train_data,
          valid_data=None,
          batch_size=8, 
          n_epochs=10,
          **kwargs):

    if not os.path.exists(self.model_path):
      os.mkdir(self.model_path)
    _X, _y = self.prepare_inputs(*train_data)

    if valid_data is not None:
      valid_X, valid_y = self.prepare_inputs(*valid_data)
      self.valid_score_callback.valid_data = (valid_X, valid_y)
      
    self.model.fit(x=_X, 
                   y=_y,
                   batch_size=batch_size,
                   epochs=n_epochs,
                   verbose=1,
                   callbacks=self.call_backs + [self.valid_score_callback],
                   validation_data=(valid_X, valid_y),
                   **kwargs)

  def predict(self, data):
    X, _ = self.prepare_inputs(*data)
    y_pred = self.model.predict(X)
    y_pred = scipy.special.softmax(y_pred, -1)
    return y_pred

  def save(self, path):
    self.model.save_weights(path)
  
  def load(self, path):
    self.model.load_weights(path)
