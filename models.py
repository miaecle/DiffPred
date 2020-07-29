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
               input_shape=(256, 256, 1),
               unet_feat=32,
               fc_layers=[64, 32],
               n_classes=1,
               freeze_encoder=False,
               model_path=None,
               **kwargs):
    self.input_shape = input_shape
    self.unet_feat = unet_feat
    self.fc_layers = fc_layers
    self.n_classes = n_classes

    self.freeze_encoder = freeze_encoder
    if model_path is None:
      self.model_path = tempfile.mkdtemp()
    else:
      self.model_path = model_path
    self.call_backs = [keras.callbacks.TerminateOnNaN(),
                 keras.callbacks.ReduceLROnPlateau(patience=5, min_lr=1e-7)]
                       #keras.callbacks.ModelCheckpoint(self.model_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5')]
    self.valid_score_callback = ValidMetrics()

    self.loss_func = l2_loss()
    self.build_model()
  
  def build_model(self):
    self.input = Input(shape=self.input_shape, dtype='float32')
    self.pre_conv = Dense(3, activation=None, name='pre_conv')(self.input)
    
    backbone = segmentation_models.backbones.get_backbone(
        'resnet34',
        input_shape=list(self.input_shape[:2]) + [3],
        weights='imagenet',
        include_top=False)
    
    if self.freeze_encoder:
      for layer in backbone.layers:
        if not isinstance(layer, BatchNormalization):
          layer.trainable=False    
    skip_connection_layers = segmentation_models.backbones.get_feature_layers('resnet34', n=4)
    self.unet = segmentation_models.unet.builder.build_unet(
        backbone,
        self.n_classes,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        block_type='upsampling',
        activation='linear',
        n_upsample_blocks=5,
        upsample_rates=(2, 2, 2, 2, 2),
        use_batchnorm=True)
    output = self.unet(self.pre_conv)
    
    self.model = Model(self.input, output)
    self.model.compile(optimizer='Adam', 
                       loss=self.loss_func,
                       metrics=[])

  def fit(self, 
          patches,
          batch_size=8, 
          n_epochs=10,
          valid_patches=None,
          **kwargs):

    if not os.path.exists(self.model_path):
      os.mkdir(self.model_path)
    X, y = preprocess(patches)
    validation_data = None
    if valid_patches is not None:
      validation_data = preprocess(valid_patches)
      self.valid_score_callback.valid_data = validation_data
      
    self.model.fit(x=X, 
                   y=y,
                   batch_size=batch_size,
                   epochs=n_epochs,
                   verbose=1,
                   callbacks=self.call_backs + [self.valid_score_callback],
                   validation_data=validation_data,
                   **kwargs)

  def predict(self, patches):
    if patches.__class__ is list:
      X, _ = preprocess(patches)
      y_pred = self.model.predict(X)
    elif patches.__class__ is np.ndarray:
      y_pred = self.model.predict(patches)
    else:
      raise ValueError("Input format not supported")
    return y_pred

  def save(self, path):
    self.model.save_weights(path)
  
  def load(self, path):
    self.model.load_weights(path)