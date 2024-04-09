#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:22:55 2019

@author: zqwu
"""
import segmentation_models
import classification_models
import numpy as np
import tempfile
import os
import scipy

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Conv2D

from layers import (
    weighted_cross_entropy,
    Conv2dBn,
    ValidMetrics,
    evaluate_segmentation,
    evaluate_classification,
    evaluate_segmentation_and_classification,
)
from data_generator import CustomGenerator


class Segment(object):
    def __init__(self,
                 input_shape=(288, 384, 1),
                 n_classes=2,
                 encoder_weights='imagenet',
                 freeze_encoder=False,
                 loss_fn=weighted_cross_entropy,
                 eval_fn=evaluate_segmentation,
                 model_structure='unet',
                 model_path=None,
                 **kwargs):
        self.input_shape = input_shape
        self.n_classes = n_classes

        self.structure = model_structure
        self.encoder_weights = encoder_weights
        self.freeze_encoder = freeze_encoder
        if model_path is None:
            self.model_path = tempfile.mkdtemp()
        else:
            self.model_path = model_path
        self.call_backs = [
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.ReduceLROnPlateau(patience=20, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(self.model_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5')]
        self.valid_score_callback = ValidMetrics(eval_fn)

        self.loss_func = loss_fn(n_classes=n_classes)
        self.build_model()
        self.compile()

    def build_model(self):
        self.input = Input(shape=self.input_shape, dtype='float32')
        self.pre_conv = Dense(3, activation=None, name='pre_conv')(self.input)
        self.net = self.get_backbone_module(self.n_classes)
        output = self.net(self.pre_conv)
        self.model = Model(self.input, output)

    def get_backbone_module(self, n_classes, activation='linear'):
        if self.structure == 'unet':
            net = segmentation_models.models.unet.Unet(
                backbone_name='resnet34',
                input_shape=list(self.input_shape[:2]) + [3],
                classes=n_classes,
                activation=activation,
                encoder_weights=self.encoder_weights,
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
        elif self.structure == 'pspnet':
            net = segmentation_models.models.pspnet.PSPNet(
                backbone_name='resnet34',
                input_shape=list(self.input_shape[:2]) + [3],
                classes=n_classes,
                activation=activation,
                encoder_weights=self.encoder_weights,
                encoder_freeze=False,
                downsample_factor=8,
                psp_conv_filters=512,
                psp_pooling_type='avg',
                psp_use_batchnorm=True,
                psp_dropout=None,
                backend=keras.backend,
                layers=keras.layers,
                models=keras.models,
                utils=keras.utils
            )
        elif self.structure == 'fpn':
            net = segmentation_models.models.fpn.FPN(
                backbone_name='resnet34',
                input_shape=list(self.input_shape[:2]) + [3],
                classes=n_classes,
                activation=activation,
                encoder_weights=self.encoder_weights,
                encoder_freeze=False,
                encoder_features='default',
                pyramid_block_filters=256,
                pyramid_use_batchnorm=True,
                pyramid_aggregation='concat',
                pyramid_dropout=None,
                backend=keras.backend,
                layers=keras.layers,
                models=keras.models,
                utils=keras.utils
                )
        else:
            raise ValueError("Structure not supported")
        return net


    def compile(self):
      self.model.compile(optimizer='Adam', 
                        loss=self.loss_func,
                        metrics=[])


    def fit(self, 
            train_gen,
            valid_gen=None,
            n_epochs=10,
            verbose=1,
            **kwargs):
      if not os.path.exists(self.model_path):
        os.mkdir(self.model_path)
      if valid_gen is not None:
        self.valid_score_callback.valid_data = valid_gen
      self.model.fit_generator(train_gen,
                              steps_per_epoch=len(train_gen), 
                              epochs=n_epochs,
                              verbose=verbose,
                              callbacks=self.call_backs + [self.valid_score_callback],
                              validation_data=valid_gen,
                              shuffle=False, 
                              initial_epoch=0,
                              **kwargs)


    def predict_on_generator(self, gen):
      y_pred = self.model.predict_generator(gen)
      y_pred = scipy.special.softmax(y_pred, -1)
      return y_pred


    def predict_on_X(self, X):
      y_pred = self.model.predict(X)
      y_pred = scipy.special.softmax(y_pred, -1)
      return y_pred


    def predict(self, inputs):
      if isinstance(inputs, (np.ndarray, np.generic)) and tuple(inputs.shape[1:]) == self.input_shape:
        preds = self.predict_on_X(inputs)
      elif isinstance(inputs, tuple) and tuple(inputs[0].shape[1:]) == self.input_shape:
        preds = self.predict_on_X(inputs[0])
      elif isinstance(inputs, CustomGenerator):
        preds = self.predict_on_generator(inputs)
      else:
        print("Data type not supported")
        return None
      return preds


    def save(self, path):
      self.model.save_weights(path)


    def load(self, path):
      self.model.load_weights(path)


class Classify(Segment):
  def __init__(self,
               input_shape=(288, 384, 1),
               fc_layers=[1024, 128],
               n_classes=2,
               encoder_weights='imagenet',
               loss_fn=weighted_cross_entropy,
               eval_fn=evaluate_classification,
               model_structure='resnet34',
               model_path=None,
               **kwargs):
    self.input_shape = input_shape
    self.fc_layers = fc_layers
    self.n_classes = n_classes

    self.encoder_weights = encoder_weights
    self.structure = model_structure
    if model_path is None:
      self.model_path = tempfile.mkdtemp()
    else:
      self.model_path = model_path
    self.call_backs = [keras.callbacks.TerminateOnNaN(),
                       # keras.callbacks.ReduceLROnPlateau(patience=5, min_lr=1e-7),
                       keras.callbacks.ModelCheckpoint(self.model_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5')]
    self.valid_score_callback = ValidMetrics(eval_fn)

    self.loss_func = loss_fn(n_classes=n_classes)
    self.build_model()
    self.compile()


  def build_model(self):
    self.input = Input(shape=self.input_shape, dtype='float32')
    self.pre_conv = Dense(3, activation=None, name='pre_conv')(self.input)
    self.net = self.get_backbone_module(self.n_classes)
    output = self.net(self.pre_conv)
    output = layers.GlobalAveragePooling2D(name='pool1')(output)
    for i, l in enumerate(self.fc_layers):
      output = Dense(l, name='fc%d' % i)(output)
    output = Dense(self.n_classes, name='fc_output')(output)
    self.model = Model(self.input, output)


  def get_backbone_module(self, n_classes):
    if self.structure == 'resnet34':
      net = classification_models.models.resnet.ResNet34(
          input_shape=list(self.input_shape[:2]) + [3],
          classes=n_classes,
          include_top=False,
          weights=self.encoder_weights,
          backend=keras.backend,
          layers=keras.layers,
          models=keras.models,
          utils=keras.utils
      )
    else:
      raise ValueError("Structure not supported")
    return net




class ClassifyOnSegment(Segment):
  def __init__(self,
               input_shape=(288, 384, 1),
               unet_feat=32,
               fc_layers=[64, 32],
               n_segment_classes=2,
               n_classify_classes=2,
               encoder_weights='imagenet',
               freeze_encoder=False,
               segment_loss_fn=weighted_cross_entropy,
               classify_loss_fn=weighted_cross_entropy,
               eval_fn=evaluate_segmentation_and_classification,
               segment_model_structure='unet',
               model_path=None,
               **kwargs):
    self.input_shape = input_shape
    self.unet_feat = unet_feat
    self.fc_layers = fc_layers

    self.n_segment_classes = n_segment_classes
    self.n_classify_classes = n_classify_classes

    self.encoder_weights = encoder_weights
    self.structure = segment_model_structure
    if model_path is None:
      self.model_path = tempfile.mkdtemp()
    else:
      self.model_path = model_path
    self.call_backs = [keras.callbacks.TerminateOnNaN(),
                       # keras.callbacks.ReduceLROnPlateau(patience=5, min_lr=1e-7),
                       keras.callbacks.ModelCheckpoint(self.model_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5')]
    self.valid_score_callback = ValidMetrics(eval_fn)

    self.loss_func = [segment_loss_fn(n_classes=n_segment_classes),
                      classify_loss_fn(n_classes=n_classify_classes)]
    self.build_model()
    self.compile()


  def build_model(self):
    self.input = Input(shape=self.input_shape, dtype='float32')
    self.pre_conv = Dense(3, activation=None, name='pre_conv')(self.input)
    self.core_net = self.get_backbone_module(self.unet_feat, activation='relu')
    core_embedding = self.core_net(self.pre_conv)

    self.segment_out = Conv2D(filters=self.n_segment_classes,
                              kernel_size=1,
                              activation='linear',
                              use_bias=True,
                              kernel_initializer='glorot_uniform',
                              name='segment_head_output')(core_embedding)

    x = core_embedding
    for i in range(5):
      x = Conv2dBn(self.unet_feat*max(1, 2**(i-2)), 
                   kernel_size=3, 
                   padding='same', 
                   activation='relu', 
                   use_batchnorm=True, 
                   name='classify_head_block%d' % i)(x)
      x = Conv2dBn(self.unet_feat*max(1, 2**(i-1)), 
                   kernel_size=3, 
                   strides=(2, 2),
                   padding='same', 
                   activation='relu',
                   use_batchnorm=True,
                   name='classify_head_block%d_downsample' % i)(x)

    output = layers.GlobalAveragePooling2D(name='classify_head_pool1')(x)
    for i, l in enumerate(self.fc_layers):
      output = Dense(l, name='classify_head_fc%d' % i)(output)
    self.classify_out = Dense(self.n_classify_classes, name='classify_head_output')(output)
    self.model = Model(self.input, [self.segment_out, self.classify_out])


  def predict_on_generator(self, gen):
    y_preds = self.model.predict_generator(gen)
    y_preds = [scipy.special.softmax(y_pred, -1) for y_pred in y_preds]
    return y_preds


  def predict_on_X(self, X):
    y_preds = self.model.predict(X)
    y_preds = [scipy.special.softmax(y_pred, -1) for y_pred in y_preds]
    return y_preds
