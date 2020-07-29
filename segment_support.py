#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:35:16 2020

@author: zqwu
"""
import os
import numpy as np
import cv2
import pickle

CHANNEL_MAX = 65535

def rotate_image(mat, angle, image_center=None):
  # angle in degrees
  height, width = mat.shape[:2]
  if image_center is None:
    image_center = (width/2, height/2)
  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])
  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)
  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]
  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat


def index_mat(mat, x_from, x_to, y_from, y_to):
  x_size = mat.shape[0]
  if x_from < 0:
    assert x_to < x_size
    s = np.concatenate([np.zeros_like(mat[x_from:]), mat[:x_to]], 0)
  elif x_to > x_size:
    assert x_from >= 0
    s = np.concatenate([mat[x_from:], np.zeros_like(mat[:(x_to-x_size)])], 0)
  else:
    s = mat[x_from:x_to]

  y_size = mat.shape[1]
  if y_from < 0:
    assert y_to < y_size
    s = np.concatenate([np.zeros_like(s[:, y_from:]), s[:, :y_to]], 1)
  elif y_to > y_size:
    assert y_from >= 0
    s = np.concatenate([s[:, y_from:], np.zeros_like(s[:, :(y_to-y_size)])], 1)
  else:
    s = s[:, y_from:y_to]
  assert s.shape[0] == (x_to - x_from)
  assert s.shape[1] == (y_to - y_from)
  return s


def extract_mat(input_mat, 
                x_center, 
                y_center, 
                x_size=256,
                y_size=256,
                angle=0, 
                flip=False):
  x_margin = int(x_size/np.sqrt(2))
  y_margin = int(y_size/np.sqrt(2))

  patch = index_mat(input_mat, 
                    (x_center - x_margin), 
                    (x_center + x_margin), 
                    (y_center - y_margin), 
                    (y_center + y_margin))
  patch = np.array(patch).astype(float)
  if angle != 0:
    patch = rotate_image(patch, angle)
  if flip:
    patch = cv2.flip(patch, 1)

  center = (patch.shape[0]//2, patch.shape[1]//2)
  patch_X = patch[(center[0] - x_size//2):(center[0] + x_size//2),
                  (center[1] - y_size//2):(center[1] + y_size//2)]
  return patch_X


def generate_patches(input_dat_pairs,
                     n_patches=1000,
                     x_size=256,
                     y_size=256,
                     rotate=False,
                     mirror=False,
                     seed=None,
                     **kwargs):  
  data = []
  if not seed is None:
    np.random.seed(seed)
  while len(data) < n_patches:
    pair = np.random.choice(input_dat_pairs)

    x_center = np.random.randint(0, pair[0].shape[0])
    y_center = np.random.randint(0, pair[0].shape[1])

    if rotate:
      angle = np.random.rand() * 360
    else:
      angle = 0

    if mirror:
      flip = np.random.rand() > 0.5
    else:
      flip = False

    patch_pc = extract_mat(pair[0], x_center, y_center, x_size=x_size, y_size=y_size, angle=angle, flip=flip)
    patch_gfp = extract_mat(pair[1], x_center, y_center, x_size=x_size, y_size=y_size, angle=angle, flip=flip)
    data.append((patch_pc, patch_gfp))
  return data

def generate_ordered_patches(input_dat_pairs,
                             x_size=256,
                             y_size=256,
                             seed=None):
  data = []
  if not seed is None:
    np.random.seed(seed)

  x_shape = input_dat_pairs[0][0].shape[0]
  y_shape = input_dat_pairs[0][0].shape[1]
  for pair in input_dat_pairs:
    x_center = np.random.randint(-x_size//2, x_size//2)
    while (x_center < x_shape+x_size//2):
      y_center = np.random.randint(-y_size//2, y_size//2)
      while (y_center < y_shape+y_size//2):
        patch_pc = extract_mat(pair[0], x_center, y_center, x_size=x_size, y_size=y_size)
        patch_gfp = extract_mat(pair[1], x_center, y_center, x_size=x_size, y_size=y_size)
        data.append((patch_pc, patch_gfp))
        y_center += y_size
      x_center += x_size
  return data

def preprocess(patches):
  Xs = []
  ys = []
  for pair in patches:
    x = np.expand_dims(pair[0].astype(float)/CHANNEL_MAX, 2)
    y = np.expand_dims(pair[1].astype(float)/CHANNEL_MAX, 2)
    Xs.append(x)
    ys.append(y)
  Xs = np.stack(Xs, 0)
  ys = np.stack(ys, 0)
  return Xs, ys
  

# def predict_whole_map_on_offset(inp, 
#                                 model, 
#                                 x_offset=0, 
#                                 y_offset=0, 
#                                 n_classes=3, 
#                                 batch_size=8):
#   x_size = model.input_shape[0]
#   y_size = model.input_shape[1]

#   assert inp.shape[0] % x_size == 0
#   assert inp.shape[1] % y_size == 0
#   assert inp.shape[2] == model.input_shape[2]
#   rows = inp.shape[0] // x_size
#   columns = inp.shape[1] // y_size

#   batch_inputs = []
#   outputs = []
#   for r in range(rows):
#     for c in range(columns):
#       if x_offset + (r+1)*x_size > inp.shape[0] or \
#          y_offset + (c+1)*y_size > inp.shape[1]:
#         continue
#       patch_inp = inp[(x_offset + r*x_size):(x_offset + (r+1)*x_size), 
#                       (y_offset + c*y_size):(y_offset + (c+1)*y_size)]
#       batch_inputs.append((patch_inp, None))
#       if len(batch_inputs) == batch_size:
#         batch_outputs = model.predict(batch_inputs, label_input=None)
#         outputs.extend(batch_outputs)
#         batch_inputs = []
#   if len(batch_inputs) > 0:
#     batch_outputs = model.predict(batch_inputs, label_input=None)
#     outputs.extend(batch_outputs)
#     batch_inputs = []
  
#   ct = 0
#   concatenated_output = -np.ones((inp.shape[0], inp.shape[1], n_classes))
#   for r in range(rows):
#     for c in range(columns):
#       if x_offset + (r+1)*x_size > inp.shape[0] or \
#          y_offset + (c+1)*y_size > inp.shape[1]:
#         continue
#       concatenated_output[(x_offset + r*x_size):(x_offset + (r+1)*x_size), 
#                           (y_offset + c*y_size):(y_offset + (c+1)*y_size)] = outputs[ct]
#       ct += 1
#   assert ct == len(outputs)
#   return concatenated_output

# def predict_whole_map(inp, 
#                       model, 
#                       n_classes=3, 
#                       batch_size=8, 
#                       n_supp=5):
#   x_size = model.input_shape[0]
#   y_size = model.input_shape[1]
#   base_mat = predict_whole_map_on_offset(inp, 
#                                          model, 
#                                          x_offset=0, 
#                                          y_offset=0, 
#                                          n_classes=n_classes, 
#                                          batch_size=batch_size)
#   ct_mat = np.ones((inp.shape[0], inp.shape[1], 1))
#   for i_supp in range(n_supp):
#     x_offset = np.random.randint(1, x_size)
#     y_offset = np.random.randint(1, y_size)
#     supp_mat = predict_whole_map_on_offset(inp, 
#                                            model, 
#                                            x_offset=x_offset, 
#                                            y_offset=y_offset, 
#                                            n_classes=n_classes, 
#                                            batch_size=batch_size)
#     supp_positions = np.where(supp_mat[:, :, 0] >= 0.)
#     base_mat[supp_positions] = (base_mat[supp_positions] * ct_mat[supp_positions] + \
#                                 supp_mat[supp_positions]) / (ct_mat[supp_positions] + 1)
#     ct_mat[supp_positions] += 1
#   return base_mat

# if __name__ == '__main__':
#   dats = load_features()
#   uv_annos = load_UV_annos()
#   bf_annos = load_BF_annos(shrink_cell_ratio=0.5)
  
#   uv_patches = generate_ordered_patches(dats[:, :, :, 2:3], uv_annos) + \
#                generate_patches(dats[:, :, :, 2:3], uv_annos, rotate=True, mirror=True, seed=123)
#   with open('uv_patches.pkl', 'wb') as f:
#     pickle.dump(uv_patches, f)

#   bf_patches = generate_ordered_patches(dats[:, :, :, 1:2], bf_annos) + \
#                generate_patches(dats[:, :, :, 1:2], bf_annos, rotate=True, mirror=True, seed=123)
#   with open('bf_patches.pkl', 'wb') as f:
#     pickle.dump(bf_patches, f)
#   