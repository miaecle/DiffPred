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
import cmath
from scipy import optimize
from scipy.signal import convolve2d
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from skimage import measure
import matplotlib.pyplot as plt
plt.switch_backend('AGG')

CHANNEL_MAX = 65535


def rotate(coords, angle):
    cs = np.cos(angle)
    sn = np.sin(angle)

    x = coords[0] * cs - coords[1] * sn;
    y = coords[0] * sn + coords[1] * cs;
    return x, y


def smooth(x, window_len=3, window='hanning'):    
    w = np.ones(window_len, 'd')
    y = np.convolve(w/w.sum(), x, mode='same')
    return y


def well_id(pair):
    f_name = pair[0].split('/')[-1]
    f_name = f_name.replace('-', ' ').replace('_', ' ')
    return f_name.split()[0]


def position_code(pair):
    return pair[0].split('/')[-1].split('_')[3]


def generate_dist_mat(mask, position_code):
    light_center = {
        '1': (1356, 1836), 
        '2': (1356, 612),
        '3': (1356, -612),
        '4': (452, 1836),
        '5': (452, 612),
        '6': (452, -612),
        '7': (-452, 1836),
        '8': (-452, 612),
        '9': (-452, -612)
    }
    center = np.array(light_center[position_code])
    dist_mat1 = np.stack([np.arange(mask.shape[0])] * mask.shape[1], 1)
    dist_mat2 = np.stack([np.arange(mask.shape[1])] * mask.shape[0], 0)
    dist_mat = np.stack([dist_mat1, dist_mat2], 2)

    dist_mat = np.sqrt(((dist_mat - center.reshape((1, 1, 2)))**2).sum(2))
    dist_mat = dist_mat * mask
    return dist_mat


def get_center(edge):
    # Deprecated, calculate plate center based on edge
    x, y = np.where(edge)
    def calc_R(xc, yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def loss(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    
    center, ier = optimize.leastsq(loss, (x.mean(), y.mean()))
    assert ier in [1,2,3,4]
    R = calc_R(*center).mean()
    return center, R


def get_long_axis(blob_mask, blob_id):
    y, x = np.where(blob_mask == blob_id)
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.stack([x, y], 0)
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    main_axis = evecs[:, np.argmax(evals)]  # Eigenvector with largest eigenvalue
    angle = cmath.polar(complex(*main_axis))[1]
    _x, _y = rotate(coords, angle)
    return max(_y.max() - _y.min(), _x.max() - _x.min())


def find_threshold(fl):
    # Find threshold between out-of-well area fl and in-well fl
    hist_vals, x = np.histogram(fl.flatten(), bins=np.arange(0, 65535, 200))
    hist_vals = hist_vals/hist_vals.sum()
    x = (x[1:] + x[:-1])/2

    smoothed_arr = smooth(hist_vals, 30)
    arr = np.r_[True, smoothed_arr[1:] > smoothed_arr[:-1]] & \
          np.r_[smoothed_arr[:-1] > smoothed_arr[1:], True] & \
          (smoothed_arr > np.max(smoothed_arr) * 0.1)
    
    local_maximas = sorted(np.where(arr)[0])
    
    if len(local_maximas) == 1:
        thr = min(0.003, np.min(hist_vals[:local_maximas[0]])) + 0.0002
    else:
        thr = min(0.003, np.min(hist_vals[local_maximas[0]:local_maximas[-1]])) + 0.0002
    
    max_val = max([hist_vals[i] for i in local_maximas])
    right_max_ind = [i for i in local_maximas if hist_vals[i] > 0.8 * max_val][-1]
    
    p_right = None
    p_right_ind = None
    for i in range(right_max_ind, 0, -1):
        if hist_vals[i] < thr:
            p_right_ind = i
            p_right = x[i]
            break
    assert not p_right is None
    
    left_max_ind = np.argmax(hist_vals[:p_right_ind])
    p_left = None
    if hist_vals[left_max_ind] < thr:
        p_left = x[0]
    else:
        for i in range(left_max_ind, p_right_ind):
            if hist_vals[i] < thr and p_left is None:
                p_left = x[i]
                break
    if p_left is None:
      p_left = x[0]
    return p_left, p_right


def generate_mask(pair_dat, plot=False):
    fl = pair_dat[1] # Should be unnormalized uint16 values

    p_left, p_right = find_threshold(fl)
    threshold = fl > ((p_left + p_right)/2)
    mask = cv2.blur(threshold.astype('uint8'), (10, 10))

    if plot:
        hist_vals, x = np.histogram(fl.flatten(), bins=np.arange(0, 65535, 200))
        hist_vals = hist_vals/hist_vals.sum()
        x = (x[1:] + x[:-1])/2
        plt.clf()
        plt.plot(x, hist_vals)
        plt.vlines([p_left, p_right], 0, 0.1, color='r')
    
        plt.clf()
        plt.imshow(mask)
        plt.show()
    return mask


def generate_fluorescence_labels(pair_dat, mask):
    fl = pair_dat[1]

    # Add blur to intensity
    intensity = ((fl * mask.astype('float'))/256).astype('uint8')
    intensity = cv2.medianBlur(intensity, 15) * 256

    # Reduce resolution to better find image gradient
    _intensity = cv2.resize(intensity, (75, 56))
    _mask = cv2.resize(mask, (75, 56))

    # Extend mask to remove edge effect (if on 1, 3, 7, 9)
    _mask_extended = 1 - np.sign(convolve2d(1 - _mask, np.ones((11, 11)), mode='same'))

    # Using Laplacian to find points with large gradient
    _lap = cv2.Laplacian(_intensity, cv2.CV_64F, ksize=17) * _mask_extended
    
    # Log transform
    _lap[np.where(_lap > 1)] = np.log(_lap[np.where(_lap > 1)])
    _lap[np.where(_lap < -1)] = - np.log(-_lap[np.where(_lap < -1)])
    lap = cv2.resize(_lap, (1224, 904))
    
    # High-conf neg and pos
    negatives = (lap > 25) * mask
    positives = (lap < -27) * mask

    if np.sum(positives) < 1200*900*0.01:
        # No significant differentiation detected
        negatives = mask - np.sign(convolve2d(positives, np.ones((3, 3)), mode='same'))
        positives = np.zeros_like(mask)
    else:
        # Average fluorescence for negative/positive pixels
        fl_neg_threshold = np.quantile(fl[np.where(negatives)], 0.6)
        fl_pos_threshold = np.quantile(fl[np.where(positives)], 0.5)

        # Assign uncovered pixels to negative if the gradient is not high and fluorescence signal is low
        negatives = np.sign(negatives + (fl < fl_neg_threshold) * mask * (1 - positives))
        positives = np.sign(positives + (fl > fl_pos_threshold) * mask * (1 - negatives))
        negatives = np.sign(convolve2d(negatives, np.ones((3, 3)), mode='same')) * (1 - positives)

    # Clean masks by removing scattered segmentations
    blobs = measure.label(positives, background=0)
    for blob_id, ct in zip(*np.unique(blobs, return_counts=True)):
        if blob_id == 0:
            continue
        elif ct < 500:
            positives[np.where(blobs == blob_id)] = 0
        elif ct < 3000:
            max_d = get_long_axis(blobs, blob_id)
            if max_d < 60:
                positives[np.where(blobs == blob_id)] = 0
    return positives - negatives + 1


def quantize_fluorescence(pair_dat, mask):
    segmentation = generate_fluorescence_labels(pair_dat, mask)
    
    fl = pair_dat[1]
    neg_intensity = fl[np.where(segmentation==0)]
    pos_intensity = fl[np.where(segmentation==2)]
    zero_standard = np.median(neg_intensity)
    one_standard = np.median(pos_intensity)
    
    segs = np.linspace(zero_standard, one_standard, 4)
    interval_seg = segs[1] - segs[0]
    fl_discretized = [np.exp(-((fl - seg)/(0.8*interval_seg))**2) for seg in segs]
    fl_discretized = np.stack(fl_discretized, 2)
    fl_discretized = fl_discretized/fl_discretized.sum(2, keepdims=True)
    return fl_discretized


def adjust_contrast(pair_dat, mask, position_code=None, linear_align=False):
    pc_mat = pair_dat[0]
    
    if linear_align:
        # Fit a linear model on phase contrast ~ distance to image center
        assert not position_code is None
        dist_mat = generate_dist_mat(mask, position_code)
        dist_segs = np.linspace(np.min(dist_mat[np.nonzero(dist_mat)])-1e-5,
                                np.max(dist_mat),
                                7)
        quantized_dist_mat = np.stack([dist_mat > seg for seg in dist_segs], 2).sum(2)

        dists = list((dist_segs[:-1] + dist_segs[1:])/2)
        ms = [np.mean(pc_mat[np.where(quantized_dist_mat == i)]) for i in range(1, 7)]
        stds = [np.std(pc_mat[np.where(quantized_dist_mat == i)]) for i in range(1, 7)]

        ms_fit = Ridge(alpha=5e5)
        ms_fit.fit(np.array(dists).reshape((-1, 1)), np.array(ms).reshape((-1, 1)))
        std_fit = Ridge(alpha=5e5)
        std_fit.fit(np.array(dists).reshape((-1, 1)), np.array(stds).reshape((-1, 1)))

        all_dist_vals = np.unique(dist_mat).reshape((-1, 1))
        ms_all_fit = ms_fit.predict(all_dist_vals)
        std_all_fit = std_fit.predict(all_dist_vals)

        mean_mat = ms_fit.predict(dist_mat.reshape((-1, 1))).reshape(dist_mat.shape)
        std_mat = std_fit.predict(dist_mat.reshape((-1, 1))).reshape(dist_mat.shape)
    else:
        mean_mat = np.ones_like(pc_mat) * np.mean(pc_mat.flatten())
        std_mat = np.ones_like(pc_mat) * np.std(pc_mat.flatten())
    
    pc_adjusted = mask * (pc_mat - mean_mat)/std_mat
    return pc_adjusted


def generate_weight(mask, position_code):
    dist_mat = generate_dist_mat(mask, position_code)
    weight_constant = (dist_mat < 1100) * 1
    weight_edge = np.clip(1900 - dist_mat, 0, 800)/800 * (dist_mat >= 1100)
    weight = weight_constant + weight_edge
    assert np.all(weight <= 1)
    return weight * mask


def preprocess(dats):
    Xs = []
    ys = []
    ws = []
    names = []
    for pair, pair_dat in dats.items():
        pair_dat = dats[pair]
        position_code = pair[0].split('/')[-1].split('_')[3]
        if position_code in ['1', '3', '7', '9']:
            mask = generate_mask(pair_dat)
        else:
            mask = np.ones_like(pair_dat[0])
        pc_adjusted = adjust_contrast(pair_dat, mask, position_code, linear_align=True)
        weight = generate_weight(mask, position_code)
        
        fluorescence = generate_fluorescence_labels(pair_dat, mask)
        # discretized_fl = quantize_fluorescence(pair_dat, mask)
        
        Xs.append(pc_adjusted)
        ys.append(fluorescence)
        ws.append(weight)
        names.append(pair[0])
        if len(names) % 100 == 0:
            print("featurized %d inputs" % len(names))
    return Xs, ys, ws, names


"""
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
"""


if __name__ == '__main__':
  dat_fs = os.listdir('data')
  for f_name in dat_fs:
    f_name = f_name.split('.')[0]
    if not 'processed' in f_name:
      print(f_name)
      dats = pickle.load(open('./data/%s.pkl' % f_name, 'rb'))
      if f_name.startswith('ex2'):
        wells = set(well_id(k) for k in dats)
        for w in wells:
          try:
            well_dats = {k:v for k,v in dats.items() if well_id(k) == w}
            processed_dats = preprocess(well_dats)
            with open('./data/%s_processed_%s.pkl' % (f_name, w), 'wb') as f:
              pickle.dump(processed_dats, f)
          except Exception as e:
            print(e)
            continue
      else:
        pos_code = '5'
        try:
          pos_code_dats = {k:v for k,v in dats.items() if position_code(k) == pos_code}
          processed_dats = preprocess(pos_code_dats)
          with open('./data/%s_processed_%s.pkl' % (f_name, pos_code), 'wb') as f:
            pickle.dump(processed_dats, f)
        except Exception as e:
          print(e)
          continue
