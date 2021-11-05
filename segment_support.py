#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:35:16 2020

@author: zqwu
"""
import os
import copy
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

from data_loader import get_identifier, load_image_pair


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

    if np.sum(positives) < 1200*900*0.005:
        # No significant differentiation detected
        negatives = mask - np.sign(convolve2d(positives, np.ones((3, 3)), mode='same'))
        positives = np.zeros_like(mask)
    else:
        # Average fluorescence for negative/positive pixels
        neg_median = np.quantile(fl[np.where(negatives)], 0.5)
        pos_median = np.quantile(fl[np.where(positives)], 0.5)
        if not pos_median > neg_median:
            print("Error in labeling")
            return None
        pos_thr = pos_median - 0.3 * (pos_median - neg_median)
        neg_thr = neg_median + 0.3 * (pos_median - neg_median)

        # Assign uncovered pixels to negative if the gradient is not high and fluorescence signal is low
        positives = np.sign(positives + (fl > pos_thr) * mask)
        negatives = np.sign((negatives + (fl < neg_thr) * mask) * (1 - positives))
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

    blobs = measure.label(negatives, background=0)
    for blob_id, ct in zip(*np.unique(blobs, return_counts=True)):
        if blob_id == 0:
            continue
        elif ct < 200:
            negatives[np.where(blobs == blob_id)] = 0

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

        ms_fit = Ridge(alpha=5e4)
        ms_fit.fit(np.array(dists).reshape((-1, 1)), np.array(ms).reshape((-1, 1)))
        std_fit = Ridge(alpha=5e4)
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
    pc_adjusted = np.clip(pc_adjusted, -3, 3)
    return pc_adjusted


def generate_weight(mask, position_code, linear_align=True):
    if linear_align:
        dist_mat = generate_dist_mat(mask, position_code)
    else:
        dist_mat = np.zeros_like(mask)
    weight_constant = (dist_mat < 1100) * 1
    weight_edge = np.clip(2300 - dist_mat, 0, 1200)/1200 * (dist_mat >= 1100)
    weight = weight_constant + weight_edge
    assert np.all(weight <= 1)
    return weight * mask


def binarized_fluorescence_label(y, w):
    if y is None:
        return None, 0
    if isinstance(y, np.ndarray):
        y_ct = np.where(y > 0)[0].size
        invalid_ct = np.where(np.sign(w) == 0)[0].size
    elif np.all(int(y) == y):
        y_ct = y
        invalid_ct = w
    else:
        raise ValueError("Data type not supported")
    if y_ct > 500:
        sample_y = 1
        sample_w = 1
    elif y_ct == 0 and invalid_ct < 600:
        sample_y = 0
        sample_w = 1
    else:
        sample_y = 0
        sample_w = 0
    return sample_y, sample_w


def plot_sample_labels(pairs, save_dir='.', raw_label_preprocess=lambda x: x, linear_align=False):
    all_views = list(set(get_identifier(p[0])[3:] for p in pairs if p[0] is not None))
    selected_views = set([all_views[i] for i in np.random.choice(np.arange(len(all_views)), (20,), replace=False)])

    data = {}
    for view in selected_views:
        view_pairs = [p for p in pairs if p[0] is not None and p[1] is not None and get_identifier(p[0])[3:] == view]
        print(view)
        for p in view_pairs:
            identifier = get_identifier(p[0])
            day = str(identifier[2])
            name = '_'.join(identifier)
            save_path = os.path.join(save_dir, day, name)
            os.makedirs(os.path.join(save_dir, day), exist_ok=True)

            pair_dat = load_image_pair(p)
            pair_dat = [pair_dat[0], raw_label_preprocess(pair_dat[1])]
            data[identifier] = pair_dat

            plt.clf()
            plt.imshow(pair_dat[1].astype(float))
            plt.savefig(save_path+'_fl.png')

            position_code = identifier[-1]
            if linear_align and position_code in ['1', '3', '7', '9']:
                mask = generate_mask(pair_dat)
            else:
                mask = np.ones_like(pair_dat[0])

            discrete_y = generate_fluorescence_labels(pair_dat, mask)
            if discrete_y is None:
                print("ERROR in labeling %s" % name)
                continue

            plt.clf()
            plt.imshow(discrete_y.astype(float), vmin=0, vmax=2)
            plt.savefig(save_path+'_fl_discrete.png')
    with open(os.path.join(save_dir, "data.pkl"), "wb") as f:
        pickle.dump(data, f)
    return

