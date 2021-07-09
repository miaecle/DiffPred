#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:31:55 2021

@author: zqwu
"""
import os
import pickle
import numpy as np
import pandas as pd
from functools import partial
import cv2
import matplotlib.pyplot as plt


from segment_support import convolve2d, quantize_fluorescence
from data_loader import load_all_pairs, get_identifier, get_fl_stats, load_image_pair, load_image
from data_assembly import preprocess, extract_samples_for_inspection, merge_dataset_soft


RAW_FOLDERS = [
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex1_new',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex3_new',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex4_new',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex5_new',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex6_new',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex7_new',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex8',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex2',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex4',
]

INTERMEDIATE_FOLDERS = [
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex1/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex3/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex4/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex5/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex6/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex7/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex8/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line3_TNNI/ex2/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line3_TNNI/ex4/0-to-0/',
]

WELL_SETTINGS = {
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex1_new': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex3_new': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex4_new': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex5_new': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex6_new': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex7_new': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex8': '6well-15',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex2': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex4': '6well-14',
}

# scale and offset parameters for raw fl preprocess
FL_PREPROCESS_SETTINGS = {
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex1_new': (0.7, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex3_new': (0.8, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex4_new': (0.7, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex5_new': (0.5, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex6_new': (0.6, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex7_new': (0.7, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex8': (1.8, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex2': (4.0, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex4': (2.5, 0.0),
}

FL_STATS = {
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex1_new': (35869, 8945),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex3_new': (29330, 8233),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex4_new': (28628, 9162),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex5_new': (46583, 12002),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex6_new': (47762, 10558),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex7_new': (43362, 9402),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex8': (10813, 4208),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex2': (3990, 1422),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex4': (4290, 2944),
}

RAW_F_FILTER = lambda f: not 'bkp' in f


def PREPROCESS_FILTER(pair, well_setting='96well-3'):
    # Remove samples without phase contrast
    if pair[0] is None:
        return False
    # Remove samples with inconsistent id
    if pair[1] is not None and get_identifier(pair[0]) != get_identifier(pair[1]):
        return False
    # Remove corner samples
    if well_setting == '6well-15':
        if get_identifier(pair[0])[-1] in \
            ['1', '2', '16', '14', '15', '30', '196', '211', '212', '210', '224', '225']:
            return False
    elif well_setting == '6well-14':
        if get_identifier(pair[0])[-1] in \
            ['1', '2', '15', '13', '14', '28', '169', '183', '184', '182', '195', '196']:
            return False
    elif well_setting == '96well-3':
        if get_identifier(pair[0])[-1] in \
            ['1', '3', '7', '9']:
            return False
    return True


def FL_PREPROCESS(fl, scale=1., offset=0.):
    if fl is None:
        return None 
    fl = fl.astype(float)
    _fl = fl * scale + offset
    _fl = np.clip(_fl, 0, 65535).astype(int).astype('uint16')
    return _fl

# %% Featurize each experiment
for raw_dir, inter_dir in zip(RAW_FOLDERS, INTERMEDIATE_FOLDERS):
    os.makedirs(inter_dir, exist_ok=True)
    
    well_setting = WELL_SETTINGS[raw_dir]
    preprocess_filter = partial(PREPROCESS_FILTER, well_setting=well_setting)
    
    fl_preprocess_setting = FL_PREPROCESS_SETTINGS[raw_dir]
    fl_preprocess_fn = partial(FL_PREPROCESS, 
                               scale=fl_preprocess_setting[0],
                               offset=fl_preprocess_setting[1])
    fl_stat = FL_STATS[raw_dir]
    fl_stat = (fl_stat[0] * fl_preprocess_setting[0] + fl_preprocess_setting[1],
               fl_stat[1] * fl_preprocess_setting[0])
    fl_nonneg_thr = fl_stat[0] + fl_stat[1]

    pairs = load_all_pairs(path=raw_dir, check_valid=RAW_F_FILTER)
    
    preprocess(pairs, 
               output_path=inter_dir, 
               preprocess_filter=preprocess_filter,
               target_size=(384, 288),
               labels=['discrete', 'continuous'], 
               raw_label_preprocess=fl_preprocess_fn,
               nonneg_thr=fl_nonneg_thr,
               well_setting=well_setting,
               linear_align=False,
               shuffle=True,
               seed=123)

# %% Select samples for manual check
for raw_dir, inter_dir in zip(RAW_FOLDERS, INTERMEDIATE_FOLDERS):
    image_output_dir = inter_dir.replace('/0-to-0/', '/sample_figs/')
    
    well_setting = WELL_SETTINGS[raw_dir]
    preprocess_filter = partial(PREPROCESS_FILTER, well_setting=well_setting)
    pairs = load_all_pairs(path=raw_dir, check_valid=RAW_F_FILTER)
    pairs = [p for p in pairs if p[0] is not None and preprocess_filter(p)]
    
    extract_samples_for_inspection(pairs, inter_dir, image_output_dir, seed=123)
    
# %% Merge datasets
output_dir = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN_READY/0-to-0/'
os.makedirs(output_dir, exist_ok=True)
merge_dataset_soft(inter_dir, output_dir, shuffle=True, seed=123)
    
