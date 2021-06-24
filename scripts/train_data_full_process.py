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
from data_assembly import preprocess, extract_samples_for_inspection


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
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex1_new': '96well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex3_new': '96well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex4_new': '96well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex5_new': '96well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex6_new': '96well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex7_new': '96well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex8': '6well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex2': '6well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex4': '6well',
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

"""
Overall SDs
/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex1_new
8945.947214908601
/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex3_new
8233.19607771444
/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex4_new
9162.170954396646
/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex5_new
12002.488909666301
/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex6_new
10558.256643280603
/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex7_new
9402.983538731476
/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex8
4208.384351393968
/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex2
1422.8580214600352
/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex4
2944.2806001509884
"""

RAW_F_FILTER = lambda f: not 'bkp' in f


def PREPROCESS_FILTER(pair, well_setting='96well'):
    # Remove samples without phase contrast
    if pair[0] is None:
        return False
    # Remove samples with inconsistent id
    if pair[1] is not None and get_identifier(pair[0]) != get_identifier(pair[1]): #
        return False
    # Remove corner samples
    if well_setting == '6well':
        if get_identifier(pair[0])[-1] in \
            ['1', '2', '16', '14', '15', '30', '196', '211', '212', '210', '224', '225']:
            return False
    elif well_setting == '96well':
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


for raw_dir, inter_dir in zip(RAW_FOLDERS, INTERMEDIATE_FOLDERS):
    os.makedirs(inter_dir, exist_ok=True)
    well_setting = WELL_SETTINGS[raw_dir]
    fl_preprocess_setting = FL_PREPROCESS_SETTINGS[raw_dir]
    pairs = load_all_pairs(path=raw_dir, check_valid=RAW_F_FILTER)
    
    fl_preprocess_fn = partial(FL_PREPROCESS, 
                               scale=fl_preprocess_setting[0],
                               offset=fl_preprocess_setting[1])
    
    preprocess(pairs, 
               output_path=inter_dir, 
               preprocess_filter=PREPROCESS_FILTER,
               target_size=(384, 288),
               labels=['discrete', 'continuous'], 
               raw_label_preprocess=fl_preprocess_fn,
               well_setting=well_setting, #'6well' or '96well'
               linear_align=False,
               shuffle=True,
               seed=123)
    