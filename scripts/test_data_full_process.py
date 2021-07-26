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

from data_loader import load_all_pairs, get_identifier, load_image_pair, load_image
from data_assembly import preprocess
from data_generator import CustomGenerator

RAW_FOLDERS = [
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_477/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_202/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_20/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_142/ex1',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_273/ex2',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_839/ex1',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_480/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_854/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_975/ex0',
]

OUTPUT_FOLDERS = [
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_477/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_202/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_20/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_142/ex1/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_273/ex2/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_839/ex1/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_480/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_854/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_975/ex0/0-to-0/',
]

WELL_SETTINGS = {
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_477/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_202/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_20/ex0': '6well-15',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_142/ex1': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_273/ex2': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_839/ex1': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_480/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_854/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_975/ex0': '6well-14',
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

# %% Featurize each experiment
for raw_dir, inter_dir in zip(RAW_FOLDERS, OUTPUT_FOLDERS):
    os.makedirs(inter_dir, exist_ok=True)
    
    well_setting = WELL_SETTINGS[raw_dir]
    preprocess_filter = partial(PREPROCESS_FILTER, well_setting=well_setting)
    
    pairs = load_all_pairs(path=raw_dir, check_valid=RAW_F_FILTER)
    
    preprocess(pairs, 
               output_path=inter_dir, 
               preprocess_filter=preprocess_filter,
               target_size=(384, 288),
               labels=[], 
               well_setting=well_setting,
               linear_align=False,
               shuffle=True,
               seed=123)


# %% Check invalid entries and remove
kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': None,
    'segment_class_weights': None,
    'segment_extra_weights': None,
    'segment_label_type': 'discrete',
    'n_classify_classes': None,
    'classify_class_weights': None,
    'classify_label_type': 'discrete',
}
for output_path in OUTPUT_FOLDERS:
    print("Checking %s" % output_path)
    n_fs = len([f for f in os.listdir(output_path) if f.startswith('X_') and f.endswith('.pkl')])
    X_filenames = [os.path.join(output_path, 'X_%d.pkl' % i) for i in range(n_fs)]
    name_file = os.path.join(output_path, 'names.pkl')

    test_gen = CustomGenerator(
        name_file,
        X_filenames, 
        augment=False,
        batch_with_name=True,
        **kwargs)

    X_valid = []
    for i in test_gen.selected_inds:
        try:
            X, _, _, name = test_gen.load_ind(i)
        except Exception as e:
            print(e)
            print("ISSUE %d" % i)
            continue
        if not X is None:
            X_valid.append(i)

    if len(X_valid) < len(test_gen.selected_inds):
        print("Found invalid entries, saving corrected dataset")
        corrected_output_path = output_path.replace('/0-to-0/', '/0-to-0_corrected/')
        os.makedirs(corrected_output_path, exist_ok=True)
        test_gen.reorder_save(np.array(X_valid),
                              save_path=corrected_output_path,
                              write_segment_labels=False,
                              write_classify_labels=False)