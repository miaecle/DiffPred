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
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex12',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex13',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex15',
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
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex12/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex13/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex14/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/INTERMEDIATE/line1_3R/ex15/0-to-0/',
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
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex12': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex13': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex14': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex15': '6well-14',
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
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex12': (3.6, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex13': (2.8, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex14': (3.0, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex15': (3.0, 0.0),
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
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex12': (5002, 1683),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex13': (5875, 2333),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex14': (5735, 1951),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex15': (5436, 1980),
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
output_dir = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-0/'
os.makedirs(output_dir, exist_ok=True)
merge_dataset_soft(inter_dir, output_dir, shuffle=True, seed=123)




# %% Extract datasets for 0-to-0 and 0-to-inf training
root = output_dir
name_file = os.path.join(root, "names.pkl")
X_ct = len([f for f in os.listdir(root) if f.startswith('X_')])
X_files = [os.path.join(root, "X_%d.pkl" % i) for i in range(X_ct)]

segment_y_files = [os.path.join(root, "segment_continuous_y_%d.pkl" % i) for i in range(X_ct)]
segment_w_files = [os.path.join(root, "segment_continuous_w_%d.pkl" % i) for i in range(X_ct)]
classify_label_file = os.path.join(root, "classify_continuous_labels.pkl")
base_dataset = CustomGenerator(
    name_file,
    X_files, 
    segment_y_files=segment_y_files, 
    segment_w_files=segment_w_files,
    n_segment_classes=4,
    segment_class_weights=[1, 1, 1, 1],
    segment_extra_weights=None,
    segment_label_type='continuous',
    classify_label_file=classify_label_file,
    n_classify_classes=4,
    classify_class_weights=[1, 1, 1, 1],
    classify_label_type='continuous',
    sample_per_file=100,
    cache_file_num=5)

# %% Save for 0-to-0 segmentation
def check_valid_for_0_to_0_training(i):
    try:
        X, y, w, name = base_dataset.load_ind(i)
    except Exception as e:
        print(e)
        print("ISSUE %d" % i)
        return False
    # "Use data after day 7 onwards"
    if int(get_identifier(name)[2]) < 7:
        return False
    if (X is None) or \
       (y is None) or \
       (w is None) or \
       (base_dataset.classify_y[i] is None) or \
       (base_dataset.classify_w[i] is None):
        return False    
    if np.all(w == 0) or (base_dataset.classify_w[i] == 0):
        return False
    return True


selected_inds = [i for i in base_dataset.selected_inds if check_valid_for_0_to_0_training(i)]
selected_inds = np.array(sorted(selected_inds))

with open(root.replace("/0-to-0/", "/0-to-0_continuous_inds.pkl"), "wb") as f:
    pickle.dump(selected_inds, f)
print("TOTAL samples: %d" % len(selected_inds))

np.random.seed(123)
np.random.shuffle(selected_inds)
save_path = root.replace("/0-to-0/", "/0-to-0_continuous/")
os.makedirs(save_path, exist_ok=True)
base_dataset.reorder_save(selected_inds, 
                          save_path=save_path,
                          write_segment_labels=True,
                          write_classify_labels=True)



# %% Save for 0-to-inf classification
def check_valid_for_0_to_inf_training(i):
    try:
        X, y, w, name = base_dataset.load_ind(i)
    except Exception as e:
        print(e)
        print("ISSUE %d" % i)
        return False, False

    if X is None:
        return False, False

    source_flag = True
    target_flag = True

    # "Source data from day 3 to 12, Target data from day 8 onwards"
    if int(get_identifier(name)[2]) < 8:
        target_flag = False
    if int(get_identifier(name)[2]) < 3 or int(get_identifier(name)[2]) > 12:
        source_flag = False
    if (y is None) or \
       (w is None) or \
       (base_dataset.classify_y[i] is None) or \
       (base_dataset.classify_w[i] is None):
        target_flag = False
    if np.all(w == 0) or (base_dataset.classify_w[i] == 0):
        target_flag = False
    return source_flag, target_flag


flags = {i: check_valid_for_0_to_inf_training(i) for i in base_dataset.selected_inds}


valid_wells = sorted(set([get_identifier(base_dataset.names[i])[:2] + get_identifier(base_dataset.names[i])[3:] for i in flags if flags[i][1]]))
id_mapping = {i: get_identifier(base_dataset.names[i]) for i in flags}

def get_pairs(inds, label=1, startday_range=(4, 12)):
    well_labels = [base_dataset.classify_y[i] for i in inds]
    well_weights = [base_dataset.classify_w[i] for i in inds]

    for i in range(len(well_labels)):
        if well_weights[i] > 0 and well_labels[i] == label:
            break
    if well_weights[i] > 0 and well_labels[i] == label:
        end_ind = inds[i]
        if int(id_mapping[end_ind][2]) < 10:
            return []
        start_inds = [ind for ind in inds if \
            int(id_mapping[ind][2]) >= startday_range[0] and \
            int(id_mapping[ind][2]) <= startday_range[1] and \
            int(id_mapping[ind][2]) <= int(id_mapping[end_ind][2]) - 3]
        return [(i, end_ind) for i in start_inds]
    else:
        return []

quest_pairs = []
extra_pairs = []
for well in valid_wells:
    related_inds = [i for i in X_valid if id_mapping[i][:2] + id_mapping[i][3:] == well]
    related_inds = [i for i in related_inds if int(id_mapping[i][2]) <= 18]
    related_inds = sorted(related_inds, key=lambda x: -int(id_mapping[x][2]))

    well_labels = [base_dataset.classify_y[i] for i in related_inds]
    well_weights = [base_dataset.classify_w[i] for i in related_inds]

    if not 1 in well_labels:
        quest_pairs.extend(get_pairs(related_inds, label=0, startday_range=(4, 12)))
        extra_pairs.extend(get_pairs(related_inds, label=0, startday_range=(0, 3)))
    else:
        _well_labels = [lab for i, lab in enumerate(well_labels) if not lab is None and well_weights[i] > 0]
        ct = [not (_well_labels[i] == _well_labels[i+1]) for i in range(len(_well_labels) - 1)]
        ct = sum(ct)
        if ct <= 1:
            quest_pairs.extend(get_pairs(related_inds, label=1, startday_range=(4, 12)))
            extra_pairs.extend(get_pairs(related_inds, label=1, startday_range=(0, 3)))
        elif _well_labels[0] == 1 and sum(_well_labels) > 2:
            quest_pairs.extend(get_pairs(related_inds, label=1, startday_range=(4, 12)))
            extra_pairs.extend(get_pairs(related_inds, label=1, startday_range=(0, 3)))
        elif _well_labels[1] == 1 and sum(_well_labels) > 3:
            quest_pairs.extend(get_pairs(related_inds, label=1, startday_range=(4, 12)))
            extra_pairs.extend(get_pairs(related_inds, label=1, startday_range=(0, 3)))
        elif sum(_well_labels) < 2:
            quest_pairs.extend(get_pairs(related_inds, label=0, startday_range=(4, 12)))
            extra_pairs.extend(get_pairs(related_inds, label=0, startday_range=(0, 3)))
        else:
            print("Exclude well %s" % str(well))