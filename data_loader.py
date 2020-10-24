import os
import numpy as np
import sys
import difflib
import tifffile
import cv2
import pickle

CHANNEL_MAX = 65535


def n_diff(s1, s2):
    ct = 0
    for s in difflib.ndiff(s1, s2):
        if s[0] != ' ':
            ct += 1
    return ct


def get_all_files(path='predict_gfp_raw'):
    fs = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        fs.extend([os.path.join(dirpath, f) for f in filenames if '.tif' in f])
    return fs


def load_all_pairs(path='predict_gfp_raw'):
    fs = get_all_files(path=path)
    pcs = sorted([f for f in fs if 'Phase' in f])
    gfps = sorted([f for f in fs if not 'Phase' in f and 'FP' in f])

    exclusions = []
    pc_file_mapping = {}
    for f in pcs:
        identifier = get_ex_day(f) + get_well(f)
        if identifier in pc_file_mapping:
            exclusions.append(identifier)
        pc_file_mapping[identifier] = f

    gfp_file_mapping = {}
    for f in gfps:
        identifier = get_ex_day(f) + get_well(f)
        if identifier in gfp_file_mapping:
            exclusions.append(identifier)
        gfp_file_mapping[identifier] = f

    pairs = []
    for identifier in (gfp_file_mapping.keys() | pc_file_mapping.keys()):
        if identifier in exclusions:
            continue
        p = [None, None]
        if identifier in pc_file_mapping:
            p[0] = pc_file_mapping[identifier]
        if identifier in gfp_file_mapping:
            p[1] = gfp_file_mapping[identifier]
        pairs.append(tuple(p))
    return pairs


def load_image(f):
    try:
        img = tifffile.imread(f)
    except Exception as e:
        img = cv2.imread(f, -1)
        if len(img.shape) == 3:
            img = img[..., 0]
    if img.dtype == 'uint16':
        return img
    elif img.dtype == 'uint8':
        return (img / 255 * 65535).astype('uint16')



def load_image_pair(pair):
    dats = [load_image(f) if not f is None else None for f in pair]
    assert len(set(d.shape for d in dats if not d is None)) == 1
    return dats


def get_ex_day(name):
    n = name.split('/')
    for i, seg in enumerate(n):
      if seg.startswith('ex'):
        ex_id = seg
        break
    detail = n[i+1]
    detail = detail.replace('-', ' ').replace('_', ' ')
    ds_sep = detail.split()
    day = None
    for d in ds_sep:
        if d.startswith('D'):
            day = d
            break
    if day is None:
        day = 'Dunknown'
    return (ex_id, day)


def get_well(f):
    f = f.split('/')[-1].split('.')[0]
    f = f.split('_')
    return (f[0], f[3])



if __name__ == '__main__':
    RAW_DATA_PATH = '../iPSC_data'
    SAVE_PATH = '/oak/stanford/groups/jamesz/zqwu/iPSC_data'

    pairs = load_all_pairs(path=os.path.join(RAW_DATA_PATH, 'predict_gfp_raw'))
    ex_day_groups = set(get_ex_day(p[0]) if not p[0] is None else get_ex_day(p[1]) for p in pairs)
    for g in ex_day_groups:
        save_file_name = os.path.join(SAVE_PATH, '%s_%s.pkl' % g)
        if os.path.exists(save_file_name):
            continue
        group_pairs = [p for p in pairs if (get_ex_day(p[0]) if not p[0] is None else get_ex_day(p[1])) == g]
        group_pair_dats = {p: load_image_pair(p) for p in group_pairs}
        with open(save_file_name, 'wb') as f:
            pickle.dump(group_pair_dats, f)

    pairs = load_all_pairs(path=os.path.join(RAW_DATA_PATH, 'predict_diff_raw'))
    ex_day_groups = set(get_ex_day(p[0]) if not p[0] is None else get_ex_day(p[1]) for p in pairs)
    for g in ex_day_groups:
        save_file_name = os.path.join(SAVE_PATH, '%s_%s.pkl' % g)
        if os.path.exists(save_file_name):
            continue
        group_pairs = [p for p in pairs if (get_ex_day(p[0]) if not p[0] is None else get_ex_day(p[1])) == g]
        group_pair_dats = {p: load_image_pair(p) for p in group_pairs}
        with open(save_file_name, 'wb') as f:
            pickle.dump(group_pair_dats, f)