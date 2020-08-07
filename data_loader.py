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
        fs.extend([os.path.join(dirpath, f) for f in filenames])
    return fs

def load_all_pairs(path='predict_gfp_raw'):
    pairs = []
    fs = get_all_files(path=path)
    pcs = sorted([f for f in fs if 'Phase' in f])
    gfps = sorted([f for f in fs if 'GFP' in f])
    pairs = []

    pc_file_mapping = {}
    for f in pcs:
        sep = f.split('_')
        sep[-2] = '' # Channel name
        sep[-4] = '' # Channel ID
        sep = tuple(sep)
        assert sep not in pc_file_mapping
        pc_file_mapping[sep] = f

    gfp_file_mapping = {}
    for f in gfps:
        sep = f.split('_')
        sep[-2] = '' # Channel name
        sep[-4] = '' # Channel ID
        sep = tuple(sep)
        assert sep not in gfp_file_mapping
        gfp_file_mapping[sep] = f
        if sep in pc_file_mapping:
            pairs.append((pc_file_mapping[sep], f))

    for p in pairs:
        assert n_diff(p[0].split('Phase')[0], p[1].split('GFP')[0]) == 2

    unmatched_pc = len(pc_file_mapping) - len(pairs)
    unmatched_gfp = len(gfp_file_mapping) - len(pairs)
    print("Unmatched files: %d Phase Contrast, %d GFP" % (unmatched_pc, unmatched_gfp))
    return pairs

def load_image(f):
    return tifffile.imread(f)

def load_image_pair(pair):
    dats = [load_image(f) for f in pair]
    assert len(set(d.shape for d in dats)) == 1
    return dats

def get_well(f):
    f = f.split('/')[-1].split('.')[0]
    f = f.split('_')
    return (f[0], f[3])

def get_keys(fs):
    if fs.__class__ is str:
        return get_well(fs)
    elif fs.__class__ is tuple and len(fs) == 2:
        return get_well(fs[0])
    elif fs.__class__ is list:
        return [get_well(f[0]) for f in fs]


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


if __name__ == '__main__':
    pairs = sorted(load_all_pairs(path='predict_gfp_raw'))
    ds1 = set([tuple(p[0].split('/')[2:-1]) for p in pairs])
    ds1 = set(ds for ds in ds1 if len(ds) == 2 or (len(ds) > 2 and not ds[2].endswith('Copy')))
    mapping1 = {route: [] for route in ds1}
    for p in pairs:
        k = tuple(p[0].split('/')[2:-1])
        if k in mapping1:
            mapping1[k].append(p)

    pairs = sorted(load_all_pairs(path='predict_diff_raw'))
    ds2 = set([tuple(p[0].split('/')[2:-1]) for p in pairs])
    mapping2 = {route: [] for route in ds2}
    for p in pairs:
        mapping2[tuple(p[0].split('/')[2:-1])].append(p)

    def get_ex_day(ds):
        n = ds[1]
        n = n.replace('-', ' ').replace('_', ' ')
        ds_sep = n.split()
        day = None
        for d in ds_sep:
            if d.startswith('D'):
                day = d
        if day is None:
            day = 'Dunknown'
        return (ds[0], day)

    merged_mapping = {}
    for ds in ds1:
        name = get_ex_day(ds)
        merged_mapping[name] = mapping1[ds]

    for ds in ds2:
        if not ds in mapping1:
            name = get_ex_day(ds)
            merged_mapping[name] = mapping2[ds]

    for k in merged_mapping:
        file_name = k[0] + '_' + k[1] + '.pkl'
        if os.path.exists(file_name):
            continue
        segment_pair_dats = {p:load_image_pair(p) for p in merged_mapping[k]}
        with open(file_name, 'wb') as f:
            pickle.dump(segment_pair_dats, f)
