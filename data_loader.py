import os
import numpy as np
import difflib
import tifffile
import cv2

CHANNEL_MAX = 65535


def n_diff(s1, s2):
    """ Check edit distance between two strings """
    ct = 0
    for s in difflib.ndiff(s1, s2):
        if s[0] != ' ':
            ct += 1
    return ct


def get_all_files(path='predict_gfp_raw'):
    """ List all .png/.tif image files in the path,

    Report files with different extensions

    """
    fs = []
    other_exts = set()
    for (dirpath, dirnames, filenames) in os.walk(path):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            ext = os.path.splitext(file_path)[-1]
            if ext == '.tif' or ext == '.png':
                fs.append(file_path)
            else:
                other_exts.add(ext)
    for ext in other_exts:
        print("Found %s files" % ext)
    return fs


def load_all_pairs(path='predict_gfp_raw', check_valid=lambda x: True):
    """ List all pairs of phase contrast/GFP image files under path

    A customizable `check_valid` function could be provided
    """
    fs = get_all_files(path=path)
    fs = [f for f in fs if check_valid(f)]
    pcs = sorted([f for f in fs if 'Phase' in f])
    gfps = sorted([f for f in fs if 'Phase' not in f and 'GFP' in f])

    exclusions = []
    pc_file_mapping = {}
    for f in pcs:
        identifier = get_identifier(f)
        if identifier in pc_file_mapping:
            # If there are multiple files having the same identifier
            exclusions.append(identifier)
        pc_file_mapping[identifier] = f

    gfp_file_mapping = {}
    for f in gfps:
        identifier = get_identifier(f)
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

    print("From %s" % path)
    print("\tTotal number of valid files %d" % len(fs))
    print("\tExcluded %d" % len(exclusions))
    print("\tTotal number of pcs %d" % len(pc_file_mapping))
    print("\tTotal number of gfps %d" % len(gfp_file_mapping))
    print("\tTotal number of pairs %d" % len(pairs))
    print("\tTotal number of pairs (with gfp) %d" % len(gfp_file_mapping.keys() & pc_file_mapping.keys()))
    return pairs


def check_pairs_by_day(pairs):
    """ Count file pairs by experiment day """
    days = set([get_identifier(p[0])[2] for p in pairs])
    print("Day\tFull\tPC\tGFP")
    for day in sorted(days, key=lambda x: int(x)):
        n_pairs = len([p for p in pairs if p[0] is not None and p[1] is not None and get_identifier(p[0])[2] == day])
        n_pcs = len([p for p in pairs if p[0] is not None and get_identifier(p[0])[2] == day])
        n_gfps = len([p for p in pairs if p[1] is not None and get_identifier(p[1])[2] == day])
        print("%s\t%d\t%d\t%d" % (day, n_pairs, n_pcs, n_gfps))


def check_positions(pairs, limit_to_day=None):
    """ Check well and position identifiers """
    if limit_to_day is not None:
        pairs = [p for p in pairs if get_identifier(p[0])[2] == str(limit_to_day)]
    unique_wells = set(get_identifier(p[0])[3] for p in pairs)
    print(len(unique_wells))
    print(sorted(set(w[0] for w in unique_wells)))
    print(sorted(set(w[1:] for w in unique_wells)))
    unique_positions = set(get_identifier(p[0])[4] for p in pairs)
    print(len(unique_positions))
    if len(unique_positions) < 20:
        print(sorted(unique_positions))
    return unique_wells, unique_positions


def load_image(f):
    try:
        img = tifffile.imread(f)
    except Exception:
        img = cv2.imread(f, -1)
        assert img is not None
        if len(img.shape) == 3:
            img = img[..., 0]
    if img.dtype == 'uint16':
        return img
    elif img.dtype == 'uint8':
        return (img / 255 * 65535).astype('uint16')


def load_image_pair(pair):
    dats = [load_image(f) if f is not None else None for f in pair]
    assert len(set(d.shape for d in dats if d is not None)) == 1
    return dats


def get_fl_distri(pairs):
    fl_files = [pair[1] for pair in pairs if pair[1] is not None]
    overall_distri = {i: 0 for i in range(65536)}
    for fl_f in fl_files:
        try:
            mat = load_image(fl_f)
            vals, cts = np.unique(mat, return_counts=True)
            for val, ct in zip(vals, cts):
                overall_distri[val] += ct
        except Exception as e:
            print(e)
            print("Error loading file %s" % fl_f)
    return [overall_distri[i] for i in range(65535)]


def get_fl_stats(distri):
    assert len(distri) == 65535
    fl_mean = sum([i * num for i, num in enumerate(distri)]) / sum(distri)
    fl_var = sum([(i - fl_mean)**2 * num for i, num in enumerate(distri)]) / sum(distri)
    fl_std = np.sqrt(fl_var)
    return fl_mean, fl_std


def get_line_name(name):
    n = name.split('/')
    for i, seg in enumerate(n):
        if seg.startswith('line'):
            return seg
    return None


def get_ex_day(name):
    try:
        n = name.split('/')
        for i, seg in enumerate(n):
            if seg.startswith('ex'):
                ex_id = seg
                break
        detail = n[i + 1]
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
    except Exception:
        return (None, None)


def well_id_from_name(f):
    try:
        f = os.path.split(f)[-1].split('.')[0]
        f = f.split('_')
        return (f[0], f[3])
    except Exception:
        return (None, None)


def exp_id_from_name(f):
    exp_id = get_ex_day(f)[0]
    if exp_id is None:
        return None
    else:
        return str(exp_id)


def exp_day_from_name(f):
    try:
        return str(get_ex_day(f)[1][1:])
    except Exception:
        return None


def get_identifier(f):
    identifier = (get_line_name(f), exp_id_from_name(f), exp_day_from_name(f)) + well_id_from_name(f)
    if identifier.count(None) <= 2:
        return identifier
    else:
        print("Cannot parse file path %s" % f)
        return tuple(f.strip().strip('/').split('/'))
