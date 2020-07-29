import os
import numpy as np
import sys
import difflib
import tifffile
import cv2
import cmath
from scipy import optimize
from scipy.signal import convolve2d
from skimage import measure

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

def get_name(f):
    f = f.split('/')[-1].split('.')[0]
    f = f.split('_')
    return (f[0], f[3])

def get_keys(fs):
    if fs.__class__ is str:
        return get_name(fs)
    elif fs.__class__ is tuple and len(fs) == 2:
        return get_name(fs[0])
    elif fs.__class__ is list:
        return [get_name(f[0]) for f in fs]

def get_center(edge):
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

def generate_mask(pair_dat):
    fl = pair_dat[1] # Should be unnormalized uint16 values
    threshold = fl > 18000
    mask = cv2.blur(threshold.astype('uint8'), (10, 10))
#     dist_mat1 = np.stack([np.arange(mask.shape[0])] * mask.shape[1], 1)
#     dist_mat2 = np.stack([np.arange(mask.shape[1])] * mask.shape[0], 0)
#     dist_mat = np.stack([dist_mat1, dist_mat2], 2)

#     dist_mat = np.sqrt(((dist_mat - center.reshape((1, 1, 2)))**2).sum(2))
#     dist_mat = dist_mat * mask
    return mask

def rotate(coords, angle):
    cs = np.cos(angle)
    sn = np.sin(angle)

    x = coords[0] * cs - coords[1] * sn;
    y = coords[0] * sn + coords[1] * cs;
    return x, y

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

def quantize_fluorescence(pair_dat, mask):
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

def adjust_contrast(pair_dat, mask, position_code):
    light_center = {
        '1':
        '2':
        '3':
        '4':
        '5':
        '6':
        '7':
        '8':
        '9':
    }
    pc_mat = pair_dat[0]
    dist_mat[np.where(dist_mat > 0)]
    

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

    def get_name(ds):
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
        name = get_name(ds)
        merged_mapping[name] = mapping1[ds]

    for ds in ds2:
        if not ds in mapping1:
            name = get_name(ds)
            merged_mapping[name] = mapping2[ds]

    for k in merged_mapping:
        file_name = k[0] + '_' + k[1] + '.pkl'
        if os.path.exists(file_name):
            continue
        segment_pair_dats = {p:load_image_pair(p) for p in merged_mapping[k]}
        with open(file_name, 'wb') as f:
            pickle.dump(segment_pair_dats, f)
