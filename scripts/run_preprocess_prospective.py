import os
import csv
import numpy as np
from data_generator import binarized_fluorescence_label
from data_loader import *
from segment_support import *


def load_assemble_test_data(data_path, dataset_path):
    # Load data under the given path
    print("Loading Data")
    fs = get_all_files(data_path)
    fs = [f for f in fs if not str(get_well(f)[1]) in ['1', '2', '16', '14', '15', '30', '196', '211', '212', '210', '224', '225']]
    fs = [f for f in fs if 'Phase' in f]
    fs_pair = [(f, None) for f in fs]
    print("Number of input images: %d" % len(fs_pair))
    pair_dats = {pair: load_image_pair(pair) for pair in fs_pair}

    # Preprocessing
    print("Start Preprocessing")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    processed_dats = preprocess(pair_dats, linear_align=False)
    _ = assemble_for_training(processed_dats, 
                              (384, 288), 
                              save_path=dataset_path,
                              label='segmentation')
    del processed_dats


def load_test_data_labels(data_path, label_save_path):
    fs = get_all_files(data_path)
    ex, day = get_ex_day(data_path)
    fs = [f for f in fs if not get_well(f)[1] in ['1', '2', '16', '14', '15', '30', '196', '211', '212', '210', '224', '225']]
    pc_fs = [f for f in fs if 'Phase' in f]
    fp_fs = [f for f in fs if 'FP' in f]
    assert len(pc_fs) == len(fp_fs)
    fs_pair = list(zip(pc_fs, fp_fs))
    pair_dats = {pair: load_image_pair(pair) for pair in fs_pair}

    fl_labels = {}
    binarized_labels = {}
    target_size = (384, 288)
    for i, pair in enumerate(fs_pair):
        if i%100 == 0:
            print(i)
        key = get_ex_day(pair[0]) + get_well(pair[0])
        if key in fl_labels:
            continue
        mask = np.ones_like(pair_dats[pair][0])
        fl_label = generate_fluorescence_labels(pair_dats[pair], mask)

        _y = cv2.resize(fl_label, target_size)
        _y[np.where((_y > 0) & (_y < 1))] = 1
        _y[np.where((_y > 1) & (_y < 2))] = 1
        fl_labels[key] = _y.astype(int)

        _w = (_y != 1) * 1
        _y[np.where(_y == 1)] = 0
        _y[np.where(_y == 2)] = 1
        labels[key] = binarized_fluorescence_label((_y, _w))

    with open(os.path.join(label_save_path, '%s_%s_labels.pkl' % (ex, day)), 'wb') as f:
        pickle.dump(labels, f)

    with open(os.path.join(label_save_path, '%s_%s_labels.csv' % (ex, day)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ex', 'day', 'well', 'well_position', 'label', 'validity'])
        for k in labels:
            writer.writerow([k[0], k[1], k[2], k[3], labels[k][0], labels[k][1]])

    with open(os.path.join(label_save_path, '%s_%s_fl.pkl' % (ex, day)), 'wb') as f:
        pickle.dump(fl_labels, f)


if __name__ == '__main__':
    folder_paths = os.listdir('/scratch/users/zqwu/iPSC_data/prospective/cms/ex1/')
    label_save_path = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/prospective/ex1/'
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)
    for folder in sorted(folder_paths):
        print(folder)
        raw_path = '/scratch/users/zqwu/iPSC_data/prospective/cms/ex1/%s' % folder
        save_path = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/prospective/ex1/%s/' % folder

        ex, day = get_ex_day(raw_path)
        _ = load_assemble_test_data(raw_path, save_path)

        if int(day[1:]) >= 8 and int(day[1:]) <= 17:
            load_test_data_labels(raw_path, label_save_path)