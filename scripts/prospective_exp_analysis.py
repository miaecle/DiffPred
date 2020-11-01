import os
import numpy as np
import pandas as pd
import pickle
import csv
from data_loader import *
from segment_support import *
from data_generator import CustomGenerator
from models import ClassifyOnSegment
from predict import load_assemble_test_data, predict_on_test_data


DATA_ROOT = '/home/zqwu/iPSC/data/prospective/ex1/'

### Run prediction ###
days = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
model_paths = ['/home/zqwu/iPSC/model_save/%s' % path for path in ['pspnet_random_0-to-10_1.model',
                                                                   'pspnet_random_0-to-10_2.model',
                                                                   'pspnet_ex67_0-to-10_0.model',
                                                                   'pspnet_random_0-to-inf_1.model',
                                                                   'pspnet_random_0-to-inf_2.model']]
output_path_root = DATA_ROOT

for day in days:
    data_path = os.path.join(DATA_ROOT, 'az6well_cm_D%d_prospective' % day)
    dataset_path = os.path.join(data_path, 'merged') + '/'
    load_assemble_test_data(data_path, dataset_path)
    for model_path in model_paths:
        model_name = os.path.split(model_path)[-1].split('.')[0]
        output_path = os.path.join(output_path_root, 'D%d_pred_%s.csv' % (day, model_name))
        predict_on_test_data(dataset_path, model_path, output_path)





### Generate labels from GFP images ###
day = 12

label_data_path = os.path.join(DATA_ROOT, 'az6well_cm_D%d_prospective' % day)
label_save_path = DATA_ROOT

fs = get_all_files(label_data_path)
fs = [f for f in fs if not str(get_well(f)[1]) in ['1', '2', '16', '14', '15', '30', '196', '211', '212', '210', '224', '225']]
pc_fs = [f for f in fs if 'Phase' in f]
fp_fs = [f for f in fs if 'FP' in f]
pc_fs = sorted(pc_fs, key=lambda x: get_ex_day(x) + get_well(x))
fp_fs = sorted(fp_fs, key=lambda x: get_ex_day(x) + get_well(x))

fs_pair = list(zip(pc_fs, fp_fs))
pair_dats = {pair: load_image_pair(pair) for pair in fs_pair}

fl_labels = {}
for i, pair in enumerate(fs_pair):
    key = get_ex_day(pair[0]) + get_well(pair[0])
    mask = np.ones_like(pair_dats[pair][0])
    fl_label = generate_fluorescence_labels(pair_dats[pair], mask)
    fl_labels[key] = fl_label

with open(os.path.join(label_save_path, 'D%d_fl.pkl' % day), 'wb') as f:
    pickle.dump(fl_labels, f)

target_size = (384, 288)
labels = {}
for key in fl_labels:
    _y = cv2.resize(fl_labels[key], target_size)
    _w = np.ones_like(_y)
    _y[np.where((_y > 0) & (_y < 1))] = 1
    _y[np.where((_y > 1) & (_y < 2))] = 1
    _w[np.where(_y == 1)] = 0
    _y[np.where(_y == 1)] = 0
    _y[np.where(_y == 2)] = 1
    labels[key] = binarized_fluorescence_label((_y, _w))

with open(os.path.join(label_save_path, 'D%d_labels.pkl' % day), 'wb') as f:
    pickle.dump(labels, f)

with open(os.path.join(label_save_path, 'D%d_labels.csv' % day), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['ex', 'day', 'well', 'well_position', 'label', 'validity'])
    for k in labels:
        writer.writerow([k[0], k[1], k[2], k[3], labels[k][0], labels[k][1]])



### Metrics ###
day = 10
label_save_path = DATA_ROOT
pred_save_path = DATA_ROOT

labels = pd.read_csv(os.path.join(label_save_path, 'D%d_labels.csv' % day))
label_dict = {}
for r in np.array(labels):
    label_dict[tuple(r[2:4].astype(str))] = r[4:]
ks = [k for k in label_dict if label_dict[k][1] == 1]

pred_fs = [f for f in os.listdir(pred_save_path) if 'pred' in f and f.endswith('.csv')]
for f in sorted(pred_fs):
    preds = pd.read_csv(f)
    preds_dict = {}
    for r in np.array(preds):
        preds_dict[tuple(r[2:4].astype(str))] = r[4:]

    y_true = np.array([label_dict[k][0] for k in ks if k in preds_dict])
    y_pred = np.array([preds_dict[k] for k in ks if k in preds_dict])

    pos_preds = [preds_dict[k] for k in preds_dict if label_dict[k][0] == 1 and label_dict[k][1] == 1]
    neg_preds = [preds_dict[k] for k in preds_dict if label_dict[k][0] == 0 and label_dict[k][1] == 1]

    print(f)
    print("\t%.3f" % roc_auc_score(y_true, y_pred[:, 0]))
    print("\t%.3f\t%.3f" % (np.mean(pos_preds), np.std(pos_preds)))
    print("\t%.3f\t%.3f" % (np.mean(neg_preds), np.std(neg_preds)))



### Plot prediction ###
day = 10
label_save_path = DATA_ROOT
pred_save_path = DATA_ROOT

labels = pd.read_csv(os.path.join(label_save_path, 'D%d_labels.csv' % day))
well_labels = {}
for r in np.array(labels):
    well = r[2]
    site = int(r[3])
    if not well in well_labels:
        well_labels[well] = np.zeros((15, 15))
    if r[5] == 1:
        well_labels[well][site//15, site%15] = r[4]
    else:
        well_labels[well][site//15, site%15] = 0.5


pred_fs = [f for f in os.listdir(pred_save_path) if 'pred' in f and f.endswith('.csv')]
for f in sorted(pred_fs):
    preds = pd.read_csv(f)
    preds_dict = {}
    for r in np.array(preds):
        preds_dict[tuple(r[2:4].astype(str))] = r[4:]

    y_true = np.array([label_dict[k][0] for k in ks if k in preds_dict])
    y_pred = np.array([preds_dict[k] for k in ks if k in preds_dict])

    pos_preds = [preds_dict[k] for k in preds_dict if label_dict[k][0] == 1 and label_dict[k][1] == 1]
    neg_preds = [preds_dict[k] for k in preds_dict if label_dict[k][0] == 0 and label_dict[k][1] == 1]

    print(f)
    print("\t%.3f" % roc_auc_score(y_true, y_pred[:, 0]))
    print("\t%.3f\t%.3f" % (np.mean(pos_preds), np.std(pos_preds)))
    print("\t%.3f\t%.3f" % (np.mean(neg_preds), np.std(neg_preds)))
