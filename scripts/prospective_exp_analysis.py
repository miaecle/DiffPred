import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import numpy as np
import pandas as pd
import pickle
import csv
from sklearn.metrics import roc_auc_score
from data_loader import *
from segment_support import *
from data_generator import CustomGenerator, binarized_fluorescence_label
from models import ClassifyOnSegment
from predict import load_assemble_test_data, predict_on_test_data
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


DATA_ROOT = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/prospective/ex1/'

### Run prediction ###
days = [4, 6, 8, 10]
model_paths = ['/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/%s' % path for path in ['temp_bkp.model',
                                                                                                  'weights.90-0.55.hdf5',
                                                                                                  'weights.120-0.49.hdf5']] \
            + ['/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex67/%s' % path for path in ['weights.35-0.82.hdf5',
                                                                                                   'weights.65-0.79.hdf5',
                                                                                                   'weights.95-1.12.hdf5']]
output_path_root = DATA_ROOT

for day in days:
    data_path = os.path.join(DATA_ROOT, 'az6well_cm_D%d_prospective2' % day)
    # load_assemble_test_data(data_path, dataset_path)
    for model_path in model_paths:
        model_name = os.path.splitext(os.path.split(model_path)[-1])[0]
        output_path = os.path.join(output_path_root, 'D%d_pred_%s.csv' % (day, model_name))
        predict_on_test_data(data_path, model_path, output_path, predict_interval=20-day)





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
label_day = 16
label_save_path = DATA_ROOT
pred_save_path = DATA_ROOT

labels = pd.read_csv(os.path.join(label_save_path, 'D%d_labels.csv' % label_day))
label_dict = {}
for r in np.array(labels):
    label_dict[tuple(r[2:4].astype(str))] = r[4:]
ks = [k for k in label_dict]# if label_dict[k][1] == 1]

pred_fs = [f for f in os.listdir(pred_save_path) if 'pred' in f and f.endswith('.csv')]
model_lines = {}
for f in sorted(pred_fs):
    model_name = os.path.splitext(f)[0].split('pred_')[1]
    if not model_name in model_lines:
        model_lines[model_name] = {}

    preds = pd.read_csv(os.path.join(pred_save_path, f))
    preds_dict = {}
    for r in np.array(preds):
        preds_dict[tuple(r[2:4].astype(str))] = r[4:]


    y_true = np.array([label_dict[k][0] for k in ks if k in preds_dict])
    y_pred = np.array([preds_dict[k] for k in ks if k in preds_dict])
    pos_preds = [preds_dict[k] for k in ks if k in preds_dict and label_dict[k][0] == 1 and label_dict[k][1] == 1]
    neg_preds = [preds_dict[k] for k in ks if k in preds_dict and label_dict[k][0] == 0 and label_dict[k][1] == 1]
    day = int(f.split('_')[0][1:])
    model_lines[model_name][day] = roc_auc_score(y_true, y_pred[:, 0])
    print("%s" % f)
    print("\t%.3f" % roc_auc_score(y_true, y_pred[:, 0]))
    print("\t%.3f\t%.3f" % (np.mean(pos_preds), np.std(pos_preds)))
    print("\t%.3f\t%.3f\n" % (np.mean(neg_preds), np.std(neg_preds)))
    # with open('results.txt', 'a') as out_f:
    #     out_f.write("%s\n" % f)
    #     out_f.write("\t%.3f\n" % roc_auc_score(y_true, y_pred[:, 0]))
    #     out_f.write("\t%.3f\t%.3f\n" % (np.mean(pos_preds), np.std(pos_preds)))
    #     out_f.write("\t%.3f\t%.3f\n" % (np.mean(neg_preds), np.std(neg_preds)))


plt.clf()
for model_name in model_lines:
    x = sorted(model_lines[model_name].keys())
    if len(x) > 4:
        y = [model_lines[model_name][_x] for _x in x]
        plt.plot(x, y, '.-', label=model_name)
plt.legend()
plt.savefig('/home/zqwu/Dropbox/fig_temp/perf_line_%d.png' % label_day, dpi=300)



### Plot prediction ###
label_day = 12
pred_day = 8
use_model = 'pspnet_random_0-to-inf_2'
label_save_path = DATA_ROOT
pred_save_path = DATA_ROOT

labels = pd.read_csv(os.path.join(label_save_path, 'D%d_labels.csv' % label_day))
well_labels = {}
for r in np.array(labels):
    well = r[2]
    site = int(r[3]) - 1
    if not well in well_labels:
        well_labels[well] = -np.ones((15, 15))
    if r[5] == 1:
        well_labels[well][site//15, site%15] = r[4]
    else:
        well_labels[well][site//15, site%15] = 0.5


for well in well_labels:
    plt.clf()
    plt.imshow(well_labels[well], vmin=0., vmax=1)
    plt.savefig('/home/zqwu/Dropbox/fig_temp/Label_%d_%s.png' % (label_day, well), dpi=300)


well_preds = {}
pred_fs = [f for f in os.listdir(pred_save_path) if 'pred' in f and f.endswith('.csv')]
for f in sorted(pred_fs):
    model_name = os.path.splitext(f)[0].split('pred_')[1]
    if not model_name == use_model:
        continue
    day = int(f.split('_')[0][1:])
    if not day == pred_day:
        continue

    preds = pd.read_csv(os.path.join(pred_save_path, f))
    preds_dict = {}
    for r in np.array(preds):
        well = r[2]
        site = int(r[3]) - 1
        if not well in well_preds:
            well_preds[well] = - np.ones((15, 15))
        well_preds[well][site//15, site%15] = r[4]

for well in well_preds:
    plt.clf()
    plt.imshow(well_preds[well], vmin=0.4, vmax=1)
    plt.savefig('/home/zqwu/Dropbox/fig_temp/Pred_%d_%s_from_day%d.png' % (label_day, well, pred_day), dpi=300)






### Fluorescence translation performance ###

day = 15
data_path = os.path.join(DATA_ROOT, 'az6well_cm_D%d_prospective' % day)

kwargs = {
    'batch_size': 16,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': None,
    'segment_class_weights': [1, 3],
    'segment_extra_weights': None,
    'segment_label_type': 'segmentation',
    'n_classify_classes': None,
}

n_fs = len([f for f in os.listdir(data_path) if f.startswith('X')])
X_filenames = [os.path.join(data_path, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'names.pkl')
valid_gen = CustomGenerator(X_filenames,
                            y_filenames,
                            w_filenames,
                            name_file,
                            **kwargs)
model = ClassifyOnSegment(
    input_shape=(288, 384, 2), 
    model_structure='pspnet', 
    model_path='.', 
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)

model_path = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/0-to-0_random/bkp.model'
model.load(model_path)

fls = pickle.load(open(os.path.join(DATA_ROOT, 'D%d_fl.pkl' % day), 'rb'))
labels = pickle.load(open(os.path.join(DATA_ROOT, 'D%d_labels.pkl' % day), 'rb'))

preds = []
classify_preds = []
for batch in valid_gen:
    pred = model.predict(batch)
    preds.append(pred[0])
    classify_preds.append(pred[1])
preds = np.concatenate(preds, 0)
classify_preds = np.concatenate(classify_preds, 0)


tp = 0
fp = 0
fn = 0
total_ct = 0
err_ct1 = 0 # Overall false positives
err_ct2 = 0 # Overall false negatives
thr = 0.01 * (288 * 384)
classify_y_trues = []
classify_y_preds = []
for i in range(len(preds)):
    y_pred = preds[i]
    y_pred_classify = classify_preds[i]
    # y_pred = np.stack([y_pred[..., :2].sum(-1), y_pred[..., 2:].sum(-1)], -1)
    # y_pred_classify = np.stack([y_pred_classify[..., :1].sum(-1), y_pred_classify[..., 1:].sum(-1)], -1)

    y_pred = scipy.special.softmax(y_pred, -1)
    y_pred_classify = scipy.special.softmax(y_pred_classify, -1)

    name = valid_gen.names[valid_gen.selected_inds[i]]
    key = get_ex_day(name) + get_well(name)
    y_true = (fls[key] > 1) * 1
    w = 1 - (fls[key] == 1)
    y_true_classify = labels[key][0]
    w_true_classify = labels[key][1]

    if w_true_classify > 0:
        classify_y_trues.append(y_true_classify)
        classify_y_preds.append(y_pred_classify)

    y_pred = y_pred[np.nonzero(w)].reshape((-1, 2))
    y_true = y_true[np.nonzero(w)].reshape((-1,))
    _tp = ((y_pred[:, 1] > 0.5) * y_true).sum()
    _fp = ((y_pred[:, 1] > 0.5) * (1 - y_true)).sum()
    _fn = ((y_pred[:, 1] <= 0.5) * y_true).sum()

    tp += _tp
    fp += _fp
    fn += _fn
    total_ct += 1
    if y_pred.shape[0] > (0.99*288*384) and (_tp + _fn) < thr and _fp > thr:
        err_ct1 += 1
    if _fn > thr and (_tp + _fp) < thr:
        err_ct2 += 1

iou = tp/(tp + fp + fn)
prec = tp/(tp + fp)
recall = tp/(tp + fn)
f1 = 2/(1/(prec + 1e-5) + 1/(recall + 1e-5))
print("Precision: %.3f\tRecall: %.3f\tF1: %.3f\tIOU: %.3f\tFP: %d/%d\tFN: %d/%d" %
      (prec, recall, f1, iou, err_ct1, total_ct, err_ct2, total_ct))

classify_y_trues = np.array(classify_y_trues)
classify_y_preds = np.stack(classify_y_preds, 0)
auc = roc_auc_score(classify_y_trues, classify_y_preds[:, 1])
prec = precision_score(classify_y_trues, classify_y_preds[:, 1] > 0.5)
recall = recall_score(classify_y_trues, classify_y_preds[:, 1] > 0.5)
f1 = f1_score(classify_y_trues, classify_y_preds[:, 1] > 0.5)
print("Precision: %.3f\tRecall: %.3f\tF1: %.3f\tAUC: %.3f" %
    (prec, recall, f1, auc))