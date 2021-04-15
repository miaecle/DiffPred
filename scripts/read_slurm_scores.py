###############################################################################

import os
with open('LOG', 'r') as f:
  lines = f.readlines()


seg_scores = {}
classify_scores = {}
val_losses = {}
epoch_now = -1
for i, line in enumerate(lines):
  if "Epoch" in line:
    epoch_now = int(line.split('/')[0].split()[1])
  if "IOU" in line:
    seg_score = float(line.split("IOU:")[1].split()[0])
    if seg_score == seg_score:
      seg_scores[epoch_now] = seg_score
  if "AUC" in line:
    classify_score = float(line.split("AUC:")[1].split()[0])
    if classify_score == classify_score:
      classify_scores[epoch_now] = classify_score
  if "val_loss" in line:
    val_loss = float(line.split("val_loss:")[1].split()[0])
    val_losses[epoch_now] = val_loss


seg_orders = sorted(seg_scores.keys(), key=lambda x: seg_scores[x])
classify_orders = sorted(classify_scores.keys(), key=lambda x: classify_scores[x])
print(seg_orders[-5:])
print(classify_orders[-5:])

saves = set(seg_orders[-5:]) | set(classify_orders[-5:])
for s in saves:
  assert os.system("cp weights.%02d-%.2f.hdf5 weights.%d-iou-%.3f-auc-%.3f.hdf5" % (s, val_losses[s], s, seg_scores[s], classify_scores[s])) == 0

weight_files = os.listdir()
for f in weight_files:
  if 'weights' in f and not 'iou' in f:
    os.remove(f)


###############################################################################

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
from data_loader import get_identifier
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer, evaluate_segmentation_and_classification
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives
from scipy.stats import spearmanr, pearsonr

### Settings ###
kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 2,
    'segment_class_weights': [1, 5],
    'segment_extra_weights': enhance_weight_for_false_positives,
    'segment_label_type': 'discrete',
    'n_classify_classes': 2,
    'classify_class_weights': [0.5, 0.15],
    'classify_label_type': 'discrete',
}

### Training ###
model = ClassifyOnSegment(
    input_shape=(288, 384, 3),
    model_structure='pspnet',
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)


print("l1ex7 (VALID)")
print("=========================")
print("SCORE by interval")

model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-0_ex/bkp.model')
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-0_discrete/l1ex7_valid/'

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')
pair_names = pickle.load(open(name_file, 'rb'))
for interval in range(1):
    selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
    print("%d: %d" % (interval, len(selected_inds)))
    if len(selected_inds) > 0:
        data = PairGenerator(
            name_file,
            X_filenames,
            segment_y_files=y_filenames,
            segment_w_files=w_filenames,
            classify_label_file=label_file,
            selected_inds=selected_inds,
            **kwargs)
        evaluate_segmentation_and_classification(data, model)



model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-3_ex/bkp.model')
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-3_discrete/l1ex7_valid/'

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')
pair_names = pickle.load(open(name_file, 'rb'))
for interval in range(1, 4):
    selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
    print("%d: %d" % (interval, len(selected_inds)))
    if len(selected_inds) > 0:
        data = PairGenerator(
            name_file,
            X_filenames,
            segment_y_files=y_filenames,
            segment_w_files=w_filenames,
            classify_label_file=label_file,
            selected_inds=selected_inds,
            **kwargs)
        evaluate_segmentation_and_classification(data, model)



model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-6_ex/bkp.model')
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-6_discrete/l1ex7_valid/'

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')
pair_names = pickle.load(open(name_file, 'rb'))
for interval in range(4, 7):
    selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
    print("%d: %d" % (interval, len(selected_inds)))
    if len(selected_inds) > 0:
        data = PairGenerator(
            name_file,
            X_filenames,
            segment_y_files=y_filenames,
            segment_w_files=w_filenames,
            classify_label_file=label_file,
            selected_inds=selected_inds,
            **kwargs)
        evaluate_segmentation_and_classification(data, model)



model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-10_ex/bkp.model')
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-10_discrete/l1ex7_valid/'

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')
pair_names = pickle.load(open(name_file, 'rb'))
for interval in range(7, 11):
    selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
    print("%d: %d" % (interval, len(selected_inds)))
    if len(selected_inds) > 0:
        data = PairGenerator(
            name_file,
            X_filenames,
            segment_y_files=y_filenames,
            segment_w_files=w_filenames,
            classify_label_file=label_file,
            selected_inds=selected_inds,
            **kwargs)
        evaluate_segmentation_and_classification(data, model)



print("Inf model below")
model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex/bkp.model')
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex7_valid/'

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')
pair_names = pickle.load(open(name_file, 'rb'))
labels = pickle.load(open(label_file, 'rb'))
for interval in range(9, 16):
    selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
    if len(selected_inds) > 50:
        related_labels = np.array([labels[i] for i in selected_inds])
        related_labels = related_labels[np.where(related_labels[:, 1] > 0)][:, 0]
        if len(np.unique(related_labels)) > 1:
            print("%d: %d" % (interval, len(selected_inds)))
            data = PairGenerator(
                name_file,
                X_filenames,
                segment_y_files=y_filenames,
                segment_w_files=w_filenames,
                classify_label_file=label_file,
                selected_inds=selected_inds,
                **kwargs)
            evaluate_segmentation_and_classification(data, model)


print("SCORE by start day")
for start_day in range(10):
    selected_inds = [i for i in range(len(pair_names)) if int(get_identifier(pair_names[i][0])[2]) == start_day]
    if len(selected_inds) > 50:
        related_labels = np.array([labels[i] for i in selected_inds])
        related_labels = related_labels[np.where(related_labels[:, 1] > 0)][:, 0]
        if len(np.unique(related_labels)) > 1:
            print("%d: %d" % (start_day, len(selected_inds)))
            data = PairGenerator(
                name_file,
                X_filenames,
                segment_y_files=y_filenames,
                segment_w_files=w_filenames,
                classify_label_file=label_file,
                selected_inds=selected_inds,
                **kwargs)
            evaluate_segmentation_and_classification(data, model)


print("=========================")



print("l1ex1 (TEST)")
print("=========================")
print("SCORE by interval")

model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-0_ex/bkp.model')
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-0_discrete/l1ex1_valid/'

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')
pair_names = pickle.load(open(name_file, 'rb'))
for interval in range(1):
    selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
    print("%d: %d" % (interval, len(selected_inds)))
    if len(selected_inds) > 0:
        data = PairGenerator(
            name_file,
            X_filenames,
            segment_y_files=y_filenames,
            segment_w_files=w_filenames,
            classify_label_file=label_file,
            selected_inds=selected_inds,
            **kwargs)
        evaluate_segmentation_and_classification(data, model)



model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-3_ex/bkp.model')
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-3_discrete/l1ex1_valid/'

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')
pair_names = pickle.load(open(name_file, 'rb'))
for interval in range(1, 4):
    selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
    print("%d: %d" % (interval, len(selected_inds)))
    if len(selected_inds) > 0:
        data = PairGenerator(
            name_file,
            X_filenames,
            segment_y_files=y_filenames,
            segment_w_files=w_filenames,
            classify_label_file=label_file,
            selected_inds=selected_inds,
            **kwargs)
        evaluate_segmentation_and_classification(data, model)



model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-6_ex/bkp.model')
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-6_discrete/l1ex1_valid/'

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')
pair_names = pickle.load(open(name_file, 'rb'))
for interval in range(4, 7):
    selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
    print("%d: %d" % (interval, len(selected_inds)))
    if len(selected_inds) > 0:
        data = PairGenerator(
            name_file,
            X_filenames,
            segment_y_files=y_filenames,
            segment_w_files=w_filenames,
            classify_label_file=label_file,
            selected_inds=selected_inds,
            **kwargs)
        evaluate_segmentation_and_classification(data, model)



model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-10_ex/bkp.model')
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-10_discrete/l1ex1_valid/'

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')
pair_names = pickle.load(open(name_file, 'rb'))
for interval in range(7, 11):
    selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
    print("%d: %d" % (interval, len(selected_inds)))
    if len(selected_inds) > 0:
        data = PairGenerator(
            name_file,
            X_filenames,
            segment_y_files=y_filenames,
            segment_w_files=w_filenames,
            classify_label_file=label_file,
            selected_inds=selected_inds,
            **kwargs)
        evaluate_segmentation_and_classification(data, model)



print("Inf model below")
model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex/bkp.model')
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex1_valid/'

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')
pair_names = pickle.load(open(name_file, 'rb'))
labels = pickle.load(open(label_file, 'rb'))
for interval in range(10, 22):
    selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
    if len(selected_inds) > 50:
        related_labels = np.array([labels[i] for i in selected_inds])
        related_labels = related_labels[np.where(related_labels[:, 1] > 0)][:, 0]
        if len(np.unique(related_labels)) > 1:
            print("%d: %d" % (interval, len(selected_inds)))
            data = PairGenerator(
                name_file,
                X_filenames,
                segment_y_files=y_filenames,
                segment_w_files=w_filenames,
                classify_label_file=label_file,
                selected_inds=selected_inds,
                **kwargs)
            evaluate_segmentation_and_classification(data, model)


print("SCORE by start day")
for start_day in range(10):
    selected_inds = [i for i in range(len(pair_names)) if int(get_identifier(pair_names[i][0])[2]) == start_day]
    if len(selected_inds) > 50:
        related_labels = np.array([labels[i] for i in selected_inds])
        related_labels = related_labels[np.where(related_labels[:, 1] > 0)][:, 0]
        if len(np.unique(related_labels)) > 1:
            print("%d: %d" % (start_day, len(selected_inds)))
            data = PairGenerator(
                name_file,
                X_filenames,
                segment_y_files=y_filenames,
                segment_w_files=w_filenames,
                classify_label_file=label_file,
                selected_inds=selected_inds,
                **kwargs)
            evaluate_segmentation_and_classification(data, model)


print("=========================")




###############################################################################

score_file = "/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/SCORES"
with open(score_file, 'r') as f:
    lines = f.readlines()

all_segments = []
segment = {}
start_flag = False
key = None
item = None

for line in lines:
    if start_flag:
        line = line.split()
        if len(line) == 0:
            continue
        if line[0].startswith('==='):
            if not key is None and not item is None:
                segment[key] = item
                item = None
                key = None
            if len(segment) > 0:
                all_segments.append(segment)
            segment = {}
        if len(line) == 2:
            if not key is None and not item is None:
                segment[key] = item
                item = None
                key = None
            # Key line
            try: 
                key = int(line[0][:-1])
                item = {"N_samples": int(line[1])}
            except Exception as e:
                key = None
                item = None
        elif len(line) > 2 and 'IOU:' in line:
            # Segmentation perf line
            if not key is None and not item is None:
                for k, v in zip(line[0::2], line[1::2]):
                    item["seg-" + k[:-1]] = v
        elif len(line) > 2 and 'AUC:' in line:
            # Classification perf line
            if not key is None and not item is None:
                for k, v in zip(line[0::2], line[1::2]):
                    item["cla-" + k[:-1]] = v
    else:
        if line.startswith('==='):
            start_flag = True



import matplotlib.pyplot as plt
import matplotlib

cmap = matplotlib.cm.get_cmap('tab10')
def get_line_xy(segment, key='cla-AUC'):
    x = segment.keys()
    x = sorted([float(_x) for _x in x])
    y = [float(segment[_x][key]) for _x in x]
    return x, y

plt.clf()
x, y = get_line_xy(all_segments[0], 'cla-AUC')
plt.plot(x, y, '.-', color=cmap(3), label='roc-auc')
x, y = get_line_xy(all_segments[1], 'cla-AUC')
plt.plot(x, y, '.--', color=cmap(3), label='roc-auc (0-to-inf)')
plt.xlabel("Prediction Interval (day)")
plt.legend()


plt.clf()
x, y = get_line_xy(all_segments[0], 'seg-F1')
plt.plot(x, y, '.-', color=cmap(7), label='f1')
x, y = get_line_xy(all_segments[1], 'seg-F1')
plt.plot(x, y, '.--', color=cmap(7), label='f1 (0-to-inf)')
x, y = get_line_xy(all_segments[0], 'seg-IOU')
plt.plot(x, y, '.-', color=cmap(8), label='IOU')
x, y = get_line_xy(all_segments[1], 'seg-IOU')
plt.plot(x, y, '.--', color=cmap(8), label='IOU (0-to-inf)')
plt.xlabel("Prediction Interval (day)")
plt.legend()





plt.clf()
x, y = get_line_xy(all_segments[2], 'cla-Precision')
plt.plot(x, y, '.--', color=cmap(0), label='precision (0-to-inf)')
x, y = get_line_xy(all_segments[2], 'cla-Recall')
plt.plot(x, y, '.--', color=cmap(1), label='recall (0-to-inf)')
x, y = get_line_xy(all_segments[2], 'cla-F1')
plt.plot(x, y, '.--', color=cmap(2), label='f1 (0-to-inf)')
x, y = get_line_xy(all_segments[2], 'cla-AUC')
plt.plot(x, y, '.--', color=cmap(3), label='roc-auc (0-to-inf)')
plt.xlabel("Input Phase Contrast (day)")
plt.legend()


plt.clf()
x, y = get_line_xy(all_segments[2], 'seg-F1')
plt.plot(x, y, '.--', color=cmap(7), label='f1 (0-to-inf)')
x, y = get_line_xy(all_segments[2], 'seg-IOU')
plt.plot(x, y, '.--', color=cmap(8), label='IOU (0-to-inf)')
plt.xlabel("Input Phase Contrast (day)")
plt.legend()














plt.clf()
x, y = get_line_xy(all_segments[3], 'cla-AUC')
plt.plot(x, y, '.-', color=cmap(3), label='roc-auc')
x, y = get_line_xy(all_segments[4], 'cla-AUC')
plt.plot(x, y, '.--', color=cmap(3), label='roc-auc (0-to-inf)')
plt.xlabel("Prediction Interval (day)")
plt.legend()


plt.clf()
x, y = get_line_xy(all_segments[3], 'seg-F1')
plt.plot(x, y, '.-', color=cmap(7), label='f1')
x, y = get_line_xy(all_segments[4], 'seg-F1')
plt.plot(x, y, '.--', color=cmap(7), label='f1 (0-to-inf)')
x, y = get_line_xy(all_segments[3], 'seg-IOU')
plt.plot(x, y, '.-', color=cmap(8), label='IOU')
x, y = get_line_xy(all_segments[4], 'seg-IOU')
plt.plot(x, y, '.--', color=cmap(8), label='IOU (0-to-inf)')
plt.xlabel("Prediction Interval (day)")
plt.legend()





plt.clf()
x, y = get_line_xy(all_segments[5], 'cla-Precision')
plt.plot(x, y, '.-', color=cmap(0), label='precision (0-to-inf)')
x, y = get_line_xy(all_segments[5], 'cla-Recall')
plt.plot(x, y, '.-', color=cmap(1), label='recall (0-to-inf)')
x, y = get_line_xy(all_segments[5], 'cla-F1')
plt.plot(x, y, '.-', color=cmap(2), label='f1 (0-to-inf)')
x, y = get_line_xy(all_segments[5], 'cla-AUC')
plt.plot(x, y, '.-', color=cmap(3), label='roc-auc (0-to-inf)')
plt.xlabel("Input Phase Contrast (day)")
plt.legend()


plt.clf()
x, y = get_line_xy(all_segments[5], 'seg-F1')
plt.plot(x, y, '.--', color=cmap(7), label='f1 (0-to-inf)')
x, y = get_line_xy(all_segments[5], 'seg-IOU')
plt.plot(x, y, '.--', color=cmap(8), label='IOU (0-to-inf)')
plt.xlabel("Input Phase Contrast (day)")
plt.legend()
