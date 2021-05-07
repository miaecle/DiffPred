import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
import scipy
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from data_loader import get_identifier
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives

### Settings ###
ROOT_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/'
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/random_valid/'
SPLIT_FILE = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/random_split.pkl'

MODEL_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/random_split/0-to-inf_random/'

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

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')

valid_gen = PairGenerator(
    name_file,
    X_filenames, 
    segment_y_files=y_filenames, 
    segment_w_files=w_filenames,
    classify_label_file=label_file,
    **kwargs)


model = ClassifyOnSegment(
    input_shape=(288, 384, 3),
    model_structure='pspnet',
    model_path=MODEL_DIR,
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)

model.load(os.path.join(MODEL_DIR, 'bkp.model'))



### Collect predictions ###
y_preds = []
for batch in valid_gen:
    _, y_pred_classify = model.model.predict(batch[0])
    y_preds.append(y_pred_classify)
y_preds = np.concatenate(y_preds, 0)
y_preds = scipy.special.softmax(y_preds, -1)



### Box plot w.r.t. predict day (from) ###
pred_0 = {i: [] for i in range(4, 13)}
pred_1 = {i: [] for i in range(4, 13)}
for i in valid_gen.selected_inds:
    names = valid_gen.names[i]
    day = int(get_identifier(names[0])[2])
    label = valid_gen.classify_y[i]

    if label == 0:
        pred_0[day].append(y_preds[i, 1])
    elif label == 1:
        pred_1[day].append(y_preds[i, 1])
    else:
        raise ValueError

x = sorted(pred_0.keys())
plt.clf()
fig, ax = plt.subplots(figsize=(7, 3))
red_diamond = dict(markerfacecolor='r', marker='D', markersize=2)
data = [pred_1[_x] for _x in x]
bplot_1 = ax.boxplot(data, 
                     notch=True, 
                     vert=True, 
                     patch_artist=True, 
                     positions=np.array(x)+0.11, 
                     flierprops=red_diamond,
                     widths=0.2,
                     manage_ticks=False)
blue_diamond = dict(markerfacecolor='b', marker='D', markersize=2)
data = [pred_0[_x] for _x in x]
bplot_0 = ax.boxplot(data, 
                     notch=True, 
                     vert=True, 
                     patch_artist=True, 
                     positions=np.array(x)-0.11, 
                     flierprops=blue_diamond,
                     widths=0.2,
                     manage_ticks=False)
ax.set_xticks(x)
ax.set_ylim([-0.03, 1.03])
ax.set_xlim([3, 13])
for patch in bplot_1['boxes']:
    patch.set_facecolor('pink')
for patch in bplot_0['boxes']:
    patch.set_facecolor('lightblue')
ax.set_title('Prediction Distribution')
plt.savefig('pred_distri.png', dpi=300, bbox_inches='tight')


plt.clf()
fig, ax = plt.subplots(figsize=(7, 1.5))
ax.plot(x, [len(pred_1[_x]) for _x in x], c='pink')
ax.plot(x, [len(pred_0[_x]) for _x in x], c='lightblue')
ax.set_xticks(x)
ax.set_xlim([3, 13])
ax.set_ylim([0, ax.get_ylim()[1]])
plt.savefig('n_samples.png', dpi=300, bbox_inches='tight')



### Prec, Recall, F1 score w.r.t. predict day (from) ###
classify_y_preds = {i: [] for i in range(4, 13)}
classify_y_trues = {i: [] for i in range(4, 13)}
tp = {i: 0 for i in range(4, 13)}
fp = {i: 0 for i in range(4, 13)}
fn = {i: 0 for i in range(4, 13)}
total_ct = {i: 0 for i in range(4, 13)}
thr = 0.01 * (288 * 384)

for batch in valid_gen:
    day = batch[0][:, 0, 0, 1]

    y_pred, y_pred_classify = model.model.predict(batch[0])
    yw_true, yw_true_classify = batch[1]

    y_pred = scipy.special.softmax(y_pred, -1)
    y_pred_classify = scipy.special.softmax(y_pred_classify, -1)
    
    y_true = yw_true[..., :-1]
    w = yw_true[..., -1]
    y_true_classify = yw_true_classify[..., :-1]
    w_true_classify = yw_true_classify[..., -1]

    classify_valid_inds = np.nonzero(w_true_classify)[0]
    for i in classify_valid_inds:
        d = day[i]
        classify_y_trues[d].append(y_true_classify[i])
        classify_y_preds[d].append(y_pred_classify[i])

    assert y_pred.shape[0] == y_true.shape[0] == w.shape[0]
    for _y_pred, _y_true, _w, d in zip(y_pred, y_true, w, day):
      _y_pred = _y_pred[np.nonzero(_w)].reshape((-1, 2))
      _y_true = _y_true[np.nonzero(_w)].reshape((-1, 2))
      _tp = ((_y_pred[:, 1] > 0.5) * _y_true[:, 1]).sum()
      _fp = ((_y_pred[:, 1] > 0.5) * _y_true[:, 0]).sum()
      _fn = ((_y_pred[:, 1] <= 0.5) * _y_true[:, 1]).sum()

      tp[d] += _tp
      fp[d] += _fp
      fn[d] += _fn
      total_ct[d] += 1

iou = {}
prec = {}
recall = {}
f1 = {}
for d in tp:
    iou[d] = tp[d]/(tp[d] + fp[d] + fn[d])
    prec[d] = tp[d]/(tp[d] + fp[d])
    recall[d] = tp[d]/(tp[d] + fn[d])
    f1[d] = 2/(1/(prec[d] + 1e-5) + 1/(recall[d] + 1e-5))

# x = sorted(f1.keys())
# plt.clf()
# plt.plot(x, [f1[_x] for _x in x], '.-', label='segmentation f1')
# plt.plot(x, [prec[_x] for _x in x], '.-', label='segmentation precision')
# plt.plot(x, [recall[_x] for _x in x], '.-', label='segmentation recall')
# plt.xlabel('Day (from)')
# plt.legend()
# plt.savefig('/home/zqwu/Dropbox/fig_temp/seg_0-to-10.png', dpi=300)

c_auc = {}
c_prec = {}
c_recall = {}
c_f1 = {}
for d in classify_y_trues:
    classify_y_trues[d] = np.stack(classify_y_trues[d], 0)
    classify_y_preds[d] = np.stack(classify_y_preds[d], 0)
    c_auc[d] = roc_auc_score(classify_y_trues[d], classify_y_preds[d])
    c_prec[d] = precision_score(classify_y_trues[d][:, 1], classify_y_preds[d][:, 1] > 0.5)
    c_recall[d] = recall_score(classify_y_trues[d][:, 1], classify_y_preds[d][:, 1] > 0.5)
    c_f1[d] = f1_score(classify_y_trues[d][:, 1], classify_y_preds[d][:, 1] > 0.5)

# x = sorted(c_f1.keys())
# plt.clf()
# plt.plot(x, [c_f1[_x] for _x in x], '.-', label='segmentation f1')
# plt.plot(x, [c_prec[_x] for _x in x], '.-', label='segmentation precision')
# plt.plot(x, [c_recall[_x] for _x in x], '.-', label='segmentation recall')
# plt.plot(x, [c_auc[_x] for _x in x], '.-', label='segmentation roc-auc')
# plt.xlabel('Day (from)')
# plt.legend()
# plt.savefig('/home/zqwu/Dropbox/fig_temp/cla_0-to-10.png', dpi=300)

