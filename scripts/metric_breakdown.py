import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import scipy
from data_loader import *
from segment_support import *
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer
from data_generator import CustomGenerator, PairGenerator, enhance_weight_fp, binarized_fluorescence_label
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

data_path = 'data/linear_aligned_patches/cross_7-to-10/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('random_valid_X')])
X_filenames = [os.path.join(data_path, 'random_valid_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'random_valid_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'random_valid_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'random_valid_names.pkl')
label_file = os.path.join(data_path, 'random_valid_labels.pkl')

kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 2,
    'segment_class_weights': [1, 3],
    'segment_extra_weights': enhance_weight_fp,
    'segment_label_type': 'segmentation',
    'n_classify_classes': 2,
    'classify_class_weights': [0.02, 0.02]
}

valid_gen = PairGenerator(X_filenames,
                          y_filenames,
                          w_filenames,
                          name_file,
                          label_file=label_file,
                          **kwargs)


model = ClassifyOnSegment(
    input_shape=(288, 384, 3),
    model_structure='pspnet',
    model_path='model_save',
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)


model.load('./model_save/pspnet_random_0-to-10_0.model')


### Prec, Recall, F1 score w.r.t. predict day (from) ###

# classify_y_preds = {}
# classify_y_trues = {}
# tp = {}
# fp = {}
# fn = {}
# total_ct = {}
# thr = 0.01 * (288 * 384)

# for batch in valid_gen:
#     day = batch[0][..., 1][:, 0, 0]
#     for d in day:
#         if not d in classify_y_trues:
#             classify_y_trues[d] = []
#             classify_y_preds[d] = []
#             tp[d] = 0
#             fp[d] = 0
#             fn[d] = 0
#             total_ct[d] = 0

#     y_pred, y_pred_classify = model.model.predict(batch[0])
#     yw_true, yw_true_classify = batch[1]
#     y_pred = scipy.special.softmax(y_pred, -1)
#     y_pred_classify = scipy.special.softmax(y_pred_classify, -1)
    
#     y_true = yw_true[..., :-1]
#     w = yw_true[..., -1]
#     y_true_classify = yw_true_classify[..., :-1]
#     w_true_classify = yw_true_classify[..., -1]


#     classify_valid_inds = np.nonzero(w_true_classify)[0]
#     for i in classify_valid_inds:
#         d = day[i]
#         classify_y_trues[d].append(y_true_classify[i])
#         classify_y_preds[d].append(y_pred_classify[i])

#     assert y_pred.shape[0] == y_true.shape[0] == w.shape[0]
#     for _y_pred, _y_true, _w, d in zip(y_pred, y_true, w, day):
#       _y_pred = _y_pred[np.nonzero(_w)].reshape((-1, 2))
#       _y_true = _y_true[np.nonzero(_w)].reshape((-1, 2))
#       _tp = ((_y_pred[:, 1] > 0.5) * _y_true[:, 1]).sum()
#       _fp = ((_y_pred[:, 1] > 0.5) * _y_true[:, 0]).sum()
#       _fn = ((_y_pred[:, 1] <= 0.5) * _y_true[:, 1]).sum()

#       tp[d] += _tp
#       fp[d] += _fp
#       fn[d] += _fn
#       total_ct[d] += 1


# iou = {}
# prec = {}
# recall = {}
# f1 = {}
# for d in tp:
#     iou[d] = tp[d]/(tp[d] + fp[d] + fn[d])
#     prec[d] = tp[d]/(tp[d] + fp[d])
#     recall[d] = tp[d]/(tp[d] + fn[d])
#     f1[d] = 2/(1/(prec[d] + 1e-5) + 1/(recall[d] + 1e-5))

# x = sorted(f1.keys())
# plt.clf()
# plt.plot(x, [f1[_x] for _x in x], '.-', label='segmentation f1')
# plt.plot(x, [prec[_x] for _x in x], '.-', label='segmentation precision')
# plt.plot(x, [recall[_x] for _x in x], '.-', label='segmentation recall')
# plt.xlabel('Day (from)')
# plt.legend()
# plt.savefig('/home/zqwu/Dropbox/fig_temp/seg_0-to-10.png', dpi=300)

# c_auc = {}
# c_prec = {}
# c_recall = {}
# c_f1 = {}
# for d in classify_y_trues:
#     classify_y_trues[d] = np.stack(classify_y_trues[d], 0)
#     classify_y_preds[d] = np.stack(classify_y_preds[d], 0)
#     c_auc[d] = roc_auc_score(classify_y_trues[d], classify_y_preds[d])
#     c_prec[d] = precision_score(classify_y_trues[d][:, 1], classify_y_preds[d][:, 1] > 0.5)
#     c_recall[d] = recall_score(classify_y_trues[d][:, 1], classify_y_preds[d][:, 1] > 0.5)
#     c_f1[d] = f1_score(classify_y_trues[d][:, 1], classify_y_preds[d][:, 1] > 0.5)

# x = sorted(c_f1.keys())
# plt.clf()
# plt.plot(x, [c_f1[_x] for _x in x], '.-', label='segmentation f1')
# plt.plot(x, [c_prec[_x] for _x in x], '.-', label='segmentation precision')
# plt.plot(x, [c_recall[_x] for _x in x], '.-', label='segmentation recall')
# plt.plot(x, [c_auc[_x] for _x in x], '.-', label='segmentation roc-auc')
# plt.xlabel('Day (from)')
# plt.legend()
# plt.savefig('/home/zqwu/Dropbox/fig_temp/cla_0-to-10.png', dpi=300)


### Accuracy of 4 class w.r.t. predict day (from) ###

all_frame_names = pickle.load(open('data/linear_aligned_patches/merged_all/permuted_names.pkl', 'rb'))
all_frame_labels = pickle.load(open('data/linear_aligned_patches/merged_all/permuted_labels.pkl', 'rb'))
name_to_label = {all_frame_names[i]: all_frame_labels[i] for i in all_frame_names}

pred_0_0 = {}
pred_0_1 = {}
pred_1_1 = {}
pred_1_0 = {}

y_preds = []
for batch in valid_gen:
    _, y_pred_classify = model.model.predict(batch[0])
    y_preds.append(y_pred_classify)
y_preds = np.concatenate(y_preds, 0)
y_preds = scipy.special.softmax(y_preds, -1)

for i in valid_gen.selected_inds:
    names = valid_gen.names[i]
    day = int(get_ex_day(names[0])[1][1:])
    if not day in pred_0_0:
        pred_0_0[day] = []
        pred_0_1[day] = []
        pred_1_0[day] = []
        pred_1_1[day] = []
    label_pre = name_to_label[names[0]]
    label_post = name_to_label[names[1]]

    if not (label_pre[1] == 1 and label_post[1] == 1):
        continue

    if label_pre[0] == 0 and label_post[0] == 0:
        pred_0_0[day].append(y_preds[i, 1])
    elif label_pre[0] == 0 and label_post[0] == 1:
        pred_0_1[day].append(y_preds[i, 1])
    elif label_pre[0] == 1 and label_post[0] == 0:
        pred_1_0[day].append(y_preds[i, 1])
    elif label_pre[0] == 1 and label_post[0] == 1:
        pred_1_1[day].append(y_preds[i, 1])
    else:
        raise ValueError



x = sorted(pred_0_0.keys())
selected_x = [_x for _x in x if _x < 13]
plt.clf()
fig, ax = plt.subplots(figsize=(7, 3))
red_diamond = dict(markerfacecolor='r', marker='D', markersize=2)
data = [pred_0_1[_x] for _x in selected_x]
bplot_0_1 = ax.boxplot(data, 
                       notch=True, 
                       vert=True, 
                       patch_artist=True, 
                       positions=np.array(selected_x)+0.11, 
                       flierprops=red_diamond,
                       widths=0.2,
                       manage_ticks=False)
blue_diamond = dict(markerfacecolor='b', marker='D', markersize=2)
data = [pred_0_0[_x] for _x in selected_x]
bplot_0_0 = ax.boxplot(data, 
                       notch=True, 
                       vert=True, 
                       patch_artist=True, 
                       positions=np.array(selected_x)-0.11, 
                       flierprops=blue_diamond,
                       widths=0.2,
                       manage_ticks=False)
ax.set_xticks(selected_x)
ax.set_ylim([0, 1])
ax.set_xlim([-1, 13])
for patch in bplot_0_1['boxes']:
    patch.set_facecolor('pink')
for patch in bplot_0_0['boxes']:
    patch.set_facecolor('lightblue')
ax.set_title('Prediction Distribution (From 0)')
plt.savefig('/home/zqwu/Dropbox/fig_temp/pred_distri_from0.png', dpi=300)


plt.clf()
fig, ax = plt.subplots(figsize=(7, 1.5))
ax.plot(selected_x, [len(pred_0_1[_x]) for _x in selected_x], c='pink')
ax.plot(selected_x, [len(pred_0_0[_x]) for _x in selected_x], c='lightblue')
ax.set_xticks(selected_x)
ax.set_xlim([-1, 13])
ax.set_ylim([0, ax.get_ylim()[1]])
plt.savefig('/home/zqwu/Dropbox/fig_temp/n_samples_from0.png', dpi=300)


x = sorted(pred_1_1.keys())
plt.clf()
fig, ax = plt.subplots(figsize=(7, 3))
green_diamond = dict(markerfacecolor='g', marker='D', markersize=2)
data = [pred_1_1[_x] for _x in x]
bplot_1_1 = ax.boxplot(data, 
                       notch=True, 
                       vert=True, 
                       patch_artist=True, 
                       positions=np.array(x), 
                       flierprops=green_diamond,
                       widths=0.2,
                       manage_ticks=False)
ax.set_xticks(x)
ax.set_ylim([0, 1])
ax.set_xlim([5, 17])
for patch in bplot_1_1['boxes']:
    patch.set_facecolor('lightgreen')
ax.set_title('Prediction Distribution (From 1)')
plt.savefig('/home/zqwu/Dropbox/fig_temp/pred_distri_from1.png', dpi=300)


plt.clf()
fig, ax = plt.subplots(figsize=(7, 1.5))
ax.plot(x, [len(pred_1_1[_x]) for _x in selected_x], c='lightgreen')
ax.set_xticks(x)
ax.set_xlim([5, 17])
ax.set_ylim([0, ax.get_ylim()[1]])
plt.savefig('/home/zqwu/Dropbox/fig_temp/n_samples_from1.png', dpi=300)