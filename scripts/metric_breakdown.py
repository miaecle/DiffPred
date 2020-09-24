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
n_fs = len([f for f in os.listdir(data_path) if f.startswith('temp_valid_X')])
X_filenames = [os.path.join(data_path, 'temp_valid_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'temp_valid_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'temp_valid_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'temp_valid_names.pkl')
label_file = os.path.join(data_path, 'temp_valid_labels.pkl')

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


classify_y_preds = {}
classify_y_trues = {}
tp = {}
fp = {}
fn = {}
total_ct = {}
thr = 0.01 * (288 * 384)

for batch in valid_gen:
    day = batch[0][..., 1][:, 0, 0]
    for d in day:
        if not d in classify_y_trues:
            classify_y_trues[d] = []
            classify_y_preds[d] = []
            tp[d] = 0
            fp[d] = 0
            fn[d] = 0
            total_ct[d] = 0

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

x = sorted(f1.keys())
plt.clf()
plt.plot(x, [f1[_x] for _x in x], '.-', label='segmentation f1')
plt.plot(x, [prec[_x] for _x in x], '.-', label='segmentation precision')
plt.plot(x, [recall[_x] for _x in x], '.-', label='segmentation recall')
plt.xlabel('Day (from)')
plt.legend()
plt.savefig('/home/zqwu/Dropbox/fig_temp/seg_0-to-10.png', dpi=300)

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

x = sorted(c_f1.keys())
plt.clf()
plt.plot(x, [c_f1[_x] for _x in x], '.-', label='segmentation f1')
plt.plot(x, [c_prec[_x] for _x in x], '.-', label='segmentation precision')
plt.plot(x, [c_recall[_x] for _x in x], '.-', label='segmentation recall')
plt.plot(x, [c_auc[_x] for _x in x], '.-', label='segmentation roc-auc')
plt.xlabel('Day (from)')
plt.legend()
plt.savefig('/home/zqwu/Dropbox/fig_temp/cla_0-to-10.png', dpi=300)