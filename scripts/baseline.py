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

def baseline(gen):
  tp = 0
  fp = 0
  fn = 0
  thrs = []
  for batch in gen:
    y_pred = batch[0][..., 0]
    y_true = batch[1][0][..., :-1]
    w = batch[1][0][..., -1]
    for _y_pred, _y_true, _w in zip(y_pred, y_true, w):
      _y_pred = _y_pred[np.nonzero(_w)].reshape((-1))
      _y_true = _y_true[np.nonzero(_w)].reshape((-1, 2))
      if _y_true[:, 1].sum() == 0:
        continue
      def local_score(thr):
        _tp = ((_y_pred < thr) * _y_true[:, 1]).sum()
        _fp = ((_y_pred < thr) * _y_true[:, 0]).sum()
        _fn = ((_y_pred < thr) * _y_true[:, 1]).sum()
        _prec = _tp/(_tp + _fp + 1e-5)
        _recall = _tp/(_tp + _fn + 1e-5)
        _f1 = 2/(1/(_prec + 1e-5) + 1/(_recall + 1e-5))
        return _f1
      scores = {thr: local_score(thr) for thr in np.arange(-3, 3, 0.05)}
      best_thr = sorted(scores.keys(), key=lambda x: scores[x])[-1]
      thrs.append(best_thr)
      _tp = ((_y_pred < best_thr) * _y_true[:, 1]).sum()
      _fp = ((_y_pred < best_thr) * _y_true[:, 0]).sum()
      _fn = ((_y_pred < best_thr) * _y_true[:, 1]).sum()
      tp += _tp
      fp += _fp
      fn += _fn
  iou = tp/(tp + fp + fn)
  prec = tp/(tp + fp)
  recall = tp/(tp + fn)
  f1 = 2/(1/(prec + 1e-5) + 1/(recall + 1e-5))
  print(prec)
  print(recall)
  print(f1)
  print(iou)
  print(np.percentile(thrs, 25))
  print(np.percentile(thrs, 50))
  print(np.percentile(thrs, 75))
  print(np.mean(thrs))
  return prec, recall, f1


def baseline_overall(gen):
  def overall_score(thr):
    tp = 0
    fp = 0
    fn = 0
    thrs = []
    for batch in gen:
      y_pred = batch[0][..., 0]
      y_true = batch[1][0][..., :-1]
      w = batch[1][0][..., -1]
      for _y_pred, _y_true, _w in zip(y_pred, y_true, w):
        _y_pred = _y_pred[np.nonzero(_w)].reshape((-1))
        _y_true = _y_true[np.nonzero(_w)].reshape((-1, 2))
        if _y_true[:, 1].sum() == 0:
          continue
        _tp = ((_y_pred < thr) * _y_true[:, 1]).sum()
        _fp = ((_y_pred < thr) * _y_true[:, 0]).sum()
        _fn = ((_y_pred < thr) * _y_true[:, 1]).sum()
        tp += _tp
        fp += _fp
        fn += _fn
    iou = tp/(tp + fp + fn)
    prec = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2/(1/(prec + 1e-5) + 1/(recall + 1e-5))
    return prec, recall, f1, iou

  scores = {thr: overall_score(thr) for thr in np.arange(-2.6, 2.0, 0.02)}
  best_thr = sorted(scores.keys(), key=lambda x: scores[x][2])[-1]
  print(scores[best_thr])
  return scores[best_thr]


if __name__ == '__main__':
  data_path = 'data/linear_aligned_patches/merged_all_fl/'

  kwargs = {
      'batch_size': 16,
      'shuffle_inds': False,
      'include_day': True,
      'n_segment_classes': 2,
      'segment_class_weights': [1, 3],
      'segment_extra_weights': enhance_weight_fp,
      'segment_label_type': 'segmentation',
      'n_classify_classes': 2,
      'classify_class_weights': [0.02, 0.02]
  }

  n_fs = len([f for f in os.listdir(data_path) if f.startswith('random_valid_X')])
  X_filenames = [os.path.join(data_path, 'random_valid_X_%d.pkl' % i) for i in range(n_fs)]
  y_filenames = [os.path.join(data_path, 'random_valid_y_%d.pkl' % i) for i in range(n_fs)]
  w_filenames = [os.path.join(data_path, 'random_valid_w_%d.pkl' % i) for i in range(n_fs)]
  name_file = os.path.join(data_path, 'random_valid_names.pkl')
  label_file = os.path.join(data_path, 'random_valid_labels.pkl')
  valid_gen = CustomGenerator(X_filenames,
                              y_filenames,
                              w_filenames,
                              name_file,
                              label_file=label_file,
                              augment=False,
                              **kwargs)

  print("Local search")
  _ = baseline(valid_gen)

  print("Global search")
  _ = baseline_overall(valid_gen)