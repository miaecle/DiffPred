import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from data_loader import get_identifier, get_ex_day
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer, evaluate_confusion_mat, summarize_conf_mat
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives


#%% Settings ###
DATA_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/random_valid/'
MODEL_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/random_split/0-to-inf_random/'
PRED_SAVE_DIR = os.path.join(MODEL_DIR, "valid_pred")
n_input_channel = 3 # 2 for 0-to-0, 3 for 0-to-inf

kwargs = {
    'batch_size': 8,
    'shuffle_inds': False, # Datasets are usually pre-shuffled
    'include_day': True,
    'n_segment_classes': 4,
    'segment_class_weights': [1, 2, 2, 2],
    'segment_extra_weights': None,
    'segment_label_type': 'continuous',
    'n_classify_classes': 4,
    'classify_class_weights': [1., 1., 2., 1.],
    'classify_label_type': 'continuous',
}

#%% Define Dataset ###
n_fs = len([f for f in os.listdir(DATA_DIR) if f.startswith('X_') and f.endswith('.pkl')])
name_file = os.path.join(DATA_DIR, 'names.pkl')
X_filenames = [os.path.join(DATA_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
if os.path.exists(os.path.join(DATA_DIR, 'classify_continuous_labels.pkl')):
    kwargs['label_file'] = os.path.join(DATA_DIR, 'classify_continuous_labels.pkl')
if os.path.exists(os.path.join(DATA_DIR, 'segment_continuous_y_%d.pkl' % (n_fs - 1))):
    kwargs['y_filenames'] = [os.path.join(DATA_DIR, 'segment_continuous_y_%d.pkl' % i) for i in range(n_fs)]
    kwargs['w_filenames'] = [os.path.join(DATA_DIR, 'segment_continuous_w_%d.pkl' % i) for i in range(n_fs)]

valid_gen = PairGenerator(
    name_file,
    X_filenames,
    **kwargs)

#%% Define Model ###
model = ClassifyOnSegment(
    input_shape=(288, 384, n_input_channel),
    model_structure='pspnet',
    model_path=MODEL_DIR,
    encoder_weights='imagenet',
    n_segment_classes=kwargs['n_segment_classes'],
    n_classify_classes=kwargs['n_classify_classes'],
    eval_fn=evaluate_confusion_mat)

model.load(os.path.join(MODEL_DIR, 'bkp.model'))


#%% Collect predictions ###
os.makedirs(PRED_SAVE_DIR, exist_ok=True)
valid_gen.batch_with_name = True
pred_save = {"seg_preds": [], "seg_trues": [], "seg_ws": [],
             "cla_preds": [], "cla_trues": [], "cla_ws": [],
             "pred_names": []}
file_ct = 0
for batch in valid_gen:
    pred = model.model.predict(batch[0])
    seg_pred = scipy.special.softmax(pred[0], -1)
    seg_pred = seg_pred[..., 1] + seg_pred[..., 2]*2 + seg_pred[..., 3] * 3
    pred_save["seg_preds"].append(seg_pred)

    cla_pred = scipy.special.softmax(pred[1], -1)
    pred_save["cla_preds"].append(cla_pred)

    pred_save["pred_names"].extend(batch[-1])

    seg_true = batch[1][0]
    seg_true = seg_true[..., 1] + seg_true[..., 2]*2 + seg_true[..., 3] * 3
    pred_save["seg_trues"].append(seg_true)
    pred_save["seg_ws"].append(batch[1][0][..., -1])
    
    pred_save["cla_trues"].append(batch[1][1][..., :-1])
    pred_save["cla_ws"].append(batch[1][1][..., -1])


    if len(pred_save["seg_preds"]) >= 100:
        with open(os.path.join(PRED_SAVE_DIR, "%d.pkl" % file_ct), 'wb') as f:
            pickle.dump(pred_save, f)
        pred_save = {"seg_preds": [], "seg_trues": [], "seg_ws": [],
                     "cla_preds": [], "cla_trues": [], "cla_ws": [],
                     "pred_names": []}
        file_ct += 1

with open(os.path.join(PRED_SAVE_DIR, "%d.pkl" % file_ct), 'wb') as f:
    pickle.dump(pred_save, f)
