import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from data_loader import get_identifier, get_ex_day
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer, evaluate_confusion_mat, summarize_conf_mat
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives


def augment_fixed_end(X, end=15):
    assert X.shape[-1] == 2
    interval_slice = np.ones_like(X[..., 1:2]) * end - X[..., 1:2]
    _X = np.concatenate([X, interval_slice], -1)
    return _X


def augment_fixed_interval(X, interval=15):
    assert X.shape[-1] == 2
    interval_slice = np.ones_like(X[..., 1:2]) * interval
    _X = np.concatenate([X, interval_slice], -1)
    return _X


def filter_for_inf_predictor(batch):
    X, labels, batch_names = batch
    inds = []
    for i, name in enumerate(batch_names):
        if isinstance(name, list) or isinstance(name, tuple):
            name = name[0]
        if 4 <= int(get_identifier(name)[2]) <= 12:
            inds.append(i)
    return np.array(inds)


def get_data_gen(data_dir, data_gen, batch_size=8):
    #%% Define Dataset ###
    n_fs = len([f for f in os.listdir(data_dir) if f.startswith('X_') and f.endswith('.pkl')])
    name_file = os.path.join(data_dir, 'names.pkl')
    X_filenames = [os.path.join(data_dir, 'X_%d.pkl' % i) for i in range(n_fs)]

    kwargs = {
        'batch_size': batch_size,
        'shuffle_inds': False, # Datasets are usually pre-shuffled
        'batch_with_name': True,
        'augment': False,

    }
    if os.path.exists(os.path.join(data_dir, 'classify_continuous_labels.pkl')):
        kwargs.update({'n_classify_classes': 4,
                       'classify_class_weights': [1., 1., 1., 1.],
                       'classify_label_type': 'continuous',
                       'classify_label_file': os.path.join(data_dir, 'classify_continuous_labels.pkl')})
    if os.path.exists(os.path.join(data_dir, 'segment_continuous_y_%d.pkl' % (n_fs - 1))):
        kwargs.update({'n_segment_classes': 4,
                       'segment_class_weights': [1., 1., 1., 1.],
                       'segment_label_type': 'continuous',
                       'segment_y_files': [os.path.join(data_dir, 'segment_continuous_y_%d.pkl' % i) for i in range(n_fs)],
                       'segment_w_files': [os.path.join(data_dir, 'segment_continuous_w_%d.pkl' % i) for i in range(n_fs)]})
    valid_gen = data_gen(
        name_file,
        X_filenames,
        **kwargs)
    valid_gen.batch_with_name = True
    return valid_gen


def get_model(model_dir):
    model_folder_name = os.path.split(model_dir)[-1] if len(os.path.split(model_dir)[-1]) > 0 else os.path.split(os.path.split(model_dir)[0])[-1]
    if '0-to-0' in model_folder_name:
        n_input_channel = 2
    elif '0-to-inf' in model_folder_name:
        n_input_channel = 3
    else:
        raise ValueError("MODEL DIR not valid")

    #%% Define Model ###
    model = ClassifyOnSegment(
        input_shape=(288, 384, n_input_channel),
        model_structure='pspnet',
        model_path=model_dir,
        encoder_weights='imagenet',
        n_segment_classes=4,
        segment_class_weights=[1., 1., 1., 1.],
        n_classify_classes=4,
        classify_class_weights=[1., 1., 1., 1.],
        eval_fn=evaluate_confusion_mat)

    model.load(os.path.join(model_dir, 'bkp.model'))
    return model


#%% Collect predictions ###
def collect_preds(valid_gen, 
                  model, 
                  pred_save_dir, 
                  input_transform=None, 
                  input_filter=None, 
                  support_gen=None,
                  support_transform=None):
    os.makedirs(pred_save_dir, exist_ok=True)

    pred_save = {"seg_preds": [], "seg_trues": [], "seg_ws": [], "pred_names": []}
    file_ct = 0
    cla_preds = []
    cla_trues = []
    cla_ws = []
    pred_names = []

    if not support_gen is None:
        support_gen = iter(support_gen)
    for batch in valid_gen:
        X = batch[0]
        names = batch[-1]
        if not input_filter is None:
            inds = input_filter(batch)
        else:
            inds = np.arange(X.shape[0])
        if len(inds) == 0:
            continue
        if not input_transform is None:
            X = input_transform(X)

        if not support_gen is None:
            support_X = next(support_gen)[0]
            if not support_transform is None:
                support_X = support_transform(support_X)
            X = np.concatenate([X, support_X])

        pred = model.model.predict(X)
        seg_pred = scipy.special.softmax(pred[0], -1)
        seg_pred = seg_pred[..., 1] + seg_pred[..., 2]*2 + seg_pred[..., 3] * 3
        pred_save["seg_preds"].extend([seg_pred[i] for i in inds])

        cla_pred = scipy.special.softmax(pred[1], -1)
        cla_preds.extend([cla_pred[i] for i in inds])

        pred_save["pred_names"].extend([names[i] for i in inds])
        pred_names.extend([names[i] for i in inds])

        if not batch[1] is None:
            seg_true, cla_true = batch[1]
            seg_y = seg_true[..., 1] + seg_true[..., 2]*2 + seg_true[..., 3] * 3
            seg_w = seg_true[..., -1]
            cla_y, cla_w = cla_true[..., :-1], cla_true[..., -1]
            pred_save["seg_trues"].extend([seg_y[i] for i in inds])
            pred_save["seg_ws"].extend([seg_w[i] for i in inds])
        
            cla_trues.extend([cla_y[i] for i in inds])
            cla_ws.extend([cla_w[i] for i in inds])

        if len(pred_save["seg_preds"]) >= 500:
            with open(os.path.join(PRED_SAVE_DIR, "seg_%d.pkl" % file_ct), 'wb') as f:
                pickle.dump(pred_save, f)
            pred_save = {"seg_preds": [], "seg_trues": [], "seg_ws": [], "pred_names": []}
            file_ct += 1

    with open(os.path.join(PRED_SAVE_DIR, "seg_%d.pkl" % file_ct), 'wb') as f:
        pickle.dump(pred_save, f)
        
    with open(os.path.join(PRED_SAVE_DIR, "cla.pkl"), 'wb') as f:
        pickle.dump({"cla_preds": np.stack(cla_preds, 0),
                     "cla_trues": np.stack(cla_trues, 0) if len(cla_trues) > 0 else cla_trues,
                     "cla_ws": np.stack(cla_ws, 0) if len(cla_ws) > 0 else cla_ws,
                     "pred_names": pred_names}, f)


if __name__ == '__main__':
    DATA_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_142/ex1/0-to-0/'
    MODEL_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/random_split/0-to-inf_random/'
    GEN_TYPE = CustomGenerator # PairGenerator for 3 channel, CustomGenerator for 2 channel

    SUPPORT_DATA_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/'
    SUPPORT_GEN_TYPE = PairGenerator

    PRED_SAVE_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_142/ex1/pred_to-18/'
    input_transform = partial(augment_fixed_end, end=18)
    input_filter = filter_for_inf_predictor

    valid_gen = get_data_gen(DATA_DIR, GEN_TYPE, batch_size=8)
    model = get_model(MODEL_DIR)
    collect_preds(valid_gen, model, PRED_SAVE_DIR, input_transform=input_transform, input_filter=input_filter)

    # PRED_SAVE_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_142/ex1/pred_to-18_withsupp/'
    # valid_gen = get_data_gen(DATA_DIR, GEN_TYPE, batch_size=2)
    # support_gen = get_data_gen(SUPPORT_DATA_DIR, SUPPORT_GEN_TYPE, batch_size=6)
    # collect_preds(valid_gen, model, PRED_SAVE_DIR, input_transform=input_transform, input_filter=input_filter, support_gen=support_gen)
