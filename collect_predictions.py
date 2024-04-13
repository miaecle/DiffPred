import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
import scipy
import argparse
import tempfile
from functools import partial
from keras.models import Model

from data_loader import get_identifier
from models import Classify, ClassifyOnSegment
from layers import evaluate_confusion_mat, evaluate_confusion_mat_classify_only
from data_generator import CustomGenerator, PairGenerator


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


def filter_for_inf_predictor(batch, day_min=4, day_max=12):
    X, labels, batch_names = batch
    inds = []
    for i, name in enumerate(batch_names):
        if isinstance(name, list) or isinstance(name, tuple):
            name = name[0]
        if day_min <= int(get_identifier(name)[2]) <= day_max:
            inds.append(i)
    return np.array(inds)


def filter_for_0_predictor(batch, day_min=7, day_max=20):
    X, labels, batch_names = batch
    inds = []
    for i, name in enumerate(batch_names):
        if isinstance(name, list) or isinstance(name, tuple):
            name = name[0]
        if day_min <= int(get_identifier(name)[2]) <= day_max:
            inds.append(i)
    return np.array(inds)


def get_data_gen(data_dir, data_gen, batch_size=8, with_label=False):
    n_fs = len([f for f in os.listdir(data_dir) if f.startswith('X_') and f.endswith('.pkl')])
    name_file = os.path.join(data_dir, 'names.pkl')
    X_filenames = [os.path.join(data_dir, 'X_%d.pkl' % i) for i in range(n_fs)]

    kwargs = {
        'batch_size': batch_size,
        'shuffle_inds': False,  # Datasets are usually pre-shuffled
        'batch_with_name': True,
        'augment': False,
    }
    if with_label:
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


def get_model(model_path):
    # Load model from path
    model_folder_name = os.path.split(model_path)[-2]
    if '0-to-0' in model_folder_name:
        n_input_channel = 2
    elif '0-to-inf' in model_folder_name:
        n_input_channel = 3
    else:
        raise ValueError("MODEL PATH not valid")

    if 'pspnet' in model_path:
        model_structure = 'pspnet'
    elif 'fpn' in model_path:
        model_structure = 'fpn'
    elif 'resnet' in model_path:
        model_structure = 'resnet34'
    else:
        model_structure = 'unet'

    if 'discrete' in model_path:
        n_classes = 2
    else:
        n_classes = 4

    if model_structure != 'resnet34':
        model = ClassifyOnSegment(
            input_shape=(288, 384, n_input_channel),
            segment_model_structure=model_structure,
            model_path=tempfile.mkdtemp(),
            encoder_weights='imagenet',
            n_segment_classes=n_classes,
            segment_class_weights=[1.] * n_classes,
            n_classify_classes=n_classes,
            classify_class_weights=[1.] * n_classes,
            eval_fn=evaluate_confusion_mat)
    else:
        model = Classify(
            input_shape=(288, 384, n_input_channel),
            fc_layers=[1024, 128],
            n_classes=n_classes,
            encoder_weights='imagenet',
            model_path=tempfile.mkdtemp(),
            eval_fn=evaluate_confusion_mat_classify_only,
            model_structure=model_structure)

    model.load(model_path)
    return model


def collect_preds(valid_gen,
                  model,
                  pred_save_dir,
                  input_transform=None,
                  input_filter=None):
    os.makedirs(pred_save_dir, exist_ok=True)

    pred_save = {"seg_preds": [], "seg_trues": [], "seg_ws": [], "pred_names": []}
    file_ct = 0
    cla_preds = []
    cla_trues = []
    cla_ws = []
    pred_names = []

    for batch in valid_gen:
        X = batch[0]
        names = batch[-1]
        if input_filter is not None:
            inds = input_filter(batch)
        else:
            inds = np.arange(X.shape[0])
        if len(inds) == 0:
            continue
        if input_transform is not None:
            X = input_transform(X)

        pred = model.model.predict(X)
        if (isinstance(pred, list) or isinstance(pred, tuple)) and len(pred) == 2:
            seg_pred, cla_pred = pred
        else:
            seg_pred = None
            cla_pred = pred
        if seg_pred is not None:
            seg_pred = scipy.special.softmax(seg_pred, -1)
            seg_pred = seg_pred[..., 1] + seg_pred[..., 2] * 2 + seg_pred[..., 3] * 3
            pred_save["seg_preds"].extend([seg_pred[i] for i in inds])
        cla_pred = scipy.special.softmax(cla_pred, -1)
        cla_preds.extend([cla_pred[i] for i in inds])

        if batch[1] is not None:
            if (isinstance(batch[1], list) or isinstance(batch[1], tuple)) and len(batch[1]) == 2:
                seg_true, cla_true = batch[1]
            else:
                seg_true = None
                cla_true = batch[1]
            cla_y, cla_w = cla_true[..., :-1], cla_true[..., -1]
            cla_trues.extend([cla_y[i] for i in inds])
            cla_ws.extend([cla_w[i] for i in inds])

            if seg_true is not None:
                seg_y = seg_true[..., 1] + seg_true[..., 2] * 2 + seg_true[..., 3] * 3
                seg_w = seg_true[..., -1]
                pred_save["seg_trues"].extend([seg_y[i] for i in inds])
                pred_save["seg_ws"].extend([seg_w[i] for i in inds])

        pred_save["pred_names"].extend([names[i] for i in inds])
        pred_names.extend([names[i] for i in inds])

        if len(pred_save["seg_preds"]) >= 500:
            with open(os.path.join(pred_save_dir, "seg_%d.pkl" % file_ct), 'wb') as f:
                pickle.dump(pred_save, f)
            pred_save = {"seg_preds": [], "seg_trues": [], "seg_ws": [], "pred_names": []}
            file_ct += 1

    with open(os.path.join(pred_save_dir, "seg_%d.pkl" % file_ct), 'wb') as f:
        pickle.dump(pred_save, f)

    with open(os.path.join(pred_save_dir, "cla.pkl"), 'wb') as f:
        pickle.dump({"cla_preds": np.stack(cla_preds, 0),
                     "cla_trues": np.stack(cla_trues, 0) if len(cla_trues) > 0 else cla_trues,
                     "cla_ws": np.stack(cla_ws, 0) if len(cla_ws) > 0 else cla_ws,
                     "pred_names": pred_names}, f)


def collect_embeddings(valid_gen,
                       model,
                       save_dir,
                       layer_names=[
                           'classify_head_pool1',
                           'classify_head_fc0',
                           'classify_head_fc1',
                       ],
                       input_transform=None,
                       input_filter=None):
    os.makedirs(save_dir, exist_ok=True)
    all_layer_names = [layer.name for layer in model.model.layers]
    assert len(all_layer_names) == len(set(all_layer_names))

    # Define models for extracting embeddings
    _models = []
    for layer_name in layer_names:
        assert layer_name in all_layer_names
        _models.append(Model(model.input, model.model.layers[all_layer_names.index(layer_name)].output))

    emb_save = {"embedding-%s" % layer_name: [] for layer_name in layer_names}
    emb_save["pred_names"] = []
    for batch in valid_gen:
        X = batch[0]
        names = batch[-1]
        if input_filter is not None:
            inds = input_filter(batch)
        else:
            inds = np.arange(X.shape[0])
        if len(inds) == 0:
            continue
        if input_transform is not None:
            X = input_transform(X)

        # Run embedding models
        for layer_name, _model in zip(layer_names, _models):
            emb = _model.predict(X)
            emb_save["embedding-%s" % layer_name].extend([emb[i] for i in inds])
        emb_save["pred_names"].extend([names[i] for i in inds])

    with open(os.path.join(save_dir, "embs.pkl"), 'wb') as f:
        pickle.dump(emb_save, f)
    return


def parse_args(cli=True):
    parser = argparse.ArgumentParser(description='Prediction script')

    # Input-output shape
    parser.add_argument('--input_data_dir',
                        type=str,
                        default="",
                        help="input data directory")
    parser.add_argument('--model_path',
                        type=str,
                        default="/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/random_split/0-to-inf_random/bkp.model",
                        help="model weight directory")
    parser.add_argument('--output_dir',
                        type=str,
                        default="",
                        help="output directory")
    parser.add_argument('--pred_target_day',
                        type=int,
                        default=18,
                        help="prediction target end point")
    parser.add_argument('--with_label',
                        action='store_true',
                        default=False,
                        help="if input contains full labels")
    if cli:
        args = parser.parse_args()
    else:
        args = parser.parse_args("")
    return args


if __name__ == '__main__':
    args = parse_args()
    data_dir = args.input_data_dir
    model_path = args.model_path
    pred_save_dir = args.output_dir
    target_day = args.pred_target_day
    with_label = args.with_label

    gen_fn = CustomGenerator if '0-to-0' in data_dir else PairGenerator
    valid_gen = get_data_gen(data_dir, gen_fn, batch_size=8, with_label=with_label)
    model = get_model(model_path)

    if '0-to-inf' in model_path:
        input_filter = partial(filter_for_inf_predictor, day_min=1, day_max=12)
        input_transform = partial(augment_fixed_end, end=target_day)
    elif '0-to-0' in model_path:
        input_filter = filter_for_0_predictor
        input_transform = None
    else:
        raise ValueError("model not supported")

    collect_preds(valid_gen, model, pred_save_dir, input_transform=input_transform, input_filter=input_filter)
