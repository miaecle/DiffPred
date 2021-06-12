import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
from functools import partial
from data_loader import get_identifier
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer, evaluate_confusion_mat
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives
from scipy.stats import spearmanr, pearsonr

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

def collect_preds(gen, model, input_process_fn=lambda x: x):
    gen.batch_with_name = True
    full_preds = {}
    for batch in gen:
        X = input_process_fn(batch[0])
        ids = [get_identifier(n) for n in batch[-1]]
        preds = model.predict_on_X(X)
        for pair in zip(ids, *preds):
            full_preds[pair[0]] = pair[1:]
    return full_preds

kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': None,
    'segment_class_weights': None,
    'segment_extra_weights': None,
    'segment_label_type': 'discrete',
    'n_classify_classes': None,
    'classify_class_weights': None,
    'classify_label_type': 'discrete',
}






### Settings ###
ROOT_DIRS = [
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_20/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_100/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_142/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_202/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_273/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_477/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_diabetes/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_LMNA_control/0-to-0/",
]
MODEL_DIR = "/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/random_split/0-to-inf_random/"
save_name = "random_split_0-to-inf"

input_process_fn = partial(augment_fixed_end, end=18)
def name_check_fn(n):
    identifier = get_identifier(n)
    day = int(identifier[2])
    return True if 4 <= day <= 12 else False

model = ClassifyOnSegment(
    input_shape=(288, 384, 3), 
    model_structure='pspnet', 
    model_path=MODEL_DIR, 
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)
model.load(os.path.join(MODEL_DIR, 'bkp.model'))

for ROOT_DIR in ROOT_DIRS:
    n_fs = len([f for f in os.listdir(ROOT_DIR) if f.startswith('X_') and f.endswith('.pkl')])
    X_filenames = [os.path.join(ROOT_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
    name_file = os.path.join(ROOT_DIR, 'names.pkl')

    names = pickle.load(open(name_file, 'rb'))
    selected_inds = [i for i, n in names.items() if name_check_fn(n)]

    test_gen = CustomGenerator(
        name_file,
        X_filenames, 
        augment=False,
        batch_with_name=True,
        selected_inds=selected_inds,
        **kwargs)

    preds = collect_preds(test_gen, model, input_process_fn=input_process_fn)
    # with open(os.path.join(ROOT_DIR, "%s_predictions.pkl" % save_name), "wb") as f:
    #     pickle.dump(preds, f)

    cla_preds = {k: preds[k][1] for k in preds}
    with open(os.path.join(ROOT_DIR, "%s_cla_predictions.pkl" % save_name), "wb") as f:
        pickle.dump(cla_preds, f)






### Settings ###
ROOT_DIRS = [
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_20/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_100/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_142/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_202/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_273/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_477/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_diabetes/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_LMNA_control/0-to-0/",
]
MODEL_DIR = "/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex/"
save_name = "ex_split_0-to-inf"

input_process_fn = partial(augment_fixed_end, end=18)
def name_check_fn(n):
    identifier = get_identifier(n)
    day = int(identifier[2])
    return True if 4 <= day <= 12 else False

model = ClassifyOnSegment(
    input_shape=(288, 384, 3), 
    model_structure='pspnet', 
    model_path=MODEL_DIR, 
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)
model.load(os.path.join(MODEL_DIR, 'bkp.model'))

for ROOT_DIR in ROOT_DIRS:
    n_fs = len([f for f in os.listdir(ROOT_DIR) if f.startswith('X_') and f.endswith('.pkl')])
    X_filenames = [os.path.join(ROOT_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
    name_file = os.path.join(ROOT_DIR, 'names.pkl')

    names = pickle.load(open(name_file, 'rb'))
    selected_inds = [i for i, n in names.items() if name_check_fn(n)]

    test_gen = CustomGenerator(
        name_file,
        X_filenames, 
        augment=False,
        batch_with_name=True,
        selected_inds=selected_inds,
        **kwargs)

    preds = collect_preds(test_gen, model, input_process_fn=input_process_fn)
    # with open(os.path.join(ROOT_DIR, "%s_predictions.pkl" % save_name), "wb") as f:
    #     pickle.dump(preds, f)

    cla_preds = {k: preds[k][1] for k in preds}
    with open(os.path.join(ROOT_DIR, "%s_cla_predictions.pkl" % save_name), "wb") as f:
        pickle.dump(cla_preds, f)






### Settings ###
ROOT_DIRS = [
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_20/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_100/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_142/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_202/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_273/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_477/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_diabetes/0-to-0/",
    "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_LMNA_control/0-to-0/",
]
MODEL_DIR = "/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/random_split/0-to-0_random/"
save_name = "random_split_0-to-0"

input_process_fn = partial(augment_fixed_interval, interval=0.)
def name_check_fn(n):
    identifier = get_identifier(n)
    day = int(identifier[2])
    return True if 10 <= day <= 16 else False

model = ClassifyOnSegment(
    input_shape=(288, 384, 3), 
    model_structure='pspnet', 
    model_path=MODEL_DIR, 
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)
model.load(os.path.join(MODEL_DIR, 'bkp.model'))

for ROOT_DIR in ROOT_DIRS:
    n_fs = len([f for f in os.listdir(ROOT_DIR) if f.startswith('X_') and f.endswith('.pkl')])
    X_filenames = [os.path.join(ROOT_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
    name_file = os.path.join(ROOT_DIR, 'names.pkl')

    names = pickle.load(open(name_file, 'rb'))
    selected_inds = [i for i, n in names.items() if name_check_fn(n)]

    test_gen = CustomGenerator(
        name_file,
        X_filenames, 
        augment=False,
        batch_with_name=True,
        selected_inds=selected_inds,
        **kwargs)

    preds = collect_preds(test_gen, model, input_process_fn=input_process_fn)
    # with open(os.path.join(ROOT_DIR, "%s_predictions.pkl" % save_name), "wb") as f:
    #     pickle.dump(preds, f)

    cla_preds = {k: preds[k][1] for k in preds}
    with open(os.path.join(ROOT_DIR, "%s_cla_predictions.pkl" % save_name), "wb") as f:
        pickle.dump(cla_preds, f)
