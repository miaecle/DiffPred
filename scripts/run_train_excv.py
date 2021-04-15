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
ROOT_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/'
VALID_DIRS = {
    ('line1_3R', 'ex1_new'): '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex1_valid/',
    ('line1_3R', 'ex3_new'): '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex3_valid/',
    ('line1_3R', 'ex4_new'): '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex4_valid/',
    ('line1_3R', 'ex5_new'): '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex5_valid/',
    ('line1_3R', 'ex6_new'): '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex6_valid/',
    ('line1_3R', 'ex7_new'): '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex7_valid/',
    ('line1_3R', 'ex8'): '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex8_valid/',
    ('line3_TNNI', 'ex2'): '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l3ex2_valid/',
    ('line3_TNNI', 'ex4'): '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l3ex4_valid/',
}
MODEL_ROOT = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/excv/0-to-inf_discrete/'
MODEL_DIRS = {
    ('line1_3R', 'ex1_new'): MODEL_ROOT + 'l1ex1_valid/',
    ('line1_3R', 'ex3_new'): MODEL_ROOT + 'l1ex3_valid/',
    ('line1_3R', 'ex4_new'): MODEL_ROOT + 'l1ex4_valid/',
    ('line1_3R', 'ex5_new'): MODEL_ROOT + 'l1ex5_valid/',
    ('line1_3R', 'ex6_new'): MODEL_ROOT + 'l1ex6_valid/',
    ('line1_3R', 'ex7_new'): MODEL_ROOT + 'l1ex7_valid/',
    ('line1_3R', 'ex8'): MODEL_ROOT + 'l1ex8_valid/',
    ('line3_TNNI', 'ex2'): MODEL_ROOT + 'l3ex2_valid/',
    ('line3_TNNI', 'ex4'): MODEL_ROOT + 'l3ex4_valid/',
}

os.makedirs(ROOT_DIR, exist_ok=True)

### Base dataset setup ###
n_fs = len([f for f in os.listdir(ROOT_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(ROOT_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(ROOT_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(ROOT_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(ROOT_DIR, 'names.pkl')
label_file = os.path.join(ROOT_DIR, 'classify_discrete_labels.pkl')

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

base_gen = PairGenerator(
    name_file,
    X_filenames,
    segment_y_files=y_filenames,
    segment_w_files=w_filenames,
    classify_label_file=label_file,
    augment=True,
    **kwargs)

get_ex = lambda x: get_identifier(x)[:2]
all_exs = set(get_ex(n[0]) for i, n in base_gen.names.items())
for ex in all_exs:
    if not ex in VALID_DIRS or not ex in MODEL_DIRS:
        continue
    save_dir = VALID_DIRS[ex]
    model_dir = MODEL_DIRS[ex]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Setup train/valid datasets
    train_inds = [i for i, n in base_gen.names.items() if get_ex(n[0]) != ex]
    valid_inds = [i for i, n in base_gen.names.items() if get_ex(n[0]) == ex]
    assert len(train_inds) + len(valid_inds) == base_gen.N
    assert len(set(train_inds) & set(valid_inds)) == 0
    print("Valid with %s: %d / %d" % (str(ex), len(train_inds), len(valid_inds)))
    
    train_gen = PairGenerator(
        name_file,
        X_filenames,
        segment_y_files=y_filenames,
        segment_w_files=w_filenames,
        classify_label_file=label_file,
        selected_inds=train_inds,
        augment=True,
        **kwargs)
    
    # valid_filenames = base_gen.reorder_save(valid_inds, save_path=save_dir)
    n_fs = len([f for f in os.listdir(save_dir) if f.startswith('X_') and f.endswith('.pkl')])
    valid_X_filenames = [os.path.join(save_dir, 'X_%d.pkl' % i) for i in range(n_fs)]
    valid_y_filenames = [os.path.join(save_dir, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
    valid_w_filenames = [os.path.join(save_dir, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
    valid_name_file = os.path.join(save_dir, 'names.pkl')
    valid_label_file = os.path.join(save_dir, 'classify_discrete_labels.pkl')

    valid_gen = PairGenerator(
        valid_name_file,
        valid_X_filenames,
        segment_y_files=valid_y_filenames,
        segment_w_files=valid_w_filenames,
        classify_label_file=valid_label_file,
        **kwargs)
    
    assert len(set(train_gen.names[i] for i in train_gen.selected_inds) & \
        set(valid_gen.names[i] for i in valid_gen.selected_inds)) == 0
    

    print("Initiate Model", flush=True)
    model = ClassifyOnSegment(
        input_shape=(288, 384, 3),
        model_structure='pspnet',
        model_path=model_dir,
        encoder_weights='imagenet',
        n_segment_classes=2,
        n_classify_classes=2)


    ### Training ###
    # print("Start Training", flush=True)
    # model.fit(train_gen,
    #           valid_gen=valid_gen,
    #           verbose=2,
    #           n_epochs=40)

    ### Validation ###
    print("=========================")
    print("Loading from: %s" % os.path.join(model_dir, 'bkp.model'))
    model.load(os.path.join(model_dir, 'bkp.model'))
    pair_names = pickle.load(open(valid_name_file, 'rb'))
    labels = pickle.load(open(valid_label_file, 'rb'))

    print("%s: SCORE by interval" % str(ex))
    for interval in range(5, 22):
        selected_inds = [i for i in range(len(pair_names)) if (int(get_identifier(pair_names[i][1])[2]) - int(get_identifier(pair_names[i][0])[2])) == interval]
        if len(selected_inds) > 50:
            related_labels = np.array([labels[i] for i in selected_inds])
            related_labels = related_labels[np.where(related_labels[:, 1] > 0)][:, 0]
            if len(np.unique(related_labels)) > 1:
                print("%d: %d" % (interval, len(selected_inds)))
                data = PairGenerator(
                    valid_name_file,
                    valid_X_filenames,
                    segment_y_files=valid_y_filenames,
                    segment_w_files=valid_w_filenames,
                    classify_label_file=valid_label_file,
                    selected_inds=selected_inds,
                    **kwargs)
                evaluate_segmentation_and_classification(data, model)

    print("=========================")
    print("%s: SCORE by start day" % str(ex))
    for start_day in range(13):
        selected_inds = [i for i in range(len(pair_names)) if int(get_identifier(pair_names[i][0])[2]) == start_day]
        if len(selected_inds) > 50:
            related_labels = np.array([labels[i] for i in selected_inds])
            related_labels = related_labels[np.where(related_labels[:, 1] > 0)][:, 0]
            if len(np.unique(related_labels)) > 1:
                print("%d: %d" % (start_day, len(selected_inds)))
                data = PairGenerator(
                    valid_name_file,
                    valid_X_filenames,
                    segment_y_files=valid_y_filenames,
                    segment_w_files=valid_w_filenames,
                    classify_label_file=valid_label_file,
                    selected_inds=selected_inds,
                    **kwargs)
                evaluate_segmentation_and_classification(data, model)
