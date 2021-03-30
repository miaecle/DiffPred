import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
from data_loader import get_identifier
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer, evaluate_confusion_mat
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives
from scipy.stats import spearmanr, pearsonr

### Settings ###
ROOT_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-0_continuous/'
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-0_continuous/random_valid/'
SPLIT_FILE = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/random_split.pkl'

MODEL_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/0-to-0_random/'
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def well_info(name):
    return get_identifier(name)[:2] + get_identifier(name)[3:]


### Train-Valid split ###

# Setting up training set
n_fs = len([f for f in os.listdir(ROOT_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(ROOT_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(ROOT_DIR, 'segment_continuous_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(ROOT_DIR, 'segment_continuous_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(ROOT_DIR, 'names.pkl')
label_file = os.path.join(ROOT_DIR, 'classify_continuous_labels.pkl')

train_wells, valid_wells = pickle.load(open(SPLIT_FILE, 'rb'))
names = pickle.load(open(name_file, 'rb'))
valid_inds = [i for i, n in names.items() if well_info(n) in valid_wells]
train_inds = [i for i, n in names.items() if well_info(n) in train_wells]
print("N(train): %d" % len(train_inds))
print("N(valid): %d" % len(valid_inds))

kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 4,
    'segment_class_weights': [1, 1, 1, 1],
    'segment_extra_weights': None,
    'segment_label_type': 'continuous',
    'n_classify_classes': 4,
    'classify_class_weights': [1, 1, 1, 1],
    'classify_label_type': 'continuous',
}

train_gen = CustomGenerator(
    name_file,
    X_filenames, 
    segment_y_files=y_filenames, 
    segment_w_files=w_filenames,
    classify_label_file=label_file,
    selected_inds=train_inds,
    augment=True,
    **kwargs)

# Setting up validation set
valid_filenames = train_gen.reorder_save(valid_inds, save_path=VALID_DIR)
n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_continuous_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_continuous_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_continuous_labels.pkl')

valid_gen = CustomGenerator(
    name_file,
    X_filenames, 
    segment_y_files=y_filenames, 
    segment_w_files=w_filenames,
    classify_label_file=label_file,
    **kwargs)

### Training ###
model = ClassifyOnSegment(
    input_shape=(288, 384, 2), 
    model_structure='pspnet', 
    model_path=MODEL_DIR, 
    encoder_weights='imagenet',
    n_segment_classes=4,
    n_classify_classes=4,
    eval_fn=evaluate_confusion_mat)

model.fit(train_gen,
          valid_gen=valid_gen,
          n_epochs=200,
          verbose=2)
model.save(os.path.join(MODEL_DIR, 'pspnet_random_0-to-0_0.model'))

