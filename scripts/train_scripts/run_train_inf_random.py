import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
from data_loader import get_identifier
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer, evaluate_confusion_mat
from layers import weighted_cross_entropy, sparse_weighted_cross_entropy
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives
from scipy.stats import spearmanr, pearsonr

### Settings ###
ROOT_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/'
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/random_valid/'
SPLIT_FILE = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/random_split.pkl'

MODEL_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/random_split/0-to-inf_random/'
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def well_info(name):
    return get_identifier(name)[:2] + get_identifier(name)[3:]

### Train-Valid split ###

# Generate split file
# names = pickle.load(open(os.path.join("/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-0/names.pkl"), 'rb'))
# unique_wells = sorted(set(well_info(n) for n in names.values()))
# np.random.seed(123)
# np.random.shuffle(unique_wells)
# valid_wells = set(unique_wells[:int(0.2*len(unique_wells))])
# train_wells = set(unique_wells[int(0.2*len(unique_wells)):])
# with open(SPLIT_FILE, 'wb') as f:
#     pickle.dump([train_wells, valid_wells], f)


# Setting up training set
n_fs = len([f for f in os.listdir(ROOT_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(ROOT_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(ROOT_DIR, 'segment_continuous_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(ROOT_DIR, 'segment_continuous_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(ROOT_DIR, 'names.pkl')
label_file = os.path.join(ROOT_DIR, 'classify_continuous_labels.pkl')

train_wells, valid_wells = pickle.load(open(SPLIT_FILE, 'rb'))
cross_names = pickle.load(open(name_file, 'rb'))
valid_inds = [i for i, n in cross_names.items() if well_info(n[0]) in valid_wells]
train_inds = [i for i, n in cross_names.items() if well_info(n[0]) in train_wells]
print("N(train): %d" % len(train_inds))
print("N(valid): %d" % len(valid_inds))

kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 4,
    'segment_class_weights': [1, 2, 2, 2],
    'segment_extra_weights': None,
    'segment_label_type': 'continuous',
    'n_classify_classes': 4,
    'classify_class_weights': [1., 1., 1., 1.],
    'classify_label_type': 'continuous',
}


train_gen = PairGenerator(
    name_file,
    X_filenames, 
    segment_y_files=y_filenames, 
    segment_w_files=w_filenames,
    classify_label_file=label_file,
    selected_inds=train_inds,
    augment=True,
    **kwargs)

# Setting up validation set
# valid_filenames = train_gen.reorder_save(valid_inds, save_path=VALID_DIR)
n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_continuous_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_continuous_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_continuous_labels.pkl')

valid_gen = PairGenerator(
    name_file,
    X_filenames, 
    segment_y_files=y_filenames, 
    segment_w_files=w_filenames,
    classify_label_file=label_file,
    **kwargs)

### Training ###
print("Initiate Model", flush=True)
model = ClassifyOnSegment(
    input_shape=(288, 384, 3),
    model_structure='pspnet',
    model_path=MODEL_DIR,
    encoder_weights='imagenet',
    n_segment_classes=4,
    n_classify_classes=4,
    eval_fn=evaluate_confusion_mat)

# model.load(os.path.join(MODEL_DIR, 'bkp.model'))

print("Start Training", flush=True)
model.fit(train_gen,
          valid_gen=valid_gen,
          verbose=2,
          n_epochs=100)
model.save(os.path.join(MODEL_DIR, 'pspnet_random_0-to-inf_0.model'))
