import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
from data_loader import get_identifier
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives
from scipy.stats import spearmanr, pearsonr

### Settings ###
ROOT_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/'
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex7_valid/'
TEST_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/l1ex1_valid/'
SPLIT_FILE = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/ex_split.pkl'

MODEL_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex/'
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def well_info(name):
    return get_identifier(name)[:2] + get_identifier(name)[3:]

### Train-Valid split ###

# Generate split file
# names = pickle.load(open(os.path.join("/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-0/names.pkl"), 'rb'))
# unique_wells = sorted(set(well_info(n) for n in names.values()))
# np.random.seed(123)
# np.random.shuffle(unique_wells)
# valid_wells = [w for w in unique_wells if w[0].startswith('line1') and w[1].startswith('ex7')]
# test_wells = [w for w in unique_wells if w[0].startswith('line1') and w[1].startswith('ex1')]
# train_wells = sorted(set(unique_wells) - set(valid_wells) - set(test_wells))
# with open(SPLIT_FILE, 'wb') as f:
#     pickle.dump([train_wells, valid_wells, test_wells], f)


# Setting up training set
n_fs = len([f for f in os.listdir(ROOT_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(ROOT_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(ROOT_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(ROOT_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(ROOT_DIR, 'names.pkl')
label_file = os.path.join(ROOT_DIR, 'classify_discrete_labels.pkl')

train_wells, valid_wells, test_wells = pickle.load(open(SPLIT_FILE, 'rb'))
cross_names = pickle.load(open(name_file, 'rb'))
train_inds = [i for i, n in cross_names.items() if well_info(n[0]) in train_wells]
valid_inds = [i for i, n in cross_names.items() if well_info(n[0]) in valid_wells]
test_inds = [i for i, n in cross_names.items() if well_info(n[0]) in test_wells]
print("N(train): %d" % len(train_inds))
print("N(valid): %d" % len(valid_inds))
print("N(test): %d" % len(test_inds))

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

train_gen = PairGenerator(
    name_file,
    X_filenames,
    segment_y_files=y_filenames,
    segment_w_files=w_filenames,
    classify_label_file=label_file,
    selected_inds=train_inds,
    augment=True,
    **kwargs)

# Setting up validation set & test set
valid_filenames = train_gen.reorder_save(valid_inds, save_path=VALID_DIR)
test_filenames = train_gen.reorder_save(test_inds, save_path=TEST_DIR)

n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_discrete_labels.pkl')

valid_gen = PairGenerator(
    name_file,
    X_filenames,
    segment_y_files=y_filenames,
    segment_w_files=w_filenames,
    classify_label_file=label_file,
    **kwargs)


n_fs = len([f for f in os.listdir(TEST_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(TEST_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(TEST_DIR, 'segment_discrete_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(TEST_DIR, 'segment_discrete_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(TEST_DIR, 'names.pkl')
label_file = os.path.join(TEST_DIR, 'classify_discrete_labels.pkl')

test_gen = PairGenerator(
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
    n_segment_classes=2,
    n_classify_classes=2)


print("Start Training", flush=True)
model.fit(train_gen,
          valid_gen=valid_gen,
          verbose=2,
          n_epochs=200)
model.save(os.path.join(MODEL_DIR, 'pspnet_ex_0-to-inf_0.model'))
