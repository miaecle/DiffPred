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
ROOT_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/'
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/l1ex15_valid/'
TEST_DIR1 = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/l1ex7_valid/'
TEST_DIR2 = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/l3ex4_valid/'
SPLIT_FILE = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/ex_split.pkl'

MODEL_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex_pspnet/'
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(TEST_DIR1, exist_ok=True)
os.makedirs(TEST_DIR2, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def well_info(name):
    return get_identifier(name)[:2] + get_identifier(name)[3:]

### Train-Valid split ###

# # Generate split file
# names = pickle.load(open(os.path.join("/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-0/names.pkl"), 'rb'))
# unique_wells = sorted(set(well_info(n) for n in names.values()))
# np.random.seed(123)
# np.random.shuffle(unique_wells)
# valid_wells = [w for w in unique_wells if w[0] == 'line1_3R' and w[1].split('_')[0] == 'ex15']
# test_wells1 = [w for w in unique_wells if w[0] == 'line1_3R' and w[1].split('_')[0] == 'ex7']
# test_wells2 = [w for w in unique_wells if w[0] == 'line3_TNNI' and w[1].split('_')[0] == 'ex4']

# train_wells = sorted(set(unique_wells) - set(valid_wells) - set(test_wells1) - set(test_wells2))
# with open(SPLIT_FILE, 'wb') as f:
#     pickle.dump([train_wells, valid_wells, test_wells1, test_wells2], f)



# Setting up training set
n_fs = len([f for f in os.listdir(ROOT_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(ROOT_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(ROOT_DIR, 'segment_continuous_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(ROOT_DIR, 'segment_continuous_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(ROOT_DIR, 'names.pkl')
label_file = os.path.join(ROOT_DIR, 'classify_continuous_labels.pkl')

train_wells, valid_wells, test_wells1, test_wells2 = pickle.load(open(SPLIT_FILE, 'rb'))
cross_names = pickle.load(open(name_file, 'rb'))
valid_inds = [i for i, n in cross_names.items() if well_info(n[0]) in valid_wells]
train_inds = [i for i, n in cross_names.items() if well_info(n[0]) in train_wells]
test_inds1 = [i for i, n in cross_names.items() if well_info(n[0]) in test_wells1]
test_inds2 = [i for i, n in cross_names.items() if well_info(n[0]) in test_wells2]
print("N(train): %d" % len(train_inds))
print("N(valid): %d" % len(valid_inds))
print("N(test): %d + %d" % (len(test_inds1), len(test_inds2)))

kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 4,
    'segment_class_weights': [1, 2, 2, 2],
    'segment_extra_weights': None,
    'segment_label_type': 'continuous',
    'n_classify_classes': 4,
    'classify_class_weights': [1., 1., 2., 1.],
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

# Setting up validation set & test sets
validation_gens = []
for inds, save_dir in zip([valid_inds, test_inds1, test_inds2], [VALID_DIR, TEST_DIR1, TEST_DIR2]):
    # _ = train_gen.reorder_save(inds, save_path=save_dir)
    n_fs = len([f for f in os.listdir(save_dir) if f.startswith('X_') and f.endswith('.pkl')])
    X_filenames = [os.path.join(save_dir, 'X_%d.pkl' % i) for i in range(n_fs)]
    y_filenames = [os.path.join(save_dir, 'segment_continuous_y_%d.pkl' % i) for i in range(n_fs)]
    w_filenames = [os.path.join(save_dir, 'segment_continuous_w_%d.pkl' % i) for i in range(n_fs)]
    name_file = os.path.join(save_dir, 'names.pkl')
    label_file = os.path.join(save_dir, 'classify_continuous_labels.pkl')

    gen = PairGenerator(
        name_file,
        X_filenames,
        segment_y_files=y_filenames,
        segment_w_files=w_filenames,
        classify_label_file=label_file,
        **kwargs)
    validation_gens.append(gen)

### Training ###
print("Initiate Model", flush=True)
model = ClassifyOnSegment(
    input_shape=(288, 384, 3),
    segment_model_structure='pspnet',
    model_path=MODEL_DIR,
    encoder_weights='imagenet',
    n_segment_classes=4,
    n_classify_classes=4,
    eval_fn=evaluate_confusion_mat)

model.load(os.path.join(MODEL_DIR, 'bkp.model'))

print("Start Training", flush=True)
model.fit(train_gen,
          valid_gen=validation_gens[0],
          verbose=2,
          n_epochs=200)
model.save(os.path.join(MODEL_DIR, 'pspnet_ex_0-to-inf_0.model'))

