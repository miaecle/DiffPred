import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from data_loader import *
from segment_support import *
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer
from data_generator import CustomGenerator, PairGenerator, enhance_weight_fp
from scipy.stats import spearmanr, pearsonr


ROOT_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/linear_aligned_patches/'
MODEL_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex67/'

if not os.path.exists(MODEL_DIR):
  os.makedirs(MODEL_DIR)


data_path = os.path.join(ROOT_DIR, 'cross_infinite')
n_fs = len([f for f in os.listdir(data_path) if f.startswith('X')])
X_filenames = [os.path.join(data_path, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'names.pkl')
label_file = os.path.join(data_path, 'labels.pkl')

cross_names = pickle.load(open(name_file, 'rb'))
train_inds = np.array([i for i, n in cross_names.items() if not get_ex_day(n[0])[0] in ['ex6', 'ex7']])
valid_inds = np.array([i for i, n in cross_names.items() if get_ex_day(n[0])[0] in ['ex6', 'ex7']])
print(len(train_inds))
print(len(valid_inds))

kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 2,
    'segment_class_weights': [1, 3],
    'segment_extra_weights': enhance_weight_fp,
    'segment_label_type': 'segmentation',
    'n_classify_classes': 2,
    'classify_class_weights': [0.5, 0.5]
}

train_gen = PairGenerator(X_filenames,
                          y_filenames,
                          w_filenames,
                          name_file,
                          label_file=label_file,
                          augment=True,
                          selected_inds=train_inds,
                          **kwargs)

valid_filenames = train_gen.reorder_save(valid_inds, save_path=os.path.join(data_path, 'ex67_'))
n_fs = len([f for f in os.listdir(data_path) if f.startswith('ex67_X')])
X_filenames = [os.path.join(data_path, 'ex67_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'ex67_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'ex67_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'ex67_names.pkl')
label_file = os.path.join(data_path, 'ex67_labels.pkl')
valid_gen = PairGenerator(X_filenames,
                          y_filenames,
                          w_filenames,
                          name_file,
                          label_file=label_file,
                          **kwargs)

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
          n_epochs=200)
model.save(os.path.join(MODEL_DIR, 'pspnet_ex67_0-to-inf_0.model'))
