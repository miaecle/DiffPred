import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from data_loader import *
from segment_support import *
from layers import *
from models import Segment, ClassifyOnSegment
from data_generator import CustomGenerator, enhance_weight_fp, binarized_fluorescence_label


DATA_ROOT = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/'
data_path = os.path.join(DATA_ROOT, 'discretized_fl', 'merged_all')
model_path = os.path.join(DATA_ROOT, 'model_save', '0-to-0_random_discretized')
os.makedirs(model_path, exist_ok=True)

kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 4,
    'segment_class_weights': [1, 2, 2, 2],
    'segment_extra_weights': None,
    'segment_label_type': 'discretized',
    'n_classify_classes': 4,
    'classify_class_weights': [0.02, 0.02, 0.02, 0.02],
    'classify_label_fn': None
}

valid_wells = pickle.load(open(os.path.join(DATA_ROOT, 'linear_aligned_patches', 'merged_all', 'random_valid_wells.pkl'), 'rb'))
train_wells = pickle.load(open(os.path.join(DATA_ROOT, 'linear_aligned_patches', 'merged_all', 'random_train_wells.pkl'), 'rb'))

n_fs = len([f for f in os.listdir(data_path) if f.startswith('permuted_X')])
X_filenames = [os.path.join(data_path, 'permuted_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'permuted_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'permuted_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'permuted_names.pkl')
label_file = os.path.join(data_path, 'permuted_labels.pkl')

names = pickle.load(open(name_file, 'rb'))
fl_inds = pickle.load(open(os.path.join(data_path, 'permuted_sample_inds_with_fl.pkl'), 'rb'))
valid_inds = [i for i, n in names.items() if get_ex_day(n)[:1] + get_well(n) in valid_wells and i in fl_inds]
train_inds = [i for i, n in names.items() if get_ex_day(n)[:1] + get_well(n) in train_wells and i in fl_inds]
print(len(train_inds))
print(len(valid_inds))

train_gen = CustomGenerator(X_filenames,
                            y_filenames,
                            w_filenames,
                            name_file,
                            label_file=label_file,
                            augment=True,
                            selected_inds=train_inds,
                            **kwargs)

# valid_filenames = train_gen.reorder_save(valid_inds, save_path=os.path.join(data_path, 'random_valid_'))
n_fs = len([f for f in os.listdir(data_path) if f.startswith('random_valid_X')])
X_filenames = [os.path.join(data_path, 'random_valid_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'random_valid_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'random_valid_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'random_valid_names.pkl')
label_file = os.path.join(data_path, 'random_valid_labels.pkl')
valid_gen = CustomGenerator(X_filenames,
                            y_filenames,
                            w_filenames,
                            name_file,
                            label_file=label_file,
                            **kwargs)

model = ClassifyOnSegment(
    input_shape=(288, 384, 2), 
    model_structure='pspnet', 
    model_path=model_path, 
    encoder_weights='imagenet',
    n_segment_classes=4,
    n_classify_classes=4,
    eval_fn=evaluate_confusion_mat)

model.fit(train_gen,
          valid_gen=valid_gen,
          n_epochs=50)
