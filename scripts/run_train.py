from data_loader import *
from segment_support import *
from layers import *
from models import Segment, ClassifyOnSegment
from data_generator import CustomGenerator, enhance_weight_fp

data_path = 'data/linear_aligned_patches/merged_all/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('permuted_X')])

X_filenames = [os.path.join(data_path, 'permuted_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'permuted_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'permuted_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'permuted_names.pkl')
label_file = os.path.join(data_path, 'permuted_labels.pkl')

# data_path = 'data/linear_aligned_patches/merged_all_in_order/'
# n_fs = len([f for f in os.listdir(data_path) if f.startswith('X')])

# X_filenames = [os.path.join(data_path, 'X_%d.pkl' % i) for i in range(n_fs)]
# y_filenames = [os.path.join(data_path, 'y_%d.pkl' % i) for i in range(n_fs)]
# w_filenames = [os.path.join(data_path, 'w_%d.pkl' % i) for i in range(n_fs)]
# name_file = os.path.join(data_path, 'names.pkl')
# label_file = os.path.join(data_path, 'labels.pkl')

names = pickle.load(open(name_file, 'rb'))
unique_wells = sorted(set(get_ex_day(n)[:1] + get_well(n) for n in names.values()))
np.random.seed(123)
np.random.shuffle(unique_wells)
valid_wells = set(unique_wells[:int(0.2*len(unique_wells))])
valid_inds = [i for i, n in names.items() if get_ex_day(n)[:1] + get_well(n) in valid_wells]
train_wells = set(unique_wells[int(0.2*len(unique_wells)):])
train_inds = [i for i, n in names.items() if get_ex_day(n)[:1] + get_well(n) in train_wells]
print(len(train_inds))
print(len(valid_inds))

train_gen = CustomGenerator(X_filenames,
                            y_filenames,
                            w_filenames,
                            name_file,
                            label_file,
                            augment=True,
                            batch_size=16,
                            segment_class_weights=[1, 3],
                            n_classify_classes=2,
                            classify_class_weights=[0.02, 0.02],
                            selected_inds=train_inds,
                            extra_weights=enhance_weight_fp)

valid_filenames = train_gen.reorder_save(valid_inds, save_path=data_path+'temp_valid_')
valid_gen = CustomGenerator(*valid_filenames, 
                            batch_size=16,
                            n_classify_classes=2,
                            classify_class_weights=[0.02, 0.02])

model = ClassifyOnSegment(
    input_shape=(288, 384, 2), 
    model_structure='pspnet', 
    model_path='model_save', 
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)

model.load('./model_save/pspnet_random_0-to-0_0.model')
model.fit(train_gen,
          valid_gen=valid_gen,
          n_epochs=50)
