from data_loader import *
from segment_support import *
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer
from data_generator import CustomGenerator, PairGenerator, enhance_weight_fp, binarized_fluorescence_label
from scipy.stats import spearmanr, pearsonr

data_path = 'data/linear_aligned_patches/cross_1-to-3/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('X')])

X_filenames = [os.path.join(data_path, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'names.pkl')
label_file = os.path.join(data_path, 'labels.pkl')

names = pickle.load(open('data/linear_aligned_patches/merged_all/permuted_names.pkl', 'rb'))
unique_wells = sorted(set(get_ex_day(n)[:1] + get_well(n) for n in names.values()))
np.random.seed(123)
np.random.shuffle(unique_wells)
valid_wells = set(unique_wells[:int(0.2*len(unique_wells))])
train_wells = set(unique_wells[int(0.2*len(unique_wells)):])

cross_names = pickle.load(open(name_file, 'rb'))
valid_inds = [i for i, n in cross_names.items() if get_ex_day(n[0])[:1] + get_well(n[0]) in valid_wells]
train_inds = [i for i, n in cross_names.items() if get_ex_day(n[0])[:1] + get_well(n[0]) in train_wells]
# train_inds = np.array([i for i, n in cross_names.items() if not get_ex_day(n[0])[0] == 'ex1'])
# valid_inds = np.array([i for i, n in cross_names.items() if get_ex_day(n[0])[0] == 'ex1'])
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
    'classify_class_weights': [0.02, 0.02],
    'classify_label_fn': binarized_fluorescence_label
}

train_gen = PairGenerator(X_filenames,
                          y_filenames,
                          w_filenames,
                          name_file,
                          label_file=label_file,
                          augment=True,
                          selected_inds=train_inds)

valid_filenames = train_gen.reorder_save(valid_inds, save_path=data_path+'temp_valid_')
valid_gen = PairGenerator(*valid_filenames, **kwargs)


model = ClassifyOnSegment(
    input_shape=(288, 384, 3), 
    model_structure='pspnet', 
    model_path='model_save', 
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)

model2 = ClassifyOnSegment(input_shape=(288, 384, 2), model_structure='pspnet')
model2.load('model_save/pspnet_ex1_0-to-0_1.model')
model = load_partial_weights(model, model2)
model = fill_first_layer(model, model2)

model.fit(train_gen,
          valid_gen=valid_gen,
          n_epochs=50)
model.save('./model_save/pspnet_ex1_0-to-3_0.model')
