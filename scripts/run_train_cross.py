from data_loader import *
from segment_support import *
from models import Segment, ClassifyOnSegment
from data_generator import CustomGenerator, PairGenerator, enhance_weight_fp
from scipy.stats import spearmanr, pearsonr

data_path = 'data/linear_aligned_patches/merged_all_in_order/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('X')])

X_filenames = [os.path.join(data_path, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'names.pkl')
label_file = os.path.join(data_path, 'labels.pkl')

names = pickle.load(open(name_file, 'rb'))
unique_wells = sorted(set(get_ex_day(n)[:1] + get_well(n) for n in names.values()))
np.random.seed(123)
np.random.shuffle(unique_wells)
valid_wells = set(unique_wells[:int(0.2*len(unique_wells))])
valid_inds = [i for i, n in names.items() if get_ex_day(n)[:1] + get_well(n) in valid_wells]
train_wells = set(unique_wells[int(0.2*len(unique_wells)):])
train_inds = [i for i, n in names.items() if get_ex_day(n)[:1] + get_well(n) in train_wells]

output_mode = {'pc': ['pre'], 'fl': ['post']}
time_interval = [1, 3]
train_gen = PairGenerator(X_filenames,
                          y_filenames,
                          w_filenames,
                          name_file,
                          label_file,
                          include_day=True,
                          batch_size=8,
                          n_classify_classes=2,
                          classify_class_weights=[1, 1],
                          augment=True,
                          selected_inds=train_inds,
                          extra_weights=enhance_weight_fp,
                          time_interval=time_interval,
                          output_mode=output_mode)

valid_filenames = train_gen.reorder_save(valid_inds, save_path=data_path+'temp_valid_')
valid_gen = PairGenerator(*valid_filenames
                          include_day=True,
                          batch_size=8,
                          n_classify_classes=2,
                          classify_class_weights=[1, 1],
                          augment=True,
                          selected_inds=train_inds,
                          extra_weights=enhance_weight_fp,
                          time_interval=time_interval,
                          output_mode=output_mode)

model = ClassifyOnSegment(
    input_shape=(288, 384, 3), 
    model_structure='pspnet', 
    model_path='model_save', 
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2,
    segment_class_weights=[1, 3],
    classify_class_weights=[1, 1])

model.fit(train_gen,
          valid_gen=train_gen,
          n_epochs=50)
model.save('./model_save/temp.model')
