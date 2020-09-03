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

names = pickle.load(open(name_file, 'rb'))
train_inds = np.array([i for i, n in names.items() if not get_ex_day(n)[0] == 'ex1'])
valid_inds = np.array([i for i, n in names.items() if get_ex_day(n)[0] == 'ex1'])
print(len(train_inds))
print(len(valid_inds))

train_gen = CustomGenerator(X_filenames,
                            y_filenames,
                            w_filenames,
                            name_file,
                            label_file,
                            include_day=True,
                            batch_size=8,
                            n_classify_classes=2,
                            classify_class_weights=[1, 1],
                            selected_inds=train_inds,
                            extra_weights=enhance_weight_fp)

valid_filenames = train_gen.reorder_save(valid_inds, save_path=data_path+'temp_valid_')
valid_gen = CustomGenerator(*valid_filenames, 
                            include_day=True, 
                            batch_size=8,
                            n_classify_classes=2,
                            classify_class_weights=[1, 1])

model = ClassifyOnSegment(
    input_shape=(288, 384, 2), 
    model_structure='pspnet', 
    model_path='model_save', 
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2,
    segment_class_weights=[1, 3],
    classify_class_weights=[1, 1])

model2 = Segment(input_shape=(288, 384, 2), model_structure='pspnet')
model2.load('model_save/pspnet_all_test_include_day_ex1_1.model')
model = load_partial_weights(model, model2)

model.fit(train_gen,
          valid_gen=valid_gen,
          n_epochs=50)
model.save('./model_save/temp.model')
