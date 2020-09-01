from data_loader import *
from segment_support import *
from models import Segment, Classify
from layers import load_partial_weights, GradualDefreeze
from data_generator import *

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

train_gen = ClassificationGenerator(X_filenames,
                        y_filenames,
                        w_filenames,
                        name_file,
                        label_file=label_file,
                        time_interval=[0, 1],
                        include_day=True,
                        batch_size=8,
                        selected_inds=train_inds)

valid_gen = ClassificationGenerator(X_filenames,
                        y_filenames,
                        w_filenames,
                        name_file,
                        label_file=label_file,
                        time_interval=[0, 1],
                        include_day=True,
                        batch_size=8,
                        selected_inds=valid_inds)

print(len(train_gen.selected_pair_inds))
print(len(valid_gen.selected_pair_inds))

model = Classify(input_shape=(288, 384, 3), model_path='model_save', encoder_weights='imagenet')
#model.call_backs.append(GradualDefreeze(order={0: 'none', 10: 'last', 20: 'full'}))
#ref_model = Segment(input_shape=(288, 384, 2))
#ref_model.load('./model_save/unet_all_include_day_test_ex1_0.model')
#model = load_partial_weights(model, ref_model)

model.fit(train_gen,
          valid_gen=valid_gen,
          n_epochs=50)
model.save('model_save/unet_classify_cross_include_day_test_ex1_0.model')

