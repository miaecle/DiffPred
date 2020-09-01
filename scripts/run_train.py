from data_loader import *
from segment_support import *
from models import Segment
from data_generator import CustomGenerator, enhance_weight_fp

data_path = 'data/linear_aligned_patches/merged_all/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('permuted_X')])

X_filenames = [os.path.join(data_path, 'permuted_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'permuted_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'permuted_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'permuted_names.pkl')

names = pickle.load(open(name_file, 'rb'))
train_inds = np.array([i for i, n in names.items() if not get_ex_day(n)[0] == 'ex1'])
valid_inds = np.array([i for i, n in names.items() if get_ex_day(n)[0] == 'ex1'])
print(len(train_inds))
print(len(valid_inds))

train_gen = CustomGenerator(X_filenames,
                            y_filenames,
                            w_filenames,
                            name_file,
                            include_day=True,
                            batch_size=8,
                            selected_inds=train_inds,
                            extra_weights=enhance_weight_fp)

valid_filenames = train_gen.reorder_save(valid_inds, save_path=data_path+'temp_valid_')
valid_gen = CustomGenerator(*valid_filenames, include_day=True, batch_size=8)

model = Segment(input_shape=(288, 384, 2), model_structure='unet', model_path='model_save', encoder_weights=None)
model.fit(train_gen,
          valid_gen=valid_gen,
          n_epochs=50)
model.save('./model_save/unet_all_include_day_test_ex1_1.model')

