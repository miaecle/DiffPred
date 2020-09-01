from data_loader import *
from segment_support import *
from models import Segment
from data_generator import CustomGenerator, PairGenerator, enhance_weight_fp

data_path = 'data/linear_aligned_patches/merged_all_in_order/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('X')])

X_filenames = [os.path.join(data_path, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'names.pkl')

names = pickle.load(open(name_file, 'rb'))
valid_inds = np.random.choice(np.arange(len(names)), (2000,), replace=False)
train_inds = np.array([i for i in np.arange(len(names)) if not i in valid_inds])
print(len(train_inds))
print(len(valid_inds))

output_mode = {'pc': ['pre'], 'fl': ['post']}
time_interval = [6, 10]
train_gen = PairGenerator(X_filenames,
                          y_filenames,
                          w_filenames,
                          name_file,
                          include_day=True,
                          batch_size=8,
                          selected_inds=train_inds,
                          extra_weights=enhance_weight_fp,
                          time_interval=time_interval,
                          output_mode=output_mode)
valid_gen = PairGenerator(X_filenames,
                          y_filenames,
                          w_filenames,
                          name_file,
                          include_day=True,
                          batch_size=8,
                          selected_inds=valid_inds,
                          extra_weights=enhance_weight_fp,
                          time_interval=time_interval,
                          output_mode=output_mode)

model = Segment(input_shape=(288, 384, 3), model_structure='pspnet', model_path='model_save')
model.fit(train_gen,
          valid_gen=valid_gen,
          n_epochs=20)
model.save('./model_save/pspnet_cross_test_ex1_0.model')

