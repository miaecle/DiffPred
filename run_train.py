from data_loader import *
from segment_support import *
from models import Segment
from data_generator import CustomGenerator

X_filenames = ['data/linear_aligned_middle_patch/merged/merged_X_%d.pkl' % i for i in range(37)]
y_filenames = ['data/linear_aligned_middle_patch/merged/merged_y_%d.pkl' % i for i in range(37)]
w_filenames = ['data/linear_aligned_middle_patch/merged/merged_w_%d.pkl' % i for i in range(37)]
name_file = 'data/linear_aligned_middle_patch/merged/merged_names_perfed.pkl'

perfed_names = pickle.load(open(name_file, 'rb'))
train_inds = np.array([i for i, n in perfed_names.items() if not n.split('/')[2].startswith('ex4')])
valid_inds = np.array([i for i, n in perfed_names.items() if n.split('/')[2].startswith('ex4')])
print(len(train_inds))
print(len(valid_inds))

train_gen = CustomGenerator(X_filenames, y_filenames, w_filenames, name_file, include_day=False, batch_size=8, selected_inds=train_inds)
valid_gen = CustomGenerator(X_filenames, y_filenames, w_filenames, name_file, include_day=False, batch_size=8, selected_inds=valid_inds)


for i in range(5):
  model = Segment(input_shape=(288, 384, 1))
  model.fit(train_gen,
            valid_gen=valid_gen,
            n_epochs=20)
  model.save('./baseline%d.model' % i)
  del model