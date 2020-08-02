from data_loader import *
from segment_support import *
from models import Segment

X = pickle.load(open('data/linear_aligned_middle_patch/merged_X.pkl', 'rb'))
y = pickle.load(open('data/linear_aligned_middle_patch/merged_y.pkl', 'rb'))
w = pickle.load(open('data/linear_aligned_middle_patch/merged_w.pkl', 'rb'))
names = pickle.load(open('data/linear_aligned_middle_patch/merged_names.pkl', 'rb'))

train_inds = np.array([i for i, n in enumerate(names) if not n.split('/')[2].startswith('ex7')])
valid_inds = np.array([i for i, n in enumerate(names) if n.split('/')[2].startswith('ex7')])

train_data = [X[train_inds], y[train_inds], w[train_inds]]
valid_data = [X[valid_inds], y[valid_inds], w[valid_inds]]

model = Segment()

model.fit(train_data,
          valid_data=valid_data,
          batch_size=8,
          epochs=100)

model.save('./temp.modelsave')

