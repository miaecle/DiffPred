import numpy as np
import keras
import os
import pickle
from data_loader import get_ex_day, get_well

def enhance_weight_fp(_X, _y, _w, ratio=5):
    for i in range(_X.shape[0]):
        X = _X[i, :, :, 0]
        y = _y[i, :, :, :2]
        if y[:, :, 1].sum() > 0:
            continue
        thr = np.median(X) - 2 * np.std(X)
        _w[i][np.where(X < thr)] *= ratio
    return _w

def binaried_fluorescence_label(y, w):
    if isinstance(y, int):
        y_ct = y
        w_ct = w
    elif isinstance(y, np.ndarray):
        y_ct = np.where(y == 1)[0].size
        w_ct = np.where(np.sign(w) == 0)[0].size
    else:
        raise ValueError("Data type not supported")
    if y_ct > 500:
        sample_y = 1
        sample_w = 1
    elif y_ct == 0 and w_ct < 600:
        sample_y = 0
        sample_w = 1
    else:
        sample_y = 0
        sample_w = 0
    return sample_y, sample_w

class CustomGenerator(keras.utils.Sequence) :
  def __init__(self, 
               X_filenames, 
               y_filenames, 
               w_filenames,
               name_file,
               N=None,
               include_day=False,
               batch_size=8,
               n_classes=2,
               class_weights=[1, 3],
               extra_weights=None,
               selected_inds=None,
               sample_per_file=100,
               allow_size=3,
               **kwargs):
    self.X_filenames = X_filenames
    self.y_filenames = y_filenames
    self.w_filenames = w_filenames
    self.names = pickle.load(open(name_file, 'rb'))
    self.batch_size = batch_size
    if N is None:
        self.N = len(self.names)
    else:
        self.N = N
    self.include_day = include_day
    self.n_classes = n_classes
    self.sample_per_file = sample_per_file
    self.class_weights = class_weights
    self.extra_weights = extra_weights

    if selected_inds is None:
        self.selected_inds = np.arange(self.N)
    else:
        self.selected_inds = selected_inds

    self.cache_X = {}
    self.cache_y = {}
    self.cache_w = {}
    self.allow_size = allow_size
    
    
  def __len__(self):
    return (np.ceil(len(self.selected_inds) / float(self.batch_size))).astype(np.int)
  
  def __getitem__(self, idx):
    batch_X = []
    batch_y = []
    batch_w = []
    batch_names = []
    for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
        if i >= len(self.selected_inds):
            break
        ind = self.selected_inds[i]
        sample_X, sample_y, sample_w, sample_name = self.load_ind(ind)
        batch_X.append(sample_X)
        batch_y.append(sample_y)
        batch_w.append(sample_w)
        batch_names.append(sample_name)

    batch_X = np.stack(batch_X, 0)
    batch_y = np.stack(batch_y, 0)
    batch_w = np.stack(batch_w, 0)
    return self.prepare_inputs(batch_X, batch_y, batch_w, batch_names)

  def load_ind(self, ind):
    self.add_to_cache(ind)
    sample_name = self.names[ind]
    if ind in self.cache_X and ind in self.cache_y and ind in self.cache_w:
        return self.cache_X[ind], self.cache_y[ind], self.cache_w[ind], sample_name
    else:
        f_ind = ind // self.sample_per_file
        sample_X = pickle.load(open(self.X_filenames[f_ind], 'rb'))[ind]
        sample_y = pickle.load(open(self.y_filenames[f_ind], 'rb'))[ind]
        sample_w = pickle.load(open(self.w_filenames[f_ind], 'rb'))[ind]
    return sample_X, sample_y, sample_w, sample_name

  def prepare_inputs(self, X, y=None, w=None, names=None):
    if self.include_day:
        day_array = []
        for name in names:
            day = get_ex_day(name)[1][1:]
            day = float(day) if day != 'unknown' else 20 # ex2 is default to be in day 20
            day_array.append(day)
        day_nums = np.array(day_array).reshape((-1, 1, 1, 1))
        _X = np.concatenate([X, np.ones_like(X) * day_nums], 3)
    else:
        _X = X
      
    if w is None:
        w = np.ones(list(X.shape[:-1]) + [1])
    if not y is None:
        _y = np.zeros(list(X.shape[:-1]) + [self.n_classes+1])
        _w = np.zeros_like(w)
        for i in range(self.n_classes):
            _y[..., i] = (y == i)
            _w += w * (y == i) * self.class_weights[i]
        if not self.extra_weights is None:
            _w = self.extra_weights(_X, _y, _w)
        _y[..., -1] = _w
    else:
        _y = None
    return _X, _y

  def add_to_cache(self, ind):
    if ind in self.cache_X and ind in self.cache_y and ind in self.cache_w:
        return
    f_ind = ind // self.sample_per_file
    if len(self.cache_X) > (self.allow_size * self.sample_per_file + 1):
        self.cache_X = pickle.load(open(self.X_filenames[f_ind], 'rb'))
        self.cache_y = pickle.load(open(self.y_filenames[f_ind], 'rb'))
        self.cache_w = pickle.load(open(self.w_filenames[f_ind], 'rb'))
    else:
        self.cache_X.update(pickle.load(open(self.X_filenames[f_ind], 'rb')))
        self.cache_y.update(pickle.load(open(self.y_filenames[f_ind], 'rb')))
        self.cache_w.update(pickle.load(open(self.w_filenames[f_ind], 'rb')))
    return

  def reorder_save(self, inds, save_path=None):
    assert np.max(inds) < self.N
    all_Xs = {}
    all_ys = {}
    all_ws = {}
    all_names = {}

    file_ind = 0
    for i, ind in enumerate(inds):
        sample_X, sample_y, sample_w, sample_name = self.load_ind(ind)
        all_Xs[i] = sample_X
        all_ys[i] = sample_y
        all_ws[i] = sample_w
        all_names[i] = sample_name
        if save_path is not None and len(all_Xs) >= 100:
            with open(save_path + 'X_%d.pkl' % file_ind, 'wb') as f:
                pickle.dump(all_Xs, f)
            with open(save_path + 'y_%d.pkl' % file_ind, 'wb') as f:
                pickle.dump(all_ys, f)
            with open(save_path + 'w_%d.pkl' % file_ind, 'wb') as f:
                pickle.dump(all_ws, f)
            with open(save_path + 'names.pkl', 'wb') as f:
                pickle.dump(all_names, f)
            file_ind += 1
            all_Xs = {}
            all_ys = {}
            all_ws = {}
    if save_path is not None and len(all_Xs) > 0:
        with open(save_path + 'X_%d.pkl' % file_ind, 'wb') as f:
            pickle.dump(all_Xs, f)
        with open(save_path + 'y_%d.pkl' % file_ind, 'wb') as f:
            pickle.dump(all_ys, f)
        with open(save_path + 'w_%d.pkl' % file_ind, 'wb') as f:
            pickle.dump(all_ws, f)
        with open(save_path + 'names.pkl', 'wb') as f:
            pickle.dump(all_names, f)
        file_ind += 1
        all_Xs = {}
        all_ys = {}
        all_ws = {}
    if save_path:
        return [save_path + 'X_%d.pkl' % i for i in range(file_ind)],\
               [save_path + 'y_%d.pkl' % i for i in range(file_ind)],\
               [save_path + 'w_%d.pkl' % i for i in range(file_ind)],\
               save_path + 'names.pkl'
    else:
        return all_Xs, all_ys, all_ws, all_names


class PairGenerator(CustomGenerator) :
  def __init__(self,
               *args,
               output_mode={'pc': ['pre'], 'fl': ['post']},
               time_interval=[6, 10],
               **kwargs):

    super().__init__(*args, **kwargs)
    self.output_mode = output_mode
    self.time_interval = time_interval
    if 'seed' in kwargs:
        np.random.seed(kwargs['seed'])
    pair_inds = self.get_all_pairs()
    self.selected_pair_inds = self.reorder_pair_inds(pair_inds)

  def __len__(self):
    return (np.ceil(len(self.selected_pair_inds) / float(self.batch_size))).astype(np.int)
  
  def __getitem__(self, idx):
    batch_X = []
    batch_y = []
    batch_w = []
    batch_names = []
    for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
        if i >= len(self.selected_pair_inds):
            break
        ind_pair = self.selected_pair_inds[i]
        sample_X_pre, sample_y_pre, sample_w_pre, _ = self.load_ind(ind_pair[0])
        sample_X_post, sample_y_post, sample_w_post, _ = self.load_ind(ind_pair[1])

        sample_X = np.concatenate([sample_X_pre, sample_X_post], 2)
        sample_y = np.stack([sample_y_pre, sample_y_post], 2)
        sample_w = np.stack([sample_w_pre, sample_w_post], 2)
        batch_X.append(sample_X)
        batch_y.append(sample_y)
        batch_w.append(sample_w)
        batch_names.append(tuple(self.names[ind] for ind in ind_pair))

    batch_X = np.stack(batch_X, 0)
    batch_y = np.stack(batch_y, 0)
    batch_w = np.stack(batch_w, 0)
    return self.prepare_inputs(batch_X, batch_y, batch_w, batch_names)

  def prepare_inputs(self, X, y=None, w=None, names=None):
    """
    X, y, w: batch * length * width * (pre+post)
    """
    _X = []
    if 'pre' in self.output_mode['pc']:
        _X.append(X[..., 0])
    if 'post' in self.output_mode['pc']:
        _X.append(X[..., 1])
    _X = np.stack(_X, 3)

    if self.include_day:
        day_array = []
        for name in names:
            day_pre = float(get_ex_day(name[0])[1][1:])
            day_post = float(get_ex_day(name[1])[1][1:])
            day_array.append([day_pre, day_post])
        day_nums = np.array(day_array).reshape((-1, 1, 1, 2))
        day_nums = day_nums * np.ones_like(_X[..., :1])
        _X = np.concatenate([_X, day_nums], 3)
    
    if w is None:
        w = np.ones(list(X.shape[:-1]) + [2])

    if not y is None:
        _y = []
        if 'pre' in self.output_mode['fl']:
            _y_pre = np.zeros(list(X.shape[:-1]) + [self.n_classes+1])
            _w_pre = np.zeros_like(w[..., 0])
            for i in range(self.n_classes):
                _y_pre[..., i] = (y[..., 0] == i)
                _w_pre += w[..., 0] * (y[..., 0] == i) * self.class_weights[i]
            if not self.extra_weights is None:
                _w_pre = self.extra_weights(_X[..., 0:1], _y_pre, _w_pre)
            _y_pre[..., -1] = _w_pre
            _y.append(_y_pre)
        if 'post' in self.output_mode['fl']:
            _y_post = np.zeros(list(X.shape[:-1]) + [self.n_classes+1])
            _w_post = np.zeros_like(w[..., 1])
            for i in range(self.n_classes):
                _y_post[..., i] = (y[..., 1] == i)
                _w_post += w[..., 1] * (y[..., 1] == i) * self.class_weights[i]
            if not self.extra_weights is None:
                _w_post = self.extra_weights(_X[..., 1:2], _y_post, _w_post)
            _y_post[..., -1] = _w_post
            _y.append(_y_post)
        _y = np.concatenate(_y, 3)
    else:
        _y = None
    return _X, _y

  def get_all_pairs(self):
    infos = {k: get_ex_day(v) + get_well(v) for k, v in self.names.items() if k in self.selected_inds}
    infos_reverse_mapping = {v: k for k, v in infos.items()}
    valid_pairs = []
    for ind_i in sorted(infos):
        d = infos[ind_i]
        if d[1] == 'Dunknown':
            continue
        for t in range(self.time_interval[0], self.time_interval[1]+1):
            new_d = (d[0], 'D%d' % (int(d[1][1:])+t), d[2], d[3])
            if new_d in infos_reverse_mapping:
                ind_j = infos_reverse_mapping[new_d]
                valid_pairs.append((ind_i, ind_j))
    return valid_pairs
  
  def reorder_pair_inds(self, pairs):
    def get_pair_group(pair):
        return (pair[0] // self.sample_per_file, pair[1] // self.sample_per_file)
    pair_groups = sorted(set(get_pair_group(p) for p in pairs))
    np.random.shuffle(pair_groups)
    return sorted(pairs, key=lambda x: pair_groups.index(get_pair_group(x)))

class ClassificationGenerator(PairGenerator) :
  def __init__(self,
               *args,
               label_file=None,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.label_file = label_file
    self.labels = pickle.load(open(self.label_file, 'rb'))

  def __getitem__(self, idx):
    batch_X = []
    batch_y = []
    batch_w = []
    batch_names = []
    batch_labels = []
    for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
        if i >= len(self.selected_pair_inds):
            break
        ind_pair = self.selected_pair_inds[i]
        sample_X, _, _, sample_name = self.load_ind(ind_pair[0])
        name_post = self.names[ind_pair[1]]

        # Customized label
        label_post = self.labels[ind_pair[1]]
        sample_y, sample_w = binaried_fluorescence_label(*label_post)

        batch_X.append(sample_X)
        batch_y.append(sample_y)
        batch_w.append(sample_w)
        batch_names.append((sample_name, name_post))

    batch_X = np.stack(batch_X, 0)
    batch_y = np.array(batch_y)
    batch_w = np.array(batch_w)
    return self.prepare_inputs(batch_X, batch_y, batch_w, batch_names)

  def prepare_inputs(self, X, y=None, w=None, names=None, labels=None):
    _X = X
    if self.include_day:
        day_array = []
        for name in names:
            day_pre = float(get_ex_day(name[0])[1][1:])
            day_post = float(get_ex_day(name[1])[1][1:])
            day_array.append([day_pre, day_post])
        day_nums = np.array(day_array).reshape((-1, 1, 1, 2))
        day_nums = day_nums * np.ones_like(_X[..., :1])
        _X = np.concatenate([_X, day_nums], 3)
    _y = np.stack([1-y, y, w], 1)
    return _X, _y

  def reorder_pair_inds(self, pairs):
    def get_pair_group(pair):
        return pair[0] // self.sample_per_file
    pair_groups = sorted(set(get_pair_group(p) for p in pairs))
    np.random.shuffle(pair_groups)
    return sorted(pairs, key=lambda x: pair_groups.index(get_pair_group(x)))
