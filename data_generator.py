import numpy as np
import keras
import os
import pickle
from data_loader import get_ex_day

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
               class_weights=[1, 10],
               selected_inds=None,
               sample_per_file=100):
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

    if selected_inds is None:
        self.selected_inds = np.arange(self.N)
    else:
        self.selected_inds = selected_inds

    self.cache_X = {}
    self.cache_y = {}
    self.cache_w = {}
    
    
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
        if not ind in self.cache_X:
            self.add_to_cache(ind)
        if ind in self.cache_X and ind in self.cache_y and ind in self.cache_w:
            batch_X.append(self.cache_X[ind])
            batch_y.append(self.cache_y[ind])
            batch_w.append(self.cache_w[ind])
        else:
            # In case cache doesn't work
            sample_X, sample_y, sample_w, _ = self.load_ind(ind)
            batch_X.append(sample_X)
            batch_y.append(sample_y)
            batch_w.append(sample_w)
        batch_names.append(self.names[ind])

    batch_X = np.stack(batch_X, 0)
    batch_y = np.stack(batch_y, 0)
    batch_w = np.stack(batch_w, 0)

    self.clean_cache()
    return self.prepare_inputs(batch_X, batch_y, batch_w, batch_names)

  def load_ind(self, ind):
    sample_name = self.names[ind]
    if ind in self.cache_X and ind in self.cache_y and ind in self.cache_w:
        return self.cache_X[ind], self.cache_y[ind], self.cache_w[ind], sample_name
    f_ind = ind // self.sample_per_file
    sample_X = pickle.load(open(self.X_filenames[f_ind], 'rb'))[ind]
    sample_y = pickle.load(open(self.y_filenames[f_ind], 'rb'))[ind]
    sample_w = pickle.load(open(self.w_filenames[f_ind], 'rb'))[ind]
    return sample_X, sample_y, sample_w, sample_name

  def prepare_inputs(self, X, y=None, w=None, names=None):
    if self.include_day:
        day_nums = np.array([float(get_ex_day(name)[1][1:]) for name in names]).reshape((-1, 1, 1, 1))
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
        _y[..., -1] = _w
    else:
        _y = None
    return _X, _y

  def add_to_cache(self, ind):
    f_ind = ind // self.sample_per_file
    self.cache_X.update(pickle.load(open(self.X_filenames[f_ind], 'rb')))
    self.cache_y.update(pickle.load(open(self.y_filenames[f_ind], 'rb')))
    self.cache_w.update(pickle.load(open(self.w_filenames[f_ind], 'rb')))
    return

  def clean_cache(self, force=False):
    if force or len(self.cache_X) > 2 * self.sample_per_file + 1:
        self.cache_X = {}
        self.cache_y = {}
        self.cache_w = {}
    return
