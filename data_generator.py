import numpy as np
import keras
import os

class CustomGenerator(keras.utils.Sequence) :
  def __init__(self, 
               X_filenames, 
               y_filenames, 
               w_filenames, 
               batch_size, 
               N, 
               n_classes=2,
               class_weights=[1, 10],
               selected_inds=None,
               sample_per_file=100):
    self.X_filenames = X_filenames
    self.y_filenames = y_filenames
    self.w_filenames = w_filenames
    self.batch_size = batch_size
    self.N = N
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
    
    
  def __len__(self) :
    return (np.ceil(len(self.selected_inds) / float(self.batch_size))).astype(np.int)
  
  def __getitem__(self, idx) :
    batch_X = []
    batch_y = []
    batch_w = []
    for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
        if i >= len(self.selected_inds):
            break
        ind = self.selected_inds[i]
        if not ind in self.cache_X:
            self.add_to_cache(ind)

        batch_X.append(self.cache_X[ind])
        batch_y.append(self.cache_y[ind])
        batch_w.append(self.cache_w[ind])

    batch_X = np.stack(batch_X, 0)
    batch_y = np.stack(batch_y, 0)
    batch_w = np.stack(batch_w, 0)

    self.clean_cache()
    return self.prepare_inputs(batch_X, batch_y, batch_w)

  def prepare_inputs(self, X, y=None, w=None):
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
    return X, _y

  def add_to_cache(self, ind):
    f_ind = ind // self.sample_per_file
    self.cache_X.update(pickle.load(open(self.X_filenames[f_ind], 'rb')))
    self.cache_y.update(pickle.load(open(self.y_filenames[f_ind], 'rb')))
    self.cache_w.update(pickle.load(open(self.w_filenames[f_ind], 'rb')))
    return

  def clean_cache(self):
    if len(self.cache_X) > 2 * self.sample_per_file + 1:
        del self.cache_X
        del self.cache_y
        del self.cache_w
        self.cache_X = {}
        self.cache_y = {}
        self.cache_w = {}
    return