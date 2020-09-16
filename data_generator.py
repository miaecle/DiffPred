import numpy as np
import keras
import os
import pickle
from data_loader import get_ex_day, get_well
from data_augmentation import Augment


def enhance_weight_fp(_X, _y, _w, ratio=5):
    for i in range(_X.shape[0]):
        X = _X[i, :, :, 0]
        y = _y[i, :, :, :2]
        if y[:, :, 1].sum() > 0:
            continue
        thr = np.median(X) - 2 * np.std(X)
        _w[i][np.where(X < thr)] *= ratio
    return _w


def binarized_fluorescence_label(inputs):
    y, w = inputs
    if isinstance(y, np.ndarray):
        y_ct = np.where(y == 1)[0].size
        w_ct = np.where(np.sign(w) == 0)[0].size
    elif np.all(int(y) == y):
        y_ct = y
        w_ct = w
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
                 label_file=None,
                 augment=False,
                 N=None,
                 selected_inds=None,
                 shuffle_inds=False,
                 include_day=True,
                 batch_size=8,
                 n_segment_classes=2,
                 segment_class_weights=[1, 3],
                 segment_extra_weights=None,
                 segment_label_type='segmentation',
                 n_classify_classes=None,
                 classify_class_weights=None,
                 classify_label_fn=None,
                 sample_per_file=100,
                 allow_size=3,
                 **kwargs):
        self.X_filenames = X_filenames
        self.y_filenames = y_filenames
        self.w_filenames = w_filenames
        self.names = pickle.load(open(name_file, 'rb'))
        if not label_file is None:
            self.labels = pickle.load(open(label_file, 'rb'))
        else:
            self.labels = None

        # If to apply data augmentation
        if augment:
            self.augment = Augment(segment_label_type=segment_label_type)
        else:
            self.augment = None

        # Input details
        self.batch_size = batch_size
        self.include_day = include_day

        # Number of samples and batches
        self.sample_per_file = sample_per_file
        if N is None:
            self.N = len(self.names)
        else:
            self.N = N
        if selected_inds is None:
            selected_inds = np.arange(self.N)
        self.selected_inds = self.reorder_inds(selected_inds, shuffle_inds=shuffle_inds)

        # Label details
        self.n_segment_classes = n_segment_classes
        self.segment_class_weights = segment_class_weights
        self.segment_extra_weights = segment_extra_weights
        self.segment_label_type = segment_label_type
        if not self.n_segment_classes is None and self.segment_label_type == 'segmentation':
            assert len(self.segment_class_weights) == self.n_segment_classes

        self.n_classify_classes = n_classify_classes
        self.classify_class_weights = classify_class_weights
        self.classify_label_fn = classify_label_fn
        if not self.n_classify_classes is None:
            assert len(self.classify_class_weights) == self.n_classify_classes

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
        batch_labels = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            if i >= len(self.selected_inds):
                break
            ind = self.selected_inds[i]
            sample_X, sample_y, sample_w, sample_name = self.load_ind(ind)
            batch_names.append(sample_name)
            # Input
            batch_X.append(sample_X)

            # Output
            if not self.n_segment_classes is None:
                batch_y.append(sample_y)
                batch_w.append(sample_w)
            if not self.n_classify_classes is None:
                if not self.classify_label_fn is None:
                    batch_labels.append(self.classify_label_fn(self.labels[ind]))
                else:
                    batch_labels.append(self.labels[ind])

        batch_X = np.stack(batch_X, 0)
        if not self.n_segment_classes is None:
            batch_y = np.stack(batch_y, 0)
            batch_w = np.stack(batch_w, 0)
        else:
            batch_y = None
            batch_w = None

        if not self.n_classify_classes is None:
            batch_labels = np.stack(batch_labels, 0)
        else:
            batch_labels = None

        return self.prepare_inputs(batch_X, batch_y, batch_w, batch_names, batch_labels)


    def load_ind(self, ind, force_augment_off=False, random_seed=None):
        self.add_to_cache(ind)
        sample_name = self.names[ind]
        if ind in self.cache_X and ind in self.cache_y and ind in self.cache_w:
            sample_X = self.cache_X[ind]
            sample_y = self.cache_y[ind]
            sample_w = self.cache_w[ind]
        else:
            f_ind = ind // self.sample_per_file
            sample_X = pickle.load(open(self.X_filenames[f_ind], 'rb'))[ind]
            sample_y = pickle.load(open(self.y_filenames[f_ind], 'rb'))[ind]
            sample_w = pickle.load(open(self.w_filenames[f_ind], 'rb'))[ind]
        if not force_augment_off and not self.augment is None:
            if not random_seed is None:
               np.random.seed(random_seed) 
            sample_X, sample_y, sample_w = self.augment(sample_X, sample_y, sample_w)
        return sample_X, sample_y, sample_w, sample_name


    def reorder_inds(self, inds, shuffle_inds=False):
        def get_group(ind):
            return ind // self.sample_per_file
        groups = sorted(set(get_group(i) for i in inds))
        if shuffle_inds:
            np.random.shuffle(groups)
        return sorted(inds, key=lambda x: groups.index(get_group(x)))


    def prepare_inputs(self, X, y=None, w=None, names=None, labels=None):
        _X = self.prepare_features(X, names=names)
        _labels = self.prepare_labels(_X, y=y, w=w, labels=labels)
        return _X, _labels


    def prepare_features(self, X, names=None):
        if self.include_day:
            day_array = []
            for name in names:
                day = get_ex_day(name)[1][1:]
                day = float(day) if day != 'unknown' else 20 # ex2 is default to be in day 20
                day_array.append(day)
            day_nums = np.array(day_array).reshape((-1, 1, 1, 1))
            return np.concatenate([X, np.ones_like(X) * day_nums], 3)
        else:
            return X


    def prepare_labels(self, _X, y=None, w=None, labels=None):
        # Segment labels
        if not y is None:
            _y = np.zeros(list(_X.shape[:-1]) + [self.n_segment_classes+1])
            _w = np.zeros_like(w)
            if self.segment_label_type == 'segmentation':
                for i in range(self.n_segment_classes):
                    _y[..., i] = (y == i)
                    _w += w * (y == i) * self.segment_class_weights[i]
            elif self.segment_label_type == 'discretized_fl':
                y = y.astype(float)
                assert y.shape[-1] == self.n_segment_classes
                _y[..., :self.n_segment_classes] = y
            if not self.segment_extra_weights is None:
                _w = self.segment_extra_weights(_X, _y, _w)
            _y[..., -1] = _w
        else:
            _y = None

        # Classify labels
        if not labels is None:
            _y2 = np.zeros((labels.shape[0], self.n_classify_classes + 1))
            _w2 = np.zeros_like(labels[:, 1], dtype=float)
            for i in range(self.n_classify_classes):
                _y2[..., i] = (labels[:, 0] == i)
                _w2 += labels[:, 1] * (labels[:, 0] == i) * self.classify_class_weights[i]
            _y2[..., -1] = _w2
        else:
            _y2 = None

        if not y is None and not labels is None:
            return [_y, _y2]
        elif y is None and not labels is None:
            return _y2
        elif not y is None and labels is None:
            return _y
        else:
            raise ValueError


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
        if not self.labels is None:
            all_labels = {}
        else:
            all_labels = None

        file_ind = 0
        for i, ind in enumerate(inds):
            sample_X, sample_y, sample_w, sample_name = self.load_ind(ind, force_augment_off=True)
            all_Xs[i] = sample_X
            all_ys[i] = sample_y
            all_ws[i] = sample_w
            all_names[i] = sample_name
            if not self.labels is None:
                all_labels[i] = self.labels[ind]
            if save_path is not None and len(all_Xs) >= 100:
                with open(save_path + 'X_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(all_Xs, f)
                with open(save_path + 'y_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(all_ys, f)
                with open(save_path + 'w_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(all_ws, f)
                with open(save_path + 'names.pkl', 'wb') as f:
                    pickle.dump(all_names, f)
                if not self.labels is None:
                    with open(save_path + 'labels.pkl', 'wb') as f:
                        pickle.dump(all_labels, f)
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
            if not self.labels is None:
                with open(save_path + 'labels.pkl', 'wb') as f:
                    pickle.dump(all_labels, f)
            file_ind += 1
            all_Xs = {}
            all_ys = {}
            all_ws = {}
        if save_path:
            paths = ([save_path + 'X_%d.pkl' % i for i in range(file_ind)],\
                     [save_path + 'y_%d.pkl' % i for i in range(file_ind)],\
                     [save_path + 'w_%d.pkl' % i for i in range(file_ind)],\
                     save_path + 'names.pkl')
            if not self.labels is None:
                paths = paths + (save_path + 'labels.pkl',)
            return paths
        else:
            return all_Xs, all_ys, all_ws, all_names, all_labels


    def get_all_pairs(self, time_interval=[1, 3]):
        infos = {k: get_ex_day(v) + get_well(v) for k, v in self.names.items() if k in self.selected_inds}
        infos_reverse_mapping = {v: k for k, v in infos.items()}
        valid_pairs = []
        for ind_i in sorted(infos):
            d = infos[ind_i]
            if d[1] == 'Dunknown':
                continue
            for t in range(time_interval[0], time_interval[1]+1):
                new_d = (d[0], 'D%d' % (int(d[1][1:])+t), d[2], d[3])
                if new_d in infos_reverse_mapping:
                    ind_j = infos_reverse_mapping[new_d]
                    valid_pairs.append((ind_i, ind_j))
        return valid_pairs


    def cross_pair_save(self, time_interval=[1, 3], seed=None, save_path=None):
        valid_pairs = self.get_all_pairs(time_interval=time_interval)
        if not seed is None:
            np.random.seed(seed)
        np.random.shuffle(valid_pairs)

        all_Xs = {}
        all_ys = {}
        all_ws = {}
        all_names = {}
        if not self.labels is None:
            all_labels = {}
        else:
            all_labels = None

        file_ind = 0
        for i, pair in enumerate(valid_pairs):
            sample_X, _, _, sample_name_pre = self.load_ind(pair[0], force_augment_off=True)
            _, sample_y, sample_w, sample_name_post = self.load_ind(pair[1], force_augment_off=True)
            all_Xs[i] = sample_X
            all_ys[i] = sample_y
            all_ws[i] = sample_w
            all_names[i] = (sample_name_pre, sample_name_post)
            if not self.labels is None:
                all_labels[i] = self.labels[pair[1]]
            if save_path is not None and len(all_Xs) >= 100:
                with open(save_path + 'X_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(all_Xs, f)
                with open(save_path + 'y_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(all_ys, f)
                with open(save_path + 'w_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(all_ws, f)
                with open(save_path + 'names.pkl', 'wb') as f:
                    pickle.dump(all_names, f)
                if not self.labels is None:
                    with open(save_path + 'labels.pkl', 'wb') as f:
                        pickle.dump(all_labels, f)
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
            if not self.labels is None:
                with open(save_path + 'labels.pkl', 'wb') as f:
                    pickle.dump(all_labels, f)
            file_ind += 1
            all_Xs = {}
            all_ys = {}
            all_ws = {}
        if save_path:
            paths = ([save_path + 'X_%d.pkl' % i for i in range(file_ind)],\
                     [save_path + 'y_%d.pkl' % i for i in range(file_ind)],\
                     [save_path + 'w_%d.pkl' % i for i in range(file_ind)],\
                     save_path + 'names.pkl')
            if not self.labels is None:
                paths = paths + (save_path + 'labels.pkl',)
            return paths
        else:
            return all_Xs, all_ys, all_ws, all_names, all_labels



class PairGenerator(CustomGenerator) :

    def prepare_features(self, X, names=None):
        if self.include_day:
            day_array = []
            for name in names:
                day_pre = get_ex_day(name[0])[1][1:]
                day_post = get_ex_day(name[1])[1][1:]
                day_pre = float(day_pre)
                day_post = float(day_post)
                day_array.append((day_pre, day_post - day_pre))
            day_nums = np.array(day_array).reshape((-1, 1, 1, 2))
            _X = np.concatenate([X, np.ones_like(X[..., 0:1]) * day_nums], 3)
        else:
            _X = X
        return _X
