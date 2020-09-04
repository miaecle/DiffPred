import numpy as np
import keras
import os
import pickle
from data_loader import get_ex_day, get_well
from data_augmentation import Augment
from segment_support import binarized_fluorescence_label


def enhance_weight_fp(_X, _y, _w, ratio=5):
    for i in range(_X.shape[0]):
        X = _X[i, :, :, 0]
        y = _y[i, :, :, :2]
        if y[:, :, 1].sum() > 0:
            continue
        thr = np.median(X) - 2 * np.std(X)
        _w[i][np.where(X < thr)] *= ratio
    return _w



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
                 include_day=True,
                 batch_size=8,
                 n_segment_classes=2,
                 segment_class_weights=[1, 3],
                 segment_extra_weights=None,
                 n_classify_classes=None,
                 classify_class_weights=None,
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
            self.augment = Augment()
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
            self.selected_inds = np.arange(self.N)
        else:
            self.selected_inds = selected_inds

        # Label details
        self.n_segment_classes = n_segment_classes
        self.segment_class_weights = segment_class_weights
        self.segment_extra_weights = segment_extra_weights
        if not self.n_segment_classes is None:
            assert len(self.segment_class_weights) == self.n_segment_classes

        self.n_classify_classes = n_classify_classes
        self.classify_class_weights = classify_class_weights
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
                batch_labels.append(binarized_fluorescence_label(*self.labels[ind]))

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


    def prepare_inputs(self, X, y=None, w=None, names=None, labels=None):
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
        
        # Segment labels
        if not y is None:
            _y = np.zeros(list(X.shape[:-1]) + [self.n_segment_classes+1])
            _w = np.zeros_like(w)
            for i in range(self.n_segment_classes):
                _y[..., i] = (y == i)
                _w += w * (y == i) * self.segment_class_weights[i]
            if not self.segment_extra_weights is None:
                _w = self.segment_extra_weights(_X, _y, _w)
            _y[..., -1] = _w
        else:
            _y = None

        # Classify labels
        if not labels is None:
            _y2 = np.zeros((labels.shape[0], self.n_classify_classes + 1))
            _w2 = np.zeros_like(labels[:, 1])
            for i in range(self.n_classify_classes):
                _y2[..., i] = (labels[:, 0] == i)
                _w2 += labels[:, 1] * (labels[:, 0] == i) * self.classify_class_weights[i]
            _y2[..., -1] = (_w2) * 0.2  # Hardcoded weight ratio for segmentation/classification task
        else:
            _y2 = None

        if not y is None and not labels is None:
            return _X, [_y, _y2]
        elif y is None and not labels is None:
            return _X, _y2
        elif not y is None and labels is None:
            return _X, _y
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
        batch_labels = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            if i >= len(self.selected_pair_inds):
                break
            ind_pair = self.selected_pair_inds[i]
            seed = int(np.random.rand() * 1e9)
            sample_X_pre, sample_y_pre, sample_w_pre, sample_name_pre = self.load_ind(ind_pair[0], random_seed=seed)
            sample_X_post, sample_y_post, sample_w_post, sample_name_post = self.load_ind(ind_pair[1], random_seed=seed)

            batch_names.append((sample_name_pre, sample_name_post))
            sample_X = np.concatenate([sample_X_pre, sample_X_post], 2)
            batch_X.append(sample_X)

            if not self.n_segment_classes is None:
                sample_y = np.stack([sample_y_pre, sample_y_post], 2)
                sample_w = np.stack([sample_w_pre, sample_w_post], 2)
                batch_y.append(sample_y)
                batch_w.append(sample_w)
            if not self.n_classify_classes is None:
                batch_labels.append(
                    binarized_fluorescence_label(*self.labels[ind_pair[0]]) + \
                    binarized_fluorescence_label(*self.labels[ind_pair[1]]))

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


    def prepare_inputs(self, X, y=None, w=None, names=None, labels=None):
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

        # Segment labels
        if not y is None:
            _y = []
            if 'pre' in self.output_mode['fl']:
                _y_pre = np.zeros(list(X.shape[:-1]) + [self.n_segment_classes+1])
                _w_pre = np.zeros_like(w[..., 0])
                for i in range(self.n_segment_classes):
                    _y_pre[..., i] = (y[..., 0] == i)
                    _w_pre += w[..., 0] * (y[..., 0] == i) * self.segment_class_weights[i]
                if not self.segment_extra_weights is None:
                    _w_pre = self.segment_extra_weights(X[..., 0:1], _y_pre, _w_pre)
                _y_pre[..., -1] = _w_pre
                _y.append(_y_pre)
            if 'post' in self.output_mode['fl']:
                _y_post = np.zeros(list(X.shape[:-1]) + [self.n_segment_classes+1])
                _w_post = np.zeros_like(w[..., 1])
                for i in range(self.n_segment_classes):
                    _y_post[..., i] = (y[..., 1] == i)
                    _w_post += w[..., 1] * (y[..., 1] == i) * self.segment_class_weights[i]
                if not self.segment_extra_weights is None:
                    _w_post = self.segment_extra_weights(X[..., 1:2], _y_post, _w_post)
                _y_post[..., -1] = _w_post
                _y.append(_y_post)
            _y = np.concatenate(_y, 3)
        else:
            _y = None

        # Classify labels
        if not labels is None:
            # 0: pre-y, 1: pre-w, 2: post-y, 3: post-w
            _y2 = np.zeros((labels.shape[0], self.n_classify_classes + 1))
            _w2 = np.zeros_like(labels[:, 3])
            for i in range(self.n_classify_classes):
                _y2[..., i] = (labels[:, 2] == i)
                _w2 += labels[:, 3] * (labels[:, 2] == i) * self.classify_class_weights[i]
            _y2[..., -1] = _w2
        else:
            _y2 = None

        if not y is None and not labels is None:
            return _X, [_y, _y2]
        elif y is None and not labels is None:
            return _X, _y2
        elif not y is None and labels is None:
            return _X, _y
        else:
            raise ValueError


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



class ClassificationGenerator(PairGenerator):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert self.n_segment_classes is None


    def __getitem__(self, idx):
        batch_X = []
        batch_names = []
        batch_labels = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            if i >= len(self.selected_pair_inds):
                break
            ind_pair = self.selected_pair_inds[i]
            sample_X_pre, _, _, _ = self.load_ind(ind_pair[0])

            batch_names.append((self.names[ind_pair[0]], self.names[ind_pair[1]]))
            sample_X = np.concatenate([sample_X_pre, sample_X_post], 2)
            batch_X.append(sample_X)
            batch_labels.append(self.labels[ind_pair[0]] + self.labels[ind_pair[1]])

        batch_X = np.stack(batch_X, 0)
        batch_labels = np.stack(batch_labels, 0)
        return self.prepare_inputs(batch_X, None, None, batch_names, batch_labels)


    def reorder_pair_inds(self, pairs):
        def get_pair_group(pair):
            return pair[0] // self.sample_per_file
        pair_groups = sorted(set(get_pair_group(p) for p in pairs))
        np.random.shuffle(pair_groups)
        return sorted(pairs, key=lambda x: pair_groups.index(get_pair_group(x)))
