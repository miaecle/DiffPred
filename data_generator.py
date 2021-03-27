import numpy as np
import keras
import os
import pickle
from data_loader import well_id_from_name, exp_id_from_name, exp_day_from_name, get_identifier
from data_augmentation import Augment


def enhance_weight_for_false_positives(X, labels, ratio=5):
    n_samples = X.shape[0]
    y = labels[0][..., :-1]
    w = labels[0][..., -1:]
    for i in range(n_samples):
        _X = X[i, ..., 0]
        _y = y[i]
        if (_y[..., 0] < 0.5).sum() > _X.size * 0.02:
            # Skipping if there are any true positives
            continue
        thr = np.median(_X) - 2 * np.std(_X)
        w[i][np.where(_X < thr)] *= ratio
    segment_labels = np.concatenate([y, w], -1)
    return [segment_labels, labels[1]]


class CustomGenerator(keras.utils.Sequence) :
    def __init__(self, 
                 name_file,
                 X_files, 
                 segment_y_files=None, 
                 segment_w_files=None,
                 n_segment_classes=2,
                 segment_class_weights=[1, 3],
                 segment_extra_weights=None,
                 segment_label_type='discrete',
                 classify_label_file=None,
                 n_classify_classes=2,
                 classify_class_weights=[1, 3],
                 classify_label_type='discrete',
                 sample_per_file=100,
                 cache_file_num=3,
                 N=None,
                 selected_inds=None,
                 shuffle_inds=False,
                 augment=False,
                 batch_size=8,
                 **kwargs):
        """ Customized generator for predicting differentiation outcome
        """

        self.names = pickle.load(open(name_file, 'rb'))
        self.X_files = X_files

        # Segmentation label details
        self.segment_y_files = segment_y_files
        self.segment_w_files = segment_w_files
        self.n_segment_classes = n_segment_classes
        self.segment_class_weights = segment_class_weights
        self.segment_extra_weights = segment_extra_weights
        self.segment_label_type = segment_label_type
        if not self.n_segment_classes is None:
            assert len(self.segment_class_weights) == self.n_segment_classes

        # Classification label details
        if not classify_label_file is None:
            classify_labels = pickle.load(open(classify_label_file, 'rb'))
            self.classify_y = {k: v[0] for k, v in classify_labels.items()}
            self.classify_w = {k: v[1] for k, v in classify_labels.items()}
        else:
            self.classify_y = None
            self.classify_w = None
        self.n_classify_classes = n_classify_classes
        self.classify_class_weights = classify_class_weights
        self.classify_label_type = classify_label_type
        if not self.n_classify_classes is None:
            assert len(self.classify_class_weights) == self.n_classify_classes

        # Cache for inputs & segmentation labels
        self.sample_per_file = sample_per_file
        self.cache_X = {}
        self.cache_segment_y = {}
        self.cache_segment_w = {}
        self.cache_file_num = cache_file_num

        # Number of samples
        self.N = N if not N is None else len(self.names)
        if selected_inds is None:
            selected_inds = np.arange(self.N)
        self.selected_inds = self.reorder_inds(selected_inds, shuffle_inds=shuffle_inds)

        # Data augmentation setup
        self.augment = Augment(segment_label_type=segment_label_type) if augment else None

        # Input details
        self.batch_size = batch_size


    def reorder_inds(self, inds, shuffle_inds=False):
        def get_group(ind):
            # Get which file the sample is from
            return ind // self.sample_per_file
        groups = sorted(set(get_group(i) for i in inds))
        if shuffle_inds:
            np.random.shuffle(groups)
        return sorted(inds, key=lambda x: groups.index(get_group(x)))

    
    def __len__(self):
        return (np.ceil(len(self.selected_inds) / float(self.batch_size))).astype(np.int)
 

    def __getitem__(self, idx):
        batch_names = []
        batch_X = []
        batch_segment_y = []
        batch_segment_w = []
        batch_classify_y = []
        batch_classify_w = []
        get_segment_label = not self.n_segment_classes is None
        get_classify_label = not self.n_classify_classes is None

        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            if i >= len(self.selected_inds):
                break
            ind = self.selected_inds[i]
            sample_X, sample_segment_y, sample_segment_w, sample_name = self.load_ind(ind)
            if sample_segment_y is None or sample_segment_w is None:
                get_segment_label = False

            # Sample name
            batch_names.append(sample_name)
            # Input
            batch_X.append(sample_X)
            # Label & weight
            if get_segment_label:
                batch_segment_y.append(sample_segment_y)
                batch_segment_w.append(sample_segment_w)
            if get_classify_label:
                batch_classify_y.append(self.classify_y[ind])
                batch_classify_w.append(self.classify_w[ind])

        batch_X = np.stack(batch_X, 0)
        n_samples_in_batch = batch_X.shape[0]
        if get_segment_label:
            batch_segment_y = np.stack(batch_segment_y, 0)
            batch_segment_w = np.stack(batch_segment_w, 0)
        else:
            batch_segment_y = None
            batch_segment_w = None

        if get_classify_label:
            batch_classify_y = np.stack(batch_classify_y, 0).reshape((n_samples_in_batch, -1))
            batch_classify_w = np.stack(batch_classify_w, 0).reshape((n_samples_in_batch, -1))
        else:
            batch_classify_y = None
            batch_classify_w = None

        return self.prepare_inputs(
            batch_names,
            batch_X, 
            batch_segment_y, 
            batch_segment_w, 
            batch_classify_y,
            batch_classify_w)


    def load_ind(self, ind, force_augment_off=False, random_seed=None):
        self.add_to_cache(ind)
        f_ind = ind // self.sample_per_file
        sample_name = self.names[ind]

        # Sample input
        if ind in self.cache_X:
            sample_X = self.cache_X[ind]
        else:
            sample_X = pickle.load(open(self.X_filenames[f_ind], 'rb'))[ind]
        
        n_rows, n_cols = sample_X.shape[:2]
        sample_X = sample_X.reshape((n_rows, n_cols, -1)).astype(float)

        # Sample segmentation label
        if not self.n_segment_classes is None:
            if ind in self.cache_segment_y and ind in self.cache_segment_w:
                sample_segment_y = self.cache_segment_y[ind]
                sample_segment_w = self.cache_segment_w[ind]
            else:
                sample_segment_y = pickle.load(open(self.segment_y_files[f_ind], 'rb'))[ind]
                sample_segment_w = pickle.load(open(self.segment_w_files[f_ind], 'rb'))[ind]
            # Cast segment label
            if not sample_segment_y is None:
                sample_segment_y = sample_segment_y.reshape((n_rows, n_cols, -1))
                sample_segment_w = sample_segment_w.reshape((n_rows, n_cols, -1))
                if self.segment_label_type == 'discrete': # Label as class
                    sample_segment_y = sample_segment_y.astype(int)
                elif self.segment_label_type == 'continuous': # Label as class prob
                    sample_segment_y = sample_segment_y.astype(float)
                else:
                    raise ValueError("segmentation label type unknown")
        else:
            sample_segment_y = None
            sample_segment_w = None

        if not force_augment_off and not self.augment is None:
            if not random_seed is None:
               np.random.seed(random_seed)
            if not sample_segment_y is None and not sample_segment_w is None:
                sample_X, sample_segment_y, sample_segment_w = \
                    self.augment(sample_X, sample_segment_y, sample_segment_w)
            else:
                sample_X = self.augment(sample_X, None, None)
        return sample_X, sample_segment_y, sample_segment_w, sample_name


    def add_to_cache(self, ind):
        if ind in self.cache_X and ind in self.cache_segment_y and ind in self.cache_segment_w:
            return
        f_ind = ind // self.sample_per_file
        
        f_X = pickle.load(open(self.X_files[f_ind], 'rb'))
        f_segment_y = pickle.load(open(self.segment_y_files[f_ind], 'rb')) if self.segment_y_files else {}
        f_segment_w = pickle.load(open(self.segment_w_files[f_ind], 'rb')) if self.segment_w_files else {}

        if len(self.cache_X) > (self.cache_file_num * self.sample_per_file + 1):
            self.cache_X = f_X
            self.cache_segment_y = f_segment_y
            self.cache_segment_w = f_segment_w
        else:
            self.cache_X.update(f_X)
            self.cache_segment_y.update(f_segment_y)
            self.cache_segment_w.update(f_segment_w)
        return


    def prepare_inputs(self, names, X, seg_y=None, seg_w=None, cl_y=None, cl_w=None):
        _X = self.prepare_features(X, names=names)
        _labels = self.prepare_labels(_X, seg_y=seg_y, seg_w=seg_w, cl_y=cl_y, cl_w=cl_w)
        if not self.segment_extra_weights is None:
            _labels = self.segment_extra_weights(_X, _labels)
        return _X, _labels


    def prepare_features(self, X, names=None):
        day_array = []
        for name in names:
            day = exp_day_from_name(name)
            day = float(day) if day != 'unknown' else 20 # Unknown da
            day_array.append(day)
        day_nums = np.array(day_array).reshape((-1, 1, 1, 1))
        return np.concatenate([X, np.ones_like(X) * day_nums], 3)


    def prepare_labels(self, X, seg_y=None, seg_w=None, cl_y=None, cl_w=None):
        def setup_label(y, w, n_classes, class_weights, label_type):
            assert n_classes == len(class_weights)
            assert ((y.shape[-1] == n_classes) or (y.shape[-1] == 1))
            assert w.shape[-1] == 1
            assert y.shape[:-1] == w.shape[:-1]
            _y = np.zeros(list(y.shape[:-1]) + [n_classes,])
            _w = np.zeros(list(y.shape[:-1]) + [1,])
            if label_type == 'discrete':
                for i in range(n_classes):
                    _y[..., i] = (y[..., 0] == i)
                    _w[..., 0] += w[..., 0] * (y[..., 0] == i) * class_weights[i]
            elif label_type == 'continuous':
                assert y.shape == _y.shape
                _y = y
                for i in range(n_classes):
                    _w[..., 0] += w[..., 0] * y[..., i] * class_weights[i]
            labels = np.concatenate([_y, _w], -1)
            return labels

        # Segment labels
        if not seg_y is None and not seg_w is None:
            segment_labels = setup_label(seg_y,
                                         seg_w,
                                         self.n_segment_classes,
                                         self.segment_class_weights,
                                         self.segment_label_type)
        else:
            segment_labels = None

        # Classify labels
        if not cl_y is None and not cl_w is None:
            classify_labels = setup_label(cl_y,
                                          cl_w,
                                          self.n_classify_classes,
                                          self.classify_class_weights,
                                          self.classify_label_type)
        else:
            classify_labels = None

        return [segment_labels, classify_labels]


    def reorder_save(self, 
                     inds, 
                     save_path=None,
                     write_segment_labels=True,
                     write_classify_labels=True):
        if self.segment_y_files is None or self.segment_w_files is None:
            print("Segmentation labels will not be saved")
            write_segment_labels = False
        if self.classify_y is None or self.classify_w is None:
            print("Classification labels will not be saved")
            write_classify_labels = False

        assert np.max(inds) < self.N
        save_names = {}
        save_Xs = {}
        save_segment_ys = {}
        save_segment_ws = {}
        save_classify_labels = {}

        file_ind = 0
        for i, ind in enumerate(inds):
            sample_X, sample_segment_y, sample_segment_w, sample_name = self.load_ind(ind, force_augment_off=True)
            save_names[i] = sample_name
            save_Xs[i] = sample_X
            if write_segment_labels:
                save_segment_ys[i] = sample_segment_y
                save_segment_ws[i] = sample_segment_w
            if write_classify_labels:
                save_classify_labels[i] = (self.classify_y[ind], self.classify_w[ind])

            if save_path is not None and (len(save_Xs) >= 100 or (i == len(inds) - 1)):
                with open(save_path + 'names.pkl', 'wb') as f:
                    pickle.dump(save_names, f)
                with open(save_path + 'X_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(save_Xs, f)
                if write_segment_labels:
                    with open(save_path + 'segment_%s_y_%d.pkl' % (self.segment_label_type, file_ind), 'wb') as f:
                        pickle.dump(save_segment_ys, f)
                    with open(save_path + 'segment_%s_w_%d.pkl' % (self.segment_label_type, file_ind), 'wb') as f:
                        pickle.dump(save_segment_ws, f)
                if write_classify_labels:
                    with open(save_path + 'classify_%s_labels.pkl' % self.classify_label_type, 'wb') as f:
                        pickle.dump(save_classify_labels, f)
                file_ind += 1
                save_Xs = {}
                save_segment_ys = {}
                save_segment_ws = {}
        return file_ind


    def get_all_pairs(self, time_interval=[1, 3]):
        valid_pcs = {}
        valid_fls = {}
        for k in sorted(self.names.keys()):
            x, y, w, n = self.load_ind(k)
            assert isinstance(n, str)
            if x is not None:
                valid_pcs[k] = get_identifier(n)
            if y is not None:
                valid_fls[k] = get_identifier(n)

        fls_reverse_mapping = {v: k for k, v in valid_fls.items()}

        valid_pairs = []
        for ind_i in sorted(valid_pcs):
            d = valid_pcs[ind_i]
            if d[2] == 'unknown':
                continue
            for t in range(time_interval[0], time_interval[1]+1):
                new_d = (d[0], d[1], str(int(d[2]) + t), d[3], d[4])
                if new_d in fls_reverse_mapping:
                    ind_j = fls_reverse_mapping[new_d]
                    valid_pairs.append((ind_i, ind_j))
        return valid_pairs


    def shrink_pairs(self, pairs):
        def pair_identifier(p):
            id_from = get_identifier(self.names[p[0]])
            id_to = get_identifier(self.names[p[1]])
            assert id_from[:2] == id_to[:2]
            assert id_from[3:] == id_to[3:]
            pair_id = tuple([id_from[0],
                             id_from[1],
                             int(id_from[2]), 
                             int(id_to[2]), 
                             id_from[3], 
                             id_from[4]])
            return pair_id

        def adjacent_identifiers(pair_id):
            well_id = (pair_id[0], pair_id[1], pair_id[4], pair_id[5])
            pair_day = (pair_id[2], pair_id[3])
            adjacent_ids = []
            for d_from in range(pair_day[0] - 2, pair_day[0] + 3):
                for d_to in range(pair_day[1] - 2, pair_day[1] + 3):
                    adjacent_ids.append((well_id[0], well_id[1], d_from, d_to, well_id[2], well_id[3]))
            return adjacent_ids

        out_pairs = []
        selected = set()
        for p in pairs:
            pair_id = pair_identifier(p)
            if not pair_id in selected:
                out_pairs.append(p)
                for _adjacent_id in adjacent_identifiers(pair_id):
                    selected.add(_adjacent_id)
        return out_pairs


    def cross_pair_save(self, 
                        time_interval=[1, 3], 
                        shrink=False, 
                        seed=None, 
                        save_path=None,
                        write_segment_labels=True,
                        write_classify_labels=True):
        valid_pairs = self.get_all_pairs(time_interval=time_interval)
        if not seed is None:
            np.random.seed(seed)
        np.random.shuffle(valid_pairs)
        if shrink:
            valid_pairs = self.shrink_pairs(valid_pairs)

        if self.segment_y_files is None or self.segment_w_files is None:
            print("Segmentation labels will not be saved")
            write_segment_labels = False
        if self.classify_y is None or self.classify_w is None:
            print("Classification labels will not be saved")
            write_classify_labels = False

        save_names = {}
        save_Xs = {}
        save_segment_ys = {}
        save_segment_ws = {}
        save_classify_labels = {}

        file_ind = 0
        for i, pair in enumerate(valid_pairs):
            sample_X, _, _, sample_name_pre = self.load_ind(pair[0], force_augment_off=True)
            _, sample_segment_y, sample_segment_w, sample_name_post = self.load_ind(pair[1], force_augment_off=True)
            save_names[i] = (sample_name_pre, sample_name_post)
            save_Xs[i] = sample_X
            save_segment_ys[i] = sample_segment_y
            save_segment_ws[i] = sample_segment_w

            if write_classify_labels:
                save_classify_labels[i] = (self.classify_y[pair[1]], self.classify_w[pair[1]])

            if save_path is not None and len(save_Xs) >= 100:
                with open(save_path + 'names.pkl', 'wb') as f:
                    pickle.dump(save_names, f)
                with open(save_path + 'X_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(save_Xs, f)
                if write_segment_labels:
                    with open(save_path + '%s_segmentation_y_%d.pkl' % (self.segment_label_type, file_ind), 'wb') as f:
                        pickle.dump(save_segment_ys, f)
                    with open(save_path + '%s_segmentation_w_%d.pkl' % (self.segment_label_type, file_ind), 'wb') as f:
                        pickle.dump(save_segment_ws, f)
                if write_classify_labels:
                    with open(save_path + '%s_classification_labels.pkl' % self.classify_label_type, 'wb') as f:
                        pickle.dump(save_classify_labels, f)
                file_ind += 1
                save_Xs = {}
                save_segment_ys = {}
                save_segment_ws = {}
        if save_path is not None and len(save_Xs) > 0:
            with open(save_path + 'names.pkl', 'wb') as f:
                pickle.dump(save_names, f)
            with open(save_path + 'X_%d.pkl' % file_ind, 'wb') as f:
                pickle.dump(save_Xs, f)
            if write_segment_labels:
                with open(save_path + '%s_segmentation_y_%d.pkl' % (self.segment_label_type, file_ind), 'wb') as f:
                    pickle.dump(save_segment_ys, f)
                with open(save_path + '%s_segmentation_w_%d.pkl' % (self.segment_label_type, file_ind), 'wb') as f:
                    pickle.dump(save_segment_ws, f)
            if write_classify_labels:
                with open(save_path + '%s_classification_labels.pkl' % self.classify_label_type, 'wb') as f:
                    pickle.dump(save_classify_labels, f)
            file_ind += 1
        return file_ind



class PairGenerator(CustomGenerator) :

    def prepare_features(self, X, names=None):
        if self.include_day:
            day_array = []
            for name in names:
                day_pre = exp_day_from_name(name[0])
                day_post = exp_day_from_name(name[1])
                day_pre = float(day_pre)
                day_post = float(day_post)
                day_array.append((day_pre, day_post - day_pre))
            day_nums = np.array(day_array).reshape((-1, 1, 1, 2))
            _X = np.concatenate([X, np.ones_like(X[..., 0:1]) * day_nums], 3)
        else:
            _X = X
        return _X
