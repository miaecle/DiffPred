import numpy as np
import os
import cv2
import pickle
import copy
from data_loader import get_identifier, load_image_pair
from segment_support import generate_mask, generate_weight, generate_fluorescence_labels, quantize_fluorescence
from segment_support import adjust_contrast, binarized_fluorescence_label


def preprocess(pairs, 
               output_path=None, 
               preprocess_filter=lambda x: True,
               target_size=(384, 288),
               labels=['discrete', 'continuous'], 
               raw_label_preprocess=lambda x: x,
               linear_align=True,
               shuffle=True,
               seed=None):
    if not seed is None:
        np.random.seed(seed)

    # Sanity check
    pairs = [p for p in pairs if p[0] is not None and preprocess_filter(p)]
    for p in pairs:
        if p[1] is not None:
            assert get_identifier(p[0]) == get_identifier(p[1])

    # Sort
    pairs = sorted(pairs)
    if shuffle:
        np.random.shuffle(pairs)

    # Featurize data
    target_shape = (target_size[1], target_size[0], -1) # Note that cv2 and numpy have reversed axis ordering
    names = {}
    Xs = {}
    segment_discrete_ys = {}
    segment_discrete_ws = {}
    segment_continuous_ys = {}
    segment_continuous_ws = {}
    classify_discrete_labels = {}
    classify_continuous_labels = {}
    file_ind = 0
    for ind, pair in enumerate(pairs):
        identifier = get_identifier(pair[0])
        names[ind] = pair[0]
        try:
            # Input feature (phase contrast image)
            pair_dat = load_image_pair(pair)
            position_code = identifier[-1]
            if linear_align and position_code in ['1', '3', '7', '9'] and pair_dat[1] is not None:
                mask = generate_mask(pair_dat)
            else:
                mask = np.ones_like(pair_dat[0])
            X = adjust_contrast(pair_dat, mask, position_code, linear_align=linear_align)
            X = cv2.resize(X, target_size)

            # Segment weights
            w = generate_weight(mask, position_code, linear_align=linear_align)
            w = cv2.resize(w, target_size)

            # Segment labels (binarized fluorescence, discrete labels)
            pair_dat = [pair_dat[0], raw_label_preprocess(pair_dat[1])]
            Xs[ind] = X.reshape(target_shape).astype(float)

        except Exception as e:
            print("ERROR in loading pair %s" % str(identifier))
            print(e)
            Xs[ind] = None


        if not pair_dat[1] is None and 'discrete' in labels:
            try:
                # 0 - bg, 2 - fg, 1 - intermediate
                discrete_y = generate_fluorescence_labels(pair_dat, mask)
                y = cv2.resize(discrete_y, target_size)
                y[np.where((y > 0) & (y < 1))] = 1
                y[np.where((y > 1) & (y < 2))] = 1

                discrete_w = copy.deepcopy(w)
                discrete_w[np.where(y == 1)] = 0
                
                y[np.where(y == 1)] = 0
                y[np.where(y == 2)] = 1
                segment_discrete_ys[ind] = y.reshape(target_shape).astype(int)
                segment_discrete_ws[ind] = discrete_w.reshape(target_shape).astype(float)
            except Exception as e:
                print("ERROR in generating fluorescence label %s" % str(identifier))
                print(e)
                segment_discrete_ys[ind] = None
                segment_discrete_ws[ind] = None
        else:
            segment_discrete_ys[ind] = None
            segment_discrete_ws[ind] = None

        # Segment labels (continuous fluorescence in 4 classes)
        if not pair_dat[1] is None and 'continuous' in labels:
            try:
                continuous_y = quantize_fluorescence(pair_dat, mask)
                y = cv2.resize(continuous_y, target_size)

                continuous_w = copy.deepcopy(w)
                continuous_w[np.where(y != y)[:2]] = 0
                y[np.where(y != y)[:2]] = np.zeros((1, y.shape[-1]))

                segment_continuous_ys[ind] = y.reshape(target_shape).astype(float)
                segment_continuous_ws[ind] = continuous_w.reshape(target_shape).astype(float)

                classify_continuous_y = segment_continuous_ys[ind].sum((0, 1))
                classify_continuous_y = classify_continuous_y / (1e-5 + np.sum(classify_continuous_y))
            except Exception as e:
                print("ERROR in generating fluorescence label %s" % str(identifier))
                print(e)
                segment_continuous_ys[ind] = None
                segment_continuous_ws[ind] = None
                classify_continuous_y = None
        else:
            segment_continuous_ys[ind] = None
            segment_continuous_ws[ind] = None
            classify_continuous_y = None

        # Classify labels
        classify_discrete_labels[ind] = binarized_fluorescence_label(
            segment_discrete_ys[ind], segment_discrete_ws[ind])

        # Continuous label (4-class) will be dependent on fluorescence intensity level
        thrs = np.array([0., 0.32, 0.57, 0.92])
        _classify_continuous_w = classify_discrete_labels[ind][1]
        if classify_discrete_labels[ind][0] is None or _classify_continuous_w == 0:
            _classify_continuous_y = None
        elif classify_discrete_labels[ind][0] == 0:
            _classify_continuous_y = np.array([1., 0., 0., 0.])
        else:
            assert classify_continuous_y is not None
            _fl_intensity_lev = (classify_continuous_y * np.array([0., 1., 2., 3.])).sum()
            _classify_continuous_y = np.exp(-np.abs(thrs - _fl_intensity_lev) / 0.2)
            _classify_continuous_y = _classify_continuous_y / _classify_continuous_y.sum()
        classify_continuous_labels[ind] = (_classify_continuous_y, _classify_continuous_w)


        # Save data
        if output_path is not None and ((ind % 100 == 99) or (ind == len(pairs) - 1)):
            assert len(Xs) <= 100
            print("Writing file %d" % file_ind)
            with open(output_path + 'names.pkl', 'wb') as f:
                pickle.dump(names, f)
            with open(output_path + 'X_%d.pkl' % file_ind, 'wb') as f:
                pickle.dump(Xs, f)
            if 'discrete' in labels:
                with open(output_path + 'segment_discrete_y_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(segment_discrete_ys, f)
                with open(output_path + 'segment_discrete_w_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(segment_discrete_ws, f)
                with open(output_path + 'classify_discrete_labels.pkl', 'wb') as f:
                    pickle.dump(classify_discrete_labels, f)
            if 'continuous' in labels:
                with open(output_path + 'segment_continuous_y_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(segment_continuous_ys, f)
                with open(output_path + 'segment_continuous_w_%d.pkl' % file_ind, 'wb') as f:
                    pickle.dump(segment_continuous_ws, f)
                with open(output_path + 'classify_continuous_labels.pkl', 'wb') as f:
                    pickle.dump(classify_continuous_labels, f)
            file_ind += 1
            Xs = {}
            segment_discrete_ys = {}
            segment_discrete_ws = {}
            segment_continuous_ys = {}
            segment_continuous_ws = {}
    return file_ind


def merge_dataset_soft(source_data_folders, target_data_folder, shuffle=True, seed=123):
    os.makedirs(target_data_folder, exist_ok=True)
    def get_X_file_ct(folder_root):
        fs = os.listdir(folder_root)
        ct = sum([1 if f.startswith('X_') else 0 for f in fs])
        for i in range(ct):
            assert os.path.exists(os.path.join(folder_root, "X_%d.pkl" % i))
        return ct

    # Automatic detect params
    source_X_cts = [get_X_file_ct(f) for f in source_data_folders]
    save_discrete_labels = True if all(
        [os.path.exists(os.path.join(f, "segment_discrete_y_%d.pkl" % (ct - 1))) for f, ct in zip(source_data_folders, source_X_cts)]
    ) else False
    save_continuous_labels = True if all(
        [os.path.exists(os.path.join(f, "segment_continuous_y_%d.pkl" % (ct - 1))) for f, ct in zip(source_data_folders, source_X_cts)]
    ) else False

    # Get resave file order
    file_order = [[(j, i) for i in range(source_X_cts[j] - 1)] for j in range(len(source_data_folders))]
    file_order = sorted(sum(file_order, []))
    if shuffle:
        if not seed is None:
            np.random.seed(seed)
        np.random.shuffle(file_order)

    def cp_source_to_target(source_file, source_ind, target_file, target_ind):
        dat = pickle.load(open(source_file, "rb"))
        assert isinstance(dat, dict)
        for k in dat:
            assert k // 100 == source_ind
        new_dat = {(target_ind*100 + k % 100):dat[k] for k in dat}
        with open(target_file, "wb") as f:
            pickle.dump(new_dat, f)
        del dat, new_dat
        return

    def cp_files_source_to_target(file_order, file_name_prefix='X_'):
        for target_ind, (source_folder_ind, source_file_ind) in enumerate(file_order):
            cp_source_to_target(
                os.path.join(source_data_folders[source_folder_ind], "%s%d.pkl" % (file_name_prefix, source_file_ind)), 
                source_file_ind, 
                os.path.join(target_data_folder, "%s%d.pkl" % (file_name_prefix, target_ind)), 
                target_ind)

        n_output_files = len(file_order)
        target_sample_ind = n_output_files * 100 # Starting from the next index
        rest_elements = {}
        for i, (source_data_folder, last_file_ind) in enumerate(zip(source_data_folders, source_X_cts)):
            elements = pickle.load(open(os.path.join(source_data_folder, "%s%d.pkl" % (file_name_prefix, last_file_ind - 1)), "rb"))
            assert not os.path.exists(os.path.join(source_data_folder, "%s%d.pkl" % (file_name_prefix, last_file_ind)))
            for k in sorted(elements.keys()):
                rest_elements[target_sample_ind] = elements[k]
                target_sample_ind += 1

        n_extra_files = int(np.ceil(len(rest_elements) / 100))
        for extra_file_ind in range(n_extra_files):
            target_ind = n_output_files + extra_file_ind
            sample_inds = set(list(range(target_ind * 100, (target_ind+1) * 100)))

            save_elements = {k: rest_elements[k] for k in rest_elements if k in sample_inds}
            with open(os.path.join(target_data_folder, "%s%d.pkl" % (file_name_prefix, target_ind)), "wb") as f:
                pickle.dump(save_elements, f)
        return target_sample_ind, target_ind


    def mv_dict_elements(source_dict, source_ind, target_dict, target_ind):
        for i in range(100):
            target_dict[target_ind*100 + i] = source_dict[source_ind * 100 + i]
        return

    def assemble_all_dict_elements(source_dicts, file_order):
        target_dict = {}
        for target_ind, (source_folder_ind, source_file_ind) in enumerate(file_order):
            mv_dict_elements(source_dicts[source_folder_ind], source_file_ind, target_dict, target_ind)

        n_output_files = len(file_order)
        target_sample_ind = n_output_files * 100 # Starting from the next index
        for i, (source_data_folder, last_file_ind) in enumerate(zip(source_data_folders, source_X_cts)):
            X = pickle.load(open(os.path.join(source_data_folder, "X_%d.pkl" % (last_file_ind - 1)), "rb"))
            for k in sorted(X.keys()):
                target_dict[target_sample_ind] = source_dicts[i][k]
                target_sample_ind += 1
        assert len(target_dict.keys()) == max(target_dict.keys()) + 1
        return target_dict


    # Patch names
    source_names = [pickle.load(open(os.path.join(f, "names.pkl"), "rb")) for f in source_data_folders]
    target_names = assemble_all_dict_elements(source_names, file_order)
    n_total_samples = len(target_names)
    with open(os.path.join(target_data_folder, "names.pkl"), "wb") as f:
        pickle.dump(target_names, f)

    # Phase contrast inputs
    res = cp_files_source_to_target(file_order, file_name_prefix='X_')
    assert res[0] == n_total_samples
    n_total_X_files = res[1]

    if save_discrete_labels:
        # (Discrete) segmentation labels
        res = cp_files_source_to_target(file_order, file_name_prefix='segment_discrete_y_')
        assert res == (n_total_samples, n_total_X_files)
        res = cp_files_source_to_target(file_order, file_name_prefix='segment_discrete_w_')
        assert res == (n_total_samples, n_total_X_files)

        # (Discrete) classification labels
        source_discrete_labels = [pickle.load(open(os.path.join(f, "classify_discrete_labels.pkl"), "rb")) for f in source_data_folders]
        target_discrete_labels = assemble_all_dict_elements(source_discrete_labels, file_order)
        assert len(target_discrete_labels) == n_total_samples
        with open(os.path.join(target_data_folder, "classify_discrete_labels.pkl"), "wb") as f:
            pickle.dump(target_discrete_labels, f)

    if save_continuous_labels:
        # (Continuous) segmentation labels
        res = cp_files_source_to_target(file_order, file_name_prefix='segment_continuous_y_')
        assert res == (n_total_samples, n_total_X_files)
        res = cp_files_source_to_target(file_order, file_name_prefix='segment_continuous_w_')
        assert res == (n_total_samples, n_total_X_files)

        # (Continuous) classification labels
        source_continuous_labels = [pickle.load(open(os.path.join(f, "classify_continuous_labels.pkl"), "rb")) for f in source_data_folders]
        target_continuous_labels = assemble_all_dict_elements(source_continuous_labels, file_order)
        assert len(target_continuous_labels) == n_total_samples
        with open(os.path.join(target_data_folder, "classify_continuous_labels.pkl"), "wb") as f:
            pickle.dump(target_continuous_labels, f)

