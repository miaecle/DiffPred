import numpy as np
import os
import cv2
import pickle
import copy
import matplotlib.pyplot as plt

from data_loader import get_identifier, load_image_pair, load_image
from segment_support import (
    generate_mask,
    generate_weight,
    generate_fluorescence_labels,
    quantize_fluorescence,
    adjust_contrast,
)


def remove_corner_views(pair, well_setting='96well-3'):
    # Remove samples without phase contrast
    if pair[0] is None:
        return False
    # Remove samples with inconsistent id
    if pair[1] is not None:
        if get_identifier(pair[0]) != get_identifier(pair[1]):
            return False

    position_code = str(get_identifier(pair[0])[-1])
    # Remove corner samples
    if well_setting == '96well-3':
        if position_code in ['1', '3', '7', '9']:
            return False
    elif well_setting == '6well-15':
        if position_code in ['1', '2', '16', '14', '15', '30', '196', '211', '212', '210', '224', '225']:
            return False
    elif well_setting == '6well-14':
        if position_code in ['1', '2', '15', '13', '14', '28', '169', '183', '184', '182', '195', '196']:
            return False
    elif well_setting == '12well-9':
        if position_code in ['1', '9', '73', '81']:
            return False
    elif well_setting == '24well-6':
        if position_code in ['1', '6', '31', '36']:
            return False
        # Multiples of 6 and '35' are added other than the 4 corners
        if position_code in ['12', '18', '24', '30', '35']:
            return False
    return True


def fluorescence_preprocess(fl, scale=1., offset=0.):
    if fl is None:
        return None
    fl = fl.astype(float)
    _fl = fl * scale + offset
    _fl = np.clip(_fl, 0, 65535).astype(int).astype('uint16')
    return _fl


def generate_discrete_segmentation_labels(pair_dat, mask, cv2_shape, weight_init, nonneg_thr=65535, **kwargs):
    # Generate binary label and weight for fluorescence
    discrete_y = generate_fluorescence_labels(pair_dat, mask, **kwargs)
    # 0 - bg, 2 - fg, 1 - intermediate
    y = cv2.resize(discrete_y, cv2_shape)
    y[np.where((y > 0) & (y < 1))] = 1
    y[np.where((y > 1) & (y < 2))] = 1

    # Intermediate will have zero weight
    discrete_w = copy.deepcopy(weight_init)
    discrete_w[np.where(y == 1)] = 0

    y[np.where(y == 1)] = 0
    y[np.where(y == 2)] = 1

    # If the overall fluorescence intensity is too high
    # slice could be a false negative
    if np.allclose(y, 0) and pair_dat[1][np.where(discrete_y == 0)].mean() > nonneg_thr:
        return y, np.zeros_like(discrete_w)
    return y, discrete_w


def generate_continuous_segmentation_labels(pair_dat, mask, cv2_shape, weight_init, nonneg_thr=65535, **kwargs):
    # Generate continuous label as a prob vector over 4 classes
    continuous_y = quantize_fluorescence(pair_dat, mask, **kwargs)
    y = cv2.resize(continuous_y, cv2_shape)
    continuous_w = copy.deepcopy(weight_init)
    if np.all(y != y):
        # full negative slice
        target_y = np.zeros((1, y.shape[-1]))
        target_y[0, 0] = 1.
        y[np.where(y != y)[:2]] = target_y
        continuous_w = generate_discrete_segmentation_labels(
            pair_dat, mask, cv2_shape, weight_init, nonneg_thr=nonneg_thr, **kwargs)[1]
    else:
        continuous_w[np.where(y != y)[:2]] = 0
        y[np.where(y != y)[:2]] = np.zeros((1, y.shape[-1]))
    return y, continuous_w


def generate_binary_classification_label(y, w):
    if y is None:
        return None, 0
    if isinstance(y, np.ndarray):
        y_ct = np.where(y > 0)[0].size
        invalid_ct = np.where(np.sign(w) == 0)[0].size
    elif np.all(int(y) == y):
        y_ct = y
        invalid_ct = w
    else:
        raise ValueError("Data type not supported")
    if y_ct > 500:
        sample_y = 1
        sample_w = 1
    elif y_ct == 0 and invalid_ct < 600:
        sample_y = 0
        sample_w = 1
    else:
        sample_y = 0
        sample_w = 0
    return sample_y, sample_w


def preprocess(pairs,
               output_path=None,
               preprocess_filter=lambda x: True,
               target_size=(384, 288),
               labels=['discrete', 'continuous'],
               raw_label_preprocess=lambda x: x,
               nonneg_thr=65535,
               well_setting='96well-3',
               linear_align=False,
               shuffle=True,
               seed=None,
               featurize_kwargs={}):
    # Save pairs of PC-GFP data as a dataset
    if seed is not None:
        np.random.seed(seed)

    # Sanity check
    pairs = [p for p in pairs if p[0] is not None and preprocess_filter(p)]

    # Sort
    pairs = sorted(pairs)
    if shuffle:
        np.random.shuffle(pairs)

    # More Settings
    linear_align_flag = (well_setting == '96well-3') and linear_align

    # Featurize data
    cv2_shape = target_size
    np_shape = (target_size[1], target_size[0], -1)  # Note that cv2 and numpy have reversed axis ordering
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
        # Name
        names[ind] = pair[0]

        # Input feature (phase contrast image)
        try:
            pair_dat = load_image_pair(pair)
            position_code = identifier[-1]
            # For corner views of 96 well plate, define foreground masks
            if well_setting == '96well-3' and position_code in ['1', '3', '7', '9'] and pair_dat[1] is not None:
                mask = generate_mask(pair_dat)
            else:
                mask = np.ones_like(pair_dat[0])
            # Normalize input
            X = adjust_contrast(
                pair_dat, mask, position_code, linear_align=linear_align_flag)
            X = cv2.resize(X, cv2_shape)
            # Segment weights
            base_weight = generate_weight(
                mask, position_code, linear_align=linear_align_flag)
            base_weight = cv2.resize(base_weight, cv2_shape)
            Xs[ind] = X.reshape(np_shape).astype(float)
        except Exception as e:
            print("ERROR in loading pair %s" % str(identifier), e)
            Xs[ind] = None

        # Segment labels
        _pair_dat = [pair_dat[0], raw_label_preprocess(pair_dat[1])]
        if pair_dat[1] is not None and 'discrete' in labels:
            try:
                y, discrete_w = generate_discrete_segmentation_labels(
                    _pair_dat, mask, cv2_shape, base_weight, nonneg_thr=nonneg_thr, **featurize_kwargs)
                segment_discrete_ys[ind] = y.reshape(np_shape).astype(int)
                segment_discrete_ws[ind] = discrete_w.reshape(np_shape).astype(float)
            except Exception as e:
                print("ERROR in generating fluorescence label %s" % str(identifier), e)
                segment_discrete_ys[ind] = None
                segment_discrete_ws[ind] = None
        else:
            segment_discrete_ys[ind] = None
            segment_discrete_ws[ind] = None

        # Segment labels (continuous fluorescence in 4 classes)
        if not pair_dat[1] is None and 'continuous' in labels:
            try:
                y, continuous_w = generate_continuous_segmentation_labels(
                    _pair_dat, mask, cv2_shape, base_weight, nonneg_thr=nonneg_thr, **featurize_kwargs)
                segment_continuous_ys[ind] = y.reshape(np_shape).astype(float)
                segment_continuous_ws[ind] = continuous_w.reshape(np_shape).astype(float)

                # Average prob vec over the whole slice
                avg_fl_vec = segment_continuous_ys[ind].sum((0, 1))
                avg_fl_vec = avg_fl_vec / (1e-5 + np.sum(avg_fl_vec))
            except Exception as e:
                print("ERROR in generating fluorescence label %s" % str(identifier), e)
                segment_continuous_ys[ind] = None
                segment_continuous_ws[ind] = None
                avg_fl_vec = None
        else:
            segment_continuous_ys[ind] = None
            segment_continuous_ws[ind] = None
            avg_fl_vec = None

        # Classify labels
        classify_discrete_labels[ind] = generate_binary_classification_label(
            segment_discrete_ys[ind], segment_discrete_ws[ind])

        # Continuous label (4-class) will be dependent on fluorescence intensity level
        thrs = np.array([0., 0.25, 0.35, 0.65])
        classify_continuous_w = classify_discrete_labels[ind][1]
        if classify_discrete_labels[ind][0] is None or classify_continuous_w == 0:
            classify_continuous_y = None
        elif classify_discrete_labels[ind][0] == 0:
            classify_continuous_y = np.array([1., 0., 0., 0.])
        else:
            assert avg_fl_vec is not None
            _fl_intensity_lev = (avg_fl_vec * np.array([0., 0.5, 1., 3.])).sum()
            classify_continuous_y = np.exp(-np.abs(thrs - _fl_intensity_lev) / 0.2)
            classify_continuous_y = classify_continuous_y / classify_continuous_y.sum()
        classify_continuous_labels[ind] = (classify_continuous_y, classify_continuous_w)

        # Save data
        if output_path is not None and ((ind % 100 == 99) or (ind == len(pairs) - 1)):
            assert len(Xs) <= 100
            print("Writing file %d" % file_ind)
            with open(os.path.join(output_path, 'names.pkl'), 'wb') as f:
                pickle.dump(names, f)
            with open(os.path.join(output_path, 'X_%d.pkl' % file_ind), 'wb') as f:
                pickle.dump(Xs, f)
            if 'discrete' in labels:
                with open(os.path.join(output_path, 'segment_discrete_y_%d.pkl' % file_ind), 'wb') as f:
                    pickle.dump(segment_discrete_ys, f)
                with open(os.path.join(output_path, 'segment_discrete_w_%d.pkl' % file_ind), 'wb') as f:
                    pickle.dump(segment_discrete_ws, f)
                with open(os.path.join(output_path, 'classify_discrete_labels.pkl'), 'wb') as f:
                    pickle.dump(classify_discrete_labels, f)
            if 'continuous' in labels:
                with open(os.path.join(output_path, 'segment_continuous_y_%d.pkl' % file_ind), 'wb') as f:
                    pickle.dump(segment_continuous_ys, f)
                with open(os.path.join(output_path, 'segment_continuous_w_%d.pkl' % file_ind), 'wb') as f:
                    pickle.dump(segment_continuous_ws, f)
                with open(os.path.join(output_path, 'classify_continuous_labels.pkl'), 'wb') as f:
                    pickle.dump(classify_continuous_labels, f)
            file_ind += 1
            Xs = {}
            segment_discrete_ys = {}
            segment_discrete_ws = {}
            segment_continuous_ys = {}
            segment_continuous_ws = {}
    return file_ind


def save_multi_panel_fig(mats, out_path):
    n_cols = 2
    n_rows = int(np.ceil(len(mats) / 2))
    _ = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
    for i, mat in enumerate(mats):
        if mat is None:
            continue
        plt.subplot(n_rows, n_cols, i + 1)
        if len(mat.shape) > 2:
            assert len(mat.shape) == 3
            if mat.shape[2] == 1:
                mat = mat[:, :, 0]
            else:
                assert mat.shape[2] == 3
        plt.imshow(mat)
        plt.axis('off')
    plt.savefig(out_path, dpi=300)
    return


def extract_samples_for_inspection(pairs, dataset_dir, image_output_dir, seed=123):
    # Randomly sample images from the dataset and save visualizations
    if seed is not None:
        np.random.seed(seed)
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir, exist_ok=True)
    raw_id_to_f_mapping = {get_identifier(p[0]): p for p in pairs}

    fs = os.listdir(dataset_dir)
    # Check existence of identifier file
    assert 'names.pkl' in fs
    names = pickle.load(open(os.path.join(dataset_dir, 'names.pkl'), 'rb'))
    for i, n in names.items():
        assert get_identifier(n) in raw_id_to_f_mapping

    # Check phase contrast files
    phase_contrast_files = [f for f in fs if f.startswith('X_') and f.endswith('.pkl')]
    for i in range(len(phase_contrast_files)):
        assert 'X_%d.pkl' % i in fs

    # Sample phase contrast image
    os.makedirs(os.path.join(image_output_dir, "phase_contrast"), exist_ok=True)
    random_inds = np.random.choice(list(names.keys()), (50,), replace=False)
    for ind in random_inds:
        file_ind = ind // 100
        identifier = get_identifier(names[ind])
        try:
            processed_img = pickle.load(open(os.path.join(dataset_dir, 'X_%d.pkl' % file_ind), 'rb'))[ind]
            raw_img = load_image(raw_id_to_f_mapping[identifier][0])
            out_path = os.path.join(image_output_dir,
                                    "phase_contrast",
                                    "%s.png" % '_'.join(identifier))
            save_multi_panel_fig([raw_img, processed_img], out_path)

        except Exception:
            print("Error saving sample %s" % '_'.join(identifier))

    try:
        # Check discrete segmentation annotations
        assert "classify_discrete_labels.pkl" in fs
        for i in range(len(phase_contrast_files)):
            assert 'segment_discrete_y_%d.pkl' % i in fs
            assert 'segment_discrete_w_%d.pkl' % i in fs

        classify_discrete_labels = pickle.load(open(os.path.join(dataset_dir, "classify_discrete_labels.pkl"), 'rb'))
        inds_by_class = {}
        for k in classify_discrete_labels:
            if classify_discrete_labels[k][0] is None or classify_discrete_labels[k][1] == 0:
                continue
            label = classify_discrete_labels[k][0]
            if label not in inds_by_class:
                inds_by_class[label] = []
            inds_by_class[label].append(k)

        # Sample discrete fl segmentation (by class)
        for cl in inds_by_class:
            os.makedirs(os.path.join(image_output_dir, "discrete_segmentation_class_%s" % str(cl)), exist_ok=True)
            if len(inds_by_class[cl]) > 20:
                random_inds = np.random.choice(list(inds_by_class[cl]), (20,), replace=False)
            else:
                random_inds = inds_by_class[cl]
            for ind in random_inds:
                file_ind = ind // 100
                identifier = get_identifier(names[ind])
                try:
                    raw_pc = load_image(raw_id_to_f_mapping[identifier][0])
                    raw_fl = load_image(raw_id_to_f_mapping[identifier][1])
                    processed_pc = pickle.load(open(os.path.join(dataset_dir, 'X_%d.pkl' % file_ind), 'rb'))[ind]
                    processed_fl_y = pickle.load(open(os.path.join(dataset_dir, 'segment_discrete_y_%d.pkl' % file_ind), 'rb'))[ind]
                    processed_fl_w = pickle.load(open(os.path.join(dataset_dir, 'segment_discrete_w_%d.pkl' % file_ind), 'rb'))[ind]
                    out_path = os.path.join(
                        image_output_dir, "discrete_segmentation_class_%s" % str(cl), "%s.png" % '_'.join(identifier))
                    save_multi_panel_fig([raw_pc, processed_pc, raw_fl, None, processed_fl_y, processed_fl_w], out_path)
                except Exception:
                    print("Error saving fl(discrete) sample %s" % '_'.join(identifier))
    except Exception:
        print("Issue locating discrete segmentation files")

    try:
        # Check continuous segmentation annotations
        assert "classify_continuous_labels.pkl" in fs
        for i in range(len(phase_contrast_files)):
            assert 'segment_continuous_y_%d.pkl' % i in fs
            assert 'segment_continuous_w_%d.pkl' % i in fs

        classify_continuous_labels = pickle.load(open(os.path.join(dataset_dir, "classify_continuous_labels.pkl"), 'rb'))
        inds_by_class = {}
        for k in classify_continuous_labels:
            if classify_continuous_labels[k][0] is None or classify_continuous_labels[k][1] == 0:
                continue
            label = np.argmax(classify_continuous_labels[k][0])
            if label not in inds_by_class:
                inds_by_class[label] = []
            inds_by_class[label].append(k)

        # Sample continuous fl segmentation (by class)
        for cl in inds_by_class:
            os.makedirs(os.path.join(image_output_dir, "continuous_segmentation_class_%s" % str(cl)), exist_ok=True)
            if len(inds_by_class[cl]) > 20:
                random_inds = np.random.choice(list(inds_by_class[cl]), (20,), replace=False)
            else:
                random_inds = inds_by_class[cl]
            for ind in random_inds:
                file_ind = ind // 100
                identifier = get_identifier(names[ind])
                try:
                    raw_pc = load_image(raw_id_to_f_mapping[identifier][0])
                    raw_fl = load_image(raw_id_to_f_mapping[identifier][1])
                    processed_pc = pickle.load(open(os.path.join(dataset_dir, 'X_%d.pkl' % file_ind), 'rb'))[ind]
                    processed_fl_y = pickle.load(open(os.path.join(dataset_dir, 'segment_continuous_y_%d.pkl' % file_ind), 'rb'))[ind]
                    processed_fl_y = (processed_fl_y * np.array([0., 0.333, 0.667, 1.]).reshape((1, 1, 4))).sum(2)
                    processed_fl_w = pickle.load(open(os.path.join(dataset_dir, 'segment_continuous_w_%d.pkl' % file_ind), 'rb'))[ind]
                    out_path = os.path.join(
                        image_output_dir, "continuous_segmentation_class_%s" % str(cl), "%s.png" % '_'.join(identifier))
                    save_multi_panel_fig([raw_pc, processed_pc, raw_fl, None, processed_fl_y, processed_fl_w], out_path)
                except Exception:
                    print("Error saving fl(continuous) sample %s" % '_'.join(identifier))
    except Exception:
        print("Issue locating continuous segmentation files")


# def merge_dataset_soft(source_data_folders, target_data_folder, shuffle=True, seed=123):
#     os.makedirs(target_data_folder, exist_ok=True)
#     def get_X_file_ct(folder_root):
#         fs = os.listdir(folder_root)
#         ct = sum([1 if f.startswith('X_') else 0 for f in fs])
#         for i in range(ct):
#             assert os.path.exists(os.path.join(folder_root, "X_%d.pkl" % i))
#         return ct

#     # Automatic detect params
#     source_X_cts = [get_X_file_ct(f) for f in source_data_folders]
#     save_discrete_labels = True if all(
#         [os.path.exists(os.path.join(f, "segment_discrete_y_%d.pkl" % (ct - 1))) for f, ct in zip(source_data_folders, source_X_cts)]
#     ) else False
#     save_continuous_labels = True if all(
#         [os.path.exists(os.path.join(f, "segment_continuous_y_%d.pkl" % (ct - 1))) for f, ct in zip(source_data_folders, source_X_cts)]
#     ) else False

#     # Get resave file order
#     file_order = [[(j, i) for i in range(source_X_cts[j] - 1)] for j in range(len(source_data_folders))]
#     file_order = sorted(sum(file_order, []))
#     if shuffle:
#         if not seed is None:
#             np.random.seed(seed)
#         np.random.shuffle(file_order)

#     def cp_source_to_target(source_file, source_ind, target_file, target_ind):
#         dat = pickle.load(open(source_file, "rb"))
#         assert isinstance(dat, dict)
#         for k in dat:
#             assert k // 100 == source_ind
#         new_dat = {(target_ind*100 + k % 100):dat[k] for k in dat}
#         with open(target_file, "wb") as f:
#             pickle.dump(new_dat, f)
#         del dat, new_dat
#         return

#     def cp_files_source_to_target(file_order, file_name_prefix='X_'):
#         for target_ind, (source_folder_ind, source_file_ind) in enumerate(file_order):
#             cp_source_to_target(
#                 os.path.join(source_data_folders[source_folder_ind], "%s%d.pkl" % (file_name_prefix, source_file_ind)),
#                 source_file_ind,
#                 os.path.join(target_data_folder, "%s%d.pkl" % (file_name_prefix, target_ind)),
#                 target_ind)

#         n_output_files = len(file_order)
#         target_sample_ind = n_output_files * 100 # Starting from the next index
#         rest_elements = {}
#         for i, (source_data_folder, last_file_ind) in enumerate(zip(source_data_folders, source_X_cts)):
#             elements = pickle.load(open(os.path.join(source_data_folder, "%s%d.pkl" % (file_name_prefix, last_file_ind - 1)), "rb"))
#             assert not os.path.exists(os.path.join(source_data_folder, "%s%d.pkl" % (file_name_prefix, last_file_ind)))
#             for k in sorted(elements.keys()):
#                 rest_elements[target_sample_ind] = elements[k]
#                 target_sample_ind += 1

#         n_extra_files = int(np.ceil(len(rest_elements) / 100))
#         for extra_file_ind in range(n_extra_files):
#             target_ind = n_output_files + extra_file_ind
#             sample_inds = set(list(range(target_ind * 100, (target_ind+1) * 100)))

#             save_elements = {k: rest_elements[k] for k in rest_elements if k in sample_inds}
#             with open(os.path.join(target_data_folder, "%s%d.pkl" % (file_name_prefix, target_ind)), "wb") as f:
#                 pickle.dump(save_elements, f)
#         return target_sample_ind, target_ind


#     def mv_dict_elements(source_dict, source_ind, target_dict, target_ind):
#         for i in range(100):
#             target_dict[target_ind*100 + i] = source_dict[source_ind * 100 + i]
#         return

#     def assemble_all_dict_elements(source_dicts, file_order):
#         target_dict = {}
#         for target_ind, (source_folder_ind, source_file_ind) in enumerate(file_order):
#             mv_dict_elements(source_dicts[source_folder_ind], source_file_ind, target_dict, target_ind)

#         n_output_files = len(file_order)
#         target_sample_ind = n_output_files * 100 # Starting from the next index
#         for i, (source_data_folder, last_file_ind) in enumerate(zip(source_data_folders, source_X_cts)):
#             X = pickle.load(open(os.path.join(source_data_folder, "X_%d.pkl" % (last_file_ind - 1)), "rb"))
#             for k in sorted(X.keys()):
#                 target_dict[target_sample_ind] = source_dicts[i][k]
#                 target_sample_ind += 1
#         assert len(target_dict.keys()) == max(target_dict.keys()) + 1
#         return target_dict


#     # Patch names
#     source_names = [pickle.load(open(os.path.join(f, "names.pkl"), "rb")) for f in source_data_folders]
#     target_names = assemble_all_dict_elements(source_names, file_order)
#     n_total_samples = len(target_names)
#     with open(os.path.join(target_data_folder, "names.pkl"), "wb") as f:
#         pickle.dump(target_names, f)

#     # Phase contrast inputs
#     res = cp_files_source_to_target(file_order, file_name_prefix='X_')
#     assert res[0] == n_total_samples
#     n_total_X_files = res[1]

#     if save_discrete_labels:
#         # (Discrete) segmentation labels
#         res = cp_files_source_to_target(file_order, file_name_prefix='segment_discrete_y_')
#         assert res == (n_total_samples, n_total_X_files)
#         res = cp_files_source_to_target(file_order, file_name_prefix='segment_discrete_w_')
#         assert res == (n_total_samples, n_total_X_files)

#         # (Discrete) classification labels
#         source_discrete_labels = [pickle.load(open(os.path.join(f, "classify_discrete_labels.pkl"), "rb")) for f in source_data_folders]
#         target_discrete_labels = assemble_all_dict_elements(source_discrete_labels, file_order)
#         assert len(target_discrete_labels) == n_total_samples
#         with open(os.path.join(target_data_folder, "classify_discrete_labels.pkl"), "wb") as f:
#             pickle.dump(target_discrete_labels, f)

#     if save_continuous_labels:
#         # (Continuous) segmentation labels
#         res = cp_files_source_to_target(file_order, file_name_prefix='segment_continuous_y_')
#         assert res == (n_total_samples, n_total_X_files)
#         res = cp_files_source_to_target(file_order, file_name_prefix='segment_continuous_w_')
#         assert res == (n_total_samples, n_total_X_files)

#         # (Continuous) classification labels
#         source_continuous_labels = [pickle.load(open(os.path.join(f, "classify_continuous_labels.pkl"), "rb")) for f in source_data_folders]
#         target_continuous_labels = assemble_all_dict_elements(source_continuous_labels, file_order)
#         assert len(target_continuous_labels) == n_total_samples
#         with open(os.path.join(target_data_folder, "classify_continuous_labels.pkl"), "wb") as f:
#             pickle.dump(target_continuous_labels, f)
