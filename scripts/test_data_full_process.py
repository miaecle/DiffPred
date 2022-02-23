import os
import pickle
import numpy as np
import pandas as pd
from functools import partial

from data_loader import load_all_pairs, get_identifier, load_image_pair, load_image
from data_assembly import preprocess, extract_samples_for_inspection
from data_generator import CustomGenerator

RAW_FOLDERS = [
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_477/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_202/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_20/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_100/ex4',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_142/ex1',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_273/ex2',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_839/ex1',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_480/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_854/ex1',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_975/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex2_other_instrument',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex2_prospective',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex1-14',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex2-14',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex1-6_12',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex2-6_12',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line1_3R/ex2-12well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line_839/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line_975/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/24well/line1_3R/ex0-24well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/24well/line_975-839/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line1_3R/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line_839/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line_975/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex0-96well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex0-96well-gfp',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex1-96well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex1-96well-gfp',
]

OUTPUT_FOLDERS = [
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_477/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_202/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_20/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_100/ex4/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_142/ex1/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_273/ex2/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_839/ex1/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_480/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_854/ex1/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_975/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex2_other_instrument/0-to-0/',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex2_prospective/0-to-0/',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_combined-for-seg/ex1-14/0-to-0/',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_combined-for-seg/ex2-14/0-to-0/',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_combined-for-seg/ex1-6_12/0-to-0/',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_combined-for-seg/ex2-6_12/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_wells/12well/line1_3R/ex2-12well/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_wells/12well/line_839/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_wells/12well/line_975/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_wells/24well/line1_3R/ex0-24well/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_wells/24well/line_975-839/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_differentiation/line1_3R/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_differentiation/line_839/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_differentiation/line_975/ex0/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/saliency/line1_3R/ex0-96well/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/saliency/line1_3R/ex0-96well-gfp/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/saliency/line1_3R/ex1-96well/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/saliency/line1_3R/ex1-96well-gfp/0-to-0/',
]

WELL_SETTINGS = {
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_477/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_202/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_20/ex0': '6well-15',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_100/ex4': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_142/ex1': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_273/ex2': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_839/ex1': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_480/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_854/ex1': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/line_975/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex2_other_instrument': '96well-3',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex2_prospective': '96well-3',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex1-14': '96well-3',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex2-14': '96well-3',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex2-6_12': '96well-3',
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex1-6_12': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line1_3R/ex2-12well': '12well-9',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line_839/ex0': '12well-9',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line_975/ex0': '12well-9',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/24well/line1_3R/ex0-24well': '24well-6',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/24well/line_975-839/ex0': '24well-6',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line1_3R/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line_839/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line_975/ex0': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex0-96well': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex0-96well-gfp': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex1-96well': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex1-96well-gfp': '96well-3',
}

# scale and offset parameters for raw fl preprocess
FL_PREPROCESS_SETTINGS = {
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex2_prospective': (3.0, 0.0),
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex1-6_12': (2.7, 0.0),
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex1-14': (2.7, 0.0),
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex2-6_12': (3.5, 0.0),
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex2-14': (3.5, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line1_3R/ex2-12well': (1.5, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/24well/line1_3R/ex0-24well': (2.0, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line1_3R/ex0': (3.5, 0.0),
}

FL_STATS = {
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex2_prospective': (4104, 2277),
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex1-6_12': (4000, 1350),
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex1-14': (4000, 1350),
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex2-6_12': (4000, 1350),
    # '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_combined-for-seg/ex2-14': (4000, 1350),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line1_3R/ex2-12well': (8200, 4000),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/24well/line1_3R/ex0-24well': (9700, 2800),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line1_3R/ex0': (6000, 1300),
}

RAW_F_FILTER = lambda f: not 'bkp' in f

def PREPROCESS_FILTER(pair, well_setting='96well-3'):
    # Remove samples without phase contrast
    if pair[0] is None:
        return False
    # Remove samples with inconsistent id
    if pair[1] is not None and get_identifier(pair[0]) != get_identifier(pair[1]):
        return False
    # Remove corner samples
    if well_setting == '6well-15':
        if get_identifier(pair[0])[-1] in \
            ['1', '2', '16', '14', '15', '30', '196', '211', '212', '210', '224', '225']:
            return False
    elif well_setting == '6well-14':
        if get_identifier(pair[0])[-1] in \
            ['1', '2', '15', '13', '14', '28', '169', '183', '184', '182', '195', '196']:
            return False
    elif well_setting == '96well-3':
        if get_identifier(pair[0])[-1] in \
            ['1', '3', '7', '9']:
            return False
    elif well_setting == '12well-9':
        if get_identifier(pair[0])[-1] in \
            ['1', '9', '73', '81']:
            return False
    elif well_setting == '24well-6':
        if get_identifier(pair[0])[-1] in \
            ['1', '6', '12', '18', '24', '30', '31', '35', '36']:
            # Multiples of 6 and '35' are added other than the 4 corners
            return False
    return True


def FL_PREPROCESS(fl, scale=1., offset=0.):
    if fl is None:
        return None
    fl = fl.astype(float)
    _fl = fl * scale + offset
    _fl = np.clip(_fl, 0, 65535).astype(int).astype('uint16')
    return _fl


# %% Featurize each experiment
for raw_dir, inter_dir in zip(RAW_FOLDERS, OUTPUT_FOLDERS):
    os.makedirs(inter_dir, exist_ok=True)

    well_setting = WELL_SETTINGS[raw_dir]
    preprocess_filter = partial(PREPROCESS_FILTER, well_setting=well_setting)

    pairs = load_all_pairs(path=raw_dir, check_valid=RAW_F_FILTER)

    kwargs = {'labels': []}
    if raw_dir in FL_PREPROCESS_SETTINGS and raw_dir in FL_STATS:
        fl_preprocess_setting = FL_PREPROCESS_SETTINGS[raw_dir]
        fl_preprocess_fn = partial(FL_PREPROCESS,
                                   scale=fl_preprocess_setting[0],
                                   offset=fl_preprocess_setting[1])
        fl_stat = FL_STATS[raw_dir]
        fl_stat = (fl_stat[0] * fl_preprocess_setting[0] + fl_preprocess_setting[1],
                   fl_stat[1] * fl_preprocess_setting[0])
        fl_nonneg_thr = fl_stat[0] + fl_stat[1]

        kwargs['labels'] = ['discrete', 'continuous']
        kwargs['raw_label_preprocess'] = fl_preprocess_fn
        kwargs['nonneg_thr'] = fl_nonneg_thr

    preprocess(pairs,
               output_path=inter_dir,
               preprocess_filter=preprocess_filter,
               target_size=(384, 288),
               well_setting=well_setting,
               linear_align=False,
               shuffle=True,
               seed=123,
               **kwargs)

    if raw_dir in FL_STATS:
        image_output_dir = inter_dir.replace('/0-to-0/', '/sample_figs/')
        extract_samples_for_inspection([p for p in pairs if p[0] is not None and preprocess_filter(p)],
                                       inter_dir,
                                       image_output_dir,
                                       seed=123)




# %% Check invalid entries and remove
kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': None,
    'segment_class_weights': None,
    'segment_extra_weights': None,
    'segment_label_type': 'discrete',
    'n_classify_classes': None,
    'classify_class_weights': None,
    'classify_label_type': 'discrete',
}
for output_path in OUTPUT_FOLDERS:
    print("Checking %s" % output_path)
    n_fs = len([f for f in os.listdir(output_path) if f.startswith('X_') and f.endswith('.pkl')])
    X_filenames = [os.path.join(output_path, 'X_%d.pkl' % i) for i in range(n_fs)]
    name_file = os.path.join(output_path, 'names.pkl')

    test_gen = CustomGenerator(
        name_file,
        X_filenames,
        augment=False,
        batch_with_name=True,
        **kwargs)

    X_valid = []
    for i in test_gen.selected_inds:
        try:
            X, _, _, name = test_gen.load_ind(i)
        except Exception as e:
            print(e)
            print("ISSUE %d" % i)
            continue
        if not X is None:
            X_valid.append(i)

    if len(X_valid) < len(test_gen.selected_inds):
        print("Found invalid entries, saving corrected dataset")
        corrected_output_path = output_path.replace('/0-to-0/', '/0-to-0_corrected/')
        os.makedirs(corrected_output_path, exist_ok=True)
        test_gen.reorder_save(np.array(X_valid),
                              save_path=corrected_output_path,
                              write_segment_labels=False,
                              write_classify_labels=False)





