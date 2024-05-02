import os
import pickle
import numpy as np
import pandas as pd
from functools import partial

from data_loader import load_all_pairs, get_identifier, load_image_pair, load_image
from data_assembly import remove_corner_views, fluorescence_preprocess, preprocess
from data_assembly import extract_samples_for_inspection
from data_generator import CustomGenerator


RAW_FOLDERS = [
    ### Original validation / test sets ###
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex7_new',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex15',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex4',
    ### Prospective 3R ###
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/prospective/line1_3R/ex0',
    ### 10 additional lines ###
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
    ### different instrument ###
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex2_other_instrument',
    ### 10 additional lines - combined for segmentation ###
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines_for_seg/line_additional-combined-24/ex0-post',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines_for_seg/line_additional-combined-24/ex1-day14',
    ### different wells ###
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line1_3R/ex2-12well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line_839/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line_975/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/24well/line1_3R/ex0-24well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/24well/line_975-839/ex0',
    ### different differentiation protocol ###
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line1_3R/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line_839/ex0',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line_975/ex0',
    ### saliency test ###
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex0-96well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex0-96well-gfp',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex1-96well',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/saliency/line1_3R/ex1-96well-gfp',
    ### MERSCOPE Spatial Transcriptomics Analysis ###
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex_Vizgen-slide15-pos',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex_Vizgen-slide17-pos',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex_Vizgen-slide7-neg',
    ### different institutions ###
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_institutions/ex_UofT',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_institutions/ex_UTexas',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_institutions/ex_UColorado',
    ### drug perturbation ###
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_drugs/ex4_Benzopyrene',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_drugs/ex_multi_line_Benzopyrene',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_drugs/ex0_Valproate',
]

OUTPUT_FOLDERS = [
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex7_full/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex15_full/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line3_TNNI/ex4_full/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/prospective_ex0/0-to-0/',
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
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/additional_lines_for_seg/line_additional-combined-24/ex0-post/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/additional_lines_for_seg/line_additional-combined-24/ex1-day14/0-to-0/',
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
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex_Vizgen-slide15-pos/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex_Vizgen-slide17-pos/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex_Vizgen-slide7-neg/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_institutions/ex_UofT/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_institutions/ex_UTexas/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_institutions/ex_UColorado/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_drugs/ex4_Benzopyrene/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_drugs/ex_multi_line_Benzopyrene/0-to-0/',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_drugs/ex0_Valproate/0-to-0/',
]

WELL_SETTINGS = {
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex7_new': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex15': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex4': '6well-14',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/prospective/line1_3R/ex0': '96well-1',
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
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines_for_seg/line_additional-combined-24/ex0-post': '24well-6',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines_for_seg/line_additional-combined-24/ex1-day14': '24well-6',
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
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex_Vizgen-slide15-pos': '1well-40by29',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex_Vizgen-slide17-pos': '1well-40by29',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex_Vizgen-slide7-neg': '1well-40by29',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_institutions/ex_UofT': '24well-5',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_institutions/ex_UTexas': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_institutions/ex_UColorado': '96well-1',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_drugs/ex4_Benzopyrene': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_drugs/ex_multi_line_Benzopyrene': '96well-3',
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_drugs/ex0_Valproate': '96well-3',
}

# scale and offset parameters for raw fl preprocess
FL_PREPROCESS_SETTINGS = {
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex7_new': (0.7, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex15': (3.0, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex4': (2.5, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/prospective/line1_3R/ex0': (3.0, 0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines_for_seg/line_additional-combined-24/ex0-post': (1.5, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line1_3R/ex2-12well': (1.5, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/24well/line1_3R/ex0-24well': (2.0, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line1_3R/ex0': (3.5, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_institutions/ex_UTexas': (3.0, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_institutions/ex_UColorado': (3.0, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_drugs/ex4_Benzopyrene': (2.0, 0.0),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_drugs/ex0_Valproate': (3.5, 0.0),
}

FL_STATS = {
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex7_new': (43362, 9402),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line1_3R/ex15': (5436, 1980),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line3_TNNI/ex4': (4290, 2944),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/prospective/line1_3R/ex0': (7500, 1700),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines_for_seg/line_additional-combined-24/ex0-post': (15000, 7000),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/12well/line1_3R/ex2-12well': (8200, 4000),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_wells/24well/line1_3R/ex0-24well': (9700, 2800),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/different_differentiation/line1_3R/ex0': (6000, 1300),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_institutions/ex_UTexas': (6831, 2497),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_institutions/ex_UColorado': (7651, 1779),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_drugs/ex4_Benzopyrene': (7119, 3445),
    '/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/line_drugs/ex0_Valproate': (8761, 1425),
}

RAW_F_FILTER = lambda f: not 'bkp' in f

# %% Featurize each experiment
for raw_dir, inter_dir in zip(RAW_FOLDERS, OUTPUT_FOLDERS):
    os.makedirs(inter_dir, exist_ok=True)

    well_setting = WELL_SETTINGS[raw_dir]
    preprocess_filter = partial(remove_corner_views, well_setting=well_setting)

    pairs = load_all_pairs(path=raw_dir, check_valid=RAW_F_FILTER)

    kwargs = {'labels': []}
    if raw_dir in FL_PREPROCESS_SETTINGS and raw_dir in FL_STATS:
        fl_preprocess_setting = FL_PREPROCESS_SETTINGS[raw_dir]
        fl_preprocess_fn = partial(fluorescence_preprocess,
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

    # if raw_dir in FL_STATS:
    #     image_output_dir = inter_dir.replace('/0-to-0/', '/sample_figs/')
    #     extract_samples_for_inspection([p for p in pairs if p[0] is not None and preprocess_filter(p)],
    #                                    inter_dir,
    #                                    image_output_dir,
    #                                    seed=123)




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







