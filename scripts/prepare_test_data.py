import os
import numpy as np
from data_loader import load_all_pairs
from data_assembly import preprocess
from data_generator import CustomGenerator

input_paths = [
  "/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/ex100/",
  "/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/ex202/",
  "/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/ex273/",
  "/oak/stanford/groups/jamesz/zqwu/iPSC_data/RAW/additional_lines/ex477/",
]
output_paths = [
  "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_100/",
  "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_202/",
  "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_273/",
  "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_477/",
]

# for input_path, output_path in zip(input_paths, output_paths):
#     os.makedirs(output_path, exist_ok=True)
#     pairs = load_all_pairs(path=input_path)
#     preprocess(pairs, 
#                output_path=output_path, 
#                preprocess_filter=lambda x: True,
#                target_size=(384, 288),
#                labels=[], 
#                linear_align=False,
#                shuffle=True,
#                seed=123)



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
for output_path in output_paths:
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

    os.makedirs(output_path +  "0-to-0/", exist_ok=True)
    test_gen.reorder_save(np.array(X_valid),
                          save_path=output_path +  "0-to-0/",
                          write_segment_labels=False,
                          write_classify_labels=False)