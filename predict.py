import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import tempfile
import csv
import argparse
from functools import partial

from data_loader import load_all_pairs, check_pairs_by_day
from data_assembly import preprocess, remove_corner_views
from data_generator import CustomGenerator
from models import ClassifyOnSegment
from collect_predictions import augment_fixed_end, collect_preds


def main(args):
    input_folder = args.input_folder
    data_dir = args.data_folder
    data_dir = data_dir if data_dir else tempfile.mkdtemp()
    output_folder = args.output_folder
    model_path = args.model_path
    well_setting = args.well_setting
    target_day = args.target_day

    # if `well_setting` is given, corner views will be removed
    if len(well_setting) > 0:
        preprocess_filter = partial(remove_corner_views, well_setting=well_setting)
    else:
        preprocess_filter = lambda x: True

    # Load all phase contrast (identified by keyword "Phase" in the file name)
    # and all GFP (identifier by keyword "GFP") images under the folder recursively.
    # Phase contrast/GFP of the same well will be matched into pairs automatically
    pairs = load_all_pairs(path=input_folder)
    print("Checking input data")
    # Function below checks all the identified phase contrast/GFP files and shows stats
    check_pairs_by_day(pairs)

    # Process all input files, save into a temp folder
    preprocess(pairs,
               output_path=data_dir,
               preprocess_filter=preprocess_filter,
               target_size=(384, 288),
               well_setting=well_setting,
               linear_align=False,
               shuffle=True,
               seed=123,
               labels=['discrete', 'continuous'])

    # n_fs = len([f for f in os.listdir(data_dir) if f.startswith('X_') and f.endswith('.pkl')])
    # X_filenames = [os.path.join(data_dir, 'X_%d.pkl' % i) for i in range(n_fs)]
    # name_file = os.path.join(data_dir, 'names.pkl')

    # # Generator instance looping through all input files
    # test_gen = CustomGenerator(
    #     name_file,
    #     X_filenames,
    #     batch_size=8,
    #     augment=False,
    #     batch_with_name=True,
    #     shuffle_inds=False,
    #     n_segment_classes=None,
    #     segment_class_weights=None,
    #     segment_extra_weights=None,
    #     n_classify_classes=None,
    #     classify_class_weights=None,)

    # # Identify model type: 0-to-0 / 0-to-inf
    # n_input_channel = 2 if '0-to-0' in model_path else 3
    # model = ClassifyOnSegment(
    #     input_shape=(288, 384, n_input_channel),
    #     model_structure='pspnet',
    #     model_path=tempfile.mkdtemp(),
    #     encoder_weights='imagenet',
    #     n_segment_classes=4,
    #     segment_class_weights=[1., 1., 1., 1.],
    #     n_classify_classes=4,
    #     classify_class_weights=[1., 1., 1., 1.])

    # # Load model weights
    # model.load(model_path)

    # # If it is 0-to-inf prediction, define the placeholder for endpoint (default at 18)
    # input_transform = None
    # if n_input_channel == 3:
    #     input_transform = partial(augment_fixed_end, end=target_day)

    # # Calculate predictions and save to designated folder
    # # refer to `collect_predictions.py` for the function below 
    # collect_preds(test_gen, model, output_folder, input_transform=input_transform)


def parse_args(cli=True):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input_folder',
        type=str,
        help="Path to the folder for raw inputs (tif files), \
        note that path should be formulated similar to $ROOT/line_*/ex?/*_D2_*/XXX",
    )
    parser.add_argument(
        '-d', '--data_folder',
        type=str,
        default='',
        help="Path to the folder for processed inputs (pkls)",
    )
    parser.add_argument(
        '-o', '--output_folder',
        type=str,
        help="Output folder for saving predictions",
    )
    parser.add_argument(
        '-m', '--model_path',
        type=str,
        default="/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex/bkp.model",
        help="path of model weight file"
    )
    parser.add_argument(
        '-w', '--well_setting',
        type=str,
        default='',
        help="Well setting for the input data, support: 96well-3, 6well-14, 6well-15, will remove corner views from input if given",
    )
    parser.add_argument(
        '--target_day',
        type=int,
        default=18,
        help="Prediction target timepoint (day), default: 18",
    )

    if cli:
        args = parser.parse_args()
    else:
        args = parser.parse_args("")
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
