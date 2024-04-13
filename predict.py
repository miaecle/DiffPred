import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import tempfile
import argparse
from functools import partial

from data_loader import load_all_pairs, check_pairs_by_day
from data_assembly import preprocess, remove_corner_views
from data_generator import CustomGenerator
from collect_predictions import augment_fixed_end, collect_preds, get_data_gen, get_model


def main(args):
    input_folder = args.input_folder
    data_dir = args.data_folder
    data_dir = data_dir if data_dir else tempfile.mkdtemp()
    output_folder = args.output_folder
    model_path = args.model_path
    well_setting = args.well_setting
    target_day = args.target_day

    # Load all phase contrast (identified by keyword "Phase" in the file name)
    # and all GFP (identifier by keyword "GFP") images under the folder recursively.
    # Phase contrast/GFP of the same well will be matched into pairs automatically
    pairs = load_all_pairs(path=input_folder)
    print("Checking input data")
    # Function below checks all the identified phase contrast/GFP files and shows stats
    check_pairs_by_day(pairs)

    preprocess_kwargs = {}
    # if `well_setting` is given, corner views will be removed
    if len(well_setting) > 0:
        preprocess_kwargs['preprocess_filter'] = partial(
            remove_corner_views, well_setting=well_setting)
    preprocess_kwargs['target_size'] = (384, 288)
    preprocess_kwargs['well_setting'] = well_setting
    preprocess_kwargs['linear_align'] = False
    preprocess_kwargs['labels'] = []

    # Process all input files, save into a temp folder
    preprocess(pairs,
               output_path=data_dir,
               shuffle=True,
               seed=123,
               **preprocess_kwargs)

    # Generator instance looping through all input files
    test_gen = get_data_gen(data_dir, CustomGenerator, batch_size=8, with_label=False)

    # Load model
    model = get_model(model_path)

    # If it is 0-to-inf prediction, define the placeholder for endpoint (default at 18)
    input_transform = None
    if '0-to-inf' in model_path:
        input_transform = partial(augment_fixed_end, end=target_day)

    # Calculate predictions and save to designated folder
    # refer to `collect_predictions.py` for the function below
    collect_preds(test_gen, model, output_folder, input_transform=input_transform)


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
