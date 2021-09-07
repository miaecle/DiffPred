import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import tempfile
import csv
import argparse
from functools import partial

from data_loader import load_all_pairs, get_identifier, load_image_pair, load_image, check_pairs_by_day
from data_assembly import preprocess
from data_generator import CustomGenerator
from models import ClassifyOnSegment
from collect_predictions import augment_fixed_end, collect_preds


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
    return True


def main(args):
    input_folder = args.input_folder
    output_folder = args.output_folder
    model_path = args.model_path
    well_setting = args.well_setting
    target_day = args.target_day

    intermediate_save_dir = tempfile.mkdtemp()
    
    preprocess_filter = partial(PREPROCESS_FILTER, well_setting=well_setting) if len(well_setting) > 0 else lambda x: True
    pairs = load_all_pairs(path=input_folder)
    print("Checking input data")
    check_pairs_by_day(pairs)
    
    preprocess(pairs, 
               output_path=intermediate_save_dir, 
               preprocess_filter=preprocess_filter,
               target_size=(384, 288),
               well_setting=well_setting,
               linear_align=False,
               shuffle=True,
               seed=123,
               labels=[])
    
    n_fs = len([f for f in os.listdir(intermediate_save_dir) if f.startswith('X_') and f.endswith('.pkl')])
    X_filenames = [os.path.join(intermediate_save_dir, 'X_%d.pkl' % i) for i in range(n_fs)]
    name_file = os.path.join(intermediate_save_dir, 'names.pkl')
    
    test_gen = CustomGenerator(
        name_file,
        X_filenames, 
        batch_size=8,
        augment=False,
        batch_with_name=True,
        shuffle_inds=False,
        n_segment_classes=None,
        segment_class_weights=None,
        segment_extra_weights=None,
        n_classify_classes=None,
        classify_class_weights=None,)

    n_input_channel = 2 if '0-to-0' in model_path else 3
    model = ClassifyOnSegment(
        input_shape=(288, 384, n_input_channel),
        model_structure='pspnet',
        model_path=tempfile.mkdtemp(),
        encoder_weights='imagenet',
        n_segment_classes=4,
        segment_class_weights=[1., 1., 1., 1.],
        n_classify_classes=4,
        classify_class_weights=[1., 1., 1., 1.])
    
    model.load(model_path)
    
    input_transform = None
    if n_input_channel == 3:
        input_transform = partial(augment_fixed_end, end=target_day)
    
    collect_preds(test_gen, model, output_folder, input_transform=input_transform)


def parse_args(cli=True):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input_folder',
        type=str,
        help="Path to the folder for raw inputs (tif files), \
        note that path should be formulated similar to XXX/ex1/XXX_D2_XXX/XXX",
    )
    parser.add_argument(
        '-o', '--output_folder',
        type=str,
        help="Output folder for saving predictions",
    )
    parser.add_argument(
        '-m', '--model_path',
        type=str,
        default="/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/random_split/0-to-inf_random/bkp.model",
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