import os
import pickle
import tempfile
import csv
import argparse
from data_loader import load_image_pair, get_ex_day, get_well, get_all_files
from segment_support import preprocess, assemble_for_training
from data_generator import CustomGenerator
from models import ClassifyOnSegment


def load_assemble_test_data(data_path, dataset_path):
    # Load data under the given path
    print("Loading Data")
    fs = get_all_files(data_path)
    fs = [f for f in fs if not get_well(f) in ['1', '2', '16', '14', '15', '30', '196', '211', '212', '210', '224', '225']]
    fs_pair = [(f, None) for f in fs]
    pair_dats = {pair: load_image_pair(pair) for pair in fs_pair}

    # Preprocessing
    print("Start Preprocessing")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    processed_dats = preprocess(pair_dats, linear_align=False)
    _ = assemble_for_training(processed_dats, 
                              (384, 288), 
                              save_path=dataset_path,
                              label='segmentation')
    del processed_dats
    return dataset_path


def predict_on_test_data(dataset_path, model_path, output_path):
    name_file = os.path.join(dataset_path, 'names.pkl')
    names = pickle.load(open(name_file, 'rb'))
    day_num = int(get_ex_day(names[0])[1][1:])
    model_name = os.path.split(model_path)[-1]
    if 'inf' in model_name:
        predict_interval = 20 - day_num
    else:
        predict_interval = int(model_name.split('0-to-')[1].split('_')[0])
    print("Predicting %d days ahead" % predict_interval)
    # Building data generator and model
    kwargs = {
        'batch_size': 16,
        'shuffle_inds': False,
        'include_day': predict_interval,
        'n_segment_classes': None,
        'segment_class_weights': None,
    }
    n_fs = len([f for f in os.listdir(dataset_path) if f.startswith('X')])
    data_gen = CustomGenerator([os.path.join(dataset_path, 'X_%d.pkl' % i) for i in range(n_fs)],
                               [os.path.join(dataset_path, 'y_%d.pkl' % i) for i in range(n_fs)],
                               [os.path.join(dataset_path, 'w_%d.pkl' % i) for i in range(n_fs)],
                               name_file = os.path.join(dataset_path, 'names.pkl'),
                               **kwargs)

    print("Loading Model")
    model = ClassifyOnSegment(
        input_shape=(288, 384, 3), 
        model_structure='pspnet', 
        model_path='model_save', 
        encoder_weights='imagenet',
        n_segment_classes=2,
        n_classify_classes=2)
    model.load(model_path)

    # Predict and save
    print("Start Prediction")
    preds = model.predict(data_gen)
    assert preds[1].shape[0] == data_gen.N
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ex', 'day', 'well', 'well_position', 'score'])
        for i in range(data_gen.N):
            ex, day = get_ex_day(data_gen.names[i])
            well, well_pos = get_well(data_gen.names[i])
            writer.writerow([ex, day, well, well_pos, preds[1][i, 1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help="Path to the folder for raw inputs (tif files), \
        note that path should be formulated similar to XXX/ex1/XXX_D2_XXX/XXX",
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        help="Path to the saved model weights",
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=False,
        default='',
        help="Output path",
    )
    parser.add_argument(
        '-s', '--step',
        type=lambda s: [str(item.strip(' ').strip("'")) for item in s.split(',')],
        default=[],
        required=False,
        help="Processing steps: assemble, predict or both (split by comma)",

    )

    args = parser.parse_args()

    data_path = args.input
    dataset_path = os.path.join(data_path, 'merged') + '/'
    model_path = args.model
    output_path = args.output
    if len(output_path) == 0:
        output_path = os.path.join(data_path, 'results.csv')

    steps = args.step
    if len(steps) == 0 or 'assemble' in steps:
        load_assemble_test_data(data_path, dataset_path)
    if len(steps) == 0 or 'predict' in steps:
        predict_on_test_data(dataset_path, model_path, output_path)