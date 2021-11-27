import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from data_loader import get_identifier
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer, evaluate_confusion_mat
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives

from keras.models import Model
from tf_explain.core import GradCAM
from tf_explain.utils.display import grid_display


def well_info(name):
    return get_identifier(name)[:2] + get_identifier(name)[3:]


def get_inputs_for_well(well, gen):
    inds = [i for i in gen.names if well_info(gen.names[i][0]) == well]
    inds = sorted(inds, key=lambda x: int(get_identifier(gen.names[x][0])[2]))
    days = [int(get_identifier(gen.names[i][0])[2]) for i in inds]
    assert len(set([gen.names[i][1] for i in inds])) == 1

    batch_X = []
    batch_names = []

    for i in inds:
        sample_X, sample_segment_y, sample_segment_w, sample_name = gen.load_ind(i)
        batch_X.append(sample_X)
        batch_names.append(sample_name)
    
    sample_classify_y = gen.classify_y[inds[0]]
    sample_classify_w = gen.classify_w[inds[1]]

    X = gen.prepare_features(np.stack(batch_X, 0), names=batch_names)
    return X, days, (sample_segment_y, sample_segment_w), (sample_classify_y, sample_classify_w)


#%% Define dataset ###
save_dir = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/l1ex15_valid/'

kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 4,
    'segment_class_weights': [1, 2, 2, 2],
    'segment_extra_weights': None,
    'segment_label_type': 'continuous',
    'n_classify_classes': 4,
    'classify_class_weights': [1., 1., 2., 1.],
    'classify_label_type': 'continuous',
}


n_fs = len([f for f in os.listdir(save_dir) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(save_dir, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(save_dir, 'segment_continuous_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(save_dir, 'segment_continuous_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(save_dir, 'names.pkl')
label_file = os.path.join(save_dir, 'classify_continuous_labels.pkl')

gen = PairGenerator(
    name_file,
    X_filenames,
    segment_y_files=y_filenames,
    segment_w_files=w_filenames,
    classify_label_file=label_file,
    **kwargs)


#%% Define model and load weights ###
print("Initiate Model", flush=True)
MODEL_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex/'
model = ClassifyOnSegment(
    input_shape=(288, 384, 3),
    model_structure='pspnet',
    model_path=MODEL_DIR,
    encoder_weights='imagenet',
    n_segment_classes=4,
    n_classify_classes=4,
    eval_fn=evaluate_confusion_mat)
model.load(os.path.join(MODEL_DIR, 'bkp.model'))

_model = Model(model.input, model.classify_out) # model instance for explain classify output

#%% 

all_wells = sorted(set([well_info(gen.names[i][0]) for i in gen.names]))
np.random.seed(123)
np.random.shuffle(all_wells)
select_wells = all_wells[:10]

explainer = GradCAM()
for w in select_wells:

    X, days, segment_labels, classify_labels = get_inputs_for_well(w, gen)
    if classify_labels[1] == 0:
        continue

    X = X[:9]
    days = days[:9]
    class_index = np.argmax(classify_labels[0])

    input_grid = grid_display(X[..., 0], num_rows=3, num_columns=3)
    time_grid = grid_display(X[:, 0:1, 0:1, 1], num_rows=3, num_columns=3)    
    cam_grid = explainer.explain((list(X), None), _model, class_index=class_index, image_weight=0.)

    preds = model.predict(X)
    classify_preds = preds[1][..., 1] + preds[1][..., 2] * 2 + preds[1][..., 3] * 3
    segment_preds = preds[0][..., 1] + preds[0][..., 2] * 2 + preds[0][..., 3] * 3
    segment_pred_grid = grid_display(segment_preds, num_rows=3, num_columns=3)
    classify_pred_grid = grid_display(classify_preds.reshape((-1, 1, 1)), 3, 3)

    plt.clf()
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow((time_grid - 4)/10, cmap='Blues')
    plt.title("Input Time Stamp")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(input_grid)
    plt.title("Input Phase Contrast")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cam_grid)
    plt.title("Grad CAM on class %d" % class_index)
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(classify_pred_grid, vmin=0., vmax=3., cmap='viridis')
    plt.title("Classification prediction")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(segment_pred_grid, vmin=0., vmax=3., cmap='viridis')
    plt.title("Segmentation prediction")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    y = segment_labels[0]
    segment_mat = y[..., 1] + y[..., 2] * 2 + y[..., 3] * 3
    plt.imshow(segment_mat, vmin=0., vmax=3., cmap='viridis')
    plt.title("Final fluorescence, class %d" % class_index)
    plt.axis('off')

    plt.savefig('%s-explain.png' % ('-'.join(w)), dpi=300)

