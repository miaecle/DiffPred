import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
from scipy.stats import spearmanr, pearsonr
from functools import partial
import matplotlib.pyplot as plt

from data_loader import get_identifier
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer, evaluate_confusion_mat
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives
from collect_predictions import get_data_gen, get_model, filter_for_inf_predictor, filter_for_0_predictor, augment_fixed_end


from keras.models import Model
from tf_explain.core import GradCAM
from tf_explain.utils.display import grid_display


def well_info(name):
    if isinstance(name, tuple):
        name = name[0]
    return get_identifier(name)[:2] + get_identifier(name)[3:]


def day_info(name):
    if isinstance(name, tuple):
        name = name[0]
    return int(get_identifier(name)[2])


def get_inputs_for_well(well, gen, input_transform=None, input_filter=None):
    inds = [i for i in gen.names if well_info(gen.names[i]) == well]
    inds = sorted(inds, key=lambda x: day_info(gen.names[x]))
    days = [day_info(gen.names[i]) for i in inds]

    batch_X = []
    batch_names = []

    for i in inds:
        sample_X, sample_segment_y, sample_segment_w, sample_name = gen.load_ind(i)
        batch_X.append(sample_X)
        batch_names.append(sample_name)

    sample_classify_y = gen.classify_y[inds[0]] if not gen.classify_y is None else None
    sample_classify_w = gen.classify_w[inds[1]] if not gen.classify_w is None else None

    X = gen.prepare_features(np.stack(batch_X, 0), names=batch_names)
    _batch = (X, None, batch_names)
    if not input_filter is None:
        use_inds = input_filter(_batch)
        X = X[use_inds]
        batch_names = [batch_names[i] for i in use_inds]
        days = [days[i] for i in use_inds]

    if not input_transform is None:
        X = input_transform(X)

    assert len(X) == len(days)
    return X, days, (sample_segment_y, sample_segment_w), (sample_classify_y, sample_classify_w)


def pick_wells(gen, n_samples=0, seed=123):
    if not seed is None:
        np.random.seed(seed)
    all_wells = sorted(set([well_info(gen.names[i]) for i in gen.names]))
    np.random.shuffle(all_wells)
    n_samples = len(all_wells) if (n_samples is None) or (n_samples == 0) else n_samples
    return all_wells[:n_samples]


def plot_well_gradCAM(X, 
                      days, 
                      segment_labels, 
                      classify_labels, 
                      model,
                      save_root='.'):
    explainer = GradCAM()
    _model = Model(model.input, model.classify_out) # model instance for explain classify output

    X = X[:9]
    days = days[:9]

    n_r = 3 if len(X) > 4 else 2

    input_grid = grid_display(X[..., 0], num_rows=n_r, num_columns=n_r)
    pos_cam_grid = explainer.explain((list(X), None), _model, class_index=3, image_weight=0.)
    neg_cam_grid = explainer.explain((list(X), None), _model, class_index=0, image_weight=0.)

    preds = model.predict(X)
    classify_preds = preds[1][..., 1] + preds[1][..., 2] * 2 + preds[1][..., 3] * 3
    segment_preds = preds[0][..., 1] + preds[0][..., 2] * 2 + preds[0][..., 3] * 3
    segment_pred_grid = grid_display(segment_preds, num_rows=n_r, num_columns=n_r)
    classify_pred_grid = grid_display(classify_preds.reshape((-1, 1, 1)), num_rows=n_r, num_columns=n_r)

    plt.clf()
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(input_grid)
    plt.title("Input Phase Contrast\ndays: %s\nsegment_label: %s" % (str(days), str(classify_labels)))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(pos_cam_grid)
    plt.title("Grad CAM on class 3")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(neg_cam_grid)
    plt.title("Grad CAM on class 0")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(classify_pred_grid, vmin=0., vmax=3., cmap='viridis')
    plt.title("Classification prediction")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(segment_pred_grid, vmin=0., vmax=3., cmap='viridis')
    plt.title("Segmentation prediction")
    plt.axis('off')

    if not segment_labels[0] is None:
        plt.subplot(2, 3, 6)
        y = segment_labels[0]
        segment_mat = y[..., 1] + y[..., 2] * 2 + y[..., 3] * 3
        plt.imshow(segment_mat, vmin=0., vmax=3., cmap='viridis')
        plt.title("Final fluorescence, class %d" % class_index)
        plt.axis('off')

    plt.savefig(os.path.join(save_root, '%s-explain.png' % ('-'.join(w))), dpi=300)
    return pos_cam_grid, neg_cam_grid


if __name__ == "__main__":
    data_dir = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/saliency/line1_3R/ex0-96well/0-to-0/'
    gen_fn = CustomGenerator if '0-to-0' in data_dir else PairGenerator
    gen = get_data_gen(data_dir, gen_fn, batch_size=8, with_label=False)

    model_path = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex/bkp.model'
    model = get_model(model_path)

    if '0-to-inf' in model_path:
        input_filter = filter_for_inf_predictor
        input_transform = partial(augment_fixed_end, end=18)
    elif '0-to-0' in model_path:
        input_filter = filter_for_0_predictor
        input_transform = None
    else:
        raise ValueError("model not supported")

    select_wells = pick_wells(gen, n_samples=0, seed=123)

    for w in select_wells:
        X, days, segment_labels, classify_labels = get_inputs_for_well(w, gen, input_filter=input_filter, input_transform=input_transform)
        _ = plot_well_gradCAM(X, days, segment_labels, classify_labels, model, save_root='saliency_test')


