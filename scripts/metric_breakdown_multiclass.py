import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from data_loader import get_identifier, get_ex_day
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer, evaluate_confusion_mat, summarize_conf_mat
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives


### Settings ###
ROOT_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/'
VALID_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/TRAIN/0-to-inf_continuous/random_valid/'
MODEL_DIR = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/random_split/0-to-inf_random/'

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

### Load dataset and model ###
n_fs = len([f for f in os.listdir(VALID_DIR) if f.startswith('X_') and f.endswith('.pkl')])
X_filenames = [os.path.join(VALID_DIR, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(VALID_DIR, 'segment_continuous_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(VALID_DIR, 'segment_continuous_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(VALID_DIR, 'names.pkl')
label_file = os.path.join(VALID_DIR, 'classify_continuous_labels.pkl')

valid_gen = PairGenerator(
    name_file,
    X_filenames,
    segment_y_files=y_filenames,
    segment_w_files=w_filenames,
    classify_label_file=label_file,
    **kwargs)

model = ClassifyOnSegment(
    input_shape=(288, 384, 3),
    model_structure='pspnet',
    model_path=MODEL_DIR,
    encoder_weights='imagenet',
    n_segment_classes=4,
    n_classify_classes=4,
    eval_fn=evaluate_confusion_mat)

model.load(os.path.join(MODEL_DIR, 'bkp.model'))


### Collect predictions ###
valid_gen.batch_with_name = True

y_preds = []
y_trues = []
pred_names = []
for batch in valid_gen:
    pred = model.model.predict(batch[0])
    y_preds.append(pred[1])
    y_trues.append(batch[1][1])
    pred_names.extend(batch[2])

y_preds = np.concatenate(y_preds, 0)
y_preds = scipy.special.softmax(y_preds, -1)
y_trues = np.concatenate(y_trues, 0)
ws = y_trues[:, -1]
y_trues = y_trues[:, :-1]

with open("pred_save/0-to-inf_random_preds.pkl", 'wb') as f:
    pickle.dump({"y_pred": y_preds, 
                 "y_trues": y_trues, 
                 "ws": ws, 
                 "pred_names": pred_names}, f)


### Metrics and plottings ###
def evaluate_conf_mat(y_trues, y_preds, plot_output=None):
    mat = confusion_matrix(np.argmax(y_trues, 1), np.argmax(y_preds, 1))
    mat = mat/mat.sum(1, keepdims=True)
    summarize_conf_mat(mat)

    if not plot_output is None:
        plt.clf()
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(mat, cmap='Blues', vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j-0.22, i-0.1, "%.2f" % mat[i, j])
        ax.set_title(day)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlim(-0.5, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(-0.5, 3.5)
        plt.savefig(plot_output, dpi=300)


def evaluate_binary_metric(y_trues, y_preds, cutoff=1):
    _y_preds = y_preds[:, cutoff:].sum(1)
    _y_trues = (np.argmax(y_trues, 1) >= cutoff) * 1

    s1 = precision_score(_y_trues, _y_preds > 0.5)
    s2 = recall_score(_y_trues, _y_preds > 0.5)
    s3 = f1_score(_y_trues, _y_preds > 0.5)
    s4 = roc_auc_score(_y_trues, _y_preds)
    print("Prec: %.3f\tRecall: %.3f\tF1: %.3f\tAUC: %.3f" % (s1, s2, s3, s4))


def boxplot_pred_distri(y_trues, 
                        y_preds, 
                        properties, 
                        cutoff=1,
                        boxplot_fig_output='pred_distri.png',
                        samplect_fig_output='sample_count.png'):
    _y_preds = y_preds[:, cutoff:].sum(1)
    _y_trues = (np.argmax(y_trues, 1) >= cutoff) * 1

    x_axis = sorted(set(properties))

    pred_neg = {i: [] for i in x_axis}
    pred_pos = {i: [] for i in x_axis}
    for pred, label, prop in zip(_y_preds, _y_trues, properties):
        if label == 0:
            pred_neg[prop].append(pred)
        elif label == 1:
            pred_pos[prop].append(pred)
        else:
            raise ValueError

    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 3))
    red_diamond = dict(markerfacecolor='r', marker='D', markersize=2)
    data = [pred_pos[_x] for _x in x_axis]
    bplot_pos = ax.boxplot(
        data,
        notch=True,
        vert=True,
        patch_artist=True,
        positions=np.arange(len(x_axis))+0.11,
        flierprops=red_diamond,
        widths=0.2,
        manage_ticks=False)
    blue_diamond = dict(markerfacecolor='b', marker='D', markersize=2)
    data = [pred_neg[_x] for _x in x_axis]
    bplot_neg = ax.boxplot(
        data,
        notch=True,
        vert=True,
        patch_artist=True,
        positions=np.arange(len(x_axis))-0.11,
        flierprops=blue_diamond,
        widths=0.2,
        manage_ticks=False)
    ax.set_xticks(np.arange(len(x_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_xlim(-1, len(x_axis))
    ax.set_ylim([-0.03, 1.03])
    for patch in bplot_pos['boxes']:
        patch.set_facecolor('pink')
    for patch in bplot_neg['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_title('Prediction Distribution')
    plt.savefig(boxplot_fig_output, dpi=300, bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 1.5))
    ax.plot(np.arange(len(x_axis)), [len(pred_pos[_x]) for _x in x_axis], c='pink')
    ax.plot(np.arange(len(x_axis)), [len(pred_neg[_x]) for _x in x_axis], c='lightblue')
    ax.set_xticks(np.arange(len(x_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_xlim(-1, len(x_axis))
    ax.set_ylim([0, ax.get_ylim()[1]])
    plt.savefig(samplect_fig_output, dpi=300, bbox_inches='tight')


get_day = lambda n: int(get_identifier(n)[2])
pred_days = np.array([get_day(n[0]) for n in pred_names])
pred_intervals = np.array([get_day(n[1]) - get_day(n[0]) for n in pred_names])

boxplot_pred_distri(y_trues, y_preds, pred_days, cutoff=1, 
    boxplot_fig_output='figs/pred_distri_by_day.png',
    samplect_fig_output='figs/sample_count_by_day.png')

boxplot_pred_distri(y_trues, y_preds, pred_intervals, cutoff=1, 
    boxplot_fig_output='figs/pred_distri_by_interval.png',
    samplect_fig_output='figs/sample_count_by_interval.png')


for day in sorted(set(pred_days)):
    print(day)
    inds = np.where((pred_days == day) * (ws > 0))
    _y_preds = y_preds[inds]
    _y_trues = y_trues[inds]

    evaluate_conf_mat(_y_trues, _y_preds, plot_output='figs/conf_mat_%s.png' % day)
    evaluate_binary_metric(_y_trues, _y_preds, cutoff=2)
