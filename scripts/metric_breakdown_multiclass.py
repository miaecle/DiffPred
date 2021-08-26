import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score

from data_loader import get_identifier, get_ex_day
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer, evaluate_confusion_mat, summarize_conf_mat
from data_generator import CustomGenerator, PairGenerator, enhance_weight_for_false_positives


#%% Settings ###
PRED_SAVE_DIRs = ['/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/random_split/0-to-inf_random/valid_pred']

#%% Classification metrics and plottings ###
def collect_classification_score():
    cla_preds = []
    cla_trues = []
    cla_ws = []
    pred_names = []
    for root in PRED_SAVE_DIRs:
        dat = pickle.load(open(os.path.join(root, 'cla.pkl'), 'rb'))
        cla_preds.append(dat['cla_preds'])
        cla_trues.append(dat['cla_trues'])
        cla_ws.append(dat['cla_ws'])
        pred_names.extend(dat['pred_names'])
    return np.concatenate(cla_preds, 0), np.concatenate(cla_trues, 0), np.concatenate(cla_ws, 0), pred_names


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
    return s1, s2, s3, s4


def plot_binary_metric(y_trues, 
                       y_preds, 
                       properties, 
                       cutoff=1.,
                       fig_output='binary_metric.png'):
    x_axis = sorted(set(properties))
    scores = []
    for _x in x_axis:
        inds = np.where(properties == _x)
        _y_preds = y_preds[inds]
        _y_trues = y_trues[inds]
        scores.append(evaluate_binary_metric(_y_trues, _y_preds, cutoff=cutoff))
    scores = np.stack(scores, 0)
    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 3))
    plt.plot(np.arange(len(x_axis)), scores[:, 0], '.-', label='Precision')
    plt.plot(np.arange(len(x_axis)), scores[:, 1], '.-', label='Recall')
    plt.plot(np.arange(len(x_axis)), scores[:, 2], '.-', label='F1')
    plt.plot(np.arange(len(x_axis)), scores[:, 3], '.-', label='ROC-AUC')
    plt.xticks(np.arange(len(x_axis)), x_axis)
    plt.legend()
    plt.ylim(0.7, 1.0)
    plt.savefig(fig_output, dpi=300, bbox_inches='tight')


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


def get_day(n):
    return int(get_identifier(n)[2])


def classification_main():
    y_preds, y_trues, ws, pred_names = collect_classification_score()
    
    
    pred_days = np.array([get_day(n[0]) if isinstance(n, tuple) else get_day(n) for n in pred_names])

    for cutoff in [1, 2]:
        boxplot_pred_distri(y_trues, y_preds, pred_days, cutoff=cutoff, 
            boxplot_fig_output='figs/pred_distri_by_day_cut%d.png' % cutoff,
            samplect_fig_output='figs/sample_count_by_day_cut%d.png' % cutoff)
        
        plot_binary_metric(y_trues, y_preds, pred_days, cutoff=cutoff,
                           fig_output='figs/binary_metric_cut%d.png' % cutoff)

    # for day in sorted(set(pred_days)):
    #     print(day)
    #     inds = np.where((pred_days == day) * (ws > 0))
    #     _y_preds = y_preds[inds]
    #     _y_trues = y_trues[inds]
    #     evaluate_conf_mat(_y_trues, _y_preds, plot_output='figs/conf_mat_%s.png' % day)
    return


#%% Regression metrics and plottings ###
def plot_seg_pearsonr(rs_seg_preds, rs_seg_trues, properties, fig_output='seg_pearsonr.png'):

    props = np.array(properties)
    x_axis = sorted(set(properties))

    prs = []
    for _x in x_axis:
        _preds = rs_seg_preds[np.where(props == _x)]
        _trues = rs_seg_trues[np.where(props == _x)]
        prs.append(pearsonr(_preds, _trues)[0])

    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(len(x_axis)), prs, '.-')
    ax.set_xticks(np.arange(len(x_axis)))
    ax.set_xticklabels(x_axis)
    plt.savefig(fig_output, dpi=300, bbox_inches='tight')


def boxplot_seg_pearsonr_distri(prs, 
                                properties,
                                boxplot_fig_output='seg_pearsonr_distri.png',
                                samplect_fig_output='seg_sample_count.png'):

    x_axis = sorted(set(properties))

    prs_by_x = {_x: [] for _x in x_axis}
    for pr, prop in zip(prs, properties):
        if pr == pr:
            prs_by_x[prop].append(pr)

    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 3))
    red_diamond = dict(markerfacecolor='r', marker='D', markersize=2)
    data = [prs_by_x[_x] for _x in x_axis]
    bplot = ax.boxplot(
        data,
        notch=True,
        vert=True,
        patch_artist=True,
        positions=np.arange(len(x_axis)),
        flierprops=red_diamond,
        widths=0.2,
        manage_ticks=False)
    ax.set_xticks(np.arange(len(x_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_xlim(-1, len(x_axis))
    plt.savefig(boxplot_fig_output, dpi=300, bbox_inches='tight')

    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 1.5))
    ax.plot(np.arange(len(x_axis)), [len(prs_by_x[_x]) for _x in x_axis], c='pink')
    ax.set_xticks(np.arange(len(x_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_xlim(-1, len(x_axis))
    ax.set_ylim([0, ax.get_ylim()[1]])
    plt.savefig(samplect_fig_output, dpi=300, bbox_inches='tight')


def segmentation_main():
    rs_seg_preds = []
    rs_seg_trues = []
    rs_days = []

    pearsonrs = []
    days = []

    samples_for_visualization = []

    for root in PRED_SAVE_DIRs:
        fs = [f for f in os.listdir(root) if f.startswith('seg_')]
        for f in sorted(fs, key=lambda x: int(x.split('_')[1].split('.')[0])):
            dat = pickle.load(open(os.path.join(root, f), 'rb'))
            seg_preds = dat['seg_preds']
            seg_trues = dat['seg_trues']
            seg_ws = dat['seg_ws']
            pred_names = dat['pred_names']
            for s_pred, s_true, s_w, name in zip(
                np.concatenate(seg_preds, 0),
                np.concatenate(seg_trues, 0),
                np.concatenate(seg_ws, 0),
                pred_names):

                name = name[0] if isinstance(name, tuple) else name
                day = get_day(name)

                _s_pred = s_pred[s_w > 0]
                _s_true = s_true[s_w > 0]
                pr = pearsonr(_s_pred, _s_true)[0]
                pearsonrs.append(pr)
                days.append(day)

                rs_inds = np.random.choice(np.arange(_s_pred.shape[0]), (2000,), replace=False)
                rs_seg_preds.append(_s_pred[rs_inds])
                rs_seg_trues.append(_s_true[rs_inds])
                rs_days.extend([day] * len(rs_inds))

                if pr == pr and np.random.rand() < 0.01:
                    samples_for_visualization.append((s_pred, s_true, s_w))
    
    boxplot_seg_pearsonr_distri(
        pearsonrs, 
        days, 
        boxplot_fig_output='figs/seg_pearsonr_distri.png', 
        samplect_fig_output='figs/seg_sample_count.png')
    rs_seg_preds = np.concatenate(rs_seg_preds)
    rs_seg_trues = np.concatenate(rs_seg_trues)
    plot_seg_pearsonr(
        rs_seg_preds,
        rs_seg_trues,
        rs_days, 
        fig_output='figs/seg_pearsonr.png')

    with open('figs/seg_visualization_save.pkl', 'wb') as f:
        pickle.dump(samples_for_visualization, f)


if __name__ == '__main__':
    classification_main()
    segmentation_main()