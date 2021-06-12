import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import roc_auc_score

ex_ids = {
    '20': "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_20/0-to-0/",
    '100': "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_100/0-to-0/",
    '142': "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_142/0-to-0/",
    '202': "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_202/0-to-0/", 
    '273': "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_273/0-to-0/",
    '477': "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_477/0-to-0/", 
    'diabetes': "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_diabetes/0-to-0/",
    'LMNA': "/oak/stanford/groups/jamesz/zqwu/iPSC_data/validation_set/line_LMNA_control/0-to-0/",
}
df = pd.read_csv("validation_labels.csv")

file_names = ["random_split_0-to-inf_cla_predictions_end%d.pkl" % i for i in [18,]]


cla_res = {}
for ex_id in ex_ids:
    _ex_cla_res = []
    for file_name in file_names:
        _ex_cla_res.append(pickle.load(open(os.path.join(ex_ids[ex_id], file_name), 'rb')))
    ex_cla_res = {k: np.stack([d[k] for d in _ex_cla_res], 0).mean(0) for k in _ex_cla_res[0]}
    cla_res.update(ex_cla_res)
    print(ex_id)
    print(np.mean([ex_cla_res[k][1] for k in ex_cla_res if int(k[2]) >= 7]))



def report_rocauc(evaluate_on):
    print(evaluate_on)
    ex_labels = {}
    for ex_id in evaluate_on:
        for row in np.array(df[['Unnamed: 0', ex_id]]):
            k = tuple(['ex' + ex_id] + row[0].split('_'))
            if row[1] == row[1]:
                ex_labels[k] = row[1]
            else:
                pass
                # ex_labels[k] = 1

    for day in range(0, 15):
        day_res = {tuple([k[1], k[3], k[4]]): v for k, v in cla_res.items() if k[2] == str(day)}
        shared_ks = sorted(set(day_res.keys()) & set(ex_labels.keys()))
        if len(shared_ks) > 10:
            preds = [day_res[k][1] for k in shared_ks]
            truth = [ex_labels[k] for k in shared_ks]
            try:
                print("%d: %d\t\t%.3f" % (day, len(shared_ks), roc_auc_score(truth, preds)))
            except Exception as e:
                print(e)
                pass


report_rocauc(['477'])
report_rocauc(['202'])
report_rocauc(['20'])
report_rocauc(['477', '202'])
report_rocauc(['477', '202', '273', '20', '142'])





def report_boxplot(evaluate_on):
    print(evaluate_on)
    ex_labels = {}
    for ex_id in evaluate_on:
        for row in np.array(df[['Unnamed: 0', ex_id]]):
            k = tuple(['ex' + ex_id] + row[0].split('_'))
            if row[1] == row[1]:
                ex_labels[k] = row[1]
            else:
                pass
                # ex_labels[k] = 1

    pred_0 = {i: [] for i in range(4, 13)}
    pred_1 = {i: [] for i in range(4, 13)}

    for k, v in cla_res.items():
        _k = tuple([k[1], k[3], k[4]])
        if _k in ex_labels.keys():
            day = int(k[2])
            if ex_labels[_k] == 0.:
                pred_0[day].append(v[1])
            elif ex_labels[_k] == 1.:
                pred_1[day].append(v[1])

    x = sorted(pred_0.keys())
    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 3))
    red_diamond = dict(markerfacecolor='r', marker='D', markersize=2)
    data = [pred_1[_x] for _x in x]
    bplot_1 = ax.boxplot(data, 
                         notch=True, 
                         vert=True, 
                         patch_artist=True, 
                         positions=np.array(x)+0.11, 
                         flierprops=red_diamond,
                         widths=0.2,
                         manage_ticks=False)
    blue_diamond = dict(markerfacecolor='b', marker='D', markersize=2)
    data = [pred_0[_x] for _x in x]
    bplot_0 = ax.boxplot(data, 
                         notch=True, 
                         vert=True, 
                         patch_artist=True, 
                         positions=np.array(x)-0.11, 
                         flierprops=blue_diamond,
                         widths=0.2,
                         manage_ticks=False)
    ax.set_xticks(x)
    ax.set_ylim([-0.03, 1.03])
    ax.set_xlim([3, 13])
    for patch in bplot_1['boxes']:
        patch.set_facecolor('pink')
    for patch in bplot_0['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_title('Prediction Distribution')
    plt.savefig('pred_distri_%s.png' % str(evaluate_on), dpi=300, bbox_inches='tight')


    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 1.5))
    ax.plot(x, [len(pred_1[_x]) for _x in x], c='pink')
    ax.plot(x, [len(pred_0[_x]) for _x in x], c='lightblue')
    ax.set_xticks(x)
    ax.set_xlim([3, 13])
    ax.set_ylim([0, ax.get_ylim()[1]])
    plt.savefig('n_samples_%s.png' % str(evaluate_on), dpi=300, bbox_inches='tight')


report_boxplot(['477'])
report_boxplot(['202'])
report_boxplot(['20'])
report_boxplot(['273'])
report_boxplot(['142'])
