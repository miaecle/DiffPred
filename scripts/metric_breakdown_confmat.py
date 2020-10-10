import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import scipy
from data_loader import *
from segment_support import *
from models import Segment, ClassifyOnSegment
from layers import load_partial_weights, fill_first_layer
from data_generator import CustomGenerator, PairGenerator, enhance_weight_fp, binarized_fluorescence_label
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

data_path = 'data/linear_aligned_patches/cross_7-to-10/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('random_valid_X')])
X_filenames = [os.path.join(data_path, 'random_valid_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'random_valid_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'random_valid_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'random_valid_names.pkl')
label_file = os.path.join(data_path, 'random_valid_labels_4class.pkl')

kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 2,
    'segment_class_weights': [1, 3],
    'segment_extra_weights': enhance_weight_fp,
    'segment_label_type': 'segmentation',
    'n_classify_classes': 4,
    'classify_class_weights': [0.02, 0.02, 0.02, 0.02]
}

valid_gen = PairGenerator(X_filenames,
                          y_filenames,
                          w_filenames,
                          name_file,
                          label_file=label_file,
                          **kwargs)


model = ClassifyOnSegment(
    input_shape=(288, 384, 3),
    model_structure='pspnet',
    model_path='model_save',
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=4)

model.load('./model_save/pspnet_random_0-to-10_4class_0.model')

### Accuracy of 4 class w.r.t. predict day (from) ###

all_frame_names = pickle.load(open('data/linear_aligned_patches/merged_all/permuted_names.pkl', 'rb'))
all_frame_labels = pickle.load(open('data/linear_aligned_patches/merged_all/permuted_labels.pkl', 'rb'))
name_to_label = {all_frame_names[i]: all_frame_labels[i] for i in all_frame_names}

y_preds = []
for batch in valid_gen:
    pred = model.model.predict(batch[0])
    y_preds.append(pred[1])
y_preds = np.concatenate(y_preds, 0)
y_preds = scipy.special.softmax(y_preds, -1)

days = set([get_ex_day(n[0])[1] for n in valid_gen.names.values()])
conf_mats = {day: np.zeros((4, 4)) for day in days}

for i in valid_gen.selected_inds:
    day = get_ex_day(valid_gen.names[i][0])[1]
    label = valid_gen.labels[i]
    if label[1] == 0:
      continue
label = label[0]
pred = np.argmax(y_preds[i])
conf_mats[day][label, pred] += 1

for day in days:
    mat = conf_mats[day]
    mat = mat/mat.sum(1, keepdims=True)
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
    plt.savefig('/home/zqwu/Dropbox/fig_temp/conf_mat_%s.png' % day, dpi=300)
