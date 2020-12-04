import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from data_loader import *
from segment_support import *
from layers import *
from models import Segment, ClassifyOnSegment
from data_generator import CustomGenerator, PairGenerator, enhance_weight_fp, binarized_fluorescence_label

DATA_ROOT = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/'

data_path = os.path.join(DATA_ROOT, 'linear_aligned_patches', 'cross_infinite')
kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 2,
    'segment_class_weights': [1, 3],
    'segment_extra_weights': enhance_weight_fp,
    'segment_label_type': 'segmentation',
    'n_classify_classes': 2,
    'classify_class_weights': [0.5, 0.5]
}
n_fs = len([f for f in os.listdir(data_path) if f.startswith('random_valid_X')])
X_filenames = [os.path.join(data_path, 'random_valid_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'random_valid_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'random_valid_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'random_valid_names.pkl')
label_file = os.path.join(data_path, 'random_valid_labels.pkl')
valid_gen = PairGenerator(X_filenames,
                          y_filenames,
                          w_filenames,
                          name_file,
                          label_file=label_file,
                          **kwargs)




data_path = os.path.join(DATA_ROOT, 'discretized_fl', 'merged_all')
kwargs = {
    'batch_size': 8,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 4,
    'segment_class_weights': [1, 2, 2, 2],
    'segment_extra_weights': None,
    'segment_label_type': 'discretized',
    'n_classify_classes': 4,
    'classify_class_weights': [0.02, 0.02, 0.02, 0.02],
    'classify_label_fn': None
}
n_fs = len([f for f in os.listdir(data_path) if f.startswith('random_valid_X')])
X_filenames = [os.path.join(data_path, 'random_valid_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'random_valid_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'random_valid_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'random_valid_names.pkl')
label_file = os.path.join(data_path, 'random_valid_labels.pkl')
ref_valid_gen = CustomGenerator(X_filenames,
                                y_filenames,
                                w_filenames,
                                name_file,
                                label_file=label_file,
                                **kwargs)




model = ClassifyOnSegment(
    input_shape=(288, 384, 3),
    model_structure='pspnet',
    model_path='.',
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)

model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/bkp/pspnet_random_0-to-inf_1.model')


seg_preds = []
cl_preds = []
for batch in valid_gen:
    seg_pred, cl_pred = model.model.predict(batch[0])
    seg_pred = scipy.special.softmax(seg_pred, -1)
    cl_pred = scipy.special.softmax(cl_pred, -1)
    seg_preds.append(seg_pred)
    cl_preds.append(cl_pred)
seg_preds = np.concatenate(seg_preds, 0)
cl_preds = np.concatenate(cl_preds, 0)

os.makedirs('./fig_save_cross', exist_ok=True)
ref_valid_gen_names = list(ref_valid_gen.names.values())
inds = [i for i, names in valid_gen.names.items() if names[1] in ref_valid_gen_names]
np.random.seed(100)
save_inds = np.random.choice(inds, (50,), replace=False)

pos_predicted = [i for i in range(len(seg_preds)) if (seg_preds[i, ..., 1] > 0.6).sum() > 100 and valid_gen.names[i][1] in ref_valid_gen_names]
save_inds = np.concatenate([save_inds, np.random.choice(pos_predicted, (20,), replace=False)])

for ind in save_inds:
    name = valid_gen.names[ind]
    ind2 = [i for i, n in ref_valid_gen.names.items() if n==name[1]][0]
    x1, _, _, _ = valid_gen.load_ind(ind)
    x2, y, w, _ = ref_valid_gen.load_ind(ind2)
    phase_before = x1
    phase_after = x2
    fl = y[..., 1]*0.333 + y[..., 2]*0.667 + y[..., 3]
    fl_pred = seg_preds[ind]
    cl_pred = cl_preds[ind]

    fig_name = '_'.join(get_ex_day(name[0]) + get_well(name[0]) + get_ex_day(name[1])[1:])
    plt.clf()
    plt.imshow(phase_before[..., 0], vmin=-3, vmax=3, cmap='gray')
    plt.savefig('./fig_save_cross/%s_phase_before.png' % fig_name, dpi=300)
    plt.clf()
    plt.imshow(phase_after[..., 0], vmin=-3, vmax=3, cmap='gray')
    plt.savefig('./fig_save_cross/%s_phase_after.png' % fig_name, dpi=300)
    plt.clf()
    plt.imshow(fl, vmin=0, vmax=1)
    plt.savefig('./fig_save_cross/%s_fl.png' % fig_name, dpi=300)
    plt.clf()
    plt.imshow(fl_pred[..., 1], vmin=0, vmax=1)
    plt.savefig('./fig_save_cross/%s_fl_pred_%.2f.png' % (fig_name, cl_pred[1]), dpi=300)

