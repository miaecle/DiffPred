import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from data_loader import *
from segment_support import *
from layers import *
from models import Segment, ClassifyOnSegment
from data_generator import CustomGenerator, enhance_weight_fp, binarized_fluorescence_label


DATA_ROOT = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/'
data_path = os.path.join(DATA_ROOT, 'linear_aligned_patches', 'merged_all')

kwargs = {
    'batch_size': 16,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 2,
    'segment_class_weights': [1, 3],
    'segment_extra_weights': enhance_weight_fp,
    'segment_label_type': 'segmentation',
    'n_classify_classes': 2,
    'classify_class_weights': [0.02, 0.02]
}

n_fs = len([f for f in os.listdir(data_path) if f.startswith('random_valid_X')])
X_filenames = [os.path.join(data_path, 'random_valid_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'random_valid_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'random_valid_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'random_valid_names.pkl')
label_file = os.path.join(data_path, 'random_valid_labels.pkl')
valid_gen = CustomGenerator(X_filenames,
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
    input_shape=(288, 384, 2), 
    model_structure='pspnet', 
    model_path='.', 
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)

model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/0-to-0_random/bkp.model')


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




model = ClassifyOnSegment(
    input_shape=(288, 384, 2), 
    model_structure='pspnet', 
    model_path='.', 
    encoder_weights='imagenet',
    n_segment_classes=4,
    n_classify_classes=4,
    eval_fn=evaluate_confusion_mat)

model.load('/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/0-to-0_random_discretized/bkp.model')
ref_seg_preds = []
ref_cl_preds = []
for batch in ref_valid_gen:
    seg_pred, cl_pred = model.model.predict(batch[0])
    seg_pred = scipy.special.softmax(seg_pred, -1)
    cl_pred = scipy.special.softmax(cl_pred, -1)
    ref_seg_preds.append(seg_pred)
    ref_cl_preds.append(cl_pred)
ref_seg_preds = np.concatenate(ref_seg_preds, 0)
ref_cl_preds = np.concatenate(ref_cl_preds, 0)


os.makedirs('./fig_save', exist_ok=True)
inds = ref_valid_gen.selected_inds
np.random.seed(123)
save_inds = np.random.choice(inds, (20,), replace=False)
for ind in save_inds:
    name = ref_valid_gen.names[ind]
    ind2 = [i for i, n in valid_gen.names.items() if n==name][0]
    x, y, w, _ = ref_valid_gen.load_ind(ind)
    phase = x
    fl = y[..., 1]*0.333 + y[..., 2]*0.667 + y[..., 3]
    fl_pred = seg_preds[ind2]
    cl_pred = cl_preds[ind2]

    ref_fl_pred = ref_seg_preds[ind]
    ref_cl_pred = ref_cl_preds[ind]

    fig_name = '_'.join(get_ex_day(name) + get_well(name))
    plt.clf()
    plt.imshow(phase[..., 0], vmin=-3, vmax=3, cmap='gray')
    plt.savefig('./fig_save/%s_phase.png' % fig_name, dpi=300)
    plt.clf()
    plt.imshow(fl, vmin=0, vmax=1)
    plt.savefig('./fig_save/%s_fl.png' % fig_name, dpi=300)
    plt.clf()
    plt.imshow(fl_pred[..., 1], vmin=0, vmax=1)
    plt.savefig('./fig_save/%s_fl_pred_%.2f.png' % (fig_name, cl_pred[1]), dpi=300)

    ref_fl_pred = ref_fl_pred[..., 1]*0.333 + ref_fl_pred[..., 2]*0.667 + ref_fl_pred[..., 3]
    ref_cl_pred = np.argmax(ref_cl_pred)
    plt.clf()
    plt.imshow(ref_fl_pred, vmin=0, vmax=1)
    plt.savefig('./fig_save/%s_fl_pred_discretized_%d.png' % (fig_name, ref_cl_pred), dpi=300)

