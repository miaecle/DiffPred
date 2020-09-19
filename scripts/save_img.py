import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from data_loader import *
from segment_support import *
from layers import *
from models import Segment, ClassifyOnSegment
from data_generator import CustomGenerator, enhance_weight_fp, binarized_fluorescence_label

data_path = 'data/linear_aligned_patches/merged_all/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('permuted_X')])

X_filenames = [os.path.join(data_path, 'permuted_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'permuted_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'permuted_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'permuted_names.pkl')
label_file = os.path.join(data_path, 'permuted_labels.pkl')

names = pickle.load(open(name_file, 'rb'))
unique_wells = sorted(set(get_ex_day(n)[:1] + get_well(n) for n in names.values()))
np.random.seed(123)
np.random.shuffle(unique_wells)
valid_wells = set(unique_wells[:int(0.2*len(unique_wells))])
valid_inds = [i for i, n in names.items() if get_ex_day(n)[:1] + get_well(n) in valid_wells]
train_wells = set(unique_wells[int(0.2*len(unique_wells)):])
train_inds = [i for i, n in names.items() if get_ex_day(n)[:1] + get_well(n) in train_wells]
print(len(train_inds))
print(len(valid_inds))

kwargs = {
    'batch_size': 16,
    'shuffle_inds': False,
    'include_day': True,
    'n_segment_classes': 2,
    'segment_class_weights': [1, 3],
    'segment_extra_weights': enhance_weight_fp,
    'segment_label_type': 'segmentation',
    'n_classify_classes': 2,
    'classify_class_weights': [0.02, 0.02],
    'classify_label_fn': binarized_fluorescence_label
}

train_gen = CustomGenerator(X_filenames,
                            y_filenames,
                            w_filenames,
                            name_file,
                            label_file=label_file,
                            augment=True,
                            selected_inds=train_inds,
                            **kwargs)

test_inds = np.random.choice(valid_inds, (100,), replace=False)
test_filenames = train_gen.reorder_save(test_inds, save_path=data_path+'temp_test_')
test_gen = CustomGenerator(*test_filenames, **kwargs)

model = ClassifyOnSegment(
    input_shape=(288, 384, 2), 
    model_structure='pspnet', 
    model_path='model_save', 
    encoder_weights='imagenet',
    n_segment_classes=2,
    n_classify_classes=2)

model.load('./model_save/pspnet_random_0-to-0_1.model')


data_path = 'data/discretized_fl/merged_all/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('permuted_X')])
X_filenames = [os.path.join(data_path, 'permuted_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'permuted_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'permuted_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'permuted_names.pkl')
label_file = os.path.join(data_path, 'permuted_labels.pkl')
ref_gen = CustomGenerator(X_filenames,
                          y_filenames,
                          w_filenames,
                          name_file,
                          label_file=label_file,
                          segment_label_type='discretized_fl')

seg_preds = []
cl_preds = []
for batch in test_gen:
    seg_pred, cl_pred = model.model.predict(batch[0])
    seg_pred = scipy.special.softmax(seg_pred, -1)
    cl_pred = scipy.special.softmax(cl_pred, -1)
    seg_preds.append(seg_pred)
    cl_preds.append(cl_pred)
seg_preds = np.concatenate(seg_preds, 0)
cl_preds = np.concatenate(cl_preds, 0)


for ind in test_gen.selected_inds:
    name = test_gen.names[ind]
    ind2 = [i for i, n in ref_gen.names.items() if n==name][0]
    x, y, w, _ = ref_gen.load_ind(ind2)
    phase = x
    fl = y[..., 1]*0.333 + y[..., 2]*0.667 + y[..., 3]
    fl_pred = seg_preds[ind]
    cl_pred = cl_preds[ind]
    fig_name = '_'.join(get_ex_day(name) + get_well(name))
    plt.clf()
    plt.imshow(phase[..., 0], vmin=-3, vmax=3)
    plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_phase.png' % fig_name, dpi=300)
    plt.clf()
    plt.imshow(fl, vmin=0, vmax=1)
    plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_fl.png' % fig_name, dpi=300)
    plt.clf()
    plt.imshow(fl_pred[..., 1], vmin=0, vmax=1)
    plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_fl_pred_%.2f.png' % (fig_name, cl_pred[1]), dpi=300)

