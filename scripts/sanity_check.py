from data_loader import *
from segment_support import *
from models import Segment
from data_generator import CustomGenerator, PairGenerator, enhance_weight_fp
from data_generator import binaried_fluorescence_label
from scipy.stats import spearmanr, pearsonr

data_path = 'data/linear_aligned_patches/merged_all_in_order/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('X')])

X_filenames = [os.path.join(data_path, 'X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'names.pkl')

names = pickle.load(open(name_file, 'rb'))
unique_wells = sorted(set(get_ex_day(n)[:1] + get_well(n) for n in names.values()))
np.random.seed(123)
np.random.shuffle(unique_wells)
valid_wells = set(unique_wells[:int(0.2*len(unique_wells))])
valid_inds = [i for i, n in names.items() if get_ex_day(n)[:1] + get_well(n) in valid_wells]
train_wells = set(unique_wells[int(0.2*len(unique_wells)):])
train_inds = [i for i, n in names.items() if get_ex_day(n)[:1] + get_well(n) in train_wells]

output_mode = {'pc': ['pre'], 'fl': ['post']}
intervals = [[0, 0], [1, 3], [4, 6], [7, 10]]
for time_interval in intervals:
    print(time_interval)
    train_gen = PairGenerator(X_filenames,
                              y_filenames,
                              w_filenames,
                              name_file,
                              include_day=True,
                              batch_size=8,
                              extra_weights=enhance_weight_fp,
                              time_interval=time_interval,
                              output_mode=output_mode)

    labels_pre = []
    labels_post = []
    for i, pair in enumerate(train_gen.selected_pair_inds):
        X0, y0, w0, name0 = train_gen.load_ind(pair[0])
        X1, y1, w1, name1 = train_gen.load_ind(pair[1])
        y0, w0 = binaried_fluorescence_label(y0, w0)
        y1, w1 = binaried_fluorescence_label(y1, w1)
        labels_pre.append((y0, w0))
        labels_post.append((y1, w1))
    labels_pre = np.stack(labels_pre, 0)
    labels_post = np.stack(labels_post, 0)
    with open('./label_time_%d_%d.pkl' % (time_interval[0], time_interval[1]), 'wb') as f:
        pickle.dump([labels_pre, labels_post], f)


for time_interval in intervals:
    labels_pre, labels_post = pickle.load(open('label_time_%d_%d.pkl' % tuple(time_interval), 'rb'))
    print(spearmanr(labels_pre[:, 0], labels_post[:, 0]))
    print(labels_pre.shape)
    print(labels_post.shape)
    print(np.nonzero(labels_pre[:, 1])[0].size)
    print(np.nonzero(labels_post[:, 1])[0].size)
    w_comb = labels_pre[:, 1] * labels_post[:, 1]
    print(np.nonzero(w_comb)[0].size)
    print(np.nonzero(w_comb)[0].size/labels_pre.shape[0])

    y_pre = labels_pre[:, 0][np.nonzero(w_comb)]
    y_post = labels_post[:, 0][np.nonzero(w_comb)]

    print(((y_pre == 0) & (y_post == 0)).sum())
    print(((y_pre == 0) & (y_post == 0)).sum()/y_pre.shape[0])
    print(((y_pre == 0) & (y_post == 1)).sum())
    print(((y_pre == 0) & (y_post == 1)).sum()/y_pre.shape[0])
    print(((y_pre == 1) & (y_post == 0)).sum())
    print(((y_pre == 1) & (y_post == 0)).sum()/y_pre.shape[0])
    print(((y_pre == 1) & (y_post == 1)).sum())
    print(((y_pre == 1) & (y_post == 1)).sum()/y_pre.shape[0])

# for i in np.random.choice(np.arange(len(train_gen.selected_pair_inds)), (20,), replace=False):
#   pair = train_gen.selected_pair_inds[i]

#   f_n = '_'.join(get_ex_day(name0)[:1] + get_well(name0) + get_ex_day(name0)[1:] + ('phase_contrast',)) + '.png'
#   plt.clf()
#   plt.imshow(X0[:, :, 0])
#   plt.savefig('/home/zqwu/Dropbox/fig_temp/%s' % f_n, dpi=120)

#   f_n = '_'.join(get_ex_day(name1)[:1] + get_well(name1) + get_ex_day(name1)[1:] + ('phase_contrast',)) + '.png'
#   plt.clf()
#   plt.imshow(X1[:, :, 0])
#   plt.savefig('/home/zqwu/Dropbox/fig_temp/%s' % f_n, dpi=120)

#   f_n = '_'.join(get_ex_day(name0)[:1] + get_well(name0) + get_ex_day(name0)[1:] + ('fluorescence',)) + '.png'
#   plt.clf()
#   plt.imshow(y0 + 0.5*(1 - np.sign(w0)), vmin=0, vmax=1)
#   plt.savefig('/home/zqwu/Dropbox/fig_temp/%s' % f_n, dpi=120)

#   f_n = '_'.join(get_ex_day(name1)[:1] + get_well(name1) + get_ex_day(name1)[1:] + ('fluorescence',)) + '.png'
#   plt.clf()
#   plt.imshow(y1 + 0.5*(1 - np.sign(w1)), vmin=0, vmax=1)
#   plt.savefig('/home/zqwu/Dropbox/fig_temp/%s' % f_n, dpi=120)


