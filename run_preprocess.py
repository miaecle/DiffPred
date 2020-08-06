from segment_support import *
from data_loader import *

### Preprocess ###
data_path = 'data'
dat_fs = [f for f in os.listdir(data_path) if f.startswith('ex')]
for f_name in dat_fs:
  f_name = f_name.split('.')[0]
  if not 'processed' in f_name:
    print(f_name)
    dats = pickle.load(open(data_path + '/%s.pkl' % f_name, 'rb'))
    if not f_name.startswith('ex2'): # ex2 is a different setting
      for pos_code in ['2', '4', '5', '6', '8']:
        try:
          pos_code_dats = {k:v for k,v in dats.items() if position_code(k) == pos_code}
          processed_dats = preprocess(pos_code_dats)
          with open(data_path + '/linear_aligned_middle_patch/%s_processed_%s.pkl' % (f_name, pos_code), 'wb') as f:
            pickle.dump(processed_dats, f)
        except Exception as e:
          print(e)
          continue

"""
### Assemble for training ###
processed_fs = ['data/linear_aligned_middle_patch/%s' % f \
                for f in os.listdir('data/linear_aligned_middle_patch')]
if True:
    X, y, w, names = assemble_for_training(processed_fs, (384, 288))

np.random.seed(123)
random_perf = np.arange(len(X), dtype=int)
np.random.shuffle(random_perf)

X = pickle.load(open('data/linear_aligned_middle_patch/merged/merged_X.pkl', 'rb'))
for i in range(np.ceil(len(X)/100).astype(int)):
    print(i)
    with open('data/linear_aligned_middle_patch/merged/merged_X_%d.pkl' % i, 'wb') as f:
        pickle.dump({i*100+j: X[random_perf[i*100+j]] for j in range(100) if (i*100+j) < len(X)}, f)

y = pickle.load(open('data/linear_aligned_middle_patch/merged/merged_y.pkl', 'rb'))
for i in range(np.ceil(len(X)/100).astype(int)):
    print(i)
    with open('data/linear_aligned_middle_patch/merged/merged_y_%d.pkl' % i, 'wb') as f:
        pickle.dump({i*100+j: y[random_perf[i*100+j]] for j in range(100) if (i*100+j) < len(X)}, f)

w = pickle.load(open('data/linear_aligned_middle_patch/merged/merged_w.pkl', 'rb'))
for i in range(np.ceil(len(X)/100).astype(int)):
    print(i)
    with open('data/linear_aligned_middle_patch/merged/merged_w_%d.pkl' % i, 'wb') as f:
        pickle.dump({i*100+j: w[random_perf[i*100+j]] for j in range(100) if (i*100+j) < len(X)}, f)

names = pickle.load(open('data/linear_aligned_middle_patch/merged/merged_names.pkl', 'rb'))
names_perfed = {i: names[random_perf[i]] for i in range(len(X))}
with open('data/linear_aligned_middle_patch/merged/merged_names_perfed.pkl', 'wb') as f:
    pickle.dump(names_perfed, f)
"""

"""
### Sample figures ###
processed_fs = [f for f in os.listdir('data/linear_aligned_middle_patch') if 'processed' in f]
np.random.shuffle(processed_fs)
for f in processed_fs[:5]:
  f_name = 'data/linear_aligned_middle_patch/%s' % f
  Xs, ys, ws, names = pickle.load(open(f_name, 'rb'))
  pair_dats = pickle.load(open('data/%s.pkl' % '_'.join(f.split('_')[:2]), 'rb'))
  if len(names) == 0:
    continue
  elif len(names) < 5:
    selected_samples = np.arange(len(names))
  else:
    selected_samples = np.random.choice(np.arange(len(names)), (5,), replace=False)
  for i in selected_samples:
    name = names[i]
    name2 = [k for k in pair_dats if k[0] == name][0]
    pc_original = pair_dats[name2][0]
    fl_original = pair_dats[name2][1]
    pc = Xs[i]
    fl = ys[i]
    plt.clf()
    plt.imshow(pc_original)
    plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_pc_original.png' % '_'.join(f.split('_')[:2] + name.split('/')[-1].split('_')[:1]))
    plt.clf()
    plt.imshow(fl_original)
    plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_fl_original.png' % '_'.join(f.split('_')[:2] + name.split('/')[-1].split('_')[:1]))
    plt.clf()
    plt.imshow(pc)
    plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_pc.png' % '_'.join(f.split('_')[:2] + name.split('/')[-1].split('_')[:1]))
    plt.clf()
    plt.imshow(fl)
    plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_fl.png' % '_'.join(f.split('_')[:2] + name.split('/')[-1].split('_')[:1]))
  """
