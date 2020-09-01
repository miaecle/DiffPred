from segment_support import *
from data_loader import *
from data_generator import CustomGenerator

### Preprocess ###
data_path = '/oak/stanford/groups/jamesz/zqwu/iPSC_data'
dat_fs = [f for f in os.listdir(data_path) if f.startswith('ex')]

for f_name in dat_fs:
  f_name = f_name.split('.')[0]
  if not 'processed' in f_name:
    print(f_name, flush=True)
    if not f_name.startswith('ex2'): # ex2 is a different setting
      dats = pickle.load(open(data_path + '/%s.pkl' % f_name, 'rb'))
      for pos_code in ['2', '4', '5', '6', '8']:
        if os.path.exists(data_path + '/discretized_fl/%s_processed_%s.pkl' % (f_name, pos_code)):
          continue
        print(pos_code)
        try:
          pos_code_dats = {k:v for k,v in dats.items() if get_well(k[0])[1] == pos_code}
          processed_dats = preprocess(pos_code_dats, linear_align=True, label='discretized')
          with open(data_path + '/discretized_fl/%s_processed_%s.pkl' % (f_name, pos_code), 'wb') as f:
            pickle.dump(processed_dats, f)
        except Exception as e:
          print(e)
          continue
    else:
      dats = pickle.load(open(data_path + '/%s.pkl' % f_name, 'rb'))
      for well in ['A1', 'A2', 'A3', 'B2', 'B3']:
        if os.path.exists(data_path + '/discretized_fl/%s_processed_%s.pkl' % (f_name, well)):
          continue
        print(well)
        try:
          pos_code_dats = {k:v for k,v in dats.items() if \
              get_well(k[0])[0] == well and \
              not get_well(k[0])[1] in ['1', '2', '16',
                                        '14', '15', '30',
                                        '196', '211', '212',
                                        '210', '224', '225']}
          processed_dats = preprocess(pos_code_dats, linear_align=False, label='discretized')
          with open(data_path + '/discretized_fl/%s_processed_%s.pkl' % (f_name, well), 'wb') as f:
            pickle.dump(processed_dats, f)
        except Exception as e:
          print(e)
          continue

"""

### Assemble for training ###
data_root = '../iPSC_data'
processed_fs = [data_root + '/linear_aligned_patches/%s' % f \
                for f in os.listdir(data_root + '/linear_aligned_patches/') \
                if 'processed' in f]

pairs = sorted(load_all_pairs(path=data_root + '/predict_gfp_raw'))
valid_names = set(get_ex_day(p[0]) + get_well(p[0]) for p in pairs)

def check_valid(name):
  return (get_ex_day(name) + get_well(name) in valid_names)

data_path = data_root + '/linear_aligned_patches/merged_all/'
_ = assemble_for_training(processed_fs, (384, 288), save_path=data_path, validity_check=check_valid, label='segmentation')
n_fs = len([f for f in os.listdir(data_path) if f.startswith('X')])
data_gen = CustomGenerator([os.path.join(data_path, 'X_%d.pkl' % i) for i in range(n_fs)],
                           [os.path.join(data_path, 'y_%d.pkl' % i) for i in range(n_fs)],
                           [os.path.join(data_path, 'w_%d.pkl' % i) for i in range(n_fs)],
                           name_file = os.path.join(data_path, 'names.pkl'),
                           include_day=False,
                           batch_size=8)
np.random.seed(123)
permuted_inds = np.random.choice(np.arange(data_gen.N), (data_gen.N,), replace=False)
_ = data_gen.reorder_save(permuted_inds, save_path=data_path+'permuted_')



data_path = data_root + '/linear_aligned_patches/merged_center/'
_ = assemble_for_training([f for f in processed_fs if 'ex2' in f or 'processed_5' in f], 
                          (384, 288), 
                          save_path=data_path, 
                          validity_check=check_valid,
                          label='segmentation')
n_fs = len([f for f in os.listdir(data_path) if f.startswith('X')])
data_gen = CustomGenerator([os.path.join(data_path, 'X_%d.pkl' % i) for i in range(n_fs)],
                           [os.path.join(data_path, 'y_%d.pkl' % i) for i in range(n_fs)],
                           [os.path.join(data_path, 'w_%d.pkl' % i) for i in range(n_fs)],
                           name_file = os.path.join(data_path, 'names.pkl'),
                           include_day=False,
                           batch_size=8)
np.random.seed(123)
permuted_inds = np.random.choice(np.arange(data_gen.N), (data_gen.N,), replace=False)
_ = data_gen.reorder_save(permuted_inds, save_path=data_path+'permuted_')


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
