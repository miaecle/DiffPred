from segment_support import *
from data_loader import *

# ### Preprocess ###
# dat_fs = [f for f in os.listdir('data') if f.startswith('ex')]
# for f_name in dat_fs:
#   f_name = f_name.split('.')[0]
#   if not 'processed' in f_name:
#     print(f_name)
#     dats = pickle.load(open('./data/%s.pkl' % f_name, 'rb'))
#     if not f_name.startswith('ex2'): # ex2 is a different setting
#       pos_code = '5'
#       try:
#         pos_code_dats = {k:v for k,v in dats.items() if position_code(k) == pos_code}
#         processed_dats = preprocess(pos_code_dats)
#         with open('./data/linear_aligned_middle_patch/%s_processed_%s.pkl' % (f_name, pos_code), 'wb') as f:
#           pickle.dump(processed_dats, f)
#       except Exception as e:
#         print(e)
#         continue

### Assemble for training ###
processed_fs = ['data/linear_aligned_middle_patch/%s' % f \
                for f in os.listdir('data/linear_aligned_middle_patch') if 'processed' in f]
X, y, w, names = assemble_for_training(processed_fs, (288, 384))
with open('data/linear_aligned_middle_patch/merged_X.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('data/linear_aligned_middle_patch/merged_y.pkl', 'wb') as f:
    pickle.dump(y, f)
with open('data/linear_aligned_middle_patch/merged_w.pkl', 'wb') as f:
    pickle.dump(w, f)
with open('data/linear_aligned_middle_patch/merged_names.pkl', 'wb') as f:
    pickle.dump(names, f)


# ### Sample figures ###
# processed_fs = [f for f in os.listdir('data/linear_aligned_middle_patch') if 'processed' in f]
# np.random.shuffle(processed_fs)
# for f in processed_fs[:5]:
#   f_name = 'data/linear_aligned_middle_patch/%s' % f
#   Xs, ys, ws, names = pickle.load(open(f_name, 'rb'))
#   pair_dats = pickle.load(open('data/%s.pkl' % '_'.join(f.split('_')[:2]), 'rb'))
#   if len(names) == 0:
#     continue
#   elif len(names) < 5:
#     selected_samples = np.arange(len(names))
#   else:
#     selected_samples = np.random.choice(np.arange(len(names)), (5,), replace=False)
#   for i in selected_samples:
#     name = names[i]
#     name2 = [k for k in pair_dats if k[0] == name][0]
#     pc_original = pair_dats[name2][0]
#     fl_original = pair_dats[name2][1]
#     pc = Xs[i]
#     fl = ys[i]
#     plt.clf()
#     plt.imshow(pc_original)
#     plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_pc_original.png' % '_'.join(f.split('_')[:2] + name.split('/')[-1].split('_')[:1]))
#     plt.clf()
#     plt.imshow(fl_original)
#     plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_fl_original.png' % '_'.join(f.split('_')[:2] + name.split('/')[-1].split('_')[:1]))
#     plt.clf()
#     plt.imshow(pc)
#     plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_pc.png' % '_'.join(f.split('_')[:2] + name.split('/')[-1].split('_')[:1]))
#     plt.clf()
#     plt.imshow(fl)
#     plt.savefig('/home/zqwu/Dropbox/fig_temp/%s_fl.png' % '_'.join(f.split('_')[:2] + name.split('/')[-1].split('_')[:1]))

