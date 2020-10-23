import os
import pickle
from segment_support import *
from data_loader import *
from data_generator import CustomGenerator

print("Loading data")
RAW_PATH = '/scratch/users/zqwu/iPSC_data_ecs/'
SAVE_PATH = '/scratch/users/zqwu/iPSC_data_ecs'
pairs = load_all_pairs(path=RAW_PATH)

### Data Loader ###

def get_identifier(f):
    return get_ex_day(f) + get_well(f)[:1]
groups = set(get_identifier(p[0]) if not p[0] is None else get_identifier(p[1]) for p in pairs)

for g in groups:
    print(g, flush=True)
    save_file_name = os.path.join(SAVE_PATH, '_'.join(g) + '.pkl')
    if os.path.exists(save_file_name):
        continue
    group_pairs = [p for p in pairs if (get_identifier(p[0]) if not p[0] is None else get_identifier(p[1])) == g]
    group_pair_dats = {p: load_image_pair(p) for p in group_pairs}
    with open(save_file_name, 'wb') as f:
        pickle.dump(group_pair_dats, f)


 # ('ex1', 'D7'): [786, 0],
 # ('ex1', 'D10'): [450, 516],


### Preprocess ###
print("PREPROCESS")
dat_fs = [f for f in os.listdir(SAVE_PATH) if f.startswith('ex') and not 'processed' in f and f.endswith('.pkl')]

processed_save_path = os.path.join(SAVE_PATH, 'linear_aligned_patches')
if not os.path.exists(processed_save_path):
    os.makedirs(processed_save_path)

for f_name in dat_fs:
    print(f_name, flush=True)
    processed_f_name = os.path.join(processed_save_path, '%s_processed.pkl' % f_name.split('.')[0])
    if os.path.exists(processed_f_name):
        continue

    dats = pickle.load(open(os.path.join(SAVE_PATH, f_name), 'rb'))
    try:
        well_dats = {k:v for k,v in dats.items() if \
              not get_well(k[0])[1] in ['1', '2', '16',
                                        '14', '15', '30',
                                        '196', '211', '212',
                                        '210', '224', '225']}
        processed_dats = preprocess(well_dats, linear_align=False, label='segmentation')
        with open(processed_f_name, 'wb') as f:
            pickle.dump(processed_dats, f)
    except Exception as e:
        print(e)
        continue



### Assemble for training ###
print("ASSEMBLE")
processed_fs = [os.path.join(processed_save_path, f)
                for f in os.listdir(processed_save_path) \
                if 'processed' in f and f.startswith('ex') and f.endswith('.pkl')]

dataset_path = os.path.join(processed_save_path, 'merged_all/')
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

_ = assemble_for_training(processed_fs, (384, 288), save_path=dataset_path, label='segmentation')
n_fs = len([f for f in os.listdir(dataset_path) if f.startswith('X')])
data_gen = CustomGenerator([os.path.join(dataset_path, 'X_%d.pkl' % i) for i in range(n_fs)],
                           [os.path.join(dataset_path, 'y_%d.pkl' % i) for i in range(n_fs)],
                           [os.path.join(dataset_path, 'w_%d.pkl' % i) for i in range(n_fs)],
                           name_file = os.path.join(dataset_path, 'names.pkl'),
                           include_day=False,
                           batch_size=8)


valid_fl_samples = [get_ex_day(p[1]) + get_well(p[1]) for p in pairs if p[1] is not None]
fl_inds = np.array([i for i, n in data_gen.names.items() if \
    get_ex_day(n) + get_well(n) in valid_fl_samples and int(get_ex_day(n)[1][1:]) >= 5])
np.random.seed(123)
np.random.shuffle(fl_inds)

dataset_fl_path = os.path.join(processed_save_path, 'merged_all_fl/')
if not os.path.exists(dataset_fl_path):
    os.makedirs(dataset_fl_path)
_ = data_gen.reorder_save(fl_inds, save_path=dataset_fl_path + 'permuted_')
