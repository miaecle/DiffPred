from data_generator import *
from segment_support import *
from data_loader import *

root = "/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-0"

name_file = os.path.join(root, "names.pkl")
X_ct = len([f for f in os.listdir(root) if f.startswith('X_')])
X_files = [os.path.join(root, "X_%d.pkl" % i) for i in range(X_ct)]

### save for segmentation usage ###
segment_y_files = [os.path.join(root, "segment_continuous_y_%d.pkl" % i) for i in range(X_ct)]
segment_w_files = [os.path.join(root, "segment_continuous_w_%d.pkl" % i) for i in range(X_ct)]
classify_label_file = os.path.join(root, "classify_continuous_labels.pkl")
base_dataset = CustomGenerator(name_file,
                 X_files, 
                 segment_y_files=segment_y_files, 
                 segment_w_files=segment_w_files,
                 n_segment_classes=4,
                 segment_class_weights=[1, 1, 1, 1],
                 segment_extra_weights=None,
                 segment_label_type='continuous',
                 classify_label_file=classify_label_file,
                 n_classify_classes=4,
                 classify_class_weights=[1, 1, 1, 1],
                 classify_label_type='continuous',
                 sample_per_file=100,
                 cache_file_num=5)

X_valid = []
segment_continuous_valid = []
classify_continuous_valid = []

for i in base_dataset.selected_inds:
    try:
        X, y, w, name = base_dataset.load_ind(i)
    except Exception as e:
        print(e)
        print("ISSUE %d" % i)
        continue
    if not X is None:
        X_valid.append(i)
    if not y is None and not w is None:
        if not np.all(w == 0):
            segment_continuous_valid.append(i)
    if not base_dataset.classify_y[i] is None and not base_dataset.classify_w[i] is None:
        if base_dataset.classify_w[i] > 0:
            classify_continuous_valid.append(i)
    if i % 1000 == 0:
        print(i)

selected_inds = set(X_valid) & set(classify_continuous_valid)
selected_inds = np.array(sorted(selected_inds))
with open("/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-0_continuous_inds.pkl", "wb") as f:
    pickle.dump(selected_inds, f)
print("TOTAL samples: %d" % len(selected_inds))

np.random.seed(123)
np.random.shuffle(selected_inds)
save_path = "/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-0_continuous/"
os.makedirs(save_path, exist_ok=True)
base_dataset.reorder_save(selected_inds, 
                          save_path=save_path,
                          write_segment_labels=True,
                          write_classify_labels=True)



# Fill in segmentation masks for pure negative samples
base_names = pickle.load(open(name_file, 'rb'))
name_to_id_mapping = {n: i for i, n in base_names.items()}
def get_original_w(name):
    ind = name_to_id_mapping[name]
    temp_w_dat = pickle.load(open(os.path.join(root, "segment_discrete_w_%d.pkl" % (ind // 100)), 'rb'))
    return temp_w_dat[ind]


_name_file = os.path.join(save_path, "names.pkl")
_X_ct = len([f for f in os.listdir(save_path) if f.startswith('X_')])
_segment_y_files = [os.path.join(save_path, "segment_continuous_y_%d.pkl" % i) for i in range(_X_ct)]
_segment_w_files = [os.path.join(save_path, "segment_continuous_w_%d.pkl" % i) for i in range(_X_ct)]
_classify_label_file = os.path.join(save_path, "classify_continuous_labels.pkl")

_names = pickle.load(open(_name_file, 'rb'))
_classify_labels = pickle.load(open(_classify_label_file, 'rb'))
for segment_y_file, segment_w_file in zip(_segment_y_files, _segment_w_files):
    y_dat = pickle.load(open(segment_y_file, 'rb'))
    w_dat = pickle.load(open(segment_w_file, 'rb'))
    assert y_dat.keys() == w_dat.keys()
    for k in y_dat:
        if _classify_labels[k][0][0] == 1:
            if not np.allclose(y_dat[k].sum(), 0.):
                print("Issue with key %d" % k)
            w_dat[k] = get_original_w(_names[k])
            y_dat[k] = np.concatenate([
                np.ones((288, 384, 1), dtype=float),
                np.zeros((288, 384, 3), dtype=float)
                ], 2)
    with open(segment_y_file, 'wb') as f:
        pickle.dump(y_dat, f)
    with open(segment_w_file, 'wb') as f:
        pickle.dump(w_dat, f)



### save for classification prediction ###
segment_y_files = [os.path.join(root, "segment_discrete_y_%d.pkl" % i) for i in range(X_ct)]
segment_w_files = [os.path.join(root, "segment_discrete_w_%d.pkl" % i) for i in range(X_ct)]
classify_label_file = os.path.join(root, "classify_discrete_labels.pkl")
base_dataset = CustomGenerator(name_file,
                 X_files, 
                 segment_y_files=segment_y_files, 
                 segment_w_files=segment_w_files,
                 n_segment_classes=2,
                 segment_class_weights=[1, 1],
                 segment_extra_weights=None,
                 segment_label_type='discrete',
                 classify_label_file=classify_label_file,
                 n_classify_classes=2,
                 classify_class_weights=[1, 1],
                 classify_label_type='discrete',
                 sample_per_file=100,
                 cache_file_num=5)

X_valid = []
segment_discrete_valid = []
classify_discrete_valid = []

for i in base_dataset.selected_inds:
    try:
        X, y, w, name = base_dataset.load_ind(i)
    except Exception as e:
        print(e)
        print("ISSUE %d" % i)
        continue
    if not X is None:
        X_valid.append(i)
    if not y is None and not w is None:
        if not np.all(w == 0):
            segment_discrete_valid.append(i)
    if not base_dataset.classify_y[i] is None and not base_dataset.classify_w[i] is None:
        if base_dataset.classify_w[i] > 0:
            classify_discrete_valid.append(i)
    if i % 1000 == 0:
        print(i)

with open(os.path.join(root, 'validities.pkl'), 'wb') as f:
    pickle.dump([X_valid, segment_discrete_valid, classify_discrete_valid], f)

# 0-to-inf data pairs
X_valid = set(X_valid)
valid_target_inds = set(X_valid) & set(segment_discrete_valid) & set(classify_discrete_valid)
valid_wells = sorted(set([get_identifier(base_dataset.names[i])[:2] + get_identifier(base_dataset.names[i])[3:] for i in valid_target_inds]))
id_mapping = {i: get_identifier(base_dataset.names[i]) for i in X_valid}

def get_pairs(inds, label=1, startday_range=(4, 12)):
    well_labels = [base_dataset.classify_y[i] for i in inds]
    well_weights = [base_dataset.classify_w[i] for i in inds]

    for i in range(len(well_labels)):
        if well_weights[i] > 0 and well_labels[i] == label:
            break
    if well_weights[i] > 0 and well_labels[i] == label:
        end_ind = inds[i]
        if int(id_mapping[end_ind][2]) < 10:
            return []
        start_inds = [ind for ind in inds if \
            int(id_mapping[ind][2]) >= startday_range[0] and \
            int(id_mapping[ind][2]) <= startday_range[1] and \
            int(id_mapping[ind][2]) <= int(id_mapping[end_ind][2]) - 3]
        return [(i, end_ind) for i in start_inds]
    else:
        return []

quest_pairs = []
extra_pairs = []
for well in valid_wells:
    related_inds = [i for i in X_valid if id_mapping[i][:2] + id_mapping[i][3:] == well]
    related_inds = [i for i in related_inds if int(id_mapping[i][2]) <= 18]
    related_inds = sorted(related_inds, key=lambda x: -int(id_mapping[x][2]))

    well_labels = [base_dataset.classify_y[i] for i in related_inds]
    well_weights = [base_dataset.classify_w[i] for i in related_inds]

    if not 1 in well_labels:
        quest_pairs.extend(get_pairs(related_inds, label=0, startday_range=(4, 12)))
        extra_pairs.extend(get_pairs(related_inds, label=0, startday_range=(0, 3)))
    else:
        _well_labels = [lab for i, lab in enumerate(well_labels) if not lab is None and well_weights[i] > 0]
        ct = [not (_well_labels[i] == _well_labels[i+1]) for i in range(len(_well_labels) - 1)]
        ct = sum(ct)
        if ct <= 1:
            quest_pairs.extend(get_pairs(related_inds, label=1, startday_range=(4, 12)))
            extra_pairs.extend(get_pairs(related_inds, label=1, startday_range=(0, 3)))
        elif _well_labels[0] == 1 and sum(_well_labels) > 2:
            quest_pairs.extend(get_pairs(related_inds, label=1, startday_range=(4, 12)))
            extra_pairs.extend(get_pairs(related_inds, label=1, startday_range=(0, 3)))
        elif _well_labels[1] == 1 and sum(_well_labels) > 3:
            quest_pairs.extend(get_pairs(related_inds, label=1, startday_range=(4, 12)))
            extra_pairs.extend(get_pairs(related_inds, label=1, startday_range=(0, 3)))
        elif sum(_well_labels) < 2:
            quest_pairs.extend(get_pairs(related_inds, label=0, startday_range=(4, 12)))
            extra_pairs.extend(get_pairs(related_inds, label=0, startday_range=(0, 3)))
        else:
            print("Exclude well %s" % str(well))

quest_pairs = sorted(quest_pairs)
np.random.seed(123)
np.random.shuffle(quest_pairs)
with open("/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete_inds.pkl", "wb") as f:
    pickle.dump(quest_pairs, f)

save_path="/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/"
os.makedirs(save_path, exist_ok=True)
base_dataset.cross_pair_save(
    quest_pairs, 
    save_path=save_path,
    write_segment_labels=True,
    write_classify_labels=True)


extra_pairs = sorted(extra_pairs)
np.random.seed(123)
np.random.shuffle(extra_pairs)
with open("/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete_inds_extra.pkl", "wb") as f:
    pickle.dump(extra_pairs, f)

save_path="/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-inf_discrete/extra_day0-3_samples/"
os.makedirs(save_path, exist_ok=True)
base_dataset.cross_pair_save(
    extra_pairs, 
    save_path=save_path,
    write_segment_labels=True,
    write_classify_labels=True)





# 0-to-(7~10) data pairs

target_range = (7, 10)

input_id_mapping = {i: get_identifier(base_dataset.names[i]) for i in X_valid}
output_id_mapping = {get_identifier(base_dataset.names[i]): i for i in segment_discrete_valid}

all_pairs = []
for i, identifier in input_id_mapping.items():
    for interval in range(target_range[0], target_range[1]+1):
        output_identifier = (identifier[0], identifier[1], str(int(identifier[2]) + interval), identifier[3], identifier[4])
        if output_identifier in output_id_mapping:
            all_pairs.append((i, output_id_mapping[output_identifier]))

all_pairs = sorted(all_pairs)
np.random.seed(123)
np.random.shuffle(all_pairs)
selected_pairs = base_dataset.shrink_pairs(all_pairs)
with open("/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-%d_discrete_inds.pkl" % target_range[1], "wb") as f:
    pickle.dump(selected_pairs, f)

save_path="/oak/stanford/groups/jamesz/zqwu/iPSC_data/train_set/0-to-%d_discrete/" % target_range[1]
os.makedirs(save_path, exist_ok=True)
base_dataset.cross_pair_save(
    selected_pairs, 
    save_path=save_path,
    write_segment_labels=True,
    write_classify_labels=True)