from data_loader import *
from segment_support import *
from layers import *
#from models import Segment, ClassifyOnSegment
from data_generator import CustomGenerator, enhance_weight_fp
import os

data_path = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/linear_aligned_patches/merged_all/'
n_fs = len([f for f in os.listdir(data_path) if f.startswith('permuted_X')])

X_filenames = [os.path.join(data_path, 'permuted_X_%d.pkl' % i) for i in range(n_fs)]
y_filenames = [os.path.join(data_path, 'permuted_y_%d.pkl' % i) for i in range(n_fs)]
w_filenames = [os.path.join(data_path, 'permuted_w_%d.pkl' % i) for i in range(n_fs)]
name_file = os.path.join(data_path, 'permuted_names.pkl')
label_file = os.path.join(data_path, 'permuted_labels.pkl')

gen = CustomGenerator(X_filenames,
                      y_filenames,
                      w_filenames,
                      name_file,
                      label_file,
                      include_day=True,
                      batch_size=8,
                      n_classify_classes=2,
                      classify_class_weights=[1, 1],
                      extra_weights=enhance_weight_fp,
                      allow_size=50)

out_data_path = '/oak/stanford/groups/jamesz/zqwu/iPSC_data/linear_aligned_patches/cross_7-to-10/'
if not os.path.exists(out_data_path):
    os.mkdir(out_data_path)
gen.cross_pair_save(time_interval=[7, 10], seed=123, save_path=out_data_path)
