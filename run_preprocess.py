from segment_support import *
from data_loader import *

dat_fs = os.listdir('data')
for f_name in dat_fs:
  f_name = f_name.split('.')[0]
  if not 'processed' in f_name:
    print(f_name)
    dats = pickle.load(open('./data/%s.pkl' % f_name, 'rb'))
    if not f_name.startswith('ex2'): # ex2 is a different setting
      pos_code = '5'
      try:
        pos_code_dats = {k:v for k,v in dats.items() if position_code(k) == pos_code}
        processed_dats = preprocess(pos_code_dats)
        with open('./data/linear_aligned_middle_patch/%s_processed_%s.pkl' % (f_name, pos_code), 'wb') as f:
          pickle.dump(processed_dats, f)
      except Exception as e:
        print(e)
        continue
