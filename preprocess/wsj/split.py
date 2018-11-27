import random
import pickle 
import sys
import random

full_ds_path = sys.argv[1]
half_ds_path = sys.argv[2]

with open(full_ds_path, 'rb') as f:
    full_ds = pickle.load(f)

keys = list(full_ds.keys())
random.shuffle(keys)

split_point = len(keys) // 2

first_half_keys = keys[:split_point]
second_half_keys = keys[split_point:]

ext = half_ds_path[::-1].split('.')[0][::-1]
first_half_path = f'{half_ds_path[:-len(ext) - 1]}-0.{ext}'
second_half_path = f'{half_ds_path[:-len(ext) - 1]}-1.{ext}'

with open(first_half_path, 'wb') as f:
    print(f'write to {first_half_path}')
    ds = {key: full_ds[key] for key in first_half_keys}
    pickle.dump(ds, f)

with open(second_half_path, 'wb') as f:
    print(f'write to {second_half_path}')
    ds = {key: full_ds[key] for key in second_half_keys}
    pickle.dump(ds, f)

