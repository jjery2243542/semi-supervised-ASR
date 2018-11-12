import random
import pickle 
import sys
import random

full_ds_path = sys.argv[1]
half_ds_path = sys.argv[2]

with open(full_ds_path, 'rb') as f:
    full_ds = pickle.load(f)

keys = list(full_ds.keys())
shuffled_keys = random.shuffle(keys)

split_point = len(keys) // 2

first_half_keys = shuffled_keys[:split_point]
second_half_keys = shuffled_keys[split_point:]

ext = half_ds_path[::-1].split('.')
first_half_path = f'{half_ds_path[:-len(ext) - 1]}-0.{ext}'
second_half_path = f'{half_ds_path[:-len(ext) - 1]}-1.{ext}'

with open(first_half_path, 'wb') as f:
    ds = {key: full_ds[key] for key in first_half_keys}
    pickle.dump(ds)

with open(second_half_path, 'wb') as f:
    ds = {key: full_ds[key] for key in second_half_keys}
    pickle.dump(ds)

