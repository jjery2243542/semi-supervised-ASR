import sys
import numpy as np 
import os 

'''
usage: python3 ark2npy.py [ark_path] [npy_dir] [dataset] [utt_id] 
'''
ark_path = sys.argv[1]
npy_dir = sys.argv[2]
dataset = sys.argv[3]
utt_id = sys.argv[4]

frames = []
with open(ark_path, 'r') as f_in:
    for i, line in enumerate(f_in):
        # feature id, skip
        if i == 0:
            continue
        frame = [float(word) for word in line.strip(' \n]').split()]
        frames.append(frame)

# convert to numpy array
features = np.array(frames)

speaker, chapter, seg_id = utt_id.split('-', maxsplit=2)
# create folder
directory = f'{npy_dir}/{dataset}/{speaker}/{chapter}/'
if not os.path.exists(directory):
    os.makedirs(directory)

# save to folder
np.save(os.path.join(directory, f'{utt_id}.npy'), features)

