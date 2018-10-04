import torch 
from torch.utils.data import Dataset
import os
import pickle 
import glob
import numpy as np

class NegativeDataset(Dataset):
    def __init__(self, pickle_path, config=None, sort=True):
        with open(pickle_path, 'rb') as f:
            self.data_dict = pickle.load(f)

        # remove the utterance out of limit
        self.keys = self.get_keys(config, sort=sort)

    def get_keys(self, config, sort):
        if config:
            max_feature_length = config['max_feature_length']
            min_feature_length = config['min_feature_length']
            max_text_length = config['max_text_length']
            min_text_length = config['min_text_length']
            keys = [key for key in self.data_dict 
                    if self.data_dict[key]['feature'].shape[0] <= max_feature_length and 
                    self.data_dict[key]['feature'].shape[0] >= min_feature_length and 
                    len(self.data_dict[key]['token_ids']) <= max_text_length and 
                    len(self.data_dict[key]['token_ids']) >= min_text_length]
        else:
            keys = [key for key in self.data_dict]

        # sort by feature length
        if sort:
            keys = sorted(keys, key=lambda x: self.data_dict[x]['feature'].shape[0])
        return keys

    def __getitem__(self, index):
        negative_index = np.random.choice(list(range(0, index)) + list(range(index + 1, len(self.keys))))
        pos_utt_id = self.keys[index]
        neg_utt_id = self.keys[negative_index]
        feature = self.data_dict[pos_utt_id]['feature']
        token_ids = self.data_dict[neg_utt_id]['token_ids']
        return feature, token_ids

    def __len__(self):
        return len(self.keys)

class PickleDataset(Dataset):
    def __init__(self, pickle_path, config=None, sort=True):
        with open(pickle_path, 'rb') as f:
            self.data_dict = pickle.load(f)

        # remove the utterance out of limit
        self.keys = self.get_keys(config, sort=sort)

    def get_keys(self, config, sort):
        if config:
            max_feature_length = config['max_feature_length']
            min_feature_length = config['min_feature_length']
            max_text_length = config['max_text_length']
            min_text_length = config['min_text_length']
            keys = [key for key in self.data_dict 
                    if self.data_dict[key]['feature'].shape[0] <= max_feature_length and 
                    self.data_dict[key]['feature'].shape[0] >= min_feature_length and 
                    len(self.data_dict[key]['token_ids']) <= max_text_length and 
                    len(self.data_dict[key]['token_ids']) >= min_text_length]
        else:
            keys = [key for key in self.data_dict]

        # sort by feature length
        if sort:
            keys = sorted(keys, key=lambda x: self.data_dict[x]['feature'].shape[0])
        return keys

    def __getitem__(self, index):
        utt_id = self.keys[index]
        feature = self.data_dict[utt_id]['feature']
        token_ids = self.data_dict[utt_id]['token_ids']
        return feature, token_ids

    def __len__(self):
        return len(self.keys)

#class PickleDataset(Dataset):
#    def __init__(self, pickle_path, keys):
#        with open(pickle_path, 'rb') as f:
#            self.data_dict = pickle.load(f)
#        self.keys = keys
#
#    def __len__(self):
#        return len(self.keys)
#
#    def __getitem__(self, index):
#        utt_id = self.keys[index]
#        features = self.data_dict[utt_id]['fbank']
#        # pad, bos, eos
#        text = self.data_dict[utt_id]['bpe'] + 3
#        return features, text
#
#class PartedDataset(Dataset):
#    def __init__(self, root_dir, config):
#        self.root_dir = os.path.expanduser(root_dir)
#        self.pkl_pathes = sorted([path for path in glob.glob(os.path.join(self.root_dir, '*.pkl'))])
#        self.part_keys = self.get_keys(config)
#
#    def get_keys(self, config):
#        # return utterance id with length within constraint
#        max_feature_length = config['max_feature_length']
#        min_feature_length = config['min_feature_length']
#        max_text_length = config['max_text_length']
#        min_text_length = config['min_text_length']
#
#        part_keys = []
#        for pkl_path in self.pkl_pathes:
#            with open(pkl_path, 'rb') as f:
#                data_dict = pickle.load(f)
#            keys = [key for key in data_dict 
#                    if data_dict[key]['fbank'].shape[0] <= max_feature_length and 
#                    data_dict[key]['fbank'].shape[0] >= min_feature_length and 
#                    data_dict[key]['bpe'].shape[0] <= max_text_length and 
#                    data_dict[key]['bpe'].shape[0] >= min_text_length]
#            #del data_dict
#            part_keys.append(keys)
#        return part_keys 
#
#    def __len__(self):
#        return len(self.pkl_pathes)
#
#    def __getitem__(self, index):
#        return PickleDataset(self.pkl_pathes[index], self.part_keys[index])
#
#    def batch_count(self, batch_size):
#        batches = 0
#        for keys in self.part_keys:
#            batches += (len(keys) + batch_size - 1) // batch_size
#        return batches

if __name__ == '__main__':
    config = {'max_feature_length': 1600,
            'min_feature_length': 30, 
            'max_text_length': 200,
            'min_text_length': 2}
    ds = PickleDataset('/storage/feature/wsj/processed/train_si84.pkl', config=config, sort=True)
    print(len(ds))
    print(ds[-1][0].shape)
    print(len(ds[-1][1]))
    #ds = PartedDataset('/storage/feature/LibriSpeech/packed/train-clean-100/', config=config)
    #part = ds[0]
    #print(ds.batch_count(32))
    #print(len(ds))
    #print(len(part))
    #features, text = part[0]
    #print(features.shape, text.shape, type(features), type(text), text)
    

