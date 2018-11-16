import numpy as np
from torch.utils.data import DataLoader
import torch
from dataset import PickleDataset

def _collate_fn(l):
    l.sort(key=lambda x: x[0].shape[0], reverse=True)
    features = [torch.from_numpy(feature) for feature, _ in l]
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    ilens = [feature.shape[0] for feature, _ in l]
    texts = [torch.from_numpy(np.array(text)) for _, text in l]
    return padded_features, ilens, texts

def _text_collate_fn(l):
    l.sort(key=lambda x: len(x[1]), reverse=True)
    texts = [torch.from_numpy(np.array(text)) for _, text in l]
    return texts

def _speech_collate_fn(l):
    l.sort(key=lambda x: x[0].shape[0], reverse=True)
    features = [torch.from_numpy(feature) for feature, _ in l]
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    ilens = [feature.shape[0] for feature, _ in l]
    return padded_features, ilens

def get_data_loader(dataset, batch_size, shuffle, drop_last, speech_only=False, text_only=False):
    if (not speech_only) and (not text_only):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn, 
                num_workers=0, drop_last=drop_last)
    elif speech_only:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_speech_collate_fn, 
                num_workers=0, drop_last=drop_last)
    elif text_only:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_text_collate_fn, 
                num_workers=0, drop_last=drop_last)

#class PartLoaderIter:
#    def __init__(self, loader):
#        assert len(loader) > 0
#        self.loader = loader
#        self.shuffle = loader.shuffle
#        self.batch_size = loader.batch_size
#        self.in_iter = iter(loader.in_loader)
#        self.part_iter = self.get_part_iter(next(self.in_iter)[0])
#
#    def get_part_iter(self, part):
#        return iter(DataLoader(part,
#                               batch_size=self.batch_size,
#                               shuffle=self.shuffle,
#                               collate_fn=_collate_fn,
#                               num_workers=0))
#
#    def __len__(self):
#        return self.loader._len
#
#    def __iter__(self):
#        return self
#
#    def __next__(self):
#        try:
#            ret = next(self.part_iter)
#        except StopIteration:
#            self.part_iter = self.get_part_iter(next(self.in_iter)[0])
#            ret = next(self.part_iter)
#        return ret
#
#class PartLoader:
#    def __init__(self, dataset, batch_size=1, shuffle=False):
#        self.dataset = dataset
#        self.batch_size = batch_size
#        self.shuffle = shuffle
#        self.in_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle,
#                                    num_workers=0, collate_fn=lambda x: x)
#
#        self._len = self.dataset.batch_count(self.batch_size)
#
#    def __len__(self):
#        return self._len
#
#    def __iter__(self):
#        return PartLoaderIter(self)

if __name__ == '__main__':
    config = {'max_feature_length': 1600,
            'min_feature_length': 30, 
            'max_text_length': 200,
            'min_text_length': 2}
    ds = PickleDataset('/storage/feature/wsj/processed/train_si84.pkl', config=config, sort=True)
    data_loader = get_data_loader(ds, batch_size=32, shuffle=False)
    print(len(data_loader))
    for data in data_loader:
        x, ilens, y = data
        print(x.size(), len(ilens), len(y), y[0].size())
