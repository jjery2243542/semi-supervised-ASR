import numpy as np
from torch.utils.data import DataLoader
import torch
from dataset import PartedDataset

def _collate_fn(l):
    l.sort(key=lambda x: x[0].shape[0], reverse=True)
    features = [torch.from_numpy(feature) for feature, _ in l]
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    ilens = [feature.shape[0] for feature, _ in l]
    texts = [torch.from_numpy(text) for _, text in l]
    return padded_features, ilens, texts

class PartLoaderIter:
    def __init__(self, loader):
        assert len(loader) > 0
        self.loader = loader
        self.shuffle = loader.shuffle
        self.batch_size = loader.batch_size
        self.in_iter = iter(loader.in_loader)
        self.part_iter = self.get_part_iter(next(self.in_iter)[0])

    def get_part_iter(self, part):
        return iter(DataLoader(part,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle,
                               collate_fn=_collate_fn,
                               num_workers=0))

    def __len__(self):
        return self.loader._len

    def __iter__(self):
        return self

    def __next__(self):
        try:
            ret = next(self.part_iter)
        except StopIteration:
            self.part_iter = self.get_part_iter(next(self.in_iter)[0])
            ret = next(self.part_iter)
        return ret

class PartLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.in_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle,
                                    num_workers=0, collate_fn=lambda x: x)

        self._len = self.dataset.batch_count(self.batch_size)

    def __len__(self):
        return self._len

    def __iter__(self):
        return PartLoaderIter(self)

if __name__ == '__main__':
    config = {'max_feature_length': 1600,
            'min_feature_length': 30, 
            'max_text_length': 100,
            'min_text_length': 2}
    ds = PartedDataset('/storage/feature/LibriSpeech/packed/train-clean-360/', config=config)
    data_loader = PartLoader(ds, batch_size=32, shuffle=True)
    print(len(data_loader))
    for data in data_loader:
        print(data[0].size(), len(data[1]), len(data[2]), data[2][0].size())
