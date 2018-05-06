import sys, os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence
from torchvision import transforms
# from base import BaseDataLoader
sys.path.append("./")
from datasets.dataset import RSC15Dataset


class RSC15DataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        super(RSC15DataLoader, self).__init__(
            RSC15Dataset(os.path.join(data_dir, 'rsc15_train_full.txt'), transforms=torch.tensor),
            batch_size=self.batch_size,
            shuffle=self.shuffle, 
            collate_fn=self._collate_fn
        )

        self._n_samples = len(self.dataset)

    def __len__(self):
        """
        return total number of batches
        """
        return self._n_samples // self.batch_size

    def _collate_fn(self, list_inputs):
        """
        arg: 
            list_inputs: list of tensors containing input sequences
        """
        # sorting sequences by length
        order = np.argsort([item.shape[0] for item in list_inputs])
        list_sorted = [list_inputs[i] for i in order[::-1]]
        return pack_sequence(list_sorted)



if __name__ == "__main__":
    dl = RSC15DataLoader('datasets/data', batch_size=4, shuffle=True)
    for batch in dl:
        print(batch)
        break

