import sys, os
import numpy as np
import pandas as pd
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
from torchvision.transforms import Compose


class RSC15Dataset(Dataset):
    def __init__(self, data_path, transforms=None, one_hot=True):
        self.data = pd.read_csv(data_path, delimiter="\t")
        
        # convert item ids into indices
        self.items = np.sort(self.data['ItemId'].unique())
        self.items_dict = {item:index for index, item in enumerate(self.items)} 
        self.num_items = len(self.items)

        item2index = lambda item: self.items_dict[item]
        self.data['ItemIndex'] = self.data['ItemId'].apply(item2index)

        # convert into sequences(np.array) of indices of items
        self.sessions = self.data.groupby('SessionId')['ItemIndex'].apply(np.array)


        self.one_hot = one_hot
        self.transforms = transforms
        self.max_length = self.sessions.apply(len).max()
        del self.data

    def _one_hot(self, array_indices):
        seq_length = array_indices.shape[0]
        result = np.zeros((seq_length, self.num_items), dtype=float)
        result[np.arange(seq_length), array_indices] = 1
        return result

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        sess = self.sessions.iloc[idx]
        if self.one_hot:
            sess =  self._one_hot(sess)

        if self.transforms is not None:
            sess = self.transforms(sess)    

        return sess


if __name__ == '__main__':
    rsc_dataset = RSC15Dataset('datasets/data/rsc15_train_full.txt', 
        transforms=Compose([torch.tensor]), one_hot=False)

    index = 10
    print(len(rsc_dataset))

    sess = rsc_dataset[index]
    print(sess)
    print(sess.shape)
    

    # loader = DataLoader(rsc_dataset, batch_size=16, collate_fn=pack_sequence, shuffle=True)

    # for i, batch in enumerate(loader):
    #     print(batch)
    #     if i == 0:
    #         break





