import sys, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class RSC15Dataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, delimiter="\t")
        
        # convert item ids into indices
        self.items = np.sort(self.data['ItemId'].unique())
        self.items_dict = {item:index for index, item in enumerate(self.items)} 
        self.num_items = len(self.items)

        item2index = lambda item: self.items_dict[item]
        self.data['ItemIndex'] = self.data['ItemId'].apply(item2index)

        # convert into sequences(np.array) of indices of items
        self.sessions = self.data.groupby('SessionId')['ItemIndex'].apply(np.array)

        self.max_length = self.sessions.apply(len).max()
        del self.data

    def _one_hot(self, list_of_indices):
        seq_length = list_of_indices.shape[0]
        result = np.zeros((seq_length, self.num_items), dtype=float)
        result[np.arange(seq_length), list_of_indices] = 1
        return result

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        return self._one_hot(self.sessions.iloc[idx])


if __name__ == '__main__':
    rsc_dataset = RSC15Dataset('datasets/data/rsc15_train_full.txt')


    index = 10
    print(len(rsc_dataset))

    print(rsc_dataset[index])
    print(rsc_dataset[index].shape)
    

    loader = DataLoader(rsc_dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(loader):
        if i == 0:
            break





