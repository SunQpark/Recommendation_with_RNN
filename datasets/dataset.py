import sys, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RSC15Dataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, delimiter="\t").groupby('SessionId')
        
        self.items = np.sort(self.data['ItemId'].unique())
        self.items_dict = {item:index for index, item in enumerate(self.items)} 
        self.num_items = len(self.items)

        item2index = lambda item: self.items_dict[item]
        self.data['ItemIndex'] = self.data.apply(item2index)

        self.data.groupby('SessionId')['ItemIndex'].apply(np.array)

    def __len__(self):
        return len(self.data['SessionId'].unique())

    def __getitem__(self, idx):
        return self.data['ItemID'][idx]



if __name__ == '__main__':
    rsc_dataset = RSC15Dataset('datasets/data/rsc15_train_full.txt')
    print(len(rsc_dataset))



