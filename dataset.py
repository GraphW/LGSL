import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from pandas import DataFrame

class fMRIDataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.filename = []
        with open(self.root, "r") as f:
            for line in f.readlines():
                self.filename.append(line.strip('\n'))

    def __getitem__(self, index):
        # 迭代的返回一个数据和一个标签
        file_name = self.filename[index][2:]
        # print(file_name)

        # 读取方式改变一下就行了
        # wb_pd = pd.read_excel(file_name, header=None, engine='openpyxl')
        wb = np.load(file_name)
        wb_pd = DataFrame(wb)
        raw_data = wb_pd.T
        raw_data = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min())
        raw_data = torch.tensor(data=raw_data.values)
        label = self.filename[index][0]
        label = torch.tensor(int(label))

        return raw_data, label

    def __len__(self):
        return len(self.filename)


def load_fMRI_dataset(fMRI_root, dataset=fMRIDataset, batch_size=32, shuffle=True, print_dataset=False):
    train_root = os.path.join(fMRI_root, 'train_set.txt')
    valid_root = os.path.join(fMRI_root, 'val_set.txt')
    test_root = os.path.join(fMRI_root, 'test_set.txt')  # 先将数据导入再进行划分

    train_dataset = dataset(root=train_root)
    valid_dataset = dataset(root=valid_root)
    test_dataset = dataset(root=test_root)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, valid_loader, test_loader
