
import numpy as np
import torch
from torch.utils.data import Dataset

class MechanicalModelDataset(Dataset):
    def __init__(self, data_source, train=True):
        self.train = train

        if isinstance(data_source, str):
            # If data_source is a string, treat it as a file path
            self.__dataset_npz = np.load(data_source, allow_pickle=True)
        elif isinstance(data_source, dict):
            # If data_source is a dict, assume it's already loaded data
            self.__dataset_npz = data_source
        else:
            raise ValueError("data_source must be either a file path or a dictionary.")

        self.data = self.__dataset_npz['data']
        self.dataset_len = len(self.data)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.train:
            return self.data[idx, :-1], self.data[idx, -1]
        else:
            return self.data[idx, :-1]

    def get_data_shape(self):
        return self.data.shape
