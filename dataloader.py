import numpy as np
import torch
from torch.utils.data import Dataset

class ThermalModelDataset(Dataset):
    def __init__(self, npz_file, train=True):
        self.train = train
        self.__dataset_npz = np.load(npz_file)
        self.temperature_train = torch.from_numpy(self.__dataset_npz['temperature_train']).to(torch.float64)
        self.temperature_test = torch.from_numpy(self.__dataset_npz['temperature_test']).to(torch.float64)
        self.ground_truth = torch.from_numpy(self.__dataset_npz['ground_truth']).to(torch.float64)
        
        self.y_train = torch.unsqueeze(torch.from_numpy(self.__dataset_npz['y_train']).to(torch.float64),1)
        self.y_test = torch.unsqueeze(torch.from_numpy(self.__dataset_npz['y_test']).to(torch.float64),1)
        

        self.temperature_train = self._standard_scale(self.temperature_train.view(-1,121,1))
        self.temperature_test = self._standard_scale(self.temperature_test.view(-1,121,1))
        self.ground_truth = self._standard_scale(self.ground_truth.view(-1,121,1))
        self.y_train = self._standard_scale(self.y_train, dim=0)
        self.y_test = self._standard_scale(self.y_train, dim=0)
        # self.temperature_tensors = torch.from_numpy(self.temperature_data).float()
        # self.y_tensors = torch.from_numpy(self.y_data).float()

        # self.temperature_scaled = self._standard_scale(self.temperature_tensors)
        # self.y_scaled = self._standard_scale(self.y_tensors, dim=0)


    def __len__(self):
        if self.train:
            return len(self.temperature_train)
        else:
            return len(self.temperature_test)

    def __getitem__(self, index):
        if self.train:
            feature, label = self.temperature_train[index], self.y_train[index]
        else:
            feature, label = self.temperature_test[index], self.y_test[index]
        
        return feature, label
    
    
    def _standard_scale(self, tensor, eps=1e-9, dim=1):
        mean = torch.mean(tensor, dim=dim, keepdim=True)
        std = torch.std(tensor, dim=dim, keepdim=True)
        scaled_tensor = (tensor - mean) / (std+eps)
        return scaled_tensor
    
    
    def __repr__(self) -> str:
        intro_string = "Thermal Model Simulation Validation/Verification Dataset:"
        return f"{intro_string}\n--------\nTraining set contains {len(self.temperature_train)} simulated temperature fields.\ntesting set contains {len(self.temperature_test)} simulated temperature fields."
    
        

#usage 
# npz_file = 'thermal_model_data.npz'

# dataset = ThermalModelDataset(npz_file)

# print(dataset)

# print(f"shape of X_train: {dataset.temperature_train.shape}")
# print(f"shape of y_train: {dataset.y_train.shape}")

# print(f"shape of X_test: {dataset.temperature_test.shape}")
# print(f"shape of y_test: {dataset.y_test.shape}")
# print(f"shape of ground truth: {dataset.ground_truth.shape}")

# print(dataset.ground_truth[6]) #prints an arbitrary ground truth temperature field



