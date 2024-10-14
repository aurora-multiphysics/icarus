import numpy as np
import torch
from torch.utils.data import Dataset


class ThermalModelDataset(Dataset):
    def __init__(self, data_source, train=True):
        self.train = train

        if isinstance(data_source, str):
            # If data_source is a string, treat it as a file path
            self.__dataset_npz = np.load(data_source, allow_pickle=True)
        elif isinstance(data_source, dict):
            # If data_source is a dict, assume it's already loaded data
            self.__dataset_npz = data_source
        else:
            raise ValueError(
                "data_source must be either a file path or a pre-loaded dataset dictionary"
            )

        self.temperature_train = torch.from_numpy(
            self.__dataset_npz["temperature_train"]
        ).to(torch.float64)
        self.temperature_test = torch.from_numpy(
            self.__dataset_npz["temperature_test"]
        ).to(torch.float64)
        self.ground_truth = torch.from_numpy(self.__dataset_npz["ground_truth"]).to(
            torch.float64
        )

        self.y_train = torch.from_numpy(self.__dataset_npz["y_train"]).long()
        self.y_test = torch.from_numpy(self.__dataset_npz["y_test"]).long()

        self.category = {
            0: "unperturbed",
            1: "heat_flux perturbed",
            2: "thermal_cond_perturbed",
            3: "both perturbed",
        }

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

    def __repr__(self) -> str:
        return f"""Thermal Model Simulation Validation/Verification Dataset:
    --------
    Training set: {len(self.temperature_train)} simulated temperature fields
    Testing set:  {len(self.temperature_test)} simulated temperature fields
    Input X data shape:           {self.temperature_train[0].shape}
    Input Y data shape:           {self.y_train[0].shape}
    Output classes:        4
    --------"""
