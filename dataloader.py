import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import create_tensor, calculate_y_data
from pathlib import Path
from utils import show_field,create_tensor
from pathlib import Path

class ThermalModelDataset(Dataset):
    def __init__(self, npz_file, train=True):
        self.train = train
        self.__dataset_npz = np.load(npz_file, allow_pickle=True)
        self.temperature_train = torch.from_numpy(self.__dataset_npz['temperature_train']).to(torch.float64)
        self.temperature_test = torch.from_numpy(self.__dataset_npz['temperature_test']).to(torch.float64)
        self.ground_truth = torch.from_numpy(self.__dataset_npz['ground_truth']).to(torch.float64)
        
        self.y_train = torch.from_numpy(self.__dataset_npz['y_train']).long()
        self.y_test = torch.from_numpy(self.__dataset_npz['y_test']).long()

        self.category = {0:'unperturbed', 1: 'heat_flux perturbed', 2:'thermal_cond_perturbed', 3:'both perturbed'}
        
        # self.temperature_train = F.normalize(self.temperature_train, dim=0)


        # self.temperature_test = F.normalize(self.temperature_test, dim=0)

        # self.y_train = self.y_train.view(-1,4,1)
        # self.y_test = self.y_test.view(-1,4,1)


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
    

    def standard_scale(self, tensor):
        # tensor = tensor.to(torch.float32)
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True)
        b = (tensor - mean) / std
        return b
    
    
    def __repr__(self) -> str:
        intro_string = "Thermal Model Simulation Validation/Verification Dataset:"
        return f"{intro_string}\n--------\nTraining set contains {len(self.temperature_train)} simulated temperature fields.\ntesting set contains {len(self.temperature_test)} simulated temperature fields.\n------------"
    
        

#usage 
npz_file = 'thermal_model_data.npz'

dataset = ThermalModelDataset(npz_file)

print(dataset)

print(f"shape of X_train: {dataset.temperature_train.shape}")
print(f"shape of y_train: {dataset.y_train.shape}")

print(f"shape of X_test: {dataset.temperature_test.shape}")
print(f"shape of y_test: {dataset.y_test.shape}")

print(f"ground_truth_temp_field: {dataset.ground_truth}")

# print(dataset.temperature_train[6]) #prints an arbitrary temperature field





# def show_difference(ground_truth_tensor, perturbed_tensor):
#     return perturbed_tensor - ground_truth_tensor

import json



with open(Path(Path.cwd(), 'dataset/sim-workdir-1/sweep-vars-1.json'), 'r') as file:
    data = json.load(file)

only_heat_flux_changed_ids = []
only_thermal_c_changed_ids = []
both_changed_ids = []
neither_changed_ids = []

for index, item in enumerate(data):
    sweep_vars = item[0]  # list of lsits containing dictionaries
    if sweep_vars['cuThermCond'] == 384.0 and sweep_vars['surfHeatFlux'] != 500000.0:
        only_heat_flux_changed_ids.append(index)
    if sweep_vars['surfHeatFlux'] == 500000.0 and sweep_vars['cuThermCond'] != 384.0:
        only_thermal_c_changed_ids.append(index)
    if sweep_vars['surfHeatFlux'] == 500000.0 and sweep_vars['cuThermCond'] == 384.0:
        both_changed_ids.append(index)
    if sweep_vars['surfHeatFlux'] != 500000.0 and sweep_vars['cuThermCond'] != 384.0:
        neither_changed_ids.append(index)
    

print(f"number of only heat flux changed ids: {len(only_heat_flux_changed_ids)}")
print(f"number of only thermal c changed ids: {len(only_thermal_c_changed_ids)}")
print(f"number of both_changed ids: {len(both_changed_ids)}")
print(f"number of neither changed ids: {len(neither_changed_ids)}")
# print("Indices where cuSpecHeat is 406.0:", shc_ids)
# print("Indices where surfHeatFlux is 500000.0:", heat_flux_ids)


# print(only_heat_flux_changed_ids)
# print()
# print(only_thermal_c_changed_ids)

# count = 0

# heat_flux_train_ids = [i for i in only_heat_flux_changed_ids if i < 8000]
# thermal_c_train_ids = [i for i in only_thermal_c_changed_ids if i < 8000]

# for x in heat_flux_train_ids:
#     for y in thermal_c_train_ids:
#         if torch.equal(dataset.temperature_train[x], dataset.temperature_train[y]):
#             # print(f"index {x} == index {y}")
#             count+=1
# print(f"number of equal tensors: {count}")




# ground_field = torch.tensor(create_tensor(Path(Path.cwd(), 'ground_truth_thermal_sim_out.e')))

# example_1 = dataset.temperature_train[1] + ground_field
# example_2 = torch.tensor(create_tensor(Path(Path.cwd(), 'dataset/sim-workdir-1/sim-1-2_out.e')))

# torch.max(example_2)

# def find_similar_peak_temps(dataset, ground_field, heat_flux_train_ids, thermal_c_train_ids, temp_tolerance=0.00179):
#     similar_pairs = []
#     min_temps = []
#     for x in heat_flux_train_ids:
#         for y in thermal_c_train_ids:
#             temp_x = dataset.temperature_train[x] + ground_field
#             temp_y = dataset.temperature_train[y] + ground_field
            
#             max_temp_x = torch.max(temp_x)
#             max_temp_y = torch.max(temp_y)
            
#             # Check if peak temperatures are similar within the tolerance
#             if abs(max_temp_x - max_temp_y) / max_temp_x <= temp_tolerance:
#                 min_temp_x = torch.min(temp_x)
#                 min_temp_y = torch.min(temp_y)

#                 similar_pairs.append((x,y,max_temp_x, max_temp_y, min_temp_x, min_temp_y))
#                 min_temps.append((min_temp_x.item(), min_temp_y.item()))
    
#     return similar_pairs

# Use the function
# similar_pairs = find_similar_peak_temps(dataset, ground_field, heat_flux_train_ids, thermal_c_train_ids)

# # Print results
# for x, y, max_temp_x, max_temp_y, min_temp_x, min_temp_y in similar_pairs:
#     print(f"Similar pair found: index {x} and {y}")
#     print(f"  Peak temperatures: {max_temp_x}||{max_temp_y}")
#     print(f"  Minimum temperatures: {min_temp_x} || {min_temp_y}")

# print(f"Total similar pairs found: {len(similar_pairs)}")

# example_x = dataset.temperature_train[116] + ground_field  #increased surfheatflux
# example_y = dataset.temperature_train[64] + ground_field #decreased thermal_cond


# print(f"max_temp when only surfheatflux increased: {torch.max(example_x)}, min temp: {torch.min(example_x)}")

# print(f"max_temp when only thermal_cond decreased: {torch.max(example_y)}, min temp: {torch.min(example_y)}")

#825000
#230

#24.000
#26.6000

# dataset