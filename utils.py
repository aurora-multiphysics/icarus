from pathlib import Path
import numpy as np
from mooseherder import ExodusReader
import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import re

def create_tensor(exodus_file):
    exodus_reader = ExodusReader(exodus_file)
    read_config = exodus_reader.get_read_config()
    read_config.node_vars = np.array([('temperature')])
    sim_data = exodus_reader.get_all_node_vars()
    temperature_tensor = np.array([value for value in sim_data.values()], dtype=np.float64)
    return temperature_tensor[:,:,1].T

BASE_DIR = Path.cwd()
OUTPUT_FILE = 'thermal_model_data.npz'
GROUND_TRUTH_TENSOR = create_tensor(Path(BASE_DIR, 'ground_truth_thermal_sim_out.e'))




def extract_number(filename):
    # Extract the number after the last dash and before _out.e
    match = re.search(r'sim-1-(\d+)_out\.e$', filename)
    if match:
        return int(match.group(1))
    return 0  # Return 0 if no match found

def preprocess_data(base_dir, output_file, test_size=0.2, random_state=42,):
    dataset_dir = Path(Path.cwd(), 'dataset/')
    working_dirs = [dir for dir in dataset_dir.iterdir() if dir.is_dir()]
    temperature_data = []
    y_data = []
    
    for working_dir in working_dirs:
        exodus_files = sorted(list(working_dir.glob('*.e')))
        sweep_vars_files= sorted(list(working_dir.glob('sweep-vars-*.json')))

    exodus_files_sorted = sorted(exodus_files, key=lambda x: extract_number(os.path.basename(x)))


    for exodus_file in exodus_files_sorted:
                temperature_tensor = create_tensor(exodus_file)
                temperature_data.append(GROUND_TRUTH_TENSOR - temperature_tensor)

    for sweep_vars_file in sweep_vars_files:
        y_data.extend(calculate_y_data(sweep_vars_file))

    temperature_data = np.array(temperature_data)
    max_abs_diff = np.max(np.abs(temperature_data))
    temperature_data_normalized = temperature_data / max_abs_diff
    y_data = np.array(y_data)


    # Split the data into train and test sets
    temperature_train, temperature_test, y_train, y_test = train_test_split(
        temperature_data, y_data, test_size=test_size, random_state=random_state, shuffle=False
    )


    np.savez(output_file, 
             temperature_train=temperature_train, 
             temperature_test=temperature_test,
             y_train=y_train, 
             y_test=y_test, 
             ground_truth=GROUND_TRUTH_TENSOR,
             max_abs_diff=max_abs_diff,
             ground_truth_thermal_c=384.0,
             ground_truth_surface_heat_flux=500e3)

# def create_tensor(exodus_file_path, scale_factor=1):
#     reader = ExodusReader(exodus_file_path)
#     coords, _ = reader.get_coords()
#     coords_2d = coords[:, :2]

#     temp_data = reader.get_node_vars(np.array(['temperature']))
#     if temp_data is None or 'temperature' not in temp_data:
#         raise ValueError("Temperature data not found in the exodus file")

#     temp_values = temp_data['temperature'][:, -1]

#     coord_temp_dict = {tuple(coord): temp for coord, temp in zip(coords_2d, temp_values)}

#     x = sorted(set(coord[0] for coord in coord_temp_dict.keys()))
#     y = sorted(set(coord[1] for coord in coord_temp_dict.keys()))

#     temp_grid = np.zeros((len(y), len(x)))
#     for (x_coord, y_coord), temp in coord_temp_dict.items():
#         i = y.index(y_coord)
#         j = x.index(x_coord)
#         temp_grid[i, j] = temp

#     # Create interpolation function
#     interp_func = RegularGridInterpolator((y, x), temp_grid)

#     # Create new coordinate arrays with higher resolution
#     x_new = np.linspace(min(x), max(x), len(x) * scale_factor)
#     y_new = np.linspace(min(y), max(y), len(y) * scale_factor)

#     # Create a meshgrid of new coordinates
#     xx_new, yy_new = np.meshgrid(x_new, y_new)
#     points_new = np.column_stack([yy_new.ravel(), xx_new.ravel()])

#     # Interpolate data
#     interpolated_data = interp_func(points_new).reshape(len(y_new), len(x_new))

#     return interpolated_data



def show_field(interpolated_data):
    plt.figure(figsize=(12, 5))
    plt.subplot(122)
    plt.imshow(interpolated_data, cmap='coolwarm', aspect='auto')
    plt.title("Interpolated Data")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# def calculate_y_data(perturbed_data_file):
#     ground_truth_thermal_c = 384.0
#     ground_truth_surface_heat_flux = 500e3
#     y_data = []

#     with open(perturbed_data_file, 'r') as file:
#         perturbed_data = json.load(file)

#     for data_list in perturbed_data:
#         data = data_list[0]  # Access the first (and only) dictionary in each inner list
#         thermal_cond_norm = (ground_truth_thermal_c - data["cuThermCond"])/(0.5*ground_truth_thermal_c)
#         heat_flux_norm = (ground_truth_surface_heat_flux - data["surfHeatFlux"])/(0.5*ground_truth_surface_heat_flux)
#         y_data.append([thermal_cond_norm, heat_flux_norm])

#     return y_data


def calculate_y_data(perturbed_data_file, thermal_cond_base=384.0, heat_flux_base=500000.0, tolerance=0.015):
    y_data = []

    with open(perturbed_data_file, 'r') as file:
        perturbed_data = json.load(file)

    for data_list in perturbed_data:
        data = data_list[0]  # Access the first (and only) dictionary in each inner list
        thermal_cond = data["cuThermCond"]
        heat_flux = data["surfHeatFlux"]

        # Check if parameters are perturbed based on tolerance
        tc_perturbed = abs(thermal_cond - thermal_cond_base) > thermal_cond_base * tolerance
        hf_perturbed = abs(heat_flux - heat_flux_base) > heat_flux_base * tolerance
        
        # Classify the sample
        if not tc_perturbed and not hf_perturbed:
            y_data.append(0)  # both unchanged
        elif not tc_perturbed and hf_perturbed:
            y_data.append(1)  # only heat flux perturbed
        elif tc_perturbed and not hf_perturbed:
            y_data.append(2)  # only thermal conductivity perturbed
        else:
            y_data.append(3)  # both perturbed

    return y_data

        
        # # Convert to binary (0 or 1)
        # thermal_c_changed = 1 if thermal_c_diff != 0 else 0
        # flux_changed = 1 if heat_flux_diff != 0 else 0
        
        # # Create class labels
        # if thermal_c_changed == 1 and flux_changed == 1:
        #     y_data.append(3)  # both_changed
        # elif thermal_c_changed == 1 and flux_changed == 0:
        #     y_data.append(2)  # only_thermal_c_changed
        # elif thermal_c_changed == 0 and flux_changed == 1:
        #     y_data.append(1)  # only_flux_changed
        # else:
        #     y_data.append(0)  # both_unchanged


preprocess_data(BASE_DIR, OUTPUT_FILE)

