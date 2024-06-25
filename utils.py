from pathlib import Path
import numpy as np
from mooseherder import ExodusReader
import json
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(base_dir, output_file, test_size=0.2, random_state=42,):
    dataset_dir = base_dir / 'dataset'
    working_dirs = [dir for dir in dataset_dir.iterdir() if dir.is_dir()]
    temperature_data = []
    y_data = []
    exodus_files_all= []
    sweep_vars_all = []


    for working_dir in working_dirs:
        exodus_files = sorted(list(working_dir.glob('*.e')))
        sweep_vars_files = sorted(list(working_dir.glob('sweep-vars-*.json')))
        exodus_files_all.append(exodus_files)
        sweep_vars_all.append(sweep_vars_files)

        if not sweep_vars_files:
            print(f"Warning: No sweep-vars-*.json files found in {working_dir}")
            continue
        
   
        for exodus_file in exodus_files:
            temperature_tensor = create_tensor(exodus_file)
            temperature_data.append(temperature_tensor)

        for sweep_vars_file in sweep_vars_files:
            y_data.extend(calculate_y_data(sweep_vars_file))

    temperature_data = np.stack(temperature_data)
    y_data = np.array(y_data)


    # Split the data into train and test sets
    temperature_train, temperature_test, y_train, y_test = train_test_split(
        temperature_data, y_data, test_size=test_size, random_state=random_state, shuffle=False
    )

    np.savez(output_file, temperature_train=temperature_train, temperature_test=temperature_test,
             y_train=y_train, y_test=y_test)


def create_tensor(exodus_file):
    exodus_reader = ExodusReader(exodus_file)
    read_config = exodus_reader.get_read_config()
    read_config.node_vars = np.array([('temperature')])
    sim_data = exodus_reader.get_all_node_vars()
    temperature_tensor = np.array([value for value in sim_data.values()], dtype=np.float64)
    return temperature_tensor[:,:,1].T


def calculate_y_data(perturbed_data_file):
    ground_truth_shc = 406.0
    ground_truth_surface_heat_flux = 500e3
    y_data = []

    with open(perturbed_data_file, 'r') as file:
        perturbed_data = json.load(file)

    for data_list in perturbed_data:
        data = data_list[0]  # Access the first (and only) dictionary in each inner list
        shc_diff = ground_truth_shc - data["cuSpecHeat"]
        heat_flux_diff = ground_truth_surface_heat_flux - data["surfHeatFlux"]
        y_data.append([shc_diff, heat_flux_diff])

    return y_data


BASE_DIR = Path.cwd()
OUTPUT_FILE = 'thermal_model_data.npz'

preprocess_data(BASE_DIR, OUTPUT_FILE)


# # dummy = create_tensor(Path(Path.cwd(), 'ground_truth_thermal_sim_out.e'))
# dummy1 = create_tensor(Path(Path.cwd(), 'dataset/sim-workdir-1/sim-1-1_out.e'))

# # print(dummy)
# # print()
# print(dummy1)

# # dummy_y = calculate_y_data(Path(BASE_DIR,'dataset/sim-workdir-1/sweep-vars-1.json'))

# # dummy_y = np.array(dummy_y)

# # print(f"dummy y shape: {dummy_y}")

