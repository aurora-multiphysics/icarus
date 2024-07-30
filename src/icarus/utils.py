from pathlib import Path
import numpy as np
from mooseherder import ExodusReader
import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import re


class CreateThermalDataset:
    def __init__(self, base_dir, output_file, ground_truth_file, dataset_dir, thermal_cond, heat_flux):
        self.base_dir = Path(base_dir)
        self.output_file = output_file
        self.dataset_dir = Path(dataset_dir)
        self.ground_truth = self.create_tensor(Path(ground_truth_file))
        self.thermal_cond_base = thermal_cond
        self.heat_flux_base = heat_flux
         
    def create_tensor(self, exodus_file):
        exodus_reader = ExodusReader(exodus_file)
        read_config = exodus_reader.get_read_config()
        read_config.node_vars = np.array([('temperature')])
        sim_data = exodus_reader.get_all_node_vars()
        temperature_tensor = np.array([value for value in sim_data.values()], dtype=np.float64)
        return temperature_tensor[:,:,1].T

    def extract_number(self, filename):
        match = re.search(r'sim-1-(\d+)_out\.e$', filename)
        return int(match.group(1)) if match else 0

    def preprocess_data(self, test_size=0.2, random_state=42):
        working_dirs = [dir for dir in self.dataset_dir.iterdir() if dir.is_dir()]
        temperature_data = []
        y_data = []
        
        for working_dir in working_dirs:
            exodus_files = sorted(list(working_dir.glob('*.e')))
            sweep_vars_files = sorted(list(working_dir.glob('sweep-vars-*.json')))

            exodus_files_sorted = sorted(exodus_files, key=lambda x: self.extract_number(os.path.basename(x)))

            for exodus_file in exodus_files_sorted:
                temperature_tensor = self.create_tensor(exodus_file)
                temperature_data.append(self.ground_truth - temperature_tensor)

            for sweep_vars_file in sweep_vars_files:
                y_data.extend(self.calculate_y_data(sweep_vars_file))

        temperature_data = np.array(temperature_data)
        max_abs_diff = np.max(np.abs(temperature_data))
        temperature_data_normalized = temperature_data / max_abs_diff
        y_data = np.array(y_data)

        temperature_train, temperature_test, y_train, y_test = train_test_split(
            temperature_data, y_data, test_size=test_size, random_state=random_state, shuffle=False
        )

        np.savez(self.output_file, 
                temperature_train=temperature_train, 
                temperature_test=temperature_test,
                y_train=y_train, 
                y_test=y_test, 
                ground_truth=self.ground_truth,
                max_abs_diff=max_abs_diff,
               )

    def show_field(self, interpolated_data):
        plt.figure(figsize=(12, 5))
        plt.subplot(122)
        plt.imshow(interpolated_data, cmap='coolwarm', aspect='auto')
        plt.title("Interpolated Data")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def calculate_y_data(self, perturbed_data_file, tolerance=0.001):
        y_data = []

        with open(perturbed_data_file, 'r') as file:
            perturbed_data = json.load(file)

        for data_list in perturbed_data:
            data = data_list[0]
            thermal_cond = data["cuThermCond"]
            heat_flux = data["surfHeatFlux"]

            tc_perturbed = abs(thermal_cond - self.thermal_cond_base) > self.thermal_cond_base * tolerance
            hf_perturbed = abs(heat_flux - self.heat_flux_base) > self.heat_flux_base * tolerance
            
            if not tc_perturbed and not hf_perturbed:
                y_data.append(0)
            elif not tc_perturbed and hf_perturbed:
                y_data.append(1)
            elif tc_perturbed and not hf_perturbed:
                y_data.append(2)
            else:
                y_data.append(3)

        return y_data

