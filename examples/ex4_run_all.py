from pathlib import Path
from icarus import SampleGenerator, ThermalModelDataset, IcarusModel
from mooseherder import MooseHerd, MooseRunner, InputModifier, DirectoryManager, MooseConfig, ExodusReader
import torch.utils.data.dataloader as data_utils
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

#####################################################################

def setup_moose_herd(config_path, input_file, dataset_dir, n_samples=20):
    moose_config = MooseConfig().read_config(config_path)
    moose_modifier = InputModifier(input_file, '#', '')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks=1, n_threads=4, redirect_out=True)
    dir_manager = DirectoryManager(n_dirs=1)
    herd = MooseHerd([moose_runner], [moose_modifier], dir_manager)
    herd.set_num_para_sims(n_para=10)
    
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
    
    dir_manager.set_base_dir(dataset_dir)
    dir_manager.clear_dirs()
    dir_manager.create_dirs()
    
    return herd

def generate_samples_and_run_moose(herd, dataset, thermal_cond, heat_flux):
    moose_vars = []
    thermal_cond_samples, surface_heat_flux_samples = dataset.generate_stratified_samples()
    for tc, hf in zip(thermal_cond_samples, surface_heat_flux_samples):
        moose_vars.append([{'cuThermCond': tc, 'surfHeatFlux': hf}])
    
    print('Herd sweep variables:')
    for vv in moose_vars:
        print(vv)
    
    print('Running MOOSE in parallel')
    herd.run_para(moose_vars)
    print(f'Run time (para) = {herd.get_sweep_time():.3f} seconds')

def create_dataset(base_dir, output_file, ground_truth_file, dataset_dir, thermal_cond, heat_flux):
    thermal_data = CreateThermalDataset(base_dir, output_file, ground_truth_file, dataset_dir, thermal_cond, heat_flux)
    thermal_data.preprocess_data()

def train_model(npz_file, batch_size=4, learning_rate=3e-4, hidden_size=512, output_size=4, num_epochs=20):
    thermal_dataset_train = ThermalModelDataset(npz_file)
    input_size = thermal_dataset_train.temperature_train[0].squeeze().numel()
    thermal_dataset_test = ThermalModelDataset(npz_file, train=False)
    train_loader = data_utils.DataLoader(dataset=thermal_dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(dataset=thermal_dataset_test, batch_size=batch_size, shuffle=False)

    model = IcarusModel(input_size, hidden_size, output_size, learning_rate)
    best_accuracy = model.fit(train_loader, test_loader, num_epochs=num_epochs, debug=True)
    test_accuracy = model.evaluate(test_loader)
    
    return best_accuracy, test_accuracy

def run_pipeline(config_name, input_name, dataset_name, output_name, ground_truth_name, thermal_cond, heat_flux):
    base_dir = Path.cwd()
    config_path = base_dir / config_name
    input_file = base_dir / input_name
    dataset_dir = base_dir / dataset_name
    output_file = output_name
    ground_truth_file = ground_truth_name

    print(f"Running pipeline for {dataset_name}")
    print("-" * 80)

    # Setup and run MOOSE simulations
    herd = setup_moose_herd(config_path, input_file, dataset_dir)
    dataset = SampleGenerator(n_samples=20, p_factor=0.8, tolerance=0.001, thermal_cond_base=thermal_cond, heat_flux_base=heat_flux)
    generate_samples_and_run_moose(herd, dataset, thermal_cond, heat_flux)

    # Create dataset
    create_dataset(base_dir, output_file, ground_truth_file, dataset_dir, thermal_cond, heat_flux)

    # Train model
    best_accuracy, test_accuracy = train_model(output_file)
    print(f"Best validation accuracy: {best_accuracy}")
    print(f"Test accuracy: {test_accuracy}")
    print("-" * 80)






def main():
    # Run pipeline for 2D thermal plate
    run_pipeline('example-moose-config.json', 'example_2d_plate.i', 'example_2d_thermal_dataset', 
                 'example_2d_plate_data.npz', 'example_2d_plate_out.e', 384.0, 500.0e3)

    # Run pipeline for thermal monoblock
    run_pipeline('example-moose-config.json', 'example_monoblock.i', 'example_monoblock_dataset', 
                 'example_thermal_monoblock_data.npz', 'example_monoblock_out.e', 384.0, 10.0e6)

if __name__ == '__main__':
    main()






