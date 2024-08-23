'''
================================================================================
Example: Run icarus pipeline end-to-end: from dataset generation to model training/eval

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
'''



from pathlib import Path
from icarus import SampleGenerator, ThermalModelDataset, IcarusModel, CreateThermalDataset
from mooseherder import MooseHerd, MooseRunner, InputModifier, DirectoryManager, MooseConfig
import torch.utils.data.dataloader as data_utils




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
                 'example_2d_plate_data.npz', 'example_2d_plate_out.e', 384.0, 500.0e3) #2d plate perturbed params

    # Run pipeline for thermal monoblock
    run_pipeline('example-moose-config.json', 'example_monoblock.i', 'example_monoblock_dataset', 
                 'example_thermal_monoblock_data.npz', 'example_monoblock_out.e', 384.0, 10.0e6) #monoblock perturbed params

if __name__ == '__main__':
    main()






