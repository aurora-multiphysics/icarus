'''
================================================================================
Example: Generate 2D Plate Temperature Field Dataset.

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
'''



from pathlib import Path
from icarus import CreateThermalDataset


if __name__ == "__main__":
    base_dir = Path.cwd()
    output_file = 'example_2d_plate_data.npz'
    ground_truth_file = 'example_2d_plate_out.e'
    dataset_dir = Path(Path.cwd(), '2d_thermal_dataset/')
    thermal_cond = 384.0
    heat_flux = 500.0e3

    thermal_data = CreateThermalDataset(base_dir, output_file, ground_truth_file, dataset_dir, thermal_cond, heat_flux)
    thermal_data.preprocess_data()
