
'''
================================================================================
Example: Generate 2D Plate Mechanical Field Dataset.

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
'''

from pathlib import Path
from icarus import CreateMechanicalDataset

if __name__ == "__main__":
    base_dir = Path("mechanical_simulations_2d")
    output_file = 'example_2d_mechanical_data.npz'
    ground_truth_file = 'example_2d_plate_out.e'
    
    elasticity_modulus = 110e9
    poisson_ratio = 0.33

    dataset_creator = CreateMechanicalDataset(base_dir=base_dir,
                                              output_file=output_file,
                                              ground_truth_file=ground_truth_file,
                                              dataset_dir=base_dir,
                                              elasticity_modulus=elasticity_modulus,
                                              poisson_ratio=poisson_ratio)
    dataset_creator.create_dataset()
    dataset_creator.split_and_save_dataset()
    dataset_creator.generate_ground_truth()
