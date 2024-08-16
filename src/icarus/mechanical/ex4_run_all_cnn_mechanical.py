
'''
================================================================================
Example: Run icarus pipeline end-to-end: from dataset generation to model training/eval (Mechanical)

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
'''

from pathlib import Path
from icarus import MechanicalSampleGenerator, MechanicalModelDataset, BaseMechanicalModel, CreateMechanicalDataset
import torch.utils.data.dataloader as data_utils
from mechanical_cnn_model import MechanicalCNN  # Import the new CNN model

# Generate 2D mechanical simulations
base_input_file = Path("example_2d_plate.i")
output_dir = Path("mechanical_simulations_2d")
elasticity_modulus_range = (90e9, 120e9)
poisson_ratio_range = (0.28, 0.35)
pressure_range = (1e6, 10e6)

generator = MechanicalSampleGenerator(base_input_file=base_input_file,
                                      output_dir=output_dir,
                                      elasticity_modulus_range=elasticity_modulus_range,
                                      poisson_ratio_range=poisson_ratio_range,
                                      pressure_range=pressure_range)
generator.generate_samples()

# Generate dataset
base_dir = Path("mechanical_simulations_2d")
output_file = 'example_2d_mechanical_data.npz'
ground_truth_file = 'example_2d_plate_out.e'

dataset_creator = CreateMechanicalDataset(base_dir=base_dir,
                                          output_file=output_file,
                                          ground_truth_file=ground_truth_file,
                                          dataset_dir=base_dir,
                                          elasticity_modulus=110e9,
                                          poisson_ratio=0.33)
dataset_creator.create_dataset()
dataset_creator.split_and_save_dataset()
dataset_creator.generate_ground_truth()

# Train and evaluate model
dataset = MechanicalModelDataset(data_source=output_file)
train_loader = data_utils.DataLoader(dataset, batch_size=32, shuffle=True)

model = BaseMechanicalModel(model=MechanicalCNN())  # Use the CNN model here
model.train(train_loader)
model.evaluate(train_loader)  # Use appropriate test data in real cases
