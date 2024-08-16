
'''
================================================================================
Example: Train and Validate a CNN Model on Chosen Perturbed Mechanical Dataset 

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
'''

from icarus import MechanicalModelDataset, BaseMechanicalModel
import torch.utils.data.dataloader as data_utils
from mechanical_cnn_model import MechanicalCNN  # Import the new CNN model

npz_files = ['example_2d_mechanical_data.npz', 'example_monoblock_mechanical_data.npz']

for npz_file in npz_files:
    dataset = MechanicalModelDataset(data_source=npz_file)
    
    train_loader = data_utils.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = BaseMechanicalModel(model=MechanicalCNN())  # Use the CNN model here
    model.train(train_loader)
    model.evaluate(train_loader)  # Use appropriate test data in real cases
