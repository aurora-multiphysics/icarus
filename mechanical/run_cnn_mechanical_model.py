
'''
================================================================================
Example: Train and Validate a CNN Model on Chosen Perturbed Mechanical Dataset 

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
'''

from icarus import MechanicalModelDataset, BaseMechanicalClassificationModel
import torch.utils.data.dataloader as data_utils
from model import MechanicalClassificationCNN 

num_classes = 3

# Load dataset
npz_files = ['example_2d_mechanical_data.npz', 'example_monoblock_mechanical_data.npz']

for npz_file in npz_files:
    dataset = MechanicalModelDataset(data_source=npz_file)
    
    train_loader = data_utils.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = MechanicalClassificationCNN(num_classes=num_classes)
    trainer = BaseMechanicalClassificationModel(model=model)
    
    # Train the model
    trainer.train(train_loader, num_epochs=5000)
    
    # Evaluate the model
    accuracy = trainer.evaluate(train_loader)  # Evaluate on the same loader for simplicity, replace with test_loader for real evaluation
