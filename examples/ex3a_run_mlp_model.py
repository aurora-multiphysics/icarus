'''
================================================================================
Example: Train and Validate an ML Model on Chosen Perturbed Dataset 

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
'''

from icarus import ThermalModelDataset, IcarusModel
import torch.utils.data.dataloader as data_utils

npz_files = ['example_2d_plate_data.npz','example_thermal_monoblock_data.npz']

FILE = npz_files[0]
dataset= ThermalModelDataset(FILE)


#Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
INPUT_SIZE = dataset.temperature_train[0].squeeze().numel()
HIDDEN_SIZE = 512
OUTPUT_SIZE = 4
NUM_EPOCHS = 20
# NUM_RUNS = 1 #best 'run' out of num_runs has model state_dict saved to be used for inference

def main(npz_file):

    thermal_dataset_train = ThermalModelDataset(npz_file) #training set
    thermal_dataset_test = ThermalModelDataset(npz_file, train=False) #validation set
    train_loader = data_utils.DataLoader(dataset=thermal_dataset_train, batch_size=BATCH_SIZE, shuffle=True) #dataloader to load the training data in batches
    test_loader = data_utils.DataLoader(dataset=thermal_dataset_test, batch_size=BATCH_SIZE, shuffle=False ) #dataloader to load the validation data in batches

    model = IcarusModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE)
    best_accuracy = model.fit(train_loader, test_loader, num_epochs=NUM_EPOCHS, debug=True) #prints class probabilties for further insight on model performance. can set to false for less verbose output
    test_accuracy = model.evaluate(test_loader)
   
    return best_accuracy, test_accuracy


if __name__ == '__main__':
    main(npz_file=FILE) #trains on 2d thermal plate data

