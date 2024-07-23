'''
================================================================================
Example: Train and Validate an ML Model on Perturbed Dataset 

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
'''


from thermal_demo import ThermalModelDataset, IcarusModel
import torch
import torch.utils.data.dataloader as data_utils


#Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
INPUT_SIZE = 66
HIDDEN_SIZE = 512
OUTPUT_SIZE = 4
NUM_EPOCHS = 20
# NUM_RUNS = 1 #best 'run' out of num_runs has model state_dict saved to be used for inference

device = 'cuda' if torch.cuda.is_available() else 'cpu'
npz_file = 'data/thermal_model_data.npz' #this is an existing thermal_toy dataset to experiment training the model on


thermal_dataset_train = ThermalModelDataset(npz_file) #training set
thermal_dataset_test = ThermalModelDataset(npz_file, train=False) #validation set

train_loader = data_utils.DataLoader(dataset=thermal_dataset_train, batch_size=BATCH_SIZE, shuffle=True) #dataloader to load the training data in batches
test_loader = data_utils.DataLoader(dataset=thermal_dataset_test, batch_size=BATCH_SIZE, shuffle=False ) #dataloader to load the validation data in batches


model = IcarusModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE)

best_accuracy = model.fit(train_loader, test_loader, num_epochs=NUM_EPOCHS, debug=True) #prints class probabilties for further insight on model performance. can set to false for less verbose output

test_accuracy = model.evaluate(test_loader)

