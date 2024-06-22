from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim
from dataloader import ThermalModelDataset
import matplotlib.pyplot as plt

'''
Dataset:

10 ground truth simulations

2500 perturbations per ground truth simulation

25,000 simulated temperature fields 

80/20 split 

20 0000 training samples 

5 000 testing samples 

1 X_train = perturbed temperature field

this field is then compared to the most similar field in the ground truth simulation data 

this is done either simply by finding the cosine similarity, or alternatively can be learned by a classifier 

after finding the most similar simulation, the simulation's variables are extracted. the X_data's variables are also extracted.

the difference between the perturbed simulation variables and the ground truth variables is computed and stored as the y_data.

the perturbed temperature field enters the regression model and a prediction for the residual is computed.

it's obviously wrong but that's okay, there are 2500 examples per ground truth simulation.
'''


BATCH_SIZE = 32
LEARNING_RATE = 3e-4
INPUT_SIZE = 121
HIDDEN_SIZE = 256
OUTPUT_SIZE = 2
NUM_EPOCHS = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

npz_file = 'thermal_model_data.npz'

thermal_dataset_train = ThermalModelDataset(npz_file)
thermal_dataset_test = ThermalModelDataset(npz_file, train=False)



train_loader = data_utils.DataLoader(dataset=thermal_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data_utils.DataLoader(dataset=thermal_dataset_test, batch_size=BATCH_SIZE, shuffle=False )



class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ... (previous code remains the same)

model = RegressionModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(torch.float64)
model = model.to(device)
criterion = nn.MSELoss()
optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(3):
    for i, (X_train, y_train) in enumerate(train_loader):
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        # X_train = X_train.view(X_train.size(0), -1)  # Flatten the input tensor

        optimiser.zero_grad()

        y_preds = model(X_train)
        loss = criterion(y_preds, y_train)
        loss.backward()
        optimiser.step()

        # if i % 25 == 0:
        #     print(f"loss at iteration {i} || {loss.item()}")


y_pred_dummy = model(X_train)
y_true = y_train

print(y_pred_dummy)
print()
print(y_true)
print(f"loss: {criterion(y_pred_dummy, y_true)}")














