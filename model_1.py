from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim
from dataloader import ThermalModelDataset
import matplotlib.pyplot as plt
import math

'''
Dataset:

1 ground truth simulation

2500 perturbations per ground truth simulation

2500 simulated temperature fields 

80/20 split 

2000 training samples 

500 testing samples 

1 X_train = perturbed temperature field

the difference between the perturbed simulation variables and the ground truth variables is stored as the y_data.

the perturbed temperature field enters the regression model and a prediction for the residual is computed.

it's obviously wrong but that's okay, there are 2500 examples per ground truth simulation.
'''


BATCH_SIZE = 128
LEARNING_RATE = 3e-4
INPUT_SIZE = 66
HIDDEN_SIZE = 512
OUTPUT_SIZE = 2
NUM_EPOCHS = 100



device = 'cuda' if torch.cuda.is_available() else 'cpu'


npz_file = 'thermal_model_data.npz'

dataset = ThermalModelDataset(npz_file)

thermal_dataset_train = ThermalModelDataset(npz_file)
thermal_dataset_test = ThermalModelDataset(npz_file, train=False)

train_loader = data_utils.DataLoader(dataset=thermal_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data_utils.DataLoader(dataset=thermal_dataset_test, batch_size=BATCH_SIZE, shuffle=False )

single_example_dataset = data_utils.Subset(thermal_dataset_train, indices=list(range(20)))
single_example_loader = data_utils.DataLoader(single_example_dataset, batch_size=1, shuffle=False)

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, hidden_size)
        self.fc10 = nn.Linear(hidden_size, hidden_size)
        self.fc11 = nn.Linear(hidden_size, hidden_size)
        self.fc12 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = self.fc12(x)  
        return x.unsqueeze(1)
# ... (previous code remains the same)

model = RegressionModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(torch.float32)
model = model.to(device)
criterion = nn.MSELoss()
optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

train_losses = []



# print(thermal_dataset_x_reduced.shape)
# print(thermal_dataset_y_reduced.shape)
# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for i, (X_train, y_train) in enumerate(train_loader):
        X_train = X_train.to(device)  
        y_train = y_train.to(device) 

        optimiser.zero_grad()

        y_preds = model(X_train)
        loss = criterion(y_preds, y_train)
        loss.backward()
        optimiser.step()
        
        epoch_loss += loss.item()

        if (i) % 2 == 0:
            print(f"epoch [{epoch+1}/{NUM_EPOCHS}] : iter [{i+1}/{len(train_loader)}] || loss : {loss}")
        
    # Calculate average loss for the epoch
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    
print()
print(f"Dummy input test:\n") 
dummy_x = dataset.temperature_train[0].T
dummy_y = dataset.y_train[0]

y_preds = model(dummy_x)

print(f"y-preds: {y_preds}")
print()
print(f"y_true: {dummy_y}")
       

# print("=====================================================================")

# val_losses =[]
# model.eval()
# with torch.no_grad():
#     for epoch in range(NUM_EPOCHS):
#         for i, (X_val, y_val) in enumerate(test_loader):
#             X_val = X_val.to(device)
#             y_val = y_val.to(device)

#             y_test_preds = model(X_val)
#             loss = criterion(y_test_preds, y_val)
#             val_losses.append(loss.item())
#             if (i + 1) % 10 == 0:
#                 print(
#                     f"epoch [{epoch+1}/{NUM_EPOCHS}] : iter [{i+1}/{len(test_loader)}] || loss : {loss}"
#                 )