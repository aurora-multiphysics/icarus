from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim
from dataloader import ThermalModelDataset
import matplotlib.pyplot as plt

WORKING_PATH = Path(Path.cwd(), 'mooseherder_examples/sim-workdir-1')
GROUND_TRUTH_PATH = Path(Path.cwd(), 'thermal_model_out.e')  # Assuming the ground truth file is in this directory
SWEEP_VARS_PATH = Path(WORKING_PATH, 'sweep-vars-1.json')


NUM_EPOCHS = 100
learning_rate = 3e-4
HIDDEN_SIZE = 256
OUTPUT_SIZE = 2
BATCH_SIZE = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

thermal_model_train = ThermalModelDataset(WORKING_PATH, GROUND_TRUTH_PATH, SWEEP_VARS_PATH)
train_loader = data_utils.DataLoader(dataset=thermal_model_train, batch_size=BATCH_SIZE, shuffle=True)


class Model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(242, HIDDEN_SIZE, dtype=torch.float64)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float64)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, dtype = torch.float64)
        self.fc4 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE, dtype = torch.float64)

    def forward(self, X):
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)

        return X

    

class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 15 * 1, 128)
        self.fc2 = nn.Linear(128, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)


    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.maxpool1(X)
        X = F.relu(self.bn2(self.conv2(X)))
        X = self.maxpool2(X)
        X = F.relu(self.bn3(self.conv3(X)))
        X = self.maxpool3(X)
        X = torch.flatten(X, 1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)

        return X
    
convnet = Model1()
convnet.to(device=device)


mlp = Model0()
mlp.to(device)


criterion = nn.MSELoss()
optimiser = optim.AdamW(mlp.parameters(), lr=learning_rate)


training_losses = []


for epoch in range(NUM_EPOCHS):
    for i, (X_train, y_train) in enumerate(train_loader):
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        y_train = y_train.squeeze()



        optimiser.zero_grad()
        y_preds = mlp(X_train)
        loss = criterion(y_preds, y_train)
        training_losses.append(loss.item())
        
    
        
        loss.backward()
        optimiser.step()
        print(f"loss: {loss.item()}")
    
    
    # print(X_train.shape)
    # print(y_preds.shape)
    # print(y_train.shape)



plt.plot(training_losses)
plt.show()




