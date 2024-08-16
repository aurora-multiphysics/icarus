
import torch
import torch.nn as nn
import logging
from tqdm import tqdm   
import torch.optim as optim
import copy
from abc import ABC

logging.basicConfig(filename='mechanical_model_training_and_eval.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class BaseMechanicalModel(ABC):
    def __init__(self, model, learning_rate=3e-4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.best_model = copy.deepcopy(self.model.state_dict())

    def train(self, train_loader, num_epochs=100):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in tqdm(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
        logging.info(f'Test Loss: {test_loss/len(test_loader)}')
        return test_loss / len(test_loader)

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))

    def update_best_model(self):
        self.best_model = copy.deepcopy(self.model.state_dict())

class MechanicalCNN(nn.Module):
    def __init__(self):
        super(MechanicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)  # Adjust depending on input size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
