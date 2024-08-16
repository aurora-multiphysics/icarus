
import torch
import torch.nn as nn
import logging
from tqdm import tqdm   
import torch.nn.functional as F
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
