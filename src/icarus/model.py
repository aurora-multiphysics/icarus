import torch
import torch.nn as nn
import logging
from tqdm import tqdm   
import torch.nn.functional as F
import torch.optim as optim
import copy
from abc import ABC


logging.basicConfig(filename='model_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class BaseModel(ABC):
    def __init__(self, model, learning_rate=3e-4):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(torch.float64).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        logging.info(f"Initialised BaseModel with learning rate: {learning_rate}")
   

    def fit(self, train_loader, test_loader, num_epochs=50, debug=True):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        print("Model initialised. Begin training sequence...")

        for epoch in tqdm(range(num_epochs), desc="Model Training Progress"):
            train_loss = self._train_epoch(train_loader, debug)
            test_acc = self.evaluate(test_loader)
            
            logging.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_wts, 'best_model.pth')
                logging.info(f"New best model saved with accuracy: {best_acc:.2f}%")

            

        logging.info(f"Model training completed. Best test accuracy: {best_acc:.2f}%")
        print(f"Training and Eval Complete. best_accuracy: {best_acc:.2f}%. Please check model_training.log for additional info.")
        self.model.load_state_dict(best_model_wts)
        return best_acc

    def _train_epoch(self, loader, debug=True):
        self.model.train()
        epoch_loss = 0.0
        for i, (X_train, y_train) in enumerate(loader):
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device).long() 

            self.optimizer.zero_grad()
            y_preds = self.model(X_train)
            loss = self.criterion(y_preds, y_train)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0 and debug:
                self._debug_output(i, loss, y_preds, y_train)

        return epoch_loss / len(loader)

    def _debug_output(self, batch_index, loss, y_preds, y_train):
        logging.debug(f"Batch {batch_index} - Loss: {loss:.4f}")
        probabilities = F.softmax(y_preds, dim=1)
        num_samples_to_log = min(5, y_preds.shape[0])
        for j in range(num_samples_to_log):
            logging.debug(f"Sample {j}:")
            logging.debug(f"Probabilities: {probabilities[j].tolist()}")
            logging.debug(f"True class: {y_train[j].item()}")

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        for X, y_true in dataloader:
            X = X.to(self.device)
            y_true = y_true.to(self.device)
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y_true.size(0)
            correct += (predicted == y_true).sum().item()
        return 100 * correct / total



class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        logging.info(f"Initializing SimpleModel - input_size: {input_size}, hidden_size: {hidden_size}, output_size: {output_size}")
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

class IcarusModel(BaseModel):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=3e-4):
        logging.info(f"Initializing IcarusModel - input_size: {input_size}, hidden_size: {hidden_size}, output_size: {output_size}, learning_rate: {learning_rate}")
        model = SimpleModel(input_size, hidden_size, output_size)
        super().__init__(model, learning_rate)