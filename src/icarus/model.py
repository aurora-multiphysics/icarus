import logging
import copy
from abc import ABC
from typing import List, Tuple
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from torch.utils.data import DataLoader
from icarus import ThermalModelDataset


@dataclass
class NNParameters:
    batch_size: int = 4
    learning_rate: float = 3e-4
    input_size: int = 0  # Will be set based on the dataset
    hidden_size: int = 512
    output_size: int = 4
    num_epochs: int = 50
    debug: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


logging.basicConfig(
    filename="model_training_and_eval.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class BaseModel(ABC):
    def __init__(self, model: nn.Module, params: NNParameters):
        self.params = params
        self.model = model.to(torch.float64).to(self.params.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.params.learning_rate
        )
        logging.info(f"Initialised BaseModel with parameters: {self.params}")

    def train(self, train_loader: DataLoader, test_loader: DataLoader) -> float:
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        print("Model initialised. Begin training sequence...")

        for epoch in tqdm(
            range(self.params.num_epochs), desc="Model Training Progress"
        ):
            train_loss, softmax_outputs = self._train_epoch(train_loader)
            test_acc = self.evaluate(test_loader)

            log_message = f"Epoch {epoch+1}/{self.params.num_epochs} - Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%"
            if self.params.debug and softmax_outputs:
                log_message += f"\nSoftmax outputs: {softmax_outputs[-1]}"  # Only log the last batch
            logging.info(log_message)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_wts, "best_model.pth")
                logging.info(f"New best model saved with accuracy: {best_acc:.2f}%")

        logging.info(f"Model training completed. Best test accuracy: {best_acc:.2f}%")
        print(
            f"Training and Eval Complete. best_accuracy: {best_acc:.2f}%. Please check model_training.log for additional info."
        )
        self.model.load_state_dict(best_model_wts)
        return best_acc

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, List[dict]]:
        self.model.train()
        epoch_loss = 0.0
        softmax_outputs = []
        for batch_idx, (X_train, y_train) in enumerate(loader):
            X_train = X_train.to(self.params.device)
            y_train = y_train.to(self.params.device)

            self.optimizer.zero_grad()
            y_preds = self.model(X_train)
            loss = self.criterion(y_preds, y_train)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0 and self.params.debug:
                softmax_outputs.append(self._debug_output(y_preds, y_train))

        return epoch_loss / len(loader), softmax_outputs

    def _debug_output(self, y_preds: torch.Tensor, y_train: torch.Tensor) -> List[dict]:
        probabilities = F.softmax(y_preds, dim=1)
        num_samples_to_log = min(5, y_preds.shape[0])
        return [
            {
                "Sample": j,
                "Probabilities": probabilities[j].tolist(),
                "True class": y_train[j].item(),
            }
            for j in range(num_samples_to_log)
        ]

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for X, y_true in dataloader:
            X = X.to(self.params.device)
            y_true = y_true.to(self.params.device)
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y_true.size(0)
            correct += (predicted == y_true).sum().item()
        return 100 * correct / total


class SimpleModel(nn.Module):
    def __init__(self, params: NNParameters):
        super(SimpleModel, self).__init__()
        logging.info(f"Initializing SimpleModel with parameters: {params}")
        self.fc1 = nn.Linear(params.input_size, params.hidden_size)
        self.fc2 = nn.Linear(params.hidden_size, params.hidden_size)
        self.fc3 = nn.Linear(params.hidden_size, params.hidden_size)
        self.fc4 = nn.Linear(params.hidden_size, params.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class IcarusModel(BaseModel):
    def __init__(self, params: NNParameters):
        model = SimpleModel(params)
        super().__init__(model, params)
        logging.info(f"Initialized IcarusModel with parameters: {params}")


def main(npz_file: str) -> Tuple[float, float]:
    dataset = ThermalModelDataset(npz_file)
    params = NNParameters(input_size=dataset.temperature_train[0].squeeze().numel())

    thermal_dataset_train = ThermalModelDataset(npz_file)
    thermal_dataset_test = ThermalModelDataset(npz_file, train=False)
    train_loader = DataLoader(
        dataset=thermal_dataset_train, batch_size=params.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=thermal_dataset_test, batch_size=params.batch_size, shuffle=False
    )

    model = IcarusModel(params)
    best_accuracy = model.train(train_loader, test_loader)
    test_accuracy = model.evaluate(test_loader)

    return best_accuracy, test_accuracy


if __name__ == "__main__":
    npz_files = ["example_2d_plate_data.npz", "example_thermal_monoblock_data.npz"]
    best_acc, test_acc = main(npz_files[0])
    print(f"Best accuracy: {best_acc:.2f}%, Final test accuracy: {test_acc:.2f}%")
