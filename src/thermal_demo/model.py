import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.optim as optim
from dataloader import ThermalModelDataset
import copy
'''
Dataset:

1 ground truth simulation

10,000 simulated temperature fields

4 output classes 

80/20 split 

8000 training samples 

4000 testing samples 

1 X_train = perturbed temperature field - ground truth field (elementwise subtraction)

y_data = 0: unchanged, 1: surfheatfluxchanged, 2:thermal_cond changed, 3: both changed

the perturbed temperature field enters the classification model and a prediction for the class is computed.

best model parameters saved in .pth file and used for inference

'''
#Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
INPUT_SIZE = 66
HIDDEN_SIZE = 512
OUTPUT_SIZE = 4
NUM_EPOCHS = 50
NUM_RUNS = 1 #best 'run' out of num_runs has model state_dict saved to be used for inference

device = 'cuda' if torch.cuda.is_available() else 'cpu'
npz_file = 'thermal_model_data.npz'


thermal_dataset_train = ThermalModelDataset(npz_file)
thermal_dataset_test = ThermalModelDataset(npz_file, train=False)


train_loader = data_utils.DataLoader(dataset=thermal_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data_utils.DataLoader(dataset=thermal_dataset_test, batch_size=BATCH_SIZE, shuffle=False )


class SimpleModel(nn.Module):
    def __init__(self, num_classes=OUTPUT_SIZE):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, num_classes)


    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x= self.fc4(x)
        return x

    
        
model = SimpleModel().to(torch.float64)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE)



def train_epoch(model, loader, optimiser, criterion, debug=True):
    model.train()
    epoch_loss = 0.0
    for i, (X_train, y_train) in enumerate(loader):
        X_train = X_train.to(device)
        y_train = y_train.to(device).long() 

        optimiser.zero_grad()
        y_preds = model(X_train)
        loss = criterion(y_preds, y_train)
        loss.backward()
        optimiser.step()

        epoch_loss += loss.item()

        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")
            
            # Convert predictions to probabilities
            probabilities = F.softmax(y_preds, dim=1)
            
            # Print probabilities and true labels for the first 5 samples in the batch
            if debug:
                num_samples_to_print = min(5, y_preds.shape[0])
                for j in range(num_samples_to_print):
                    print(f"  Sample {j}:")
                    print(f"    Probabilities: {probabilities[j].tolist()}")
                    print(f"    True class: {y_train[j].item()}")
                print()

    return epoch_loss / len(loader)


     
@torch.no_grad()
def get_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    for X, y_true in dataloader:
        X = X.to(device)
        y_true = y_true.to(device)
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        total += y_true.size(0)
        correct += (predicted == y_true).sum().item()
    return 100 * correct / total



def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, debug=True)
        print(f"Epoch {epoch+1} complete. Average loss: {train_loss:.4f}")

        # Calculate accuracy on test set
        test_acc = get_accuracy(model, test_loader)
        print(f"Test Accuracy: {test_acc:.2f}%")

        # If the model performs better, save it
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            # Save the best model
            torch.save(best_model_wts, 'best_model.pth')
            print(f"New best model saved with accuracy: {best_acc:.2f}%")

        print()

    print(f"Best test accuracy: {best_acc:.2f}%")
    return best_model_wts


best_model_state_dict = train_model(model, train_loader, test_loader, criterion, optimiser, NUM_EPOCHS, num_runs=NUM_RUNS)


