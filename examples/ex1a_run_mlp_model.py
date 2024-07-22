# from thermal_demo import SimpleModel, train_model, get_accuracy
# import torch.optim as optim
# import torch.nn as nn
# from thermal_demo import ThermalModelDataset, generate_thermal_data

# # Generate data (if not already generated)
# generate_thermal_data(n_samples=10000, output_file='thermal_model_data.npz')

# # Load the dataset
# train_dataset = ThermalModelDataset('thermal_model_data.npz', train=True)
# test_dataset = ThermalModelDataset('thermal_model_data.npz', train=False)

# # Initialize model and training parameters
# model = SimpleModel()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# # Train the model
# best_model = train_model(model, train_dataset, test_dataset, criterion, optimizer, num_epochs=50)



# test_accuracy = get_accuracy(best_model, test_dataset)
# print(f"Final Test Accuracy: {test_accuracy:.2f}%")