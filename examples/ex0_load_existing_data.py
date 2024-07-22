# from . import ThermalModelDataset


# npz_file = 'data/thermal_model_data.npz'

# dataset = ThermalModelDataset(npz_file)

# print(dataset)

# print(f"shape of X_train: {dataset.temperature_train.shape}")
# print(f"shape of y_train: {dataset.y_train.shape}")

# print(f"shape of X_test: {dataset.temperature_test.shape}")
# print(f"shape of y_test: {dataset.y_test.shape}")

# print(f"X_test: {dataset.temperature_test[0].shape}")
# print()
# print(f"y_train: {dataset.y_train}")



# classes = dataset.y_train

# print("First few classifications:")
# print(classes[:10])

# # Get class distribution
# class_counts = [0, 0, 0, 0]
# for label in classes:
#     class_counts[label] += 1

# print("\nClass distribution:")
# for i, count in enumerate(class_counts):
#     print(f"Class {i}: {count} samples")
