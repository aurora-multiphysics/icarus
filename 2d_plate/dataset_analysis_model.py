# Extract data points from generated datasets + ground truth
# Generate labelled dataset - formatted | T1 | T2 | ... | TN | label |
# where N is the number of sampled data points (6), and label is different for each perturbed parameter
# eg ground truth label = 0, perturbed xmax label = 1, perturbed ymax label = 2, etc.

import mooseherder as mh 
from pathlib import Path
import numpy as np
import pyvale
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def generate_labelled_dataset(folder_path):
    labelled_dataset = np.empty((0, 7))
    parameters = ["ground_truth", "xmax", "ymax", "init_temp", "max_temp", "thermal_conductivity", "specific_heat", "prop_values"]

    for file_path in folder_path.rglob('*.e'):
        for i in range(len(parameters)):
            if parameters[i] in str(file_path):
                label = i
        sim_data = mh.ExodusReader(file_path).read_all_sim_data()
        field_key = "temperature"

        sim_data.coords = sim_data.coords*1000.0 # type: ignore

        xmax = np.max(sim_data.coords[:, 0])
        ymax = np.max(sim_data.coords[:, 1])

        n_sens = (3,2,1)
        x_lims = (0.0,xmax)
        y_lims = (0.0,ymax)
        z_lims = (0.0,0.0)
        sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
        sens_data = pyvale.SensorData(positions=sens_pos)

        tc_array = pyvale.SensorArrayFactory \
            .thermocouples_no_errs(sim_data,
                                        sens_data,
                                        field_key,
                                        spat_dims=2)

        measurements = tc_array.get_measurements()[:, 0, 1] 

        measurements = np.append(measurements, label) # add label to measurements 
        labelled_dataset = np.vstack([labelled_dataset, measurements]) # add to labelled dataset 

    return labelled_dataset
    
training_folder_path = Path('2d_plate/perturbed_datasets')
training_dataset = generate_labelled_dataset(training_folder_path)
validation_folder_path = Path('2d_plate/validation_datasets')
validation_dataset = generate_labelled_dataset(validation_folder_path)

# Split the dataset into training and validation sets
X_train = training_dataset[:, :-1]
y_train = training_dataset[:, -1] 
X_val = validation_dataset[:, :-1]
y_val = validation_dataset[:, -1]

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(6,)),  # Input layer (6 features)
    Dense(32, activation='relu'),                   # Hidden layer
    Dense(8, activation='softmax')    # Output layer (one neuron per class)
])

# Compile the model
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy for integer labels
            metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Generate predictions for the validation set
y_pred = model.predict(X_val)

# Convert predictions from probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Compare predicted labels with true labels
for i in range(len(X_val)):
    print(f"Row {i+1}: Actual Label = {y_val[i]}, Predicted Label = {y_pred_labels[i]}")

# Check the model's accuracy on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"NN validation Accuracy: {val_accuracy * 100:.2f}%")