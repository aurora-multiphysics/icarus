from pathlib import Path
import math
from icarus import (DatasetGenerator,
                    ModelBuilder,
                    MooseSetup,
                    UserInterface)

def main():
    """main: runs all of the other functions.
    """
    # Required parameters - change to desired values
    # Input and output paths
    input_file_path = "scripts/moose/plate_2d_thermal.i"
    output_file_path = "examples/example_outputs/ex2_outputs/" 
    # Parallelisation options
    n_tasks, n_threads = 1, 2
    num_para_runs = 2
    # Ratio of invalid:valid datasets in training data
    ground_truths_per_dataset = 3

    # Modelling parameters - change to desired values:
    # Modelling framework:
        # rf = Random Forest
        # svm = Support Vector Machine
        # dt = Decision Tree
    framework = "rf"
    # Analysis field (temperature, displacement, or strain)
    field_key = "temperature"
    # Sensor arrangement (x_sensors, y_sensors, z_sensors)
    sensors = (3,2,1)
    # Number of spatial dimensions being used 
    dims = 2
    # Whether the sensors should include basic errors or not
    errors = False
    # Whether the model should be a multi-classifier rather than binary, so it can 
    # distinguish between perturbations to different classes of invalid parameters 
    # (geometry, BCs, material properties) rather than just valid and invalid results
    multi = True 
    # Whether the unlabelled data should be deleted 
    delete_datasets = True
    # Whether the model should be saved as a .pkl file
    save = False

    # Setup MOOSE aspects of Icarus
    moose_setup = MooseSetup(input_file_path, n_tasks=n_tasks, n_threads=n_threads)
    moose_runner, moose_modifier = moose_setup.setup_moose_runner()

    # Accesses the available parameters and allow user to select which are perturbed, what
    # values they should take, and how many validation values to use for each
    found_vars = moose_modifier.get_vars()
    num_validation_values, parameters = UserInterface().accept_parameters(found_vars)
    
    # Initialise dataset generator and generate perturbed, validation and ground truths datasets
    dataset_generator = DatasetGenerator(moose_runner, moose_modifier, parameters, output_file_path, num_para_runs)

    # Sets up the paths to where the outputted unlabelled datasets will be saved
    perturbed_path, perturbed_vals = Path(str(output_file_path+"perturbed_datasets/")), None
    validation_path, validation_vals = Path(str(output_file_path+"validation_datasets/")), None
    
    paths = {perturbed_path: perturbed_vals, validation_path: validation_vals}

    # Generates the unlabelled datasets
    for name, param_class, values, n_valid in zip(dataset_generator.param_names, dataset_generator.param_classes, \
                                                  dataset_generator.param_values, num_validation_values):
        # Generates valid validation values for each parameter 
        validation_values = dataset_generator.generate_validation_values(values, n_valid)
        
        paths[perturbed_path] = values
        paths[validation_path] = validation_values

        # Generates perturbed and validation datasets based on chosen values 
        for path, vals in paths.items():
            dataset_generator.generate_dataset(path, name, param_class, vals)

    # Generates the ground truth datasets based on the ratio of perturbed:ground truth specified
    for path in paths.keys():
        num_datasets = sum(1 for d in path.iterdir() if d.is_dir())
        num_ground_truths = math.ceil(num_datasets/ground_truths_per_dataset)
        dataset_generator.generate_ground_truths(path, num_ground_truths)

    # Sets up, runs, and (optionally) saves the chosen model:
    model = ModelBuilder(output_file_path, framework, field_key, sensors, dims, errors, multi, delete_datasets, save)
    
    # Generates labelled training and validation datasets 
    training_dataset = model.generate_labelled_dataset(perturbed_path)
    validation_dataset = model.generate_labelled_dataset(validation_path)

    # Deletes unlabelled datasets (and saves labelled) if specified
    if delete_datasets:
        model.delete_data(perturbed_path, training_dataset, validation_path, validation_dataset)
    
    # Splits data according to modelling requirements
    X_train, y_train, X_val, y_val = training_dataset[:, :-1], training_dataset[:, -1], validation_dataset[:, :-1], validation_dataset[:, -1]

    # Builds the specified model and predicts the labels of the validation datasets
    classifier = model.classifier_model(X_train, y_train)
    y_pred = classifier.predict(X_val)
    
    # Outputs model metrics 
    model.output_model_metrics(y_val, y_pred)

    # Saves model if specified
    if save:
        model.save_model(classifier)
    
if __name__ == "__main__":
    main()