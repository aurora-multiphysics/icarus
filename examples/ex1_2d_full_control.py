from pathlib import Path
import math
import sklearn
import inspect
import sys
import joblib
import numpy as np
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
    output_file_path = "examples/example_outputs/ex1_outputs/"  
    # Perturbation parameters, leave blank (ie parameters = {}) to use auto-generated
    # tkinter interface
    # Format as {param_name, [param_class, [param_values]]}
    # where param_name is the name of the parameter from the input file
    # param_class is the classification of the parameter
        # geom - geometry, bc - boundary condtion, mat_prop - material property
    # and param_values is a list of values the parameter should take on
    parameters = {
        "max_temp": ["bc", [750, 1000, 1250, 1500, 1750]],
        "thermal_conductivity": ["mat_prop", [55, 65, 76, 85, 95]]
    }
    # Number of validation values to use for each parameter, respectively
    num_validation_values = [3, 3]
    # Parallelisation options
    n_tasks, n_threads = 1, 2
    num_para_runs = 2
    # Ratio of invalid:valid datasets in training data
    ground_truths_per_dataset = 3

    # Modelling parameters - change to desired values:
    # Classifier framework
    # For a comprehensive list of available classifiers, please refer
    # to the relevant documentation:
    # https://scikit-learn.org/stable/supervised_learning.html
    classifier_framework = sklearn.ensemble.RandomForestClassifier
    # Classifier parameters
    # Note that not all parameters are required for every classifier. 
    # Any irrelevant or unsupported parameters for the selected model 
    # will be automatically ignored during model creation. 
    # For a comprehensive list of available parameters for each classifier, 
    # please refer to the relevant documentation:
    # https://scikit-learn.org/stable/modules/classes.html#classifier
    classifier_params = {
        "n_estimators": 100,
        "random_state": 42,
        "kernel": "linear",
        "C": 0.025,
        "max_depth": 5
    }
    # Filter parameters according to chosen classifier 
    sig = inspect.signature(classifier_framework.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_params = {k: v for k, v in classifier_params.items() if k in valid_params}
    # Make sure there are some filtered parameters to be used
    if len(filtered_params) == 0:
        print(f"Invalid parameters for {classifier_framework}: {classifier_params}")
        sys.exit()
    # Print ignored parameters 
    ignored = set(classifier_params) - set(filtered_params)
    print(f"Ignored parameters for {classifier_framework}: {ignored}")
    # The field being analysed as used by your input script 
    field_key = "temperature"
    # Analysis sensor type 
    # (thermocouples for temperature, disp_sensors for displacement, or strain_gauges for strain)
    sensor_type = "thermocouples"
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
    # Whether the model should be saved as a .pkl file and what it should be called
    save = False
    model_name = "ex1_2d_model"

    # Setup MOOSE aspects of Icarus
    moose_setup = MooseSetup(input_file_path, n_tasks=n_tasks, n_threads=n_threads)
    moose_runner, moose_modifier = moose_setup.setup_moose_runner()

    # Accesses the available parameters 
    found_vars = moose_modifier.get_vars()
    if len(parameters) == 0:
        # If none have been manually specified, allow user to select which parameters are perturbed, 
        # what values they should take, and how many validation values to use for each
        num_validation_values, parameters = UserInterface().accept_parameters(found_vars)
    else:
        # Ensure parameters dictionary and num_validation_values list are valid 
        if len(num_validation_values) != len(parameters) or 0 in num_validation_values:
            print("Invalidation number of validation values specified")
            sys.exit()
        for param_name, param_data in parameters.items():
            if param_name not in found_vars.keys():
                print(f"Parameter {param_name} not found in input file. Exiting.")
                sys.exit() 
            if param_data[0].lower() not in ["geom", "bc", "mat_prop"]:
                print(f"Invalid parameter class. Exiting.")
                sys.exit()
            if len(param_data[1]) < 1:
                print("Insufficient parameter values. Exiting.")
                sys.exit()
    
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
    model = ModelBuilder(output_file_path, field_key, sensor_type, sensors, dims, errors, multi, delete_datasets)
    
    # Generates labelled training and validation datasets and saves them
    header = ",".join([f"Sensor {i} reading" for i in range(1, sensors[0]*sensors[1]*sensors[2]+1)] + ["Label"])
    training_dataset = model.generate_labelled_dataset(perturbed_path)
    np.savetxt(perturbed_path/"labelled_dataset.txt", training_dataset, fmt="%d", delimiter=",", header=header, comments='')
    validation_dataset = model.generate_labelled_dataset(validation_path)
    np.savetxt(validation_path/"labelled_dataset.txt", training_dataset, fmt="%d", delimiter=",", header=header, comments='')

    # Deletes unlabelled datasets if specified
    if delete_datasets:
        model.delete_data(perturbed_path, validation_path)
    
    # Splits data according to modelling requirements
    X_train, y_train, X_val, y_val = training_dataset[:, :-1], training_dataset[:, -1], validation_dataset[:, :-1], validation_dataset[:, -1]

    # Builds the specified model and predicts the labels of the validation datasets
    classifier = classifier_framework(**filtered_params)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_val)
    
    # Outputs model metrics 
    model.output_model_metrics(y_val, y_pred)

    # Saves model if specified
    if save:
        joblib.dump(classifier, str(output_file_path)+model_name+'_model.pkl')
    
if __name__ == "__main__":
    main()

# Next steps:
    # Continue building test suite
    # Improve classifiers
    # Stretch goals: 
        # More complex input files, e.g. 3D monoblock
        # Accepting multiple simultaneous perturbations - generate datasets class
        # Allowing user to define sensor positions - labelled dataset function