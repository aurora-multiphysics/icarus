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
    classifier_framework = "sklearn.ensemble.RandomForestClassifier"
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
    delete_datasets = False
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
    dataset_generator.generate_datasets(num_validation_values, ground_truths_per_dataset)

    # Sets up, runs, and (optionally) saves the chosen model:
    model = ModelBuilder(output_file_path, classifier_framework, classifier_params, field_key,
                         sensor_type, sensors, dims, errors, multi, delete_datasets, save)
    model.run_model()
    
if __name__ == "__main__":
    main()

# Next steps:  
    # Docstrings + error handling for recent changes
    # Test suite using PyTest
    # Optimise usability and structure of classes/dicts/input files, etc
    # Improve classifiers
    # Allow user control of hyperparameters
    # Fully configureable example (decoupled steps)
    # Stretch goals: 
        # Make tkinter interface optional
        # More complex input files, e.g. 3D monoblock
        # Accepting multiple simultaneous perturbations - generate datasets class
        # Allowing user to define sensor positions - labelled dataset function
