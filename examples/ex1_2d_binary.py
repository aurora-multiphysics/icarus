from pathlib import Path
from icarus import (DatasetGenerator,
                    ModelBuilder,
                    MooseSetup,
                    UserInterface)

def main():
    """main: runs all of the other functions.

    Raises
    ------
    FileNotFoundError
        If the file submission window is closed (instead of submitting file paths)
    ValueError
        If the input file is improperly formatted and no parameters are found as expected
    ValueError
        If the parameter submission window is closed (instead of submitting parameter data)
    """
    # Change to required input and output paths
    input_file_path = "scripts/moose/plate_2d_thermal.i"
    output_file_path = "examples/example_outputs/ex1_outputs/"

    # Ensures both paths exist and that the input file path ends with an input file name
    if not Path(input_file_path).exists() or not Path(output_file_path).exists():
        if input_file_path[-2:] != ".i":
            raise FileNotFoundError(f"Specified input and/or output file path not found.")

    # Setup MOOSE aspects of Icarus - change n_tasks and n_threads to desired parallelisation options
    n_tasks, n_threads = 1, 2
    moose_setup = MooseSetup(input_file_path, n_tasks=n_tasks, n_threads=n_threads)
    moose_runner, moose_modifier = moose_setup.setup_moose_runner()

    # Accesses the available parameters
    found_vars = moose_modifier.get_vars()

    # Ensures there are some available parameters
    if len(found_vars) == 0:
        raise ValueError(f"No parameters found in input file. Check input file and try again.")
    else:
        # Accepts user-defined parameters for perturbation and validation
        num_validation_values, parameters = UserInterface().accept_parameters(found_vars)

    # Ensures acceptable parameters
    if parameters == None or len(parameters) == 0:
        raise ValueError(f"Unacceptable parameters. Exiting.")
    
    # Initialises dataset generator - change num_para_runs to desired parallelisation option
    num_para_runs = 2
    dataset_generator = DatasetGenerator(moose_runner, moose_modifier, parameters, num_para_runs=num_para_runs)

    # Generates perturbed, validation and ground truths datasets
    # Change ground_truths_per_dataset to desired value - determines ratio of invalid:valid datasets in training data
    ground_truths_per_dataset = 3
    dataset_generator.generate_datasets(output_file_path, num_validation_values, ground_truths_per_dataset=ground_truths_per_dataset)

    # Sets up and runs the chosen model - change:
        # framework to desired framework:
            # rf = Random Forest
            # svm = Support Vector Machine
            # dt = Decision Tree
        # field_key to desired analysis field (temperature, displacement, or strain)
        # sensors to desired sensor arrangement (x_sensors, y_sensors, z_sensors)
        # dims to number of spatial dimensions being used 
        # errors to True if the sensors should include basic errors
        # multi to True if it should use a multi-classifier rather than binary
            # this allows the model to distinguish between perturbations to different
            # classes of invalid parameters (geometry, BCs, material properties) rather
            # than just valid and invalid results
        # delete_datasets to True if the unlabelled data should be deleted 
    model = ModelBuilder(output_file_path, 
                         framework="rf", 
                         field_key="temperature",
                         sensors=(3,2,1), 
                         dims=2, 
                         errors=False,
                         multi=True)
    classifier = model.run_model(delete_datasets=True)

    # Allows the model to be saved if desired 
    save = False
    if save:
        model.save_model(classifier)
    
if __name__ == "__main__":
    main()


# Next steps:  
    # Test suite using PyTest
    # Optimise usability and structure of classes/dicts/input files, etc
    # Improve classifiers
    # Allow user control of hyperparameters
    # Fully configureable example (decoupled steps)
    # Stretch goals: 
        # More complex input files, e.g. 3D monoblock
        # Accepting multiple simultaneous perturbations - generate datasets class
        # Allowing user to define sensor positions - labelled dataset function