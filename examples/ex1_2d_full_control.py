"""
Example file showing how to use Icarus and detailing customisable options.

This example runs an input file to produce a 2D temperature field with
different parameters perturbed. Any results from the default input file
are considered valid, while any other results are invalid. The datasets
generated are split into training and validation datasets and labelled,
and then a machine learning model is trained using the training dataset
to distinguish between valid and invalid datasets (and classify the
invalid sets depending on which parameter has been altered, if desired).
The model is then validated using the validation datasets to see if it
correctly predicted the labels.

(c) Copyright UKAEA 2025.
"""

import sklearn
from icarus import (DatasetGenerator,
                    ModelBuilder)


def main():
    """Run all of the other functions.
    """
    # Required parameters - change to desired values

    # Input and output paths
    input_file_path = "scripts/moose/plate_2d_thermal.i"
    output_file_path = "examples/example_outputs/ex1_outputs/"

    # Perturbation parameters, leave blank (ie parameters = {}) to use
    # auto-generated tkinter interface
    # Format as {param_name, [param_class, [param_values]]}
    # where param_name is the name of the parameter from the input file
    # param_class is the classification of the parameter
    #   geom - geometry, bc - boundary condtion, mat_prop - material property
    # and param_values is a list of values the parameter should take on
    parameters = {
        "max_temp": ["bc", [750, 1000, 1250, 1500, 1750]],
        "thermal_conductivity": ["mat_prop", [55, 65, 76, 85, 95]]
    }

    # Number of validation values to use for each parameter, respectively.
    # This defines how many different perturbed values each parameter will
    # take to create validation datasets to assess the quality of the model.
    num_validation_values = [3, 3]

    # Parallelisation options
    n_tasks, n_threads = 1, 2  # Parallelisation of each simulation
    num_para_runs = 2  # Number of concurrent simulations

    # Ratio of invalid:valid datasets
    # This defines how many datasets will be generated from the default
    # input file (no perturbations -> labelled 0 for valid dataset),
    # as the right ratio is important to ensure the model can recognise
    # a valid dataset without assuming all datasets are valid.
    datasets_per_ground_truth = 3

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

    # The field being analysed as used by your input script
    field_key = "temperature"

    # Analysis sensor type
    # (thermocouples for temperature, disp_sensors for displacement,
    # or strain_gauges for strain)
    sensor_type = "thermocouples"

    # Sensor arrangement (x_sensors, y_sensors, z_sensors)
    # Defines how many sensors are present in each dimension,
    # with sensors being uniformly distributed.
    # e.g. (3,2,1) would result in a 2D arrangement as follows:
    #  _______________
    # |               |
    # |   x   x   x   |
    # |   x   x   x   |
    # |_______________|
    sensors = (3, 2, 1)

    # Number of spatial dimensions being used
    dims = 2

    # Whether the sensors should include errors or give exact values.
    # Including errors means the model is more likely to miscategorise,
    # but a successful model will be much better in reality, where errors
    # are unavoidable.
    errors = False

    # Whether the model should be a multi-classifier rather than binary,
    # so it can distinguish between perturbations to different classes of
    # invalid parameters (geometry, BCs, material properties) rather than
    # just valid and invalid results
    multi = True

    # Whether the unlabelled data should be deleted
    delete_datasets = False

    # Whether the model should be saved as a .pkl file
    # and what it should be called
    save = False
    model_name = "ex1_2d_model"

    # Initialise dataset generator to generate unlabelled datasets
    dataset_generator = DatasetGenerator(input_file_path,
                                         n_tasks, n_threads,
                                         parameters, num_validation_values,
                                         datasets_per_ground_truth,
                                         output_file_path,
                                         num_para_runs)

    # Generate the unlabelled perturbed, validation and ground truths datasets
    dataset_generator.generate_datasets()

    # Initialise model builder to set up, run, and (optionally)
    # save the chosen model
    model = ModelBuilder(output_file_path, field_key, sensor_type, sensors,
                         dims, errors, multi, delete_datasets, save)

    # Set up, run, and (optionally) save the chosen model
    model.build_model(classifier_framework, classifier_params, model_name)

if __name__ == "__main__":
    main()
