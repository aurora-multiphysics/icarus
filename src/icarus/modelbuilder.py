"""
ModelBuilder module to create and analyse the Icarus machine learning model.

(c) Copyright UKAEA 2025.
"""

from pathlib import Path
import shutil
import inspect
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from mooseherder import ExodusReader, SimData


class ModelBuilder:
    """Used to create and analyse the machine learning model
    """
    def __init__(self, output_file_path: str, field_key: str, sensor_type: str,
                 sensors: list, dims: int = 2, errors: bool = False,
                 multi: bool = False, delete_datasets: bool = False,
                 save: bool = True) -> None:
        """__init__

        Parameters
        ----------
        output_file_path : str
            Path to the location that all outputs should be saved.
        field_key : str
            The field key used for the analysis field in the input file.
        sensor_type : str, optional
            The sensor type used to analyse the field in the
            experiment/simulation.
        sensors : list, optional
            The number of sensors to in each dimension (x,y,z).
        dims : int, optional
            The dimensionality of the problem, by default 2
        errors : bool, optional
            Allows the user to determine whether or not the sensor array
            should incorporate errors,
            by default False.
        multi : bool, optional
            Allows the user to specify whether to use a multi-classifier,
            by default False (meaning use a binary classifier).
        delete_datasets : bool
            Allows the user to determine whether or not to delete the
            unlabelled datasets.
        save : bool
            Allows the user to determine whether or not to save the model
            and the labelled datasets.

        Raises
        ----------
        FileNotFoundError
            If any of the required output file paths don't exist.
        ValueError
            If any of sensor_type, sensors, dims, errors, multi,
            delete_datasets, or save are unacceptable.
        """
        import pyvale
        self.pyvale = pyvale

        if not Path(output_file_path).exists():
            raise FileNotFoundError(
                f"Output file path {output_file_path} not found. Exiting.")
        self.output_file_path = output_file_path

        self.field_key = field_key

        if sensor_type not in ["thermocouples",
                               "disp_sensors",
                               "strain_gauges"]:
            raise ValueError(f"Unacceptable sensor type {sensor_type}. \
                             Must be either thermocouples, disp_sensors, \
                             or strain_gauges. Exiting.")
        self.sensor_type = sensor_type

        for sensor in sensors:
            if sensor == 0:
                raise ValueError("Invalid sensor array {sensors}. \
                                 Can't have a dimension with zero sensors. \
                                 Exiting.")
        if len(sensors) != 3 or sensors is None:
            raise ValueError("Invalid sensor array {sensors}. \
                             Must have a number of sensors for exactly \
                             3 dimensions. Exiting.")
        self.sensors = sensors

        if dims not in [1, 2, 3]:
            raise ValueError(f"Dimensions must be 1, 2 or 3, not {dims}. \
                             Exiting.")
        self.dims = dims

        self.errors = errors
        self.multi = multi
        self.delete_datasets = delete_datasets
        self.save = save

        self.labelled_dataset_cols = (sensors[0]*sensors[1]*sensors[2])+1


    def sensor_array(self, sim_data: SimData) -> "pyvale.SensorArrayPoint":
        """Generate the array of sensors used to generate the labelled
        datasets required for training the model.

        Parameters
        ----------
        sim_data : SimData
            The unlabelled datasets.

        Returns
        -------
        sens_array : pyvale.SensorArrayPoint
            The desired sensor array used to extract the values of the chosen
            field at given points.
        """
        sensx, sensy, sensz = self.sensors[0], self.sensors[1], self.sensors[2]

        sim_data.coords = sim_data.coords*1000.0

        xmax = np.max(sim_data.coords[:, 0])
        xmin = np.min(sim_data.coords[:, 0])

        ymax, ymin, zmax, zmin = 0.0, 0.0, 0.0, 0.0
        if self.dims > 1:
            ymax = np.max(sim_data.coords[:, 1])
            ymin = np.min(sim_data.coords[:, 1])
            if self.dims > 2:
                zmax = np.max(sim_data.coords[:, 2])
                zmin = np.min(sim_data.coords[:, 2])

        n_sens = (sensx, sensy, sensz)
        x_lims = (xmin, xmax)
        y_lims = (ymin, ymax)
        z_lims = (zmin, zmax)
        sens_pos = self.pyvale.create_sensor_pos_array(n_sens, x_lims,
                                                       y_lims, z_lims)
        sens_data = self.pyvale.SensorData(positions=sens_pos)

        errors_map = {
            True: "basic_errs",
            False: "no_errs"
        }

        func_name = f"{self.sensor_type}_{errors_map[self.errors]}"
        factory = self.pyvale.SensorArrayFactory
        func = getattr(factory, func_name)
        sens_array = func(sim_data, sens_data, elem_dims=self.dims,
                          field_name=self.field_key)

        return sens_array


    def generate_training_dataset(self) \
            -> list[list[list[float], int]]:
        """Create a dataset by extracting values from the outputs of each run,
        using ExodusReader class to read the data. Assigns a label to each
        dataset depending on which (if any) parameter has been perturbed.

        Raises
        ----------
        ValueError
            If the training and/or validation datasets failed to generate.

        Returns
        ----------
        labelled_dataset: np.array[np.array[list[float], int]]
            2D array containing the list of extracted measurements for each
            dataset and its corresponding label. To be used to train/validate
            the model.
        """
        labelled_dataset = np.empty((0, self.labelled_dataset_cols))
        folder_path = Path(self.output_file_path+"perturbed_datasets/")

        for file_path in folder_path.rglob('*.e'):
            if "ground_truth" in str(file_path):
                label = 0
            else:
                if self.multi:
                    labels = {"ground_truth": 0, "geom": 1,
                              "bc": 2, "mat_prop": 3}
                    for param_class, class_label in labels.items():
                        if param_class in str(file_path):
                            label = class_label
                else:
                    label = 1

            sim_data = ExodusReader(file_path).read_all_sim_data()
            sens_array = self.sensor_array(sim_data)
            measurements = sens_array.get_measurements()[:, 0, 1]

            measurements = np.append(measurements, label)
            labelled_dataset = np.vstack([labelled_dataset, measurements])

        if len(labelled_dataset) <= 1:
            raise ValueError("Training and/or validation datasets \
                             failed to generate. Exiting.")

        return labelled_dataset


    def generate_validation_dataset(self) \
            -> list[list[list[float], int]]:
        """Create a dataset by extracting values from the outputs of each run,
        using ExodusReader class to read the data. Assigns a label to each
        dataset depending on which (if any) parameter has been perturbed.

        Raises
        ----------
        ValueError
            If the training and/or validation datasets failed to generate.

        Returns
        ----------
        labelled_dataset: np.array[np.array[list[float], int]]
            2D array containing the list of extracted measurements for each
            dataset and its corresponding label. To be used to train/validate
            the model.
        """
        labelled_dataset = np.empty((0, self.labelled_dataset_cols))
        folder_path = Path(self.output_file_path+"validation_datasets/")

        for file_path in folder_path.rglob('*.e'):
            if "ground_truth" in str(file_path):
                label = 0
            else:
                if self.multi:
                    labels = {"ground_truth": 0, "geom": 1,
                              "bc": 2, "mat_prop": 3}
                    for param_class, class_label in labels.items():
                        if param_class in str(file_path):
                            label = class_label
                else:
                    label = 1

            sim_data = ExodusReader(file_path).read_all_sim_data()
            sens_array = self.sensor_array(sim_data)
            measurements = sens_array.get_measurements()[:, 0, 1]

            measurements = np.append(measurements, label)
            labelled_dataset = np.vstack([labelled_dataset, measurements])

        if len(labelled_dataset) <= 1:
            raise ValueError("Training and/or validation datasets \
                             failed to generate. Exiting.")

        return labelled_dataset


    def generate_labelled_datasets(self):
        """Generate both the training and validation labelled datasets.

        Returns
        ----------
        training_dataset : np.array[np.array[list[float], int]]
            The labelled training dataset.
        validation_dataset : np.array[np.array[list[float], int]]
            The labelled validation dataset.
        """
        training_dataset, validation_dataset = \
            self.generate_training_dataset(), \
            self.generate_validation_dataset()

        return training_dataset, validation_dataset


    def filter_parameters(self, classifier_framework, classifier_params):
        """Filter the parameters to only use those which are valid for the
        chosen classifier framework.

        Returns
        ----------
        filtered_params : dict
            The selected parameters that apply to the chosen classifier.
        """
        # Filter parameters according to chosen classifier
        sig = inspect.signature(classifier_framework.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        filtered_params = {k: v for k, v in classifier_params.items()
                           if k in valid_params}

        # Make sure there are some filtered parameters to be used
        if len(filtered_params) == 0:
            raise ValueError("Invalid parameters for {classifier_framework}: \
                {classifier_params}")

        # Print ignored parameters
        ignored = set(classifier_params) - set(filtered_params)
        print(f"Ignored parameters for {classifier_framework}: {ignored}")

        return filtered_params


    def delete_data(self, perturbed_path: Path, validation_path: Path) -> None:
        """Delete the unlabelled datasets and just saves the labelled dataset

        Parameters
        ----------
        perturbed_path : Path
            Path to the save location for the perturbed datasets.
        validation_path : Path
            Path to the save location for the validation datasets.
        """
        for path in [perturbed_path, validation_path]:
            for folder in path.iterdir():
                if folder.is_dir():
                    shutil.rmtree(folder)


    def output_model_metrics(self, y_val: list[float], y_pred: list[float]) \
            -> None:
        """Output the actual + predicted labels, as well as the final
        accuracy score for the model.

        Parameters
        ----------
        y_val : list[float]
            The actual labels for the validation dataset.
        y_pred : list[float]
            The predicted labels for the validation dataset.
        """
        for i, (actual, predicted) in enumerate(zip(y_val, y_pred), start=1):
            print(f"Row {i}: Actual Label = {actual}, Predicted Label = \
                  {predicted}")

        val_accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")


    def save_datasets_and_model(self, classifier, model_name: str,
                                training_dataset: list[list[list[float], int]],
                                validation_dataset: list[list[list[float],
                                                              int]]):
        """Save the labelled datasets and the model if specified.

        Parameters
        ----------
        classifier
            The machine learning model classifier.
        model_name : str
            The name of the model the saved.
        training_dataset : np.array[np.array[list[float]
            The labelled training dataset.
        validation_dataset : np.array[np.array[list[float]
            The labelled validation dataset.
        """
        if self.save:
            header = ",".join([f"Sensor {i} reading" for i in range(
                1, self.labelled_dataset_cols)] + ["Label"])

            joblib.dump(classifier,
                        str(self.output_file_path)+model_name+'_model.pkl')

            perturbed_path = Path(self.output_file_path+"/perturbed_datasets")
            validation_path = Path(self.output_file_path+"/labelled_datasets")

            np.savetxt(
                perturbed_path/"labelled_dataset.txt",
                training_dataset, fmt="%d", delimiter=",", header=header,
                comments='')

            np.savetxt(
                validation_path/"labelled_dataset.txt",
                validation_dataset, fmt="%d", delimiter=",", header=header,
                comments='')


    def build_model(self, classifier_framework,
                    classifier_params: dict[str, float],
                    model_name: str):
        """Convenience function to build the model using the other functions
        present in the ModelBuilder class.

        Parameters
        ----------
        classifier_framework
            The chosen framework for the machine learning classifier.
        classifier_params : dict[str, float]
            The list of parameters for the classifier specified by the user.
        model_name : str
            The name of the model to be saved.
        """
        # Generate labelled training and validation datasets and saves them
        training_dataset, validation_dataset = \
            self.generate_labelled_datasets()

        # Split data according to modelling requirements
        x_train, y_train, x_val, y_val = training_dataset[:, :-1], \
            training_dataset[:, -1], validation_dataset[:, :-1], \
            validation_dataset[:, -1]

        # Build the specified model and predict the labels of the validation
        # datasets
        filtered_params = self.filter_parameters(classifier_framework,
                                                 classifier_params)
        classifier = classifier_framework(**filtered_params)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_val)

        # Output model metrics
        self.output_model_metrics(y_val, y_pred)

        # Save model and labelled datasets if specified
        self.save_datasets_and_model(classifier, model_name,
                                     training_dataset,
                                     validation_dataset)
