import numpy as np
import pyvale
from pyvale import SensorArrayPoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from mooseherder import ExodusReader, SimData
from pathlib import Path
import shutil

class ModelBuilder:
    """Used to create a sensor array, generate the labelled datasets the sensor array and the 
        simulation data, train the chosen model on the training datasets, use the model 
        to make predictions for the validation dataset, and verify the accuracy of the model. 
        Outputs the pertinent information to the user, and allows the user to save the model,
        and to delete unlabelled datasets
    """
    def __init__(self, output_file_path: str, framework: str, field_key: str="temperature", 
                 sensors: list=[3,2,1], dims: int=2, errors: bool=False, multi: bool=False):
        """__init__

        Parameters
        ----------
        output_file_path : str
            Path to the location that all outputs should be saved.
        framework : str
            The type of ML model to use (RandomForest, SVM, etc.)
        field_key : str, optional
            The field being analysed in the experiment/simulation, by default "temperature"
        sensors : list, optional
            The number of sensors to in each dimension (x,y,z), by default [3,2,1] for a 2d
            simulation.
        dims : int, optional
            The dimensionality of the problem, by default 2
        errors : bool, optional
            Allows the user to determine whether or not the sensor array should incorporate errors, 
            by default False.
        multi : bool, optional
            Allows the user to specify whether to use a multi-classifier, by default
            False (meaning use a binary classifier).
        """
        self.output_file_path = output_file_path 
        self.framework = framework
        self.field_key = field_key
        self.sensors = sensors
        self.dims = dims
        self.errors = errors
        self.multi = multi
        self.labelled_dataset_cols = (sensors[0]*sensors[1]*sensors[2])+1


    def sensor_array(self, sim_data: SimData) -> SensorArrayPoint:
        """sensor_array: used to generate the array of sensors used to generate the labelled
            datasets required for training the model. 

        Parameters
        ----------
        sim_data : SimData
            The unlabelled datasets.

        Returns
        -------
        SensorArrayPoint
            The desired sensor array used to extract the values of the chosen field at given points.
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

        n_sens = (sensx,sensy,sensz)
        x_lims = (xmin,xmax)
        y_lims = (ymin,ymax)
        z_lims = (zmin,zmax)
        sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
        sens_data = pyvale.SensorData(positions=sens_pos)

        field_key_map = {
            "temperature": "thermocouples",
            "displacement": "disp_sensors",
            "strain": "strain_gauges"
        }

        errors_map = {
            True: "basic_errs",
            False: "no_errs"
        }

        func_name = f"{field_key_map[self.field_key]}_{errors_map[self.errors]}"
        factory = pyvale.SensorArrayFactory
        func = getattr(factory, func_name)

        sens_array = func(sim_data, sens_data, self.field_key, spat_dims=self.dims)
        
        return sens_array
    

    def generate_labelled_dataset(self, folder_path: Path) -> list[list[list[float], int]]:
        """generate_labelled_dataset: used to create a dataset by extracting values from the outputs
            of each run, using ExodusReader class to read the data.
            Assigns a label to each dataset depending on which (if any) parameter has been perturbed.    

        Parameters
        ----------
        folder_path : Path
            Specifies the location of the unlabelled datasets from which the information for the 
            labelled dataset should be extracted. Should be the base directory - i.e. either 
            perturbed_datasets/ or validation_datasets/.

        Returns
        -------
        labelled_dataset: np.array[np.array[list[float], int]]
            2D array containing the list of extracted measurements for each dataset and its 
            corresponding label. To be used to train/validate the model.
        """
        labelled_dataset = np.empty((0, self.labelled_dataset_cols))

        for file_path in folder_path.rglob('*.e'):
            if "ground_truth" in str(file_path):
                label = 0
            else:
                if self.multi:
                    labels = {"ground_truth": 0, "geom": 1, "bc": 2, "mat_prop": 3}
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

        return labelled_dataset
    

    def delete_data(self, perturbed_path: Path, training_dataset: list[list[list[float],int]], 
                    validation_path: Path, validation_dataset: list[list[list[float],int]]) -> None:
        """delete_data: deletes the unlabelled datasets and just saves the labelled dataset

        Parameters
        ----------
        perturbed_path : Path
            Path to the save location for the perturbed datasets.
        training_dataset : list[list[list[float],int]]
            The labelled training dataset.
        validation_path : Path
            Path to the save location for the validation datasets.
        validation_dataset : list[list[list[float],int]]
            The labelled validation dataset.
        """
        paths = {perturbed_path: training_dataset, validation_path: validation_dataset}
        for path in paths:
            for folder in path.iterdir():
                if folder.is_dir():
                    shutil.rmtree(folder)
            dataset = paths[path]
            np.savetxt(path/"labelled_dataset.txt", dataset, fmt="%d", delimiter=",")
    

    def classifier_model(self, X_train: list[float], y_train: list[float]) \
           -> RandomForestClassifier | SVC | DecisionTreeClassifier:
        """classifier_model: generates a classifier of the selected framework for the
            given data.

        Parameters
        ----------
        X_train : list[float]
            The training data from the labelled training dataset.
        y_train : list[float]
            The labels for each row of the training dataset.

        Returns
        -------
        classifier : RandomForestClassifier | SVC | DecisionTreeClassifier
            The classifier itself.
        """
        classifiers = {
            "rf": RandomForestClassifier(n_estimators=100, random_state=42),
            "svm": SVC(kernel="linear", C=0.025, random_state=42),
            "dt": DecisionTreeClassifier(max_depth=5, random_state=42)
        }

        for classifier_framework, classifier_function in classifiers.items():
            if self.framework == classifier_framework:
                classifier = classifier_function
                
        classifier.fit(X_train, y_train)

        return classifier

    def output_model_metrics(self, y_val: list[float], y_pred: list[float]) -> None:
        """output_model_metrics: prints the actual + predicted labels, as well as the final
        accuracy score for the model.

        Parameters
        ----------
        y_val : list[float]
            The actual labels for the validation dataset.
        y_pred : list[float]
            The predicted labels for the validation dataset.
        """
        for i, (actual, predicted) in enumerate(zip(y_val, y_pred), start=1):
            print(f"Row {i}: Actual Label = {actual}, Predicted Label = {predicted}")

        val_accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    
    def save_model(self, classifier: RandomForestClassifier) -> None:
        """save_model: used to save the model as a .pkl file

        Parameters
        ----------
        classifier : RandomForestClassifier
            The classifier model generated to be saved.
        """
        joblib.dump(classifier, str(self.output_file_path)+str(self.framework)+'_model.pkl')


    def run_model(self, delete_datasets: bool=True) -> RandomForestClassifier:
        """run_model: Convenience function to run all aspects of the ModelBuilder class.
            
        Parameters
        ----------
        delete_datasets : bool
            Allows the user to determine whether or not to delete the unlabelled datasets.
        
        Returns 
        ----------
        classifier : RandomForestClassifier 
            The classifier model itself.
        """
        perturbed_path, validation_path = Path(self.output_file_path+"perturbed_datasets/"), Path(self.output_file_path+"validation_datasets/")
        training_dataset = self.generate_labelled_dataset(perturbed_path)
        validation_dataset = self.generate_labelled_dataset(validation_path)

        if delete_datasets:
            self.delete_data(perturbed_path, training_dataset, validation_path, validation_dataset)
        
        X_train, y_train, X_val, y_val = training_dataset[:, :-1], training_dataset[:, -1], validation_dataset[:, :-1], validation_dataset[:, -1]

        classifier = self.classifier_model(X_train, y_train)
        y_pred = classifier.predict(X_val)
        
        self.output_model_metrics(y_val, y_pred)

        return classifier