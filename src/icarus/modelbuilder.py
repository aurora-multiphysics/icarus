import numpy as np
import pyvale
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from mooseherder import ExodusReader
from pathlib import Path
import shutil

class ModelBuilder:
    def __init__(self):
        pass 

    def generate_labelled_dataset(self, folder_path: Path) -> list[list[list[float], int]]:
        """generate_labelled_dataset: used to create a dataset by extracting values from the outputs
            of each run, using ExodusReader class to read the data and PyVale to create an array
            of sensors used to extract measurements of the required field at given points.
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
        sensx, sensy, sensz = 3, 2, 1
        labelled_dataset_cols = (sensx*sensy*sensz)+1
        labelled_dataset = np.empty((0, labelled_dataset_cols))

        for file_path in folder_path.rglob('*.e'):
            if "ground_truth" in str(file_path):
                label = 0
            else:
                label = 1

            sim_data = ExodusReader(file_path).read_all_sim_data()
            field_key = "temperature"

            sim_data.coords = sim_data.coords*1000.0

            xmax = np.max(sim_data.coords[:, 0])
            ymax = np.max(sim_data.coords[:, 1])

            n_sens = (sensx,sensy,sensz)
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

            measurements = np.append(measurements, label) 
            labelled_dataset = np.vstack([labelled_dataset, measurements]) 

        return labelled_dataset


    def model(self, output_file_path: str) -> None:
        """model: used to train the Random Forest model on the training datasets, use the model 
            to make predictions for the validation dataset, and verify the accuracy of the model. 
            Outputs the pertinent information to the user, and then allows them to decide whether 
            or not to save the model as a .pkl file.
            
        Parameters
        ----------
        output_file_path : str
            Contains the path to the folder where the datasets and model(s) will be stored.
        """
        perturbed_path, validation_path = Path(output_file_path+"perturbed_datasets/"), Path(output_file_path+"validation_datasets/")
        training_dataset = self.generate_labelled_dataset(perturbed_path)
        validation_dataset = self.generate_labelled_dataset(validation_path)

        paths = {perturbed_path: training_dataset, validation_path: validation_dataset}
        for path in paths:
            for folder in path.iterdir():
                if folder.is_dir():
                    shutil.rmtree(folder)
            dataset = paths[path]
            np.savetxt(path/"labelled_dataset.txt", dataset, fmt="%d", delimiter=",")
        
        X_train, y_train, X_val, y_val = training_dataset[:, :-1], training_dataset[:, -1], validation_dataset[:, :-1], validation_dataset[:, -1]  

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_val)

        for i in range(len(X_val)):
            print(f"Row {i + 1}: Actual Label = {y_val[i]}, Predicted Label = {y_pred[i]}")

        val_accuracy = accuracy_score(y_val, y_pred)
        print(f"Random Forest Validation Accuracy: {val_accuracy * 100:.2f}%")
        
        save_model = input("Save model? (Y/N) ")
        while save_model.lower() not in ["y", "n"]:
            print("Invalid input")
            save_model = input("Save model? (Y/N) ")
            
        if save_model.lower() == "y":
            joblib.dump(rf_classifier, str(output_file_path)+'model.pkl')