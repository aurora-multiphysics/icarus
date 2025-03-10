from pathlib import Path
import random
import math
import numpy as np
import pyvale
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tkinter import *
import joblib
from mooseherder import (MooseHerd,
                         MooseRunner,
                         MooseConfig,
                         InputModifier,
                         DirectoryManager,
                         SweepReader,
                         ExodusReader)


NUM_PARA_RUNS = 2
USER_DIR = Path.home()


def setup_moose_runner(path, filename):
    """setup_moose_runner: Constructor for MOOSE runner taking a MooseConfig object
        that contains the paths to the main MOOSE install, the MOOSE app and
        the MOOSE app name. Sets parallelisation options to 1 task
        and 2 threads. Sets environment variables required for MPI setup.

    Parameters
    ----------
    path : str
        Contains the path to the folder where the input file is saved and the datasets 
        and model(s) will be stored
    filename : str
        Contains the name of the input file being used to generate the model.

    Returns
    -------
    moose_runner : MooseRunner
        Constructed MOOSE runner used to run the input file with modified variables 
    moose_modifier : InputModifier
        Used to extract and modify the variables in the input file.
        Specifies the comment character. Variable definition blocks should begin 
        #comment character#* and end #comment character#**, e.g. #_* and #** for
        moose.
    """
    moose_input = Path(str(path + filename))
    moose_modifier = InputModifier(moose_input, '#', '')
    moose_config = MooseConfig().read_config(Path.cwd() / 'moose-config.json')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks=1, n_threads=2, redirect_out=False)
    return moose_runner, moose_modifier


def setup_directory_manager(base_dir, sub_dir_name, n_dirs = 1):
    """setup_directory_manager: sets up directory manager to manage directories for running 
        simulations in parallel with the mooseherd. Clears existing directories and creates 
        specified new ones with given names

    Parameters
    ----------
    base_dir : Path
        Sets the base directory to create sub-directories for running the simulations. 
        The base directory must exist.
    sub_dir_name : str
        String to be used at the start of the created sub-directores. 
        Default on creation is 'sim-workdir'. Populates the list of run directories using 
        the new sub directory name.
    n_dirs : int, optional
        Number of directories to be created., by default 1

    Returns
    -------
    dir_manager : DirectoryManager
        Used to control how many and which directories are used to run the simulations.
    """
    dir_manager = DirectoryManager(n_dirs=n_dirs)
    dir_manager.set_base_dir(base_dir)
    dir_manager.set_sub_dir_name(sub_dir_name)
    dir_manager.clear_dirs()
    dir_manager.create_dirs()
    return dir_manager


def run_herd(moose_runner, moose_modifier, dir_manager, moose_vars, n_para=1, keep_flag=False):
    """run_herd: used to run parametric sweeps of simulation chains in
        parallel with configurable parallelisation options. Takes a list of
        SimRunner objects and a corresponding list of InputModifiers to insert the
        variables into the input scripts for the SimRunners. Will first call all InputModifiers 
        in the specified order and then call run on all the SimRunners in order. Uses the 
        DirectoryManager class to log the directories in which each parallel worker is
        creating input files and running simulations. Uses the SweepReader class to read the 
        output from one or more calls to mooseherd.run_para().
        Has configurable options for reading in the variable sweep in parallel.

    Parameters
    ----------
    moose_runner : MooseRunner
        Constructed MOOSE runner used to run the input file with modified variables. 
    moose_modifier : InputModifier
        Used to extract and modify the variables in the input file. Specifies the comment 
        character. Variable definition blocks should begin #comment character#* and end 
        #comment character#**, e.g. #_* and #** for moose.
    dir_manager : DirectoryManager
        Used to control how many and which directories are used to run the simulations.
    moose_vars : list[InputModifier]
        Used to extract and modify the variables in the input file.
        Specifies the comment character. Variable definition blocks should begin 
        #comment character#* and end #comment character#**, e.g. #_* and #** for
        moose.
    n_para : int, optional
        Sets the number of simulation chains to run in parallel. , by default 1
    keep_flag : bool, optional
        Flag used for allowing multiple calls to run to keep everything or to 
        overwrite each time, by default False - overwrite inputs and outputs with multiple calls
    """
    
    herd = MooseHerd([moose_runner], [moose_modifier], dir_manager)
    herd.set_num_para_sims(n_para=n_para)
    herd.set_keep_flag(keep_flag)
    for _ in range(NUM_PARA_RUNS):
        herd.run_para(moose_vars)

    sweep_reader = SweepReader(dir_manager, num_para_read=4)
    sweep_reader.read_all_output_keys()
    read_all = sweep_reader.read_results_para()


def generate_ground_truths(moose_runner, moose_modifier, base_dir, param_values):
    """generate_ground_truths: used to generate the ground truth by running the input file
        with no modifications, and save the results to the required base_dir under the 
        sub_dir_name "ground_truth". Creates 1 ground_truth dataset for every 5 perturbed 
        datasets

    Parameters
    ----------
    moose_runner : MooseRunner
        Constructed MOOSE runner used to run the input file with modified variables. 
    moose_modifier : InputModifier
        Used to extract and modify the variables in the input file. Specifies the comment 
        character. Variable definition blocks should begin #comment character#* and end 
        #comment character#**, e.g. #_* and #** for moose.
    base_dir : Path
        Contains the base directory to save ground_truth sub-directory for running the 
        simulations. 
    param_values : list[float]
        List of values for the currently selected parameter(s). Used to determine how many 
        ground_truth datasets to generate.
    """
    for i in range(math.ceil(len(param_values[0]) / 5)):
        dir_manager = setup_directory_manager(base_dir, 'ground_truth', 1)
        run_herd(moose_runner, moose_modifier, dir_manager, [[{}]], 1)


def generate_dataset(moose_runner, moose_modifier, base_dir, param_names, param_values):
    """generate_dataset: used to generate the perturbed and validation datasets by running the 
        input file with modifications to specified parameter(s), and save the results to the 
        required base_dir under a sub_dir_named for the perturbed parameter.

    Parameters
    ----------
    moose_runner : MooseRunner
        Constructed MOOSE runner used to run the input file with modified variables. 
    moose_modifier : InputModifier
        Used to extract and modify the variables in the input file. Specifies the comment 
        character. Variable definition blocks should begin #comment character#* and end 
        #comment character#**, e.g. #_* and #** for moose.
    base_dir : Path
        Contains the base directory to save ground_truth sub-directory for running the 
        simulations. 
    param_names : list[string]
        List of the names of the perturbed parameters.
    param_values : list[float]
        List of values for the currently selected parameter(s).
    """
    moose_vars = list([])
    n_dirs = 1
    for i in range(len(param_names)):
        n_dirs *= len(param_values[i])
    dir_manager = setup_directory_manager(base_dir, str(param_names[0]), n_dirs)
    for param in param_values[0]:
        moose_vars.append([{str(param_names[0]): param}]) 
    run_herd(moose_runner, moose_modifier, dir_manager, moose_vars, n_dirs)


def generate_validation_values(param_values, num_validation_values=2):
    """generate_validation_values: used to generate a specified number of validation values 
        for the selected parameter, so that the model can be tested to see if it can correctly
        determine when the parameter has been perturbed to a value that was not present in the 
        training dataset.

    Parameters
    ----------
    param_values : list[float]
        List of training values for the currently selected parameter(s).
    num_validation_values : int, optional
        Number of validation values to generate for each parameter, by default 2

    Returns
    -------
    validation_values : list[list[float]]
        List containing a list(s) of validation values for each parameter, to be used to
        determine the value of the InputModifier for that parameter for the run to be saved
        under validation_datasets/param_name
    """
    validation_values = [[]]
    for i in range(num_validation_values):
        distinct_val = False
        while not distinct_val:
            validation_value = random.uniform(min([val for sublist in param_values for val in sublist]),
                                             max([val for sublist in param_values for val in sublist]))
            if validation_value not in param_values and validation_value not in validation_values:
                distinct_val = True
        validation_values[0].append(validation_value)  

    return validation_values


def generate_datasets(path, parameters, moose_runner, moose_modifier):
    """generate_datasets: used to generate the unlabelled ground truth, perturbed, and 
        validation datasets by running the required functions with the necessary parameter
        names and values, and save paths.

    Parameters
    ----------
    path : str
        Contains the path to the folder where the input file is saved and the datasets 
        and model(s) will be stored.
    parameters : dict{str : list[float]}
        Contains the names of the parameters to be perturbed and the values they should take
        for each run.
    moose_runner : MooseRunner
        Constructed MOOSE runner used to run the input file with modified variables. 
    moose_modifier : InputModifier
        Used to extract and modify the variables in the input file. Specifies the comment 
        character. Variable definition blocks should begin #comment character#* and end 
        #comment character#**, e.g. #_* and #** for moose.
    """
    param_names = [[key] for key in parameters.keys()]
    param_values = [[value] for value in parameters.values()]

    for i in range(len(param_names)):
        validation_values = generate_validation_values(param_values[i])
        paths = {Path(str(path+"perturbed_datasets/")) : param_values[i], Path(str(path+"validation_datasets/")) : validation_values}
        for path, values in paths.items():
            generate_ground_truths(moose_runner, moose_modifier, path, values)
            generate_dataset(moose_runner, moose_modifier, path, param_names[i], values)

def accept_file():
    """accept_file: used to allow user to input path to input file and input file name
        via a tkinter user interface

    Returns
    -------
    path : str
        String containing the path inputted to the user interface.
    filename : str
        String containing the name of the input file inputted to the user interface.
    """
    
    def submit_file():
        """submit_file: specifies what should happen when the submit button is pressed.
            The values within the Entry boxes for path and filename should be saved to 
            their corresponding variables, and the tkinter window should close.
        """
        nonlocal path, filename
        path = str(file_path.get())
        filename = str(file_name.get())

        if Path(path).exists() and Path (path+filename).exists():
            file_root.quit()
            file_root.destroy()
        else:
            error_label.config(text="Specified path and/or input file not found.", fg="red")

    path, filename = None, None

    file_root = Tk()

    Label(file_root, text='File path:').grid(row=0)
    Label(file_root, text='Input file name:').grid(row=1)
    default_path = StringVar(value="full_functionality/")
    file_path = Entry(file_root, textvariable=default_path)
    file_path.grid(row=0, column=1)
    default_file = StringVar(value="plate_2d_thermal.i")
    file_name = Entry(file_root, textvariable=default_file)
    file_name.grid(row=1, column=1)

    submit_button = Button(file_root, text="Submit", command=submit_file)
    submit_button.grid(row=2, column=0, columnspan=2, pady=10)

    error_label = Label(file_root, text="", fg="red")
    error_label.grid(row=3, column=0, columnspan=2)

    file_root.mainloop()    

    return path, filename 


def accept_parameters(parameters):
    """accept_parameters: used to allow user to select which parameters to perturb and the range
        and interval of values to be used for each parameter.

    Parameters
    ----------
    parameters : dict{str : float}
        Dictionary of parameter names and their corresponding values in the default input file.
        Used to allow users to select which parameters to modify, and show them the default
        values so they don't include them again.

    Returns
    ----------
    parameters : dict{str : list[float]}
        Dictionary of parameter names and their corresponding list of values for perturbation.
    """

    def submit_parameters():
        """submit_parameter: specifies what should happen when the submit button is pressed.
            The values within the Entry boxes for min_val, max_val and interval for the selected 
            parameters should be saved to their corresponding variables to allow creation of 
            param_values list, and the tkinter window should close.

        Raises 
        ----------
        ValueError
            If the range or interval specified are invalid
        Value Error
            If there are fewer than 2 values for perturbation 
        """
        nonlocal parameters
        parameters = {}

        for row in rows:
            if row['checkbox_var'].get():  # Check if checkbox is ticked
                param_name = row['param_name']
                try:
                    default_val = float(row['default_val'])
                    min_val = float(row['min_val'].get())
                    max_val = float(row['max_val'].get())
                    interval = float(row['interval'].get())

                    if min_val >= max_val or interval <= 0:
                        raise ValueError("Invalid range or interval")

                    param_values = []
                    current_val = min_val
                    while current_val <= max_val:
                        if current_val != default_val:
                            param_values.append(current_val)
                        current_val += interval

                    if len(param_values) < 2:
                        raise ValueError("Insufficient perturbation values")

                    parameters[param_name] = param_values

                except ValueError as e:
                    error_label.config(text=f"Error: {str(e)}", fg="red")
                    return  # Stop execution if an error occurs

        if len(parameters) == 0:
            error_label.config(text="No parameters submitted", fg="red")
            return
        
        param_root.destroy()

    param_root = Tk()

    table_frame = Frame(param_root)
    table_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    headers = ["Select", "Param Name", "Default Value", "Min Value", "Max Value", "Interval"]
    for col, header in enumerate(headers):
        Label(table_frame, text=header).grid(row=0, column=col, padx=5, pady=5)

    params = []
    for parameter_name, parameter_value in parameters.items():
        try:
            if float(parameter_value) <= 10:
                x = parameter_value
            elif float(parameter_value) > 10 and parameter_value < 100:
                x = 10
            else:
                x = parameter_value/2
            params.append({"param_name": parameter_name,"default_val": parameter_value,"min_val": parameter_value+x, 
                        "max_val": parameter_value+5*x,"interval": x})
        except ValueError:
            params.append({"param_name": parameter_name,"default_val": parameter_value,"min_val": "", 
                        "max_val": "","interval": ""})

    rows = []
    for i, param in enumerate(params):
        row = {}

        row['checkbox_var'] = BooleanVar(value=False)

        checkbox = Checkbutton(table_frame, variable=row['checkbox_var'])
        checkbox.grid(row=i+1, column=0, padx=5, pady=5)

        Label(table_frame, text=param['param_name']).grid(row=i+1, column=1, padx=5, pady=5)
        row['param_name'] = param['param_name']

        Label(table_frame, text=param['default_val']).grid(row=i+1, column=2, padx=5, pady=5)
        row['default_val'] = param['default_val']

        row['min_val'] = StringVar(value=param['min_val'])
        min_val_entry = Entry(table_frame, textvariable=row['min_val'], width=10)
        min_val_entry.grid(row=i+1, column=3, padx=5, pady=5)

        row['max_val'] = StringVar(value=param['max_val'])
        max_val_entry = Entry(table_frame, textvariable=row['max_val'], width=10)
        max_val_entry.grid(row=i+1, column=4, padx=5, pady=5)

        row['interval'] = StringVar(value=param['interval'])
        interval_entry = Entry(table_frame, textvariable=row['interval'], width=10)
        interval_entry.grid(row=i+1, column=5, padx=5, pady=5)

        rows.append(row)

    parameters = None

    submit_button = Button(param_root, text="Submit", command=submit_parameters)
    submit_button.grid(row=3, column=0, columnspan=2, pady=10)

    error_label = Label(param_root, text="", fg="red")
    error_label.grid(row=4, column=0, columnspan=2) 

    param_root.mainloop()

    return parameters


def generate_labelled_dataset(folder_path):
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


def model(path, filename):
    """model: used to train the Random Forest model on the training datasets, use the model 
        to make predictions for the validation dataset, and verify the accuracy of the model. 
        Outputs the pertinent information to the user, and then allows them to decide whether 
        or not to save the model as a .pkl file."""
    X_train = generate_labelled_dataset(Path(str(path+'perturbed_datasets/')))[:, :-1]
    y_train = generate_labelled_dataset(Path(str(path+'perturbed_datasets/')))[:, -1] 
    X_val = generate_labelled_dataset(Path(str(path+'validation_datasets/')))[:, :-1]
    y_val = generate_labelled_dataset(Path(str(path+'validation_datasets/')))[:, -1]

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
        model_filename = str(filename).replace('.i', '')
        joblib.dump(rf_classifier, str(path)+model_filename+'_model.pkl')


def main():
    """main: runs all of the other functions.

    Raises
    ------
    FileNotFoundError
        If the file submission window is closed (instead of submitting path and filename)
    ValueError
        If the input file is improperly format and no parameters are found as expected
    ValueError
        If the parameter submission window is closed (instead of submitting parameter data)
    """
    path, filename = accept_file()

    if path == None or filename == None:
        raise FileNotFoundError(f"Specified path and/or input file name not found. Exiting.")

    moose_runner, moose_modifier = setup_moose_runner(path, filename)

    found_vars = moose_modifier.get_vars()
    if len(found_vars) == 0:
        raise ValueError(f"No parameters found in input file. Check input file and try again.")
    else:
        parameters = accept_parameters(found_vars)

    if parameters == None:
        raise ValueError(f"Unacceptable parameters. Exiting.")

    generate_datasets(path, parameters, moose_runner, moose_modifier)

    model(path, filename)


if __name__ == "__main__":
    main()


# Next steps:  
    # Refactor to OOP and restructure for project layout
    # Test suite using PyTest (develop as you go)
    # Improve binary classifier by:
        # Allow choice of modelling frameworks - modelling class
        # Incorporating errors - labelled_dataset function
        # Accepting multiple simultaneous perturbations - generate datasets class
        # Allowing user to define number of sensors + positions - labelled dataset function
    # Expand to multi-classifier
    # Examples + tutorials
    # Packaging for pip distribution
    # Stretch goal: more complex input files, e.g. 3D monoblock