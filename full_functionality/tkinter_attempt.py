from pathlib import Path
import random
import math
import numpy as np
import pyvale
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tkinter import *
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
    moose_input = Path(str(path + filename))
    moose_modifier = InputModifier(moose_input, '#', '')
    moose_config = MooseConfig().read_config(Path.cwd() / 'moose-config.json')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks=1, n_threads=2, redirect_out=False)
    return moose_runner, moose_modifier


def setup_directory_manager(base_dir, sub_dir_name, n_dirs):
    dir_manager = DirectoryManager(n_dirs=n_dirs)
    dir_manager.set_base_dir(base_dir)
    dir_manager.set_sub_dir_name(sub_dir_name)
    dir_manager.clear_dirs()
    dir_manager.create_dirs()
    return dir_manager


def run_herd(moose_runner, moose_modifier, dir_manager, moose_vars, n_para, keep_flag=False):
    herd = MooseHerd([moose_runner], [moose_modifier], dir_manager)
    herd.set_num_para_sims(n_para=n_para)
    herd.set_keep_flag(keep_flag)
    for _ in range(NUM_PARA_RUNS):
        herd.run_para(moose_vars)

    sweep_reader = SweepReader(dir_manager, num_para_read=4)
    sweep_reader.read_all_output_keys()
    read_all = sweep_reader.read_results_para()


def generate_ground_truths(moose_runner, moose_modifier, base_dir, param_values):
    for i in range(math.ceil(len(param_values[0]) / 5)):
        dir_manager = setup_directory_manager(base_dir, 'ground_truth', 1)
        run_herd(moose_runner, moose_modifier, dir_manager, [{}], 1)


def generate_perturbed_datasets(moose_runner, moose_modifier, base_dir, param_names, param_values):
    n_dirs = 1
    for i in range(len(param_names)):
        n_dirs *= len(param_values[i])
    dir_manager = setup_directory_manager(base_dir, str(param_names[0]), n_dirs)
    moose_vars = [{str(param_names[0]): param} for param in param_values[0]]
    run_herd(moose_runner, moose_modifier, dir_manager, moose_vars, n_dirs)


def generate_validation_values(param_values, num_validation_values=2):
    validation_values = [[]]
    for i in range(num_validation_values):
        distinct_val = False
        while not distinct_val:
            validation_value = random.uniform(min([val for sublist in param_values for val in sublist]),
                                             max([val for sublist in param_values for val in sublist]))
            if validation_value not in param_values and validation_value not in validation_values:
                distinct_val = True
        validation_values.append(validation_value)  

    return validation_values


def generate_validation_datasets(moose_runner, moose_modifier, base_dir, param_names, validation_values):
    n_dirs = 1
    for i in range(len(param_names)):
        n_dirs *= len(validation_values[i])
    dir_manager = setup_directory_manager(base_dir, str(param_names[0]), n_dirs)
    moose_vars = [{str(param_names[0]): param} for param in validation_values[0]]
    run_herd(moose_runner, moose_modifier, dir_manager, moose_vars, n_dirs)


def generate_datasets(path, filename, parameters, moose_runner, moose_modifier):
    param_names = [[key] for key in parameters.keys()]
    param_values = [[value] for value in parameters.values()]

    for i in range(len(param_names)):
        param_names = param_names[i]
        param_values = param_values[i]

        generate_ground_truths(moose_runner, moose_modifier, Path(str(path+'perturbed_datasets/')), param_values)
        generate_perturbed_datasets(moose_runner, moose_modifier, Path(str(path+'perturbed_datasets/')), param_names, param_values)

        validation_values = generate_validation_values(param_values)
        generate_validation_datasets(moose_runner, moose_modifier, Path(str(path+'validation_datasets/')), param_names, validation_values)
        generate_ground_truths(moose_runner, moose_modifier, Path(str(path+'validation_datasets/')), validation_values)


def generate_labelled_dataset(folder_path):
    labelled_dataset = np.empty((0, 7))

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

        measurements = np.append(measurements, label) 
        labelled_dataset = np.vstack([labelled_dataset, measurements]) 

    return labelled_dataset


def accept_file():
    
    def submit_file():
        nonlocal path, filename
        path = str(file_path.get())
        filename = str(file_name.get())
        file_root.destroy()

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

    file_root.mainloop()    

    return path, filename 


def accept_parameters(parameters):

    def submit_parameters():
        nonlocal parameters
        parameters = {}
        
        for row in rows:
            if row['checkbox_var'].get():  # Check if the checkbox is ticked
                param_name = row['param_name']
                default_val = float(row['default_val'])
                min_val = float(row['min_val'].get())
                max_val = float(row['max_val'].get())
                interval = float(row['interval'].get())
                
                param_values = []
                current_val = min_val
                while current_val <= max_val:
                    if current_val != default_val:
                        param_values.append(current_val)
                        current_val += interval

                parameters[param_name] = param_values

        print("Destroy")
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

        row['checkbox_var'] = BooleanVar(value=True)

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

    submit_button = Button(param_root, text="Submit", command=submit_parameters)
    submit_button.grid(row=3, column=0, columnspan=2, pady=10)

    param_root.mainloop()

    return parameters


def model():
    path, filename = accept_file()

    moose_runner, moose_modifier = setup_moose_runner(path, filename)

    parameters = accept_parameters(moose_modifier.get_vars())

    generate_datasets(path, filename, parameters, moose_runner, moose_modifier)

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


if __name__ == "__main__":
    model()


# Next steps:
    # Debug multiprocessing error
    # Delete unlabelled datasets
    # Explain + enforce suitable user inputs 
    # Refine valid/invalid classifier model (incorporating errors, etc)
    # Expand to multi-classification model
    # Expand to non-thermal solves (eg mechanical)
    # Testing, examples + tutorials, packaging, etc