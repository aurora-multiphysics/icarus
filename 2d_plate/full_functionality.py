from pathlib import Path
import random
import math
import numpy as np
import pyvale
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from mooseherder import (MooseHerd,
                         MooseRunner,
                         MooseConfig,
                         InputModifier,
                         DirectoryManager,
                         SweepReader,
                         ExodusReader)


NUM_PARA_RUNS = 1
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
        run_herd(moose_runner, moose_modifier, dir_manager, [[{}]], 1)


def generate_perturbed_datasets(moose_runner, moose_modifier, base_dir, param_names, param_values):
    moose_vars = list([])
    n_dirs = 1
    for i in range(len(param_names)):
        n_dirs *= len(param_values[i])
    dir_manager = setup_directory_manager(base_dir, str(param_names[0]), n_dirs)
    for param in param_values[0]:
        moose_vars.append([{str(param_names[0]): param}]) 
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
        validation_values[0].append(validation_value)  

    return validation_values


def generate_validation_datasets(moose_runner, moose_modifier, base_dir, param_names, validation_values):
    moose_vars = list([])
    n_dirs = 1
    for i in range(len(param_names)):
        n_dirs *= len(validation_values[i])
    dir_manager = setup_directory_manager(base_dir, str(param_names[0]), n_dirs)
    for param in validation_values[0]:
        moose_vars.append([{str(param_names[0]): param}]) 
    run_herd(moose_runner, moose_modifier, dir_manager, moose_vars, n_dirs)


def generate_datasets(path, filename):
    moose_runner, moose_modifier = setup_moose_runner(path, filename)

    parameters = moose_modifier.get_vars()
    parameter_names = [[key] for key in parameters.keys()]
    default_values = [value for value in parameters.values()]

    param_names = [["max_temp"], ["init_temp"], ["xmax"], ["ymax"], ["thermal_conductivity"],
                   ["specific_heat"], ["prop_values"]]
    param_values = [[[750,1000,1250,1500,1750]],[[30,40,50,60,70]],[[20,30,40,50,60]],
                    [[4,6,8,10,12]],[[55,65,75,85,95]],[[1,1.5,2,2.5,3]],[[12000,16000,20000,24000,28000]]]
    
    '''
    for i in range (len(parameter_names)):
        parameter_name = parameter_names[i]
        default_value = default_values[i]
        include = input('Include '+str(parameter_name)+'? (Y/N) ')
        if include.upper() == 'Y':
            print('Default value:', str(default_value))
            param_names.append([parameter_name])
            parameter_values = input('Enter values for '+str(parameter_name)+' separated by commas ').split(",")
            if default_value in parameter_values:
                parameter_values.remove(default_value)
            param_values.append([parameter_values])
    '''

    for i in range(len(param_names)):
        generate_ground_truths(moose_runner, moose_modifier, Path(str(path+'perturbed_datasets/')), param_values[i])
        generate_perturbed_datasets(moose_runner, moose_modifier, Path(str(path+'perturbed_datasets/')), param_names[i], param_values[i])

        validation_values = generate_validation_values(param_values[i])
        generate_validation_datasets(moose_runner, moose_modifier, Path(str(path+'validation_datasets/')), param_names[i], validation_values)
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


def model(path, filename):
    generate_datasets(path, filename)

    X_train = generate_labelled_dataset(Path(str(path+'perturbed_datasets/')))[:, :-1]
    y_train = generate_labelled_dataset(Path(str(path+'perturbed_datasets/')))[:, -1] 
    X_val = generate_labelled_dataset(Path(str(path+'validation_datasets/')))[:, :-1]
    y_val = generate_labelled_dataset(Path(str(path+'validation_datasets/')))[:, -1]

    # Need to delete unlabelled datasets

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_val)

    for i in range(len(X_val)):
        print(f"Row {i + 1}: Actual Label = {y_val[i]}, Predicted Label = {y_pred[i]}")

    val_accuracy = accuracy_score(y_val, y_pred)
    print(f"Random Forest Validation Accuracy: {val_accuracy * 100:.2f}%")


#path = input('Enter the desired path ')
#filename = input('Enter the input file name ')
path = "2d_plate/"
filename = "plate_2d_thermal.i"
if __name__ == "__main__":
    model(path, filename)