import random
import math 
from pathlib import Path
from icarus import MooseSetup
from mooseherder import (MooseHerd,
                         SweepReader)

class DatasetGenerator:
    def __init__(self, moose_runner, moose_modifier):
        self.moose_runner = moose_runner
        self.moose_modifier = moose_modifier
    
    def generate_ground_truths(self, moose_runner, moose_modifier, base_dir, num_ground_truths):
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
        num_ground_truth : list[float]
            Number of ground_truth datasets to generate.
        """
        moose_vars = list([])
        dir_manager = MooseSetup.setup_directory_manager(base_dir, 'ground_truth', num_ground_truths)
        for i in range(num_ground_truths):
            moose_vars.append([{}])
        self.run_herd(moose_runner, moose_modifier, dir_manager, moose_vars, num_ground_truths)


    def generate_dataset(self, moose_runner, moose_modifier, base_dir, param_names, param_values):
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
            Contains the base directory to save perturbed_param sub-directory for running the 
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
        dir_manager = MooseSetup.setup_directory_manager(base_dir, str(param_names[0]), n_dirs)
        for param in param_values[0]:
            moose_vars.append([{str(param_names[0]): param}]) 
        self.run_herd(moose_runner, moose_modifier, dir_manager, moose_vars, n_dirs)


    def generate_validation_values(self, param_values, num_validation_values=2):
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


    def generate_datasets(self, output_file_path, parameters, num_validation_values, moose_runner, moose_modifier):
        """generate_datasets: used to generate the unlabelled ground truth, perturbed, and 
            validation datasets by running the required functions with the necessary parameter
            names and values, and save paths.

        Parameters
        ----------
        output_file_path : str
            Contains the path to the folder where the datasets and model(s) will be stored.
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

        perturbed_path, perturbed_vals = Path(str(output_file_path+"perturbed_datasets/")), None
        validation_path, validation_vals = Path(str(output_file_path+"validation_datasets/")), None
        paths = {perturbed_path : perturbed_vals, validation_path: validation_vals}

        for i in range(len(param_names)):
            validation_values = self.generate_validation_values(param_values[i], num_validation_values[i])
            paths[perturbed_path] = param_values[i]
            paths[validation_path] = validation_values
            for path, values in paths.items():
                self.generate_dataset(moose_runner, moose_modifier, path, param_names[i], values)

        for path in paths.keys():
            num_datasets = sum(1 for d in path.iterdir() if d.is_dir())
            num_ground_truths = math.ceil(num_datasets/3)
            self.generate_ground_truths(moose_runner, moose_modifier, path, num_ground_truths)


    def run_herd(self, moose_runner, moose_modifier, dir_manager, moose_vars, n_para=1, keep_flag=False):
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
        for _ in range(NUM_PARA_RUNS=2):
            herd.run_para(moose_vars)

        sweep_reader = SweepReader(dir_manager, num_para_read=4)
        sweep_reader.read_all_output_keys()
        read_all = sweep_reader.read_results_para()