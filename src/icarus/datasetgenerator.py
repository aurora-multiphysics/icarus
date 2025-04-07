import random
import math 
from pathlib import Path
from icarus import MooseSetup
from mooseherder import (MooseHerd,
                         MooseRunner,
                         InputModifier,
                         DirectoryManager,
                         SweepReader) 

class DatasetGenerator:
    """Used to generate the unlabelled ground truth, perturbed, and 
        validation datasets by running the required functions with the necessary parameter
        names and values, and save paths.

        NB: To run the input file, mooseherder requires that the moose_vars be structured as 
        [[{param_name: [param_values]}], [{param_name: param_values}], ...] for each parameter.
    """
    def __init__(self, moose_runner: MooseRunner, moose_modifier: InputModifier, 
                 parameters: dict[str, list], output_file_path: str, num_para_runs: int=2) -> None:
        """__init__

        Parameters
        ----------
        moose_runner : MooseRunner
            Constructed MOOSE runner used to run the input file with modified variables 
        moose_modifier : InputModifier
            Used to extract and modify the variables in the input file.
            Specifies the comment character. Variable definition blocks should begin 
            #comment character#* and end #comment character#**, e.g. #_* and #** for
            moose.
        parameters : dict[str, list]
            Dictionary containing the name of the parameter and the list of values for
            that parameter to take for each simulation to be run.
        output_file_path : str
            Contains the path to the folder where the datasets and model(s) will be stored.
        num_para_runs : int, optional
            Number of parallel runs for running the simulations, by default 2.

        Raises 
        ----------
        ValueError
            If there are no parameters entered.
        FileNotFoundError
            If any of the required output file paths don't exist.
        """
        self.moose_runner = moose_runner
        self.moose_modifier = moose_modifier
        self.num_para_runs = num_para_runs

        if parameters == None or len(parameters) == 0:            
            raise ValueError(f"Unacceptable parameters. Exiting.")
        
        self.parameters = parameters
        self.param_names = [key for key in parameters.keys()]
        self.param_classes = [value[0] for value in parameters.values()]
        self.param_values = [value[1] for value in parameters.values()]

        if not Path(output_file_path).exists() or \
            not Path(str(output_file_path+"perturbed_datasets/")).exists() or \
            not Path(str(output_file_path+"validation_datasets/")).exists():
            raise FileNotFoundError(f"At least one required output file path not found. Exiting.")
        
        self.output_file_path = output_file_path 
    
    
    def generate_ground_truths(self, base_dir: Path, num_ground_truths: list[int]) -> None:
        """generate_ground_truths: used to generate the ground truth by running the input file
            with no modifications, and save the results to the required base_dir under the 
            sub_dir_name "ground_truth". Creates a specified number of ground_truth datasets

        Parameters
        ----------
        base_dir : Path
            Contains the base directory to save ground_truth sub-directory for running the 
            simulations. 
        num_ground_truths : list[int]
            Number of ground_truth datasets to generate.
        """
        if num_ground_truths <= 0:
            raise ValueError(f"Some ground truths must be generated. Exiting.")
        dir_manager = MooseSetup.setup_directory_manager(base_dir, 'ground_truth', num_ground_truths)
        moose_vars = [[{}]] * num_ground_truths
        self.run_herd(dir_manager, moose_vars, num_ground_truths)


    def generate_dataset(self, base_dir: Path, param_name: str, param_class: str, param_values: list[float]) -> None:
        """generate_dataset: used to generate the perturbed and validation datasets by running the 
            input file with modifications to specified parameter, and save the results to the 
            required base_dir under a sub_dir_named for the perturbed parameter.

        Parameters
        ----------
        base_dir : Path
            Contains the base directory to save perturbed_param sub-directory for running the 
            simulations. 
        param_name : str
            The name of the perturbed parameter.
        param_class : str
            The class of the perturbed parameter (geometry, BC or material property).
        param_values : list[float]
            List of values for the currently selected parameter.
        """
        moose_vars = []
        n_dirs = len(param_values)
        dir_manager = MooseSetup.setup_directory_manager(base_dir, str(f"{param_class}_{param_name}"), n_dirs)
        for param in param_values:
            moose_vars.append([{str(param_name): param}])
        self.run_herd(dir_manager, moose_vars, n_dirs)


    def generate_validation_values(self, param_values: list[float], num_validation_values: int=2) -> list[float]:
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
        validation_values : list[float]
            List containing a list(s) of validation values for each parameter, to be used to
            determine the value of the InputModifier for that parameter for the run to be saved
            under validation_datasets/param_name
        """
        validation_values = []

        min_val = min(param_values)
        max_val = max(param_values)
        
        while len(validation_values) < num_validation_values:
            validation_value = random.uniform(min_val, max_val)
            if validation_value not in param_values and validation_value not in validation_values:
                validation_values.append(validation_value)

        return validation_values


    def generate_datasets(self, num_validation_values: list[int], ground_truths_per_dataset: int) -> None:
        """generate_datasets: convenience function  to run all aspects of the DatasetGenerator class.

        Parameters
        ----------
        num_validation_values : list[int]
            Contains a list of the number of validation datasets to generate for each parameter
        ground_truth_per_dataset : int
            Contains the number of ground truth datasets to include per perturbed dataset
        """
        perturbed_path, perturbed_vals = Path(str(self.output_file_path+"perturbed_datasets/")), None
        validation_path, validation_vals = Path(str(self.output_file_path+"validation_datasets/")), None
        
        paths = {perturbed_path: perturbed_vals, validation_path: validation_vals}

        for name, param_class, values, n_valid in zip(self.param_names, self.param_classes, self.param_values, num_validation_values):
            validation_values = self.generate_validation_values(values, n_valid)
            
            paths[perturbed_path] = values
            paths[validation_path] = validation_values

            for path, vals in paths.items():
                self.generate_dataset(path, name, param_class, vals)

        for path in paths.keys():
            num_datasets = sum(1 for d in path.iterdir() if d.is_dir())
            num_ground_truths = math.ceil(num_datasets/ground_truths_per_dataset)
            self.generate_ground_truths(path, num_ground_truths)


    def run_herd(self,  dir_manager: DirectoryManager, 
                 moose_vars: InputModifier, n_para: int=1, keep_flag: bool=False) -> None:
        """run_herd: used to run parametric sweeps of simulation chains in
            parallel with configurable parallelisation options. Takes a list of
            SimRunner objects and a corresponding list of InputModifiers to insert the
            variables into the input scripts for the SimRunners. Will first call all InputModifiers 
            in the specified order and then call run on all the SimRunners in order. Uses the 
            DirectoryManager class to log the directories in which each parallel worker is
            creating input files and running simulations. Uses the SweepReader class to read the 
            output from one or more calls to mooseherd.run_para().

        Parameters
        ----------
        dir_manager : DirectoryManager
            Used to control how many and which directories are used to run the simulations.
        moose_vars : InputModifier
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
        
        herd = MooseHerd([self.moose_runner], [self.moose_modifier], dir_manager)
        herd.set_num_para_sims(n_para=n_para)
        herd.set_keep_flag(keep_flag)
        for _ in range(self.num_para_runs):
            herd.run_para(moose_vars)

        sweep_reader = SweepReader(dir_manager, num_para_read=4)
        sweep_reader.read_all_output_keys()
        read_all = sweep_reader.read_results_para()