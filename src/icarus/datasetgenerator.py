"""
DatasetGenerator module to generate unlabelled datasets for Icarus.

(c) Copyright UKAEA 2025.
"""

import random
import math
from pathlib import Path
from mooseherder import (MooseHerd,
                         InputModifier,
                         DirectoryManager,
                         SweepReader)
from icarus import MooseSetup, UserInterface


class DatasetGenerator:
    """Generate the required unlabelled datasets
    """
    def __init__(self, input_file_path: str, n_tasks: int, n_threads: int,
                 parameters: dict[str, list], num_validation_values: list[int],
                 datasets_per_ground_truth: int, output_file_path: str,
                 num_para_runs: int = 2) -> None:
        """__init__

        Parameters
        ----------
        input_file_path : str
            The path to the input file used.
        n_tasks : int
            The number of tasks for parallelisation.
        n_threads : int
            The number of threads for parallelisation.
        parameters : dict[str, list]
            Dictionary containing the name of the parameter and the list of
            values for that parameter to take for each simulation to be run.
        num_validation_values : list[int]
            Contains a list of the number of validation datasets to generate
            for each parameter.
        datasets_per_ground_truth : int
            Contains the ratio of perturbed:ground truth datasets.
        output_file_path : str
            Contains the path to the folder where the datasets and model(s)
            will be stored.
        num_para_runs : int, optional
            Number of parallel runs for running the simulations, by default 2.

        Raises
        ----------
        ValueError
            If there are no parameters entered.
        FileNotFoundError
            If any of the required output file paths don't exist.
        """
        moose_setup = MooseSetup(input_file_path, output_file_path,
                                 n_tasks, n_threads)
        self.moose_runner, self.moose_modifier = \
            moose_setup.setup_moose_runner()

        self.num_para_runs = num_para_runs

        found_vars = self.moose_modifier.get_vars()
        if parameters == {}:
            num_validation_values, parameters = \
                UserInterface().accept_parameters(found_vars)

        else:
            if len(num_validation_values) != len(parameters) \
                    or 0 in num_validation_values:
                raise ValueError(
                    "Invalid number of validation values specified")
            for param_name, param_data in parameters.items():
                if param_name not in found_vars:
                    raise ValueError(
                        f"Parameter {param_name} not in input file. Exiting.")
                if param_data[0].lower() not in ["geom", "bc", "mat_prop"]:
                    raise ValueError(
                        f"Invalid parameter class {param_data[0]}. Exiting.")
                if len(param_data[1]) < 1:
                    raise ValueError("Insufficient parameter values. Exiting.")

        if not parameters:
            raise ValueError(f"Unacceptable parameters {parameters}. \
                             Must be defined. Exiting.")

        self.parameters = parameters
        self.param_names = [key for key in parameters.keys()]
        self.param_classes = [value[0] for value in parameters.values()]
        self.param_values = [value[1] for value in parameters.values()]

        self.num_validation_values = num_validation_values

        self.datasets_per_ground_truth = datasets_per_ground_truth

        if not Path(output_file_path).exists():
            raise FileNotFoundError(f"Output file path {output_file_path} \
                                    not found. Exiting.")

        self.output_file_path = output_file_path


    def generate_ground_truths(self, base_dir: Path, num_ground_truths: int) \
            -> None:
        """Generate the ground truth by running the input file
        with no modifications, and save the results to the required base_dir
        under the sub_dir_name "ground_truth". Creates a specified number of
        ground_truth datasets

        Parameters
        ----------
        base_dir : Path
            Contains the base directory to save ground_truth sub-directory for
            running the simulations.
        num_ground_truths : list[int]
            Number of ground_truth datasets to generate.
        """
        if num_ground_truths <= 0:
            raise ValueError("Some ground truths must be generated. Exiting.")
        dir_manager = MooseSetup.setup_directory_manager(base_dir,
                                                         'ground_truth',
                                                         num_ground_truths)
        moose_vars = [[{}]] * num_ground_truths
        self.run_herd(dir_manager, moose_vars, num_ground_truths)


    def generate_dataset(self, base_dir: Path, param_name: str,
                         param_class: str, param_values: list[float]) -> None:
        """Generate the perturbed and validation datasets by running the
        input file with modifications to specified parameter, and save the
        results to the required base_dir under a sub_dir_named for the
        perturbed parameter.

        Parameters
        ----------
        base_dir : Path
            Contains the base directory to save perturbed_param sub-directory
            for running the simulations.
        param_name : str
            The name of the perturbed parameter.
        param_class : str
            The class of the perturbed parameter
            (geometry, BC or material property).
        param_values : list[float]
            List of values for the currently selected parameter.
        """
        moose_vars = []
        n_dirs = len(param_values)
        dir_manager = MooseSetup.setup_directory_manager(
            base_dir, str(f"{param_class}_{param_name}"), n_dirs)
        for param in param_values:
            moose_vars.append([{str(param_name): param}])
        self.run_herd(dir_manager, moose_vars, n_dirs)


    def generate_validation_values(self, param_values: list[float],
                                    num_validation_values: int) -> list[float]:
        """Generate a specified number of validation values
        for the selected parameter, so that the model can be tested to see if
        it can correctly determine when the parameter has been perturbed to a
        value that was not present in the training dataset.

        Parameters
        ----------
        param_values : list[float]
            List of training values for the currently selected parameter(s).
        num_validation_values : int
            Number of validation values to generate for each parameter.

        Returns
        -------
        validation_values : list[float]
            List containing a list(s) of validation values for each parameter,
            to be used to determine the value of the InputModifier for that
            parameter for the run to be saved under
            validation_datasets/param_name.
        """
        validation_values = []

        min_val = min(param_values)
        max_val = max(param_values)

        while len(validation_values) < num_validation_values:
            validation_value = random.uniform(min_val, max_val)
            if validation_value not in param_values and \
                    validation_value not in validation_values:
                validation_values.append(validation_value)

        return validation_values


    def generate_datasets(self) -> None:
        """Convenience function to run all aspects of the DatasetGenerator
        class.

        Parameters
        ----------
        num_validation_values : list[int]
            Contains a list of the number of validation datasets to generate
            for each parameter
        datasets_per_ground_truth : int
            Contains the ratio of perturbed:ground truth datasets.
        """
        perturbed_path, perturbed_vals = Path(
            str(self.output_file_path+"perturbed_datasets/")), None
        validation_path, validation_vals = Path(
            str(self.output_file_path+"validation_datasets/")), None

        paths = {perturbed_path: perturbed_vals,
                 validation_path: validation_vals}

        for name, param_class, values, n_valid in \
            zip(self.param_names, self.param_classes, self.param_values,
                self.num_validation_values):
            validation_values = self.generate_validation_values(values,
                                                                n_valid)

            paths[perturbed_path] = values
            paths[validation_path] = validation_values

            for path, vals in paths.items():
                self.generate_dataset(path, name, param_class, vals)

        for path in paths:
            num_datasets = sum(1 for d in path.iterdir() if d.is_dir())
            num_ground_truths = math.ceil(
                num_datasets/self.datasets_per_ground_truth)
            self.generate_ground_truths(path, num_ground_truths)


    def run_herd(self,  dir_manager: DirectoryManager,
                 moose_vars: InputModifier, n_para: int = 1,
                 keep_flag: bool = False) -> None:
        """Run parametric sweeps of simulation chains in
        parallel with configurable parallelisation options.

        Takes a list of SimRunner objects and a corresponding list of
        InputModifiers to insert the variables into the input scripts for the
        SimRunners. Will first call all InputModifiers in the specified order
        and then call run on all the SimRunners in order. Uses the
        DirectoryManager class to log the directories in which each parallel
        worker is creating input files and running simulations. Uses the
        SweepReader class to read the output from one or more calls to
        mooseherd.run_para().

        Parameters
        ----------
        dir_manager : DirectoryManager
            Used to control how many and which directories are used to run the
            simulations.
        moose_vars : InputModifier
            Used to extract and modify the variables in the input file.
            Specifies the comment character. Variable definition blocks should
            begin #comment character#* and end #comment character#**
        n_para : int, optional
            Sets the number of simulation chains to run in parallel,
            by default 1.
        keep_flag : bool, optional
            Flag used for allowing multiple calls to run to keep everything or
            to overwrite each time, by default False - overwrite inputs and
            outputs with multiple calls
        """

        herd = MooseHerd([self.moose_runner],
                         [self.moose_modifier],
                         dir_manager)
        herd.set_num_para_sims(n_para=n_para)
        herd.set_keep_flag(keep_flag)
        for _ in range(self.num_para_runs):
            herd.run_para(moose_vars)

        sweep_reader = SweepReader(dir_manager, num_para_read=4)
        sweep_reader.read_all_output_keys()
        sweep_reader.read_results_para()
