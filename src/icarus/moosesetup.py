"""
MooseSetup module to carry out the setup required for Icarus.

(c) Copyright UKAEA 2025.
"""

from pathlib import Path
import os
from mooseherder import (MooseRunner,
                         MooseConfig,
                         InputModifier,
                         DirectoryManager)


class MooseSetup:
    """Setup the file system, MOOSE Runners and Directory Managers required to
    run the simulation and generate the datasets.
    """
    def __init__(self, input_file_path: str, output_file_path: str,
                 n_tasks: int, n_threads: int) -> None:
        """__init__

        Parameters
        ----------
        input_file_path : str
            The path to the input file used.
        output_file_path : str
            The path to where the outputs will be stored.
        n_tasks : int
            The number of tasks for parallelisation.
        n_threads : int
            The number of threads for parallelisation.
        """
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.n_tasks, self.n_threads = n_tasks, n_threads

        config_file_path = Path.cwd()/'moose-config.json'
        if not os.path.exists(config_file_path):
            with open(config_file_path, "w", encoding="utf-8") as file:
                file.write("""
                    {
                        "main_path": "path/to/projects/moose",
                        "app_path": "path/to/projects/babbler",
                        "app_name": "babbler-opt"
                    }
                """)

        for file_path in [Path(self.output_file_path+"perturbed_datasets"),
                          Path(self.output_file_path+"validation_datasets")]:
            os.makedirs(file_path, exist_ok=True)

    def setup_moose_runner(self) -> tuple[MooseRunner, InputModifier]:
        """Constructor for MOOSE runner taking a MooseConfig object
        that contains the paths to the main MOOSE install, the MOOSE app and
        the MOOSE app name.

        Raises
        ----------
        FileNotFoundError
            If the input file path is unacceptable (not found or not an input
            file), or if the moose-config.json file is unacceptable.
        ValueError
            If there are no parameters found in the input file.

        Returns
        ----------
        moose_runner : MooseRunner
            Constructed MOOSE runner used to run the input file with modified
            variables.
        moose_modifier : InputModifier
            Used to extract and modify the variables in the input file.
            Specifies the comment character. Variable definition blocks should
            begin #comment character#* and end #comment character#**
        """
        if self.input_file_path[-2:] != ".i" or \
                not Path(self.input_file_path).exists():
            raise FileNotFoundError(f"Unacceptable input file path {self.input_file_path}. \
                                    Must both exist and end in .i")

        moose_input = Path(self.input_file_path)

        moose_modifier = InputModifier(moose_input, '#', '')
        if len(moose_modifier.get_vars()) == 0:
            raise ValueError("No parameters found in input file. \
                             Check input file and try again.")

        try:
            moose_config = MooseConfig().read_config(
                Path.cwd()/'moose-config.json')
        except FileNotFoundError as e:
            raise FileNotFoundError("JSON file moose-config.json not found, \
                                    or points to non-existent MOOSE app.") \
                                    from e

        moose_runner = MooseRunner(moose_config)
        moose_runner.set_run_opts(n_tasks=self.n_tasks,
                                  n_threads=self.n_threads, redirect_out=False)

        return moose_runner, moose_modifier

    @staticmethod
    def setup_directory_manager(base_dir: str, sub_dir_name: str,
                                n_dirs: int = 1) -> DirectoryManager:
        """Set up directory manager to manage directories for running
        simulations in parallel with the mooseherd. Clears existing
        directories and creates specified new ones with given names

        Parameters
        ----------
        base_dir : Path
            Sets the base directory to create sub-directories for running the
            simulations. The base directory must exist.
        sub_dir_name : str
            String to be used at the start of the created sub-directores.
            Default on creation is 'sim-workdir'. Populates the list of run
            directories using the new sub directory name.
        n_dirs : int, optional
            Number of directories to be created., by default 1

        Returns
        -------
        dir_manager : DirectoryManager
            Used to control how many and which directories are used to run the
            simulations.
        """
        dir_manager = DirectoryManager(n_dirs=n_dirs)
        dir_manager.set_base_dir(base_dir)
        dir_manager.set_sub_dir_name(sub_dir_name)
        dir_manager.clear_dirs()
        dir_manager.create_dirs()

        return dir_manager
