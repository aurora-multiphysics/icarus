from pathlib import Path
from mooseherder import (MooseRunner,
                         MooseConfig,
                         InputModifier,
                         DirectoryManager)

class MooseSetup:
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.moose_runner, self.moose_modifier = self.setup_moose_runner(input_file_path)

    def setup_moose_runner(self, input_file_path):
        """setup_moose_runner: Constructor for MOOSE runner taking a MooseConfig object
            that contains the paths to the main MOOSE install, the MOOSE app and
            the MOOSE app name. Sets parallelisation options to 1 task
            and 2 threads. Sets environment variables required for MPI setup.

        Parameters
        ----------
        input_file_path : str
            Contains the path to the  input file.

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
        moose_input = Path(str(input_file_path))
        moose_modifier = InputModifier(moose_input, '#', '')
        moose_config = MooseConfig().read_config(Path.cwd() / 'moose-config.json')
        moose_runner = MooseRunner(moose_config)
        moose_runner.set_run_opts(n_tasks=1, n_threads=2, redirect_out=False)
        return moose_runner, moose_modifier


    def setup_directory_manager(self, base_dir, sub_dir_name, n_dirs = 1):
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