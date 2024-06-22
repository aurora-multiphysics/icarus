

import torch
from pathlib import Path
from mooseherder import (MooseHerd,
                         MooseRunner,
                         InputModifier,
                         DirectoryManager,
                         MooseConfig,
                         SweepReader)
import random


def generate_values(initial_condition=293.15,
                    p=0.7,
                    n_dims=1, 
                    n_samples=10):
    """
    generate_values: generate perturbed values using Sobol sampling.\n 
    Args:
      initial_condition (Float): ground truth value for initial condition
    
    Returns:
      Self: returns a list of perturbed initial condition values.
    """
    soboleng = torch.quasirandom.SobolEngine(n_dims)
    sobol_samples = soboleng.draw(n_samples)

    initial_condition_samples = (initial_condition * (1 + (2 * sobol_samples[:, 0] - 1) * p)).tolist()

    random.shuffle(initial_condition_samples)


    return initial_condition_samples


def main():
    """main: run moose once, sequential then parallel.
    """
    print("-"*80)
    print('EXAMPLE: Herd Setup')
    print("-"*80)

    config_path = Path.cwd() / 'moose-config.json'
    moose_config = MooseConfig().read_config(config_path)
    moose_input = Path(Path.cwd(), 'thermal_model_2.i')

    moose_modifier = InputModifier(moose_input,'#','')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 3,
                          redirect_out = True)

    dir_manager = DirectoryManager(n_dirs=1)

    # Start the herd and create working directories
    herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)

    # Set the parallelisation options, we have 10 combinations of variables and
    # 5 MOOSE intances running, so ?? runs will be saved in each working directory
    herd.set_num_para_sims(n_para=5)

     # Send all the output to the examples directory and clear out old output
    dir_manager.set_base_dir(Path('ground_truth_sims/'))
    dir_manager.clear_dirs()
    dir_manager.create_dirs()

    # Create variables to sweep in a list of dictionaries, 8 combinations possible.
    initial_conditions = generate_values()
    moose_vars = list([])
    for ic in initial_conditions:
        # Needs to be list[list[dict]] - outer list is simulation iteration,
        # inner list is what is passed to each runner/inputmodifier
        moose_vars.append([{'initial_condition':ic}])

    print('Herd sweep variables:')
    for vv in moose_vars:
        print(vv)


    print('EXAMPLE: Run MOOSE in parallel')
    print("-"*80)

    # # Run all variable combinations across 4 MOOSE instances with two runs saved in
    # # each moose-workdir
    herd.run_para(moose_vars)

    print(f'Run time (para) = {herd.get_sweep_time():.3f} seconds')
    print("-"*80)
    print()

    sweep_reader = SweepReader(dir_manager,num_para_read=4)
    sweep_reader.read_all_output_keys()
    read_all = sweep_reader.read_results_para()
    # print(read_all)

if __name__ == '__main__':
    main()

