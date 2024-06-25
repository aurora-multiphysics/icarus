'''
==============================================================================
EXAMPLE: Run MOOSE in sequential then parallel mode with mooseherder

'''



import torch
from pathlib import Path
from mooseherder import (MooseHerd,
                         MooseRunner,
                         InputModifier,
                         DirectoryManager,
                         MooseConfig)
import random


def generate_values(shc_ground_truth=406.0, 
                    heat_flux_ground_truth=500.0e3,
                    p=0.57,
                    n_dims=2, 
                    n_samples=30):
    """
    generate_values: generate perturbed values using sobol sampling.\n 
    Args:
      shc_ground truth (Int): ground truth value for specific heat capacity
      thermal_c_ground truth
    
    Returns:
      Self: returns a list of lists with first sublist being shc values, and second sublist being thermal_c.
    """
    soboleng = torch.quasirandom.SobolEngine(n_dims)
    sobol_samples = soboleng.draw(n_samples)

    shc_samples = (shc_ground_truth * (1 + (2 * sobol_samples[:, 0] - 1) * p)).tolist()
    heat_flux_samples = (heat_flux_ground_truth * (1 + (2 * sobol_samples[:, 1] - 1) * p)).tolist()

    random.shuffle(shc_samples)
    random.shuffle(heat_flux_samples)

    return (shc_samples, heat_flux_samples)



def main():
    """main: run moose once, sequential then parallel.
    """
    print("-"*80)
    print('EXAMPLE: Herd Setup')
    print("-"*80)

    config_path = Path.cwd() / 'moose-config.json'
    moose_config = MooseConfig().read_config(config_path)
    moose_input = Path(Path.cwd(), 'ground_truth_thermal_sim.i')

    moose_modifier = InputModifier(moose_input,'#','')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 4,
                          redirect_out = True)

    dir_manager = DirectoryManager(n_dirs=1)

    # Start the herd and create working directories
    herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)

    # Set the parallelisation options, we have 100 combinations of variables and
    # 10 MOOSE intances running, so 10 runs will be saved in each working directory
    herd.set_num_para_sims(n_para=10)

     # Send all the output to the examples directory and clear out old output
    dir_manager.set_base_dir(Path('dataset/'))
    dir_manager.clear_dirs()
    dir_manager.create_dirs()

    # Create variables to sweep in a list of dictionaries, 8 combinations possible.
    specific_heat = generate_values()[0]
    surface_heat_flux = generate_values()[1]
    moose_vars = list([])
    for sh in specific_heat:
        for hf in surface_heat_flux:
            # Needs to be list[list[dict]] - outer list is simulation iteration,
            # inner list is what is passed to each runner/inputmodifier
            moose_vars.append([{'cuSpecHeat':sh,'surfHeatFlux':hf}])

    print('Herd sweep variables:')
    for vv in moose_vars:
        print(vv)


    # print("-"*80)
    print('EXAMPLE: Run MOOSE in parallel')
    print("-"*80)

    # # Run all variable combinations across 4 MOOSE instances with two runs saved in
    # # each moose-workdir
    herd.run_para(moose_vars)

    print(f'Run time (para) = {herd.get_sweep_time():.3f} seconds')
    print("-"*80)
    print()


if __name__ == '__main__':
    main()

