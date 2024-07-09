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



import torch
import random

N_SAMPLES =100

#Mooseherder can generate 125 sims per min 

def generate_values(thermal_cond_ground_truth=384, 
                    heat_flux_ground_truth=500.0e3,
                    p=0.8,
                    n_samples=N_SAMPLES):
    """
    generate_values: generate perturbed values using sobol sampling with balanced classes.
    
    Args:
      thermal_cond_ground_truth (float): ground truth value for thermal conductivity
      heat_flux_ground_truth (float): ground truth value for surface heat flux
      p (float): perturbation factor (as a decimal)
      n_samples (int): the square root of number of samples to generate
    
    Returns:
      tuple: returns two lists, one for thermal_cond values and one for heat flux values
    """
    # Ensure n_samples is divisible by 4 for balanced classes
    n_samples = (n_samples // 4) * 4
    
    soboleng = torch.quasirandom.SobolEngine(2)
    sobol_samples = soboleng.draw(n_samples)

    thermal_cond_samples = []
    heat_flux_samples = []

    for i in range(n_samples):
        class_type = i % 4  # This ensures an even distribution across 4 classes

        if class_type == 0:  # Both unchanged
            thermal_cond = thermal_cond_ground_truth
            heat_flux = heat_flux_ground_truth
        elif class_type == 1:  # Only thermal_c changed
            thermal_cond = float(thermal_cond_ground_truth * (1 + (2 * sobol_samples[i, 0] - 1) * p))
            heat_flux = heat_flux_ground_truth
        elif class_type == 2:  # Only heat flux changed
            thermal_cond = thermal_cond_ground_truth
            heat_flux = float(heat_flux_ground_truth * (1 + (2 * sobol_samples[i, 1] - 1) * p))
        else:  # Both changed
            thermal_cond = float(thermal_cond_ground_truth * (1 + (2 * sobol_samples[i, 0] - 1) * p))
            heat_flux = float(heat_flux_ground_truth * (1 + (2 * sobol_samples[i, 1] - 1) * p))

        thermal_cond_samples.append(thermal_cond)
        heat_flux_samples.append(heat_flux)

    # Shuffle the samples to avoid any pattern
    combined = list(zip(thermal_cond_samples, heat_flux_samples))
    random.shuffle(combined)
    thermal_cond_samples, heat_flux_samples = zip(*combined)

    return list(thermal_cond_samples), list(heat_flux_samples)



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
    thermal_cond = generate_values()[0]
    surface_heat_flux = generate_values()[1]
    moose_vars = list([])
    for tc in thermal_cond:
        for hf in surface_heat_flux:
            # Needs to be list[list[dict]] - outer list is simulation iteration,
            # inner list is what is passed to each runner/inputmodifier
            moose_vars.append([{'cuThermCond':tc,'surfHeatFlux':hf}])

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

