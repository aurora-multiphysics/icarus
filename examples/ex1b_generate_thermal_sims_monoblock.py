'''
================================================================================
Example: Generate Perturbed 3D Monoblock Temperature Field Simulations using MOOSEHERDER and Stratified Sobol Sampling of parameter space.

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
'''


from pathlib import Path
import os
from icarus import SampleGenerator
from mooseherder import (MooseHerd,
                         MooseRunner,
                         InputModifier,
                         DirectoryManager,
                         MooseConfig)


THERMAL_COND = 384.0
HEAT_FLUX = 10.0e6

dataset = SampleGenerator(n_samples=8, p_factor=0.8, tolerance=0.001, thermal_cond_base=THERMAL_COND, heat_flux_base=HEAT_FLUX) 



def main():
    """main: run moose once, sequential then parallel.
    """
    print("-"*80)
    print('EXAMPLE: Herd Setup')
    print("-"*80)

    config_path = Path.cwd() / 'example-moose-config.json'
    moose_config = MooseConfig().read_config(config_path)
    moose_input = Path(Path.cwd(), 'example_monoblock.i')

    moose_modifier = InputModifier(moose_input,'#','')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 4,
                          redirect_out = True)

    dir_manager = DirectoryManager(n_dirs=1)

    # Start the herd and create working directories
    herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)

    # Set the parallelisation options
    herd.set_num_para_sims(n_para=10)

    dataset_dir = Path(Path.cwd() / 'monoblock_dataset/')
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
    
    dir_manager.set_base_dir(dataset_dir)
    dir_manager.clear_dirs()
    dir_manager.create_dirs()

    # Create variables to sweep in a list of dictionaries
    moose_vars = list([])
    thermal_cond, surface_heat_flux = dataset.generate_stratified_samples()
    assert len(thermal_cond) == len(surface_heat_flux), "Mismatched lengths in generated samples"
    for tc,hf in zip(thermal_cond, surface_heat_flux):    
            moose_vars.append([{'cuThermCond':tc,'surfHeatFlux':hf}])

    print('Herd sweep variables:')
    for vv in moose_vars:
        print(vv)

    print('EXAMPLE: Run MOOSE in parallel')
    print("-"*80)

    herd.run_para(moose_vars)

    print(f'Run time (para) = {herd.get_sweep_time():.3f} seconds')
    print("-"*80)
    print()


if __name__ == '__main__':
    main()
