from pathlib import Path
from itertools import product
from mooseherder import (MooseHerd,
                         MooseRunner,
                         MooseConfig,
                         InputModifier,
                         DirectoryManager,
                         SweepReader)

NUM_PARA_RUNS = 2
USER_DIR = Path.home()

def main(param_names, param_values) -> None:
    print('Start minimal full functionality example') 
    print(param_values)


    moose_input = Path('2d_plate/plate_2d_thermal.i')
    moose_modifier = InputModifier(moose_input,'#','')

    moose_config = MooseConfig().read_config(Path.cwd() / 'moose-config.json')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                              n_threads = 2,
                              redirect_out = False)

    n_dirs = 1
    for i in range(len(param_names)):
        n_dirs *= len(param_values[i])
    dir_manager = DirectoryManager(n_dirs=n_dirs)

    herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)
    herd.set_num_para_sims(n_para=n_dirs)
    herd.set_keep_flag(False)

    dir_manager.set_base_dir(Path('2d_plate/validation_datasets/'))
    dir_manager.set_sub_dir_name(str(param_names[0]))
    dir_manager.clear_dirs()
    dir_manager.create_dirs()

    '''
    param_combinations = product(*param_values)
    moose_vars = []
    for combination in param_combinations:
        params = {param_names[i]: combination[i] for i in range(len(param_names))}
        moose_vars.append([params])
    '''

    moose_vars = []
    for param in param_values[0]:
        moose_vars.append([{str(param_names[0]):param}])

    for _ in range(NUM_PARA_RUNS):
        herd.run_para(moose_vars)

    sweep_reader = SweepReader(dir_manager,num_para_read=4)
    sweep_reader.read_all_output_keys()
    read_all = sweep_reader.read_results_para()

    print('Finished.')


if __name__ == '__main__':
    param_names = [["xmax"], ["ymax"], ["init_temp"], ["max_temp"], 
                   ["thermal_conductivity"], ["specific_heat"], ["prop_values"]]
    param_values = [[[8,22]],
                    [[5,9]],
                    [[25,45]],
                    [[300,900]],
                    [[18,52]], 
                    [[1.25,2.75]],
                    [[3000,11000]]]
    for i in range(len(param_names)):
        main(param_names[i], param_values[i])

