from pathlib import Path
from mooseherder import (MooseHerd,
                         MooseRunner,
                         MooseConfig,
                         InputModifier,
                         DirectoryManager,
                         SweepReader)

NUM_PARA_RUNS = 2
USER_DIR = Path.home()

def main() -> None:
    print('Start minimal full functionality example')

    moose_input = Path('monoblock_3d_thermal.i')
    moose_modifier = InputModifier(moose_input,'#','')

    moose_config = MooseConfig().read_config(Path.cwd() / 'moose-config.json')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                              n_threads = 2,
                              redirect_out = False)

    dir_manager = DirectoryManager(n_dirs=4)

    herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)
    herd.set_num_para_sims(n_para=4)
    herd.set_keep_flag(False)

    dir_manager.set_base_dir(Path('perturbed_datasets/geometry'))
    dir_manager.clear_dirs()
    dir_manager.create_dirs()

    xmax = [5,10]
    ymax = [2,4]
    moose_vars = list([])
    for x in xmax:
        for y in ymax:
            moose_vars.append([{'xmax':x,'ymax':y}])


    for _ in range(NUM_PARA_RUNS):
        herd.run_para(moose_vars)

    sweep_reader = SweepReader(dir_manager,num_para_read=4)
    sweep_reader.read_all_output_keys()
    read_all = sweep_reader.read_results_para()

    print('Finished.')


if __name__ == '__main__':
    main()