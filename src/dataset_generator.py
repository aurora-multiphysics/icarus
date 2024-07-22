#Mooseherder can generate 125 sims per min 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Sobol
from mooseherder import (MooseHerd,
                         MooseRunner,
                         InputModifier,
                         DirectoryManager,
                         MooseConfig)

N_SAMPLES = 100
np.random.seed(123)

class GeneratePerturbedSamples:
    '''generates perturbed parameter samples. using quasirandom (sobol) sampling within stratified samples to ensure dataset balance and 
       and sufficient coverage of the continous sample space, as specified by the perturbation factor,  and the tolerance level, t. both hyperparamterters can be altered by user 
    '''
    def __init__(self, n_samples=N_SAMPLES, thermal_cond_base=384.0, heat_flux_base=500000.0, p_factor=0.8, tolerance=0.001):
        self.n_samples = n_samples
        self.thermal_cond_base = thermal_cond_base
        self.heat_flux_base = heat_flux_base
        self.p_factor = p_factor
        self.tolerance = tolerance
        self.thermal_c_samples, self.heat_flux_samples = self.generate_stratified_samples()
        self.classified_samples = self.classify_samples()

    def generate_stratified_samples(self):
        samples_per_class = self.n_samples // 4
        
        # define the ranges for each parameter
        tc_min = self.thermal_cond_base * (1 - self.p_factor)
        tc_max = self.thermal_cond_base * (1 + self.p_factor)
        hf_min = self.heat_flux_base * (1 - self.p_factor)
        hf_max = self.heat_flux_base * (1 + self.p_factor)
        
        # define the boundaries for "unperturbed" regions
        tc_low = self.thermal_cond_base * (1 - self.tolerance)
        tc_high = self.thermal_cond_base * (1 + self.tolerance)
        hf_low = self.heat_flux_base * (1 - self.tolerance)
        hf_high = self.heat_flux_base * (1 + self.tolerance)
        
        # Generate samples for each class
        class_0 = self.sobol_samples(tc_low, tc_high, hf_low, hf_high, samples_per_class)
        class_1 = self.sobol_samples(tc_low, tc_high, hf_min, hf_low, samples_per_class // 2) + \
                  self.sobol_samples(tc_low, tc_high, hf_high, hf_max, samples_per_class // 2)
        class_2 = self.sobol_samples(tc_min, tc_low, hf_low, hf_high, samples_per_class // 2) + \
                  self.sobol_samples(tc_high, tc_max, hf_low, hf_high, samples_per_class // 2)
        class_3 = self.sobol_samples(tc_min, tc_low, hf_min, hf_low, samples_per_class // 4) + \
                  self.sobol_samples(tc_min, tc_low, hf_high, hf_max, samples_per_class // 4) + \
                  self.sobol_samples(tc_high, tc_max, hf_min, hf_low, samples_per_class // 4) + \
                  self.sobol_samples(tc_high, tc_max, hf_high, hf_max, samples_per_class // 4)
        
        # Combine all samples
        all_samples = class_0 + class_1 + class_2 + class_3
        np.random.shuffle(all_samples)
        
        thermal_cond_vals = [sample[0] for sample in all_samples]
        heat_flux_vals = [sample[1] for sample in all_samples]
        
        return thermal_cond_vals, heat_flux_vals

    def sobol_samples(self, tc_min, tc_max, hf_min, hf_max, n_samples):
        space = Space([(tc_min, tc_max), (hf_min, hf_max)])
        samples = Sobol().generate(space.dimensions, n_samples)
        return [(sample[0], sample[1]) for sample in samples]

    def classify_samples(self):
        classified_samples = []
        for thermal_cond, heat_flux in zip(self.thermal_c_samples, self.heat_flux_samples):
            tc_perturbed = abs(thermal_cond - self.thermal_cond_base) > self.thermal_cond_base * self.tolerance
            hf_perturbed = abs(heat_flux - self.heat_flux_base) > self.heat_flux_base * self.tolerance
            
            class_label = tc_perturbed * 2 + hf_perturbed
            classified_samples.append((thermal_cond, heat_flux, class_label))
        
        return classified_samples

    def get_class_distribution(self):
        class_counts = [0, 0, 0, 0]
        for _, _, label in self.classified_samples:
            class_counts[label] += 1
        return class_counts

    def print_sample_info(self, num_samples=10):
        print(f"First {num_samples} samples:")
        for i, (tc, hf, cl) in enumerate(self.classified_samples[:num_samples]):
            print(f"Sample {i+1}: Thermal Conductivity = {tc:.2f}, Heat Flux = {hf:.2f}, Class = {cl}")

        class_counts = self.get_class_distribution()
        print("\nClass distribution:")
        for i, count in enumerate(class_counts):
            print(f"Class {i}: {count} samples")

    def plot_searchspace(self):
        fig, ax = plt.subplots()
        plt.plot(self.thermal_c_samples, self.heat_flux_samples, 'bo', label='samples')
        plt.plot(self.thermal_c_samples, self.heat_flux_samples, 'bo', markersize=5, alpha=0.2)
        ax.set_xlabel("Thermal Conductivity")
        ax.set_xlim([self.thermal_cond_base * (1-self.p_factor), self.thermal_cond_base * (1+self.p_factor)])
        ax.set_ylabel("Heat Flux")
        ax.set_ylim([self.heat_flux_base * (1-self.p_factor), self.heat_flux_base * (1+self.p_factor)])
        plt.title('Stratified Sobol Samples')
        plt.show()


dataset = GeneratePerturbedSamples(n_samples=1000, p_factor=0.8, tolerance=0.015)


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

    # Set the parallelisation options
    herd.set_num_para_sims(n_para=10)

     # Send all the output to the examples directory and clear out old output
    dir_manager.set_base_dir(Path('dataset/'))
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

