
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Sobol
from mooseherder import (MooseHerd,
                         MooseRunner,
                         InputModifier,
                         DirectoryManager,
                         MooseConfig)

warnings.filterwarnings("ignore", category=UserWarning, module="skopt.sampler.sobol")

N_SAMPLES = 100

class MechanicalSampleGenerator:
    def __init__(self, base_input_file, output_dir, elasticity_modulus_range, poisson_ratio_range, pressure_range):
        self.base_input_file = Path(base_input_file)
        self.output_dir = Path(output_dir)
        self.elasticity_modulus_range = elasticity_modulus_range
        self.poisson_ratio_range = poisson_ratio_range
        self.pressure_range = pressure_range
        self.space = Space([self.elasticity_modulus_range, self.poisson_ratio_range, self.pressure_range])
        self.sampler = Sobol()
        
    def generate_samples(self):
        samples = self.sampler.generate(self.space.dimensions, N_SAMPLES)
        for i, sample in enumerate(samples):
            mod_input = InputModifier(self.base_input_file, {"elasticity_modulus": sample[0],
                                                             "poisson_ratio": sample[1],
                                                             "pressure": sample[2]})
            sim_dir = self.output_dir / f"sim_{i}"
            sim_dir.mkdir(parents=True, exist_ok=True)
            mod_input.write_modified_input(sim_dir / self.base_input_file.name)
            self.run_simulation(sim_dir)

    def run_simulation(self, sim_dir):
        runner = MooseRunner(input_file=sim_dir / self.base_input_file.name)
        runner.run()

    def visualize_samples(self):
        # Add any visualization methods specific to the mechanical model
        pass
