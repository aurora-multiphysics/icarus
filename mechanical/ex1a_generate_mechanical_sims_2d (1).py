
'''
================================================================================
Example: Generate Perturbed 2D Plate Mechanical Field Simulations using MOOSEHERDER and Stratified Sobol Sampling of parameter space.

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
'''

from pathlib import Path
import os
from icarus import MechanicalSampleGenerator
from mooseherder import (MooseHerd, MooseConfig)

if __name__ == "__main__":
    base_input_file = Path("example_2d_plate.i")
    output_dir = Path("mechanical_simulations_2d")
    
    elasticity_modulus_range = (90e9, 120e9)
    poisson_ratio_range = (0.28, 0.35)
    pressure_range = (1e6, 10e6)

    generator = MechanicalSampleGenerator(base_input_file=base_input_file,
                                          output_dir=output_dir,
                                          elasticity_modulus_range=elasticity_modulus_range,
                                          poisson_ratio_range=poisson_ratio_range,
                                          pressure_range=pressure_range)
    
    generator.generate_samples()
