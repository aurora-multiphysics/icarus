
from pathlib import Path
import numpy as np
from mooseherder import ExodusReader
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import re

class CreateMechanicalDataset:
    def __init__(self, base_dir, output_file, ground_truth_file, dataset_dir, elasticity_modulus, poisson_ratio):
        self.base_dir = Path(base_dir)
        self.output_file = output_file
        self.dataset_dir = Path(dataset_dir)
        self.ground_truth_file = Path(ground_truth_file)
        self.elasticity_modulus = elasticity_modulus
        self.poisson_ratio = poisson_ratio

    def process_simulation_output(self, exodus_file):
        reader = ExodusReader(str(exodus_file))
        disp_x = reader.get_data("disp_x")
        disp_y = reader.get_data("disp_y")
        stress_xx = reader.get_data("stress_xx")
        stress_yy = reader.get_data("stress_yy")
        return np.stack([disp_x, disp_y, stress_xx, stress_yy], axis=-1)

    def create_dataset(self):
        simulation_files = list(self.base_dir.glob("*.e"))
        data = []
        for sim_file in simulation_files:
            simulation_data = self.process_simulation_output(sim_file)
            data.append(simulation_data)
        data = np.array(data)
        np.savez(self.output_file, data=data)

    def split_and_save_dataset(self, test_size=0.2):
        data = np.load(self.output_file)['data']
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        np.savez(self.dataset_dir / "train_data.npz", data=train_data)
        np.savez(self.dataset_dir / "test_data.npz", data=test_data)

    def generate_ground_truth(self):
        ground_truth_data = self.process_simulation_output(self.ground_truth_file)
        np.savez(self.dataset_dir / "ground_truth_data.npz", data=ground_truth_data)
