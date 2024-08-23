from pathlib import Path
import warnings
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Space
from skopt.sampler import Sobol
from mooseherder import (
    MooseHerd,
    MooseRunner,
    InputModifier,
    DirectoryManager,
    MooseConfig,
    ExodusReader,
)
from sklearn.model_selection import train_test_split
import json
import re


warnings.filterwarnings("ignore", category=UserWarning, module="skopt.sampler.sobol")

N_SAMPLES = 100
np.random.seed(123)


@dataclass
class SimulationParameters:

    n_samples: int = 100
    p_factor: float = 0.8
    tolerance: float = 0.001
    thermal_cond_base: float = 384.0
    heat_flux_base: float = 500e3
    moose_filename: str = "example_2d_plate.i"
    config_filename: str = "example-moose-config.json"
    n_para: int = 10

    n_tasks: int = 1
    n_threads: int = 4

    output_file: str = "example_2d_plate_data.npz"
    ground_truth_file: str = "example_2d_plate_out.e"
    dataset_dir: Path = Path(Path.cwd(), "2d_thermal_dataset/")


class CreateThermalDataset:
    """generates perturbed parameter samples. using quasirandom (sobol) sampling
     within stratified samples to ensure dataset balance and
    and sufficient coverage of the continous sample space, as specified by
    the perturbation factor,  and the tolerance level, t.
    both hyperaparameters can be altered by user
    """

    def __init__(self, params: SimulationParameters):
        """_summary_

        Parameters
        ----------
        params : SimulationParameters
            _description_
        """
        self.params = params
        self.thermal_c_samples, self.heat_flux_samples = (
            self._generate_stratified_samples()
        )
        self.classified_samples = self._classify_samples()

    def _generate_stratified_samples(self):
        """generates sample paprameters using stratified sobol sampling"""
        samples_per_class = self.params.n_samples // 4

        # define the ranges for each parameter
        tc_min = self.params.thermal_cond_base * (1 - self.params.p_factor)
        tc_max = self.params.thermal_cond_base * (1 + self.params.p_factor)
        hf_min = self.params.heat_flux_base * (1 - self.params.p_factor)
        hf_max = self.params.heat_flux_base * (1 + self.params.p_factor)

        # define the boundaries for "unperturbed" regions
        tc_low = self.params.thermal_cond_base * (1 - self.params.tolerance)
        tc_high = self.params.thermal_cond_base * (1 + self.params.tolerance)
        hf_low = self.params.heat_flux_base * (1 - self.params.tolerance)
        hf_high = self.params.heat_flux_base * (1 + self.params.tolerance)

        # Generate samples for each class
        class_0 = self._sobol_samples(
            tc_low, tc_high, hf_low, hf_high, samples_per_class
        )
        class_1 = self._sobol_samples(
            tc_low, tc_high, hf_min, hf_low, samples_per_class // 2
        ) + self._sobol_samples(
            tc_low, tc_high, hf_high, hf_max, samples_per_class // 2
        )
        class_2 = self._sobol_samples(
            tc_min, tc_low, hf_low, hf_high, samples_per_class // 2
        ) + self._sobol_samples(
            tc_high, tc_max, hf_low, hf_high, samples_per_class // 2
        )
        class_3 = (
            self._sobol_samples(tc_min, tc_low, hf_min, hf_low, samples_per_class // 4)
            + self._sobol_samples(
                tc_min, tc_low, hf_high, hf_max, samples_per_class // 4
            )
            + self._sobol_samples(
                tc_high, tc_max, hf_min, hf_low, samples_per_class // 4
            )
            + self._sobol_samples(
                tc_high, tc_max, hf_high, hf_max, samples_per_class // 4
            )
        )

        # Combine all samples
        all_samples = class_0 + class_1 + class_2 + class_3
        np.random.shuffle(all_samples)

        thermal_cond_vals = [sample[0] for sample in all_samples]
        heat_flux_vals = [sample[1] for sample in all_samples]

        return thermal_cond_vals, heat_flux_vals

    def _sobol_samples(self, tc_min, tc_max, hf_min, hf_max, n_samples):
        """_summary_

        Parameters
        ----------
        tc_min : _type_
            _description_
        tc_max : _type_
            _description_
        hf_min : _type_
            _description_
        hf_max : _type_
            _description_
        n_samples : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        space = Space([(tc_min, tc_max), (hf_min, hf_max)])
        samples = Sobol().generate(space.dimensions, n_samples)
        return [(sample[0], sample[1]) for sample in samples]

    def _classify_samples(self):
        """calculates which class each perturbed parameter belongs to then
        ascribes parameter to that class
        """
        classified_samples = []
        for thermal_cond, heat_flux in zip(
            self.thermal_c_samples, self.heat_flux_samples
        ):
            tc_perturbed = (
                abs(thermal_cond - self.params.thermal_cond_base)
                > self.params.thermal_cond_base * self.params.tolerance
            )
            hf_perturbed = (
                abs(heat_flux - self.params.heat_flux_base)
                > self.params.heat_flux_base * self.params.tolerance
            )

            class_label = tc_perturbed * 2 + hf_perturbed
            classified_samples.append((thermal_cond, heat_flux, class_label))

        return classified_samples

    def get_class_distribution(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        class_counts = [0, 0, 0, 0]
        for _, _, label in self.classified_samples:
            class_counts[label] += 1
        return class_counts

    def print_sample_info(self, num_samples=10):
        """Prints sample information"""
        print(f"First {num_samples} samples:")
        for i, (tc, hf, cl) in enumerate(self.classified_samples[:num_samples]):
            print(
                f"Sample {i+1}: Thermal Conductivity = {tc:.2f}, Heat Flux = {hf:.2f}, Class = {cl}"
            )

        class_counts = self.get_class_distribution()
        print("\nClass distribution:")
        for i, count in enumerate(class_counts):
            print(f"Class {i}: {count} samples")

    def plot_searchspace(self):
        """generates scatter plot denoting coverage of stratified sobol sample
        space
        """
        _, ax = plt.subplots()
        plt.plot(self.thermal_c_samples, self.heat_flux_samples, "bo", label="samples")
        plt.plot(
            self.thermal_c_samples,
            self.heat_flux_samples,
            "bo",
            markersize=5,
            alpha=0.2,
        )
        ax.set_xlabel("Thermal Conductivity")
        ax.set_xlim(
            [
                self.params.thermal_cond_base * (1 - self.params.p_factor),
                self.params.thermal_cond_base * (1 + self.params.p_factor),
            ]
        )
        ax.set_ylabel("Heat Flux")
        ax.set_ylim(
            [
                self.params.heat_flux_base * (1 - self.params.p_factor),
                self.params.heat_flux_base * (1 + self.params.p_factor),
            ]
        )
        plt.title("Stratified Sobol Samples")
        plt.show()

    def run_simulations(self):
        config_path = Path(Path.cwd(), self.params.config_filename)
        moose_config = MooseConfig().read_config(config_path)
        moose_input = Path(Path.cwd(), self.params.moose_filename)

        moose_modifier = InputModifier(moose_input, "#", "")
        moose_runner = MooseRunner(moose_config)
        moose_runner.set_run_opts(
            n_tasks=self.params.n_tasks,
            n_threads=self.params.n_threads,
            redirect_out=True,
        )

        dir_manager = DirectoryManager(n_dirs=1)

        herd = MooseHerd([moose_runner], [moose_modifier], dir_manager)
        herd.set_num_para_sims(n_para=self.params.n_para)

        dataset_dir = Path(Path.cwd() / "2d_thermal_dataset/")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        dir_manager.set_base_dir(dataset_dir)
        dir_manager.clear_dirs()
        dir_manager.create_dirs()

        moose_vars = [
            [{"cuThermCond": tc, "surfHeatFlux": hf}]
            for tc, hf in zip(self.thermal_c_samples, self.heat_flux_samples)
        ]

        print("EXAMPLE: Run MOOSE in parallel")
        print("-" * 80)

        herd.run_para(moose_vars)

        print(f"Run time (para) = {herd.get_sweep_time():.3f} seconds")
        print("-" * 80)

    def preprocess_data(self, test_size=0.2, random_state=42):
        ground_truth = self._create_tensor(Path(self.params.ground_truth_file))

        temperature_data = []
        y_data = []

        for working_dir in self.params.dataset_dir.iterdir():
            if working_dir.is_dir():
                exodus_files = sorted(
                    working_dir.glob("*.e"), key=lambda x: self._extract_number(x.name)
                )
                sweep_vars_files = sorted(working_dir.glob("sweep-vars-*.json"))

                for exodus_file in exodus_files:
                    temperature_tensor = self._create_tensor(exodus_file)
                    temperature_data.append(ground_truth - temperature_tensor)

                for sweep_vars_file in sweep_vars_files:
                    y_data.extend(self._calculate_y_data(sweep_vars_file))

        temperature_data = np.array(temperature_data)
        y_data = np.array(y_data)

        temperature_train, temperature_test, y_train, y_test = train_test_split(
            temperature_data,
            y_data,
            test_size=test_size,
            random_state=random_state,
            shuffle=False,
        )

        np.savez(
            self.params.output_file,
            temperature_train=temperature_train,
            temperature_test=temperature_test,
            y_train=y_train,
            y_test=y_test,
            ground_truth=ground_truth,
            heat_fluxes=self.heat_flux_samples,
            thermal_conds=self.thermal_c_samples,
        )

    def _create_tensor(self, exodus_file):
        exodus_reader = ExodusReader(exodus_file)
        read_config = exodus_reader.get_read_config()
        read_config.node_vars = np.array([("temperature")])
        sim_data = exodus_reader.get_all_node_vars()
        temperature_tensor = np.array(
            [value for value in sim_data.values()], dtype=np.float64
        )
        return temperature_tensor[:, :, 1].T

    def _extract_number(self, filename):
        match = re.search(r"sim-1-(\d+)_out\.e$", filename)
        return int(match.group(1)) if match else 0

    def _calculate_y_data(self, perturbed_data_file):
        with open(perturbed_data_file, "r", encoding="utf-8") as file:
            perturbed_data = json.load(file)

        y_data = []
        for data_list in perturbed_data:
            data = data_list[0]
            thermal_cond = data["cuThermCond"]
            heat_flux = data["surfHeatFlux"]

            tc_perturbed = (
                abs(thermal_cond - self.params.thermal_cond_base)
                > self.params.thermal_cond_base * self.params.tolerance
            )
            hf_perturbed = (
                abs(heat_flux - self.params.heat_flux_base)
                > self.params.heat_flux_base * self.params.tolerance
            )

            y_data.append(tc_perturbed * 2 + hf_perturbed)

        return y_data

    def run_workflow(self):
        print("Generating stratified samples...")
        self._generate_stratified_samples()

        print("Running MOOSE simulations...")
        self.run_simulations()

        print("Preprocessing data and creating .npz file...")
        self.preprocess_data()

        print(f"Workflow completed. Output saved to {self.params.output_file}")
