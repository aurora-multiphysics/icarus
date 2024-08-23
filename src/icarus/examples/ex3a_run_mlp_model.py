"""
================================================================================
Example: Train and Validate an ML Model on Chosen Perturbed Dataset 

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
"""

from typing import Tuple
from icarus import NNParameters, IcarusModel
from torch.utils.data import DataLoader
from icarus import ThermalModelDataset as td


npz_files = ["example_2d_plate_data.npz", "example_thermal_monoblock_data.npz"]


def main(npz_file: str) -> Tuple[float, float]:

    thermal_dataset_train = td(npz_file)
    params = NNParameters(
        input_size=thermal_dataset_train.temperature_train[0].squeeze().numel()
    )
    thermal_dataset_test = td(npz_file, train=False)
    train_loader = DataLoader(
        dataset=thermal_dataset_train, batch_size=params.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=thermal_dataset_test, batch_size=params.batch_size, shuffle=False
    )

    model = IcarusModel(params)
    best_accuracy = model.train(train_loader, test_loader)
    test_accuracy = model.evaluate(test_loader)

    return best_accuracy, test_accuracy


if __name__ == "__main__":
    best_acc, test_acc = main(npz_files[0])
    print(f"Best accuracy: {best_acc:.2f}%, Final test accuracy: {test_acc:.2f}%")
