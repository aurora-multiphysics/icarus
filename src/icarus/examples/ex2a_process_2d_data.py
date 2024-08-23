"""
================================================================================
Example: Generate 2D Plate Temperature Field Dataset.

icarus:  Intelligent Compatibility Analyser for Reactor Uncertain Simulations
License: GPL 3.0
================================================================================
"""

from icarus import CreateThermalDataset, SimulationParameters


if __name__ == "__main__":
    parameters = SimulationParameters()
    workflow = CreateThermalDataset(parameters)
    workflow.run_workflow()
