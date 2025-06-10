# Icarus

A fully customisable set of simulation-driven machine learning tools that interacts with MOOSE and can be used to assess the agreement between an experiment and simulation; that is, to validate the simulation with experimental data and provide the engineer with a probable reason for any mismatches to allow further investigation and diagnosis. 

## Installation
### Virtual Environment 

Icarus is intended for use in a virtual environment. You can create a new virtual environment with 

```
python -m venv venv
```

and activate it with 
```
source venv/bin/activate
```

### Standard & Developer Installation

Clone `icarus` to your local system along with submodules using 

```
git clone --recurse-submodules git@github.com:aurora-multiphysics/icarus.git
```

`cd` to the root directory of `icarus` and run
```
pip install .
```

Alternatively, to create an editable/developer installation of `icarus`
```
pip install -e .
```

### MOOSE App

`icarus` is intended for use with any MOOSE app 
It has been developed and tested using the `proteus` MOOSE app which can be found here: https://github.com/aurora-multiphysics/proteus. Follow the build instructions found on this page to install `proteus`.

## Getting Started

The examples folder shows how to use `icarus` for different input files and with different levels of user control. Make sure to include a moose_config.json file wherever you run it, and change the values contained to match your system.

Icarus requires a moose_config.json file to be present wherever it is run; if there is not one already present, it will create one with the following values:
```
{
    "main_path": "path/to/moose",
    "app_path": "path/to/YourMooseApp",
    "app_name": "YourMooseApp-opt"
}
```

Make sure to change these values to match your system before running Icarus.

## Contributors

- Isaac Magee, UK Atomic Energy Authority, (iomags2211)
- Lloyd Fletcher, UK Atomic Energy Authority, (TheScepticalRabbit)
- Luke Humphrey, UK Atomic Energy Authority, (lukethehuman)

