# Icarus
Machine learning for simulation validation. Icarus provide a set of machine learning tools that engineers can use to assess the agreement between an experiment and simulation; that is, to validate the simulation with experimental data.


## Installation


### Standard Installation (PyPI) 

You can install icarus from PyPi as follows:

```
pip install icarus-fusion
```

### Developer Installation

Clone `icarus` to your local system and `cd` to the root directory of `icarus`. Ensure that your virtual environment is activated and run from the `icarus` root directory:

```
pip install -e .
```


### PyTorch

Icarus requires the latest stable version of PyTorch. The installation process varies depending on your hardware and operating system. Please follow the appropriate instructions below:

#### CPU Installation:

If you do not have access to NVIDIA GPUs, install the CPU version of PyTorch. Use the following commands based on your operating system:

- Windows/macOS:

```
pip3 install torch
```

- Linux: 

```
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```

#### GPU Installation (NVIDIA CUDA):

If you have access to NVIDIA GPUs and want to leverage CUDA for faster computation, use these commands (note: CUDA is not available on MacOS):

- Linux:

```
pip3 install torch
```

- Windows:
```
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

*Note: The CUDA version (`cu121` in this example) may change. Always check the official [PyTorch](https://pytorch.org/get-started/locally/) website for the most up-to-date installation instructions and CUDA version compatibility.*


#### Verifying Installation:

After installation, you can verify that PyTorch is installed correctly by running:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())  # returns True if CUDA available and properly installed
```

## Getting Started

The examples folder includes a sequence of examples using `icarus` : to generate the dataset and train an ml model from the suite available on the generated data.

#### Moose Config File

here are instructions for hoe to setup the moose config file.

```python
from pathlib import Path
from mooseherder import MooseConfig


def main() -> None:
    """main: create moose config json
    """
    config = {'main_path': Path.home()/ 'moose',
            'app_path': Path.home() / 'proteus',
            'app_name': 'proteus-opt'}

    moose_config = MooseConfig(config)

    save_path = Path.cwd() / 'example-moose-config.json'
    moose_config.save_config(save_path)


if __name__ == "__main__":
    main()
```

## Contributors

- Arjav Poudel, UK Atomic Energy Authority, (arjavp-ukaea)
- Baris Cavusoglu, UK Atomic Energy Authority, (barisc-ukaea)
- Luke Humphrey, UK Atomic Energy Authority, (lukethehuman)