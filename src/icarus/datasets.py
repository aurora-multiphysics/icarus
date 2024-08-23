import os
import numpy as np

def load_2d_thermal_dataset():
    """Load the built-in thermal model dataset."""
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'thermal_model_data.npz')
    return dict(np.load(file_path, allow_pickle=True))

def load_monoblock_thermal_dataset():
    """Load the built-in thermal model dataset."""
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'thermal_monoblock_data.npz')
    return dict(np.load(file_path, allow_pickle=True))