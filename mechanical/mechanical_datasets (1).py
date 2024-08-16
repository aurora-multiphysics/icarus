
import os
import numpy as np

def load_2d_mechanical_dataset():
    """Load the built-in mechanical model dataset."""
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'mechanical_model_data.npz')
    return dict(np.load(file_path, allow_pickle=True))

def load_monoblock_mechanical_dataset():
    """Load the built-in mechanical model dataset."""
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'mechanical_monoblock_data.npz')
    return dict(np.load(file_path, allow_pickle=True))
