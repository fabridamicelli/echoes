"""
Auxiliar functions
"""
from typing import Union

import numpy as np


def set_spectral_radius(matrix: np.ndarray, target_radius: float):
    """Rescale weights matrix to match target spectral radius"""
    current_radius = np.max(np.abs(np.linalg.eigvals(matrix)))
    matrix *= target_radius / current_radius
    return matrix


def check_arrays(inputs: np.ndarray, outputs: np.ndarray=None):
    if inputs.ndim < 2:
        raise ValueError(
            """Input must be 2D array, got 1D array instead.
            Reshape your data to (n_samples, 1)""")
    if outputs:
        if outputs.ndim < 2:
            raise ValueError(
                """Output must be 2D array, got 1D array instead.
                Reshape your data to (n_samples, 1)""")

def identity(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x
