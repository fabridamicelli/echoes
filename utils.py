"""
Auxiliar functions
"""
from typing import Union

import numpy as np


def set_spectral_radius(matrix: np.ndarray, target_radius: float) -> np.ndarray:
    """Rescale weights matrix to match target spectral radius"""
    current_radius = np.max(np.abs(np.linalg.eigvals(matrix)))
    matrix *= target_radius / current_radius
    return matrix


def check_arrays_dimensions(inputs: np.ndarray, outputs: np.ndarray = None) -> None:
    """check that input and/or outputs shape is 2D"""
    if inputs.ndim < 2:
        raise ValueError("""
            Input must be 2D array, got 1D array instead.
            If n_inputs is one reshape your data to (n_samples, 1).
            """)
    if outputs is not None:
        if outputs.ndim < 2:
            raise ValueError("""
                Output must be 2D array, got 1D array instead.
                If your n_outputs is one, reshape your data to (n_samples, 1).
                """)

def identity(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x
