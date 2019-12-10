"""
Auxiliar functions
"""
from typing import Union, Callable

import numpy as np

from esn import EchoStateNetwork


def set_spectral_radius(matrix: np.ndarray, target_radius: float) -> np.ndarray:
    """Rescale weights matrix to match target spectral radius"""
    current_radius = np.max(np.abs(np.linalg.eigvals(matrix)))
    matrix *= target_radius / current_radius
    return matrix


def check_arrays_dimensions(inputs: np.ndarray = None, outputs: np.ndarray = None) -> None:
    """check that input and/or outputs shape is 2D"""
    if inputs is not None:
        assert isinstance(inputs, np.ndarray), \
            "wrong inputs type; must be np.ndarray or None"
        if inputs.ndim < 2:
            raise ValueError("""
                Input must be 2D array, got 1D array instead.
                If n_inputs is one reshape your data with .reshape(-1, 1).
                """)
    if outputs is not None:
        if outputs.ndim < 2:
            raise ValueError("""
                Output must be 2D array, got 1D array instead.
                If your n_outputs is one, reshape your data with .reshape(-1, 1).
                """)

def identity(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x

def check_func_inverse(func: Callable, inv_func: Callable) -> None:
    """check that func and inv_func are indeed inverse of each other"""
    x = np.linspace(-2, 2, 10)
    y = func(x)
    mismatch = np.where(inv_func(y) != x)[0]
    assert np.isclose(inv_func(y), x).all(),\
        f"function {inv_func.__name__} is not the inverse of {func.__name__}"
