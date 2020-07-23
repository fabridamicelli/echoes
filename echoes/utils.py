"""
Auxiliar functions
"""
from typing import Union, Callable, Mapping
import warnings

from numba import njit
import numpy as np


def set_spectral_radius(matrix: np.ndarray, target_radius: float) -> np.ndarray:
    """Rescale weights matrix to match target spectral radius"""
    current_radius = np.max(np.abs(np.linalg.eigvals(matrix)))
    matrix *= target_radius / current_radius
    return matrix


def identity(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return x


def check_func_inverse(func: Callable, inv_func: Callable) -> None:
    """check that func and inv_func are indeed inverse of each other"""
    x = np.linspace(-2, 2, 10)
    y = func(x)
    mismatch = np.where(inv_func(y) != x)[0]
    assert np.isclose(
        inv_func(y), x
    ).all(), f"function {inv_func.__name__} is not the inverse of {func.__name__}"


def check_matrices_shapes(W_in, W, W_fb, n_inputs, n_reservoir, n_outputs, feedback):
    """Check shapes of W, W_in, W_fb"""
    assert W.shape[0] == W.shape[1], "W must be square"
    assert len(W) == n_reservoir, "W does not match n_reservoir"
    assert W_in.shape[0] == n_reservoir, "W_in first dimension must equal n_reservoir"
    assert W_in.shape[1] == n_inputs, "W_in second dimension must equal n_inputs"
    if feedback:
        assert W_fb is not None, "W_fb must be specified if feedback=True"
    if W_fb is not None:
        assert (
            W_fb.shape[0] == n_reservoir
        ), "W_fb first dimension must equal n_reservoir"
        assert W_fb.shape[1] == n_outputs, "W_fb second dimension must equal n_outputs"


def check_input_shift(input_shift, n_inputs):
    if isinstance(input_shift, np.ndarray):
        assert (
            input_shift.ndim == 1
        ), "if input_shift is array, it must be one dimensiona"
        assert len(input_shift) == n_inputs, "length of input_shift must equal n_inputs"


def check_input_scaling(input_scaling, n_inputs):
    if isinstance(input_scaling, np.ndarray):
        assert (
            input_scaling.ndim == 1
        ), "if input_scaling is array, it must be one dimensiona"
        assert (
            len(input_scaling) == n_inputs
        ), "length of input_scaling must equal n_inputs"


def check_sparsity(sparsity):
    assert 0 <= sparsity < 1, "sparsity must be in [0-1)"


def check_model_params(params: Mapping,) -> None:
    """check consistency of parameters, shapes, sensible warnings"""
    W_in = params["W_in_"]
    W = params["W_"]
    W_fb = params["W_fb_"]
    n_reservoir = params["n_reservoir_"]
    n_inputs = params["n_inputs_"]
    n_outputs = params["n_outputs_"]
    feedback = params["feedback"]
    input_scaling = params["input_scaling"]
    input_shift = params["input_shift"]

    check_matrices_shapes(W_in, W, W_fb, n_inputs, n_reservoir, n_outputs, feedback)
    check_sparsity(params["sparsity"])
    check_input_scaling(input_scaling, n_inputs)
    check_input_shift(input_scaling, n_inputs)

    # Warnings
    if params["leak_rate"] == 0:
        warnings.warn(
            "leak_rate == 0 is total leakeage, you probably meant 1. See documentation."
        )
    if (
        params["regression_method"] != "ridge"
        and params["ridge_sample_weight"] is not None
    ):
        warnings.warn(
            "ridge_sample_weight will be ignored since regression_method is not ridge"
        )


@njit
def tanh(x: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """Numba jitted tanh function. Return tanh(x)"""
    return np.tanh(x)


@njit
def relu(x: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """Numba jitted ReLu (rectified linear unit) function. Return ReLu(x)"""
    return np.maximum(0, x)


@njit
def sigmoid(x: Union[float, int, np.ndarray], a: float = 1) -> Union[float, np.ndarray]:
    """
    Return  f(x) = 1 / (1 + np.exp(-x * a)). Numba jitted sigmoid function.
    """
    return 1 / (1 + np.exp(-x * a))
