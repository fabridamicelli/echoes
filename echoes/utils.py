"""
Auxiliar functions
"""
from typing import Union, Callable, Dict
import warnings

import numpy as np


def set_spectral_radius(matrix: np.ndarray, target_radius: float) -> np.ndarray:
    """Rescale weights matrix to match target spectral radius"""
    current_radius = np.max(np.abs(np.linalg.eigvals(matrix)))
    matrix *= target_radius / current_radius
    return matrix

def check_arrays_dimensions(
    inputs: np.ndarray = None, outputs: np.ndarray = None
) -> None:
    """check that input and/or outputs shape is 2D"""
    if inputs is not None:
        assert isinstance(
            inputs, np.ndarray
        ), "wrong inputs type; must be np.ndarray or None"
        if inputs.ndim < 2:
            raise ValueError(
                "Input must be 2D array, got 1D array instead"
                "If n_inputs is one reshape your data with .reshape(-1, 1)."
            )
        if outputs is not None:
            assert (
                inputs.shape[0] == outputs.shape[0]
            ), "inputs and outputs must have same length"
    if outputs is not None:
        if outputs.ndim < 2:
            raise ValueError(
                "Output must be 2D array, got 1D array instead."
                "If your n_outputs is one, reshape your data with .reshape(-1, 1)."
            )
    # TODO: assert dimensions inputs match n_inputs and dimensions output matchs n_outputs


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


def check_model_params(params: Dict, esn_type: str) -> None:
    """check consistency of parameters, shapes, sensible warnings"""
    W = params["W"]
    W_in = params["W_in"]
    W_feedb = params["W_feedb"]
    n_reservoir = params["n_reservoir"]
    n_inputs = params["n_inputs"]
    n_outputs = params["n_outputs"]
    teacher_forcing = params["teacher_forcing"]

    # Check shapes of W,W_in, W_feedb
    assert W.shape[0] == W.shape[1], "W must be square"
    assert len(W) == n_reservoir, "W does not match n_reservoir"
    assert W_in.shape[0] == n_reservoir, "W_in first dimension must equal n_reservoir"
    assert (
        W_in.shape[1] == n_inputs + 1
    ), "W_in second dimension must equal n_inputs + 1 (bias)"
    if teacher_forcing:
        assert W_feedb is not None, "W_feedb must be specified if teacher_forcing==True"
    if W_feedb is not None:
        assert (
            W_feedb.shape[0] == n_reservoir
        ), "W_feedb first dimension must equal n_reservoir"
        assert (
            W_feedb.shape[1] == n_outputs
        ), "W_feedb second dimension must equal n_outputs"

    check_func_inverse(params["activation_out"], params["inv_activation_out"])

    assert params["spectral_radius"] is not None, "spectral_radius must be specified"

    assert 0 <= params["sparsity"] < 1, "sparsity must be in [0-1)"

    input_scaling = params["input_scaling"]
    if isinstance(input_scaling, np.ndarray):
        assert (
            len(input_scaling) == n_inputs
        ), "length of input_scaling must equal n_inputs"

    input_shift = params["input_shift"]
    if isinstance(input_shift, np.ndarray):
        assert len(input_shift) == n_inputs, "length of input_shift must equal n_inputs"

    if esn_type == "ESNGenerative":
        assert teacher_forcing == True, "generative ESN requires teacher forcing"

    # Warnings
    if params["leak_rate"] == 0:
        warnings.warn(
            "leak_rate == 0 is total leakeage, you probably meant 1. See documentation."
        )
    if esn_type == "ESNGenerative":
        if input_scaling is not None:
            warnings.warn("input scaling will be ignored, since it is a generative ESN")
        if input_shift is not None:
            warnings.warn("input shift will be ignored, since it is a generative ESN")
