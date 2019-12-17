import numpy as np

from echoes.esn import EchoStateNetwork
from echoes.datasets import load_mackeyglasst17


def test_W_out_shape_fullstates():
    """check shape of W_out when trained with input and bias (fit_only_states=False)"""
    data = load_mackeyglasst17().reshape(-1, 1)
    trainlen = 100
    esn = EchoStateNetwork(
        n_inputs=1,
        n_outputs=1,
        n_reservoir=50,
        spectral_radius=1.25,
        teacher_forcing=True,
        fit_only_states=False,
        regression_params={
            "method": "pinv"
        },
        random_seed=42,
        verbose=False
    ).fit(None, data[:trainlen])

    assert esn.W_out_.shape == (1, esn.n_reservoir+esn.n_inputs+1)

def test_W_out_shape_onlystates():
    """check shape of W_out when trained without input and bias (fit_only_states=True)"""
    data = load_mackeyglasst17().reshape(-1, 1)
    trainlen = 100
    esn = EchoStateNetwork(
        n_inputs=1,
        n_outputs=1,
        n_reservoir=50,
        spectral_radius=1.25,
        teacher_forcing=True,
        fit_only_states=True,
        regression_params={
            "method": "pinv"
        },
        random_seed=42,
        verbose=False
    ).fit(None, data[:trainlen])

    assert esn.W_out_.shape == (1, esn.n_reservoir)
