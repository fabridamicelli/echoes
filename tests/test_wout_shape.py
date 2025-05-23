from echoes import ESNGenerator
from echoes.datasets import load_mackeyglasst17


def test_W_out_shape_fullstates():
    """check shape of W_out when trained with input and bias (fit_only_states=False)"""
    data = load_mackeyglasst17().reshape(-1, 1)
    trainlen = 100
    esn = ESNGenerator(
        n_reservoir=50,
        fit_only_states=False,
        regression_method="pinv",
        random_state=42,
    ).fit(None, data[:trainlen])

    assert esn.W_out_.shape == (1, esn.n_reservoir_ + esn.n_inputs_)


def test_W_out_shape_onlystates():
    """check shape of W_out when trained without input and bias (fit_only_states=True)"""
    data = load_mackeyglasst17().reshape(-1, 1)
    trainlen = 100
    esn = ESNGenerator(
        n_reservoir=50,
        fit_only_states=True,
        random_state=42,
    ).fit(None, data[:trainlen])

    assert esn.W_out_.shape == (1, esn.n_reservoir_)
