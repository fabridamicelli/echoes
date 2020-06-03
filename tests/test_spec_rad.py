import numpy as np

from echoes import ESNGenerator, ESNRegressor
from echoes.datasets import load_mackeyglasst17


def test_spectral_rad():
    """check that actual spectral radius is the one specified by parameters"""
    decimals = 6
    radia = np.linspace(.5, 2.5).round(decimals=decimals)
    y_train = load_mackeyglasst17()

    for radius in radia:
        esn = ESNRegressor(
            spectral_radius=radius,
            feedback=True
        ).fit(X=np.random.random(size=(100, 1)), y=y_train[:100]) # we fit only to call esn.fit()

        assert (np.max(np.abs(np.linalg.eigvals(esn.W_))).round(decimals=decimals)
                == np.round(radius, decimals=decimals))

        esn = ESNGenerator(
            spectral_radius=radius,
            feedback=True
        ).fit(X=None, y=y_train[:100])  # we fit only to call esn.fit()

        assert (np.max(np.abs(np.linalg.eigvals(esn.W_))).round(decimals=decimals)
                == np.round(radius, decimals=decimals))

def test_spec_rad_inputW():
    """check that spectral radius is corrected for W passed as input"""
    decimals = 6
    n_reservoir = 50
    radia = np.linspace(.5, 2.5).round(decimals=decimals)
    y_train = load_mackeyglasst17()
    for radius in radia:
        for _ in range(10):
            W = np.random.rand(n_reservoir, n_reservoir)

            esn = ESNGenerator(
                n_reservoir=n_reservoir,
                W=W,
                spectral_radius=radius,
                feedback=True,
            ).fit(X=None, y=y_train[:100])  # we fit only to call esn.fit()

            assert (np.max(np.abs(np.linalg.eigvals(esn.W_))).round(decimals=decimals)
                    == np.round(radius, decimals=decimals))

            esn = ESNRegressor(
                n_reservoir=n_reservoir,
                W=W,
                spectral_radius=radius,
                feedback=True,
            ).fit(X=np.random.random(size=(100, 1)), y=y_train[:100]) # we fit only to call esn.fit()

            assert (np.max(np.abs(np.linalg.eigvals(esn.W_))).round(decimals=decimals)
                    == np.round(radius, decimals=decimals))
