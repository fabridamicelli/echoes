import numpy as np

from echoes import ESNGenerator, ESNRegressor
from echoes.datasets import load_mackeyglasst17


def test_spectral_rad_regressor():
    """check that actual spectral radius is the one specified by parameters"""
    decimals = 6
    radia = np.linspace(.5, 2.5, 20).round(decimals=6)
    y_train = load_mackeyglasst17()

    for radius in radia:
        esn = ESNRegressor(spectral_radius=radius, feedback=True)
        n_steps = 50
        # Mock fit to call esn.fit()
        esn.fit(X=np.random.random(size=(n_steps, 1)), y=y_train[:n_steps])

        assert (np.max(np.abs(np.linalg.eigvals(esn.W_))).round(decimals=decimals)
                == np.round(radius, decimals=decimals))

        #TODO without feedback


def test_spectral_rad_generator():
    """check that actual spectral radius is the one specified by parameters"""
    decimals = 6
    radia = np.linspace(.5, 2.5, 20).round(decimals=decimals)
    y_train = load_mackeyglasst17()

    for radius in radia:
        esn = ESNGenerator(spectral_radius=radius)
        esn.fit(X=None, y=y_train[:50])  # mock fit to call esn.fit()

        assert (np.max(np.abs(np.linalg.eigvals(esn.W_))).round(decimals=decimals)
                == np.round(radius, decimals=decimals))


def test_spec_rad_inputW_regressor():
    """check that spectral radius is corrected for W passed as input"""
    decimals = 6
    n_reservoir = 50
    radia = np.linspace(.5, 2.5).round(decimals=decimals)
    y_train = load_mackeyglasst17()
    for radius in radia:
        for _ in range(5):
            W = np.random.rand(n_reservoir, n_reservoir)

            esn = ESNRegressor(
                n_reservoir=n_reservoir,
                W=W,
                spectral_radius=radius,
                feedback=True,
            )
            # Mock fit only to call esn.fit()
            n_steps = 50
            esn.fit(X=np.random.random(size=(n_steps, 1)), y=y_train[:n_steps])

            assert (np.max(np.abs(np.linalg.eigvals(esn.W_))).round(decimals=decimals)
                    == np.round(radius, decimals=decimals))
        #TODO without feedback


def test_spec_rad_inputW_generator():
    """check that spectral radius is corrected for W passed as input"""
    decimals = 6
    n_reservoir = 50
    radia = np.linspace(.5, 2.5).round(decimals=decimals)
    y_train = load_mackeyglasst17()
    for radius in radia:
        for _ in range(5):
            W = np.random.rand(n_reservoir, n_reservoir)

            esn = ESNGenerator(n_reservoir=n_reservoir, W=W, spectral_radius=radius)
            # Mock fit to call esn.fit()
            esn.fit(X=None, y=y_train[:50])

            assert (np.max(np.abs(np.linalg.eigvals(esn.W_))).round(decimals=decimals)
                    == np.round(radius, decimals=decimals))
