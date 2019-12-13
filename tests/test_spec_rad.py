import numpy as np

from echoes.esn import EchoStateNetwork


def test_spec_rad():
    """check that actual spectral radius is the one specified by parameters"""
    decimals = 6
    radia = np.linspace(.5, 2.5).round(decimals=decimals)

    for radius in radia:
        esn = EchoStateNetwork(
            n_inputs=1,
            n_outputs=1,
            n_reservoir=100,
            spectral_radius=radius,
            teacher_forcing=True,
            regression_params={
                "method": "pinv"
            },
        )

        assert (np.max(np.abs(np.linalg.eigvals(esn.W))).round(decimals=decimals)
                == np.round(radius, decimals=decimals))

def test_spec_rad_inputW():
    """check that spectral radius is corrected for W passed as input"""
    decimals = 6
    n_reservoir = 50
    radia = np.linspace(.5, 2.5).round(decimals=decimals)

    for radius in radia:
        for _ in range(10):
            W = np.random.rand(n_reservoir, n_reservoir)

            esn = EchoStateNetwork(
                n_inputs=1,
                n_outputs=1,
                n_reservoir=n_reservoir,
                W=W,
                spectral_radius=radius,
                teacher_forcing=True,
                regression_params={
                    "method": "pinv"
                },
            )

            assert (np.max(np.abs(np.linalg.eigvals(esn.W))).round(decimals=decimals)
                    == np.round(radius, decimals=decimals))
