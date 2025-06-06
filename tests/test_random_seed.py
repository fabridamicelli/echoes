import numpy as np

from echoes.esn import ESNGenerator, ESNRegressor


def test_random_seed():
    """test that random seed affects only the weight generation and then gets reset"""

    seed = 1
    ESNGenerator(random_state=seed)

    after_seed = np.random.choice(10000000, size=10000)

    ESNRegressor(random_state=seed)

    after_seed2 = np.random.choice(10000000, size=10000)
    assert (after_seed != after_seed2).sum() > 0, (
        "random_seed has global scope, "
        "please report this. As a work around you can"
        "set the random seed manually in your script."
    )


if __name__ == "__main__":
    test_random_seed()
