import pytest

from echoes import ESNGenerator
from echoes.datasets import load_mackeyglasst17


def test_generator_errors_on_input():
    data = load_mackeyglasst17().reshape(-1, 1)
    with pytest.raises(ValueError):
        ESNGenerator(n_reservoir=10).fit(1, data[:50])
