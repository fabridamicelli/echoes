import numpy as np


def load_mackeyglasst17():
    """
    Mackey glass chaotic oscillator time series.
    Resource: http://minds.jacobs-university.de/mantas/code
    """
    return np.load('../data/mackey_glass_t17.npy')
