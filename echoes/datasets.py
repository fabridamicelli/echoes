import numpy as np


def load_mackeyglasst17():
    """
    Mackey glass chaotic oscillator time series.
    Resource: http://minds.jacobs-university.de/mantas/code
    """
    try:
        data = np.load('./data/mackey_glass_t17.npy')
    except FileNotFoundError:
        data = np.load('../data/mackey_glass_t17.npy')
    return data

