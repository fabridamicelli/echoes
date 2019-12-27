import os

import numpy as np


def load_mackeyglasst17():
    module_path = os.path.dirname(__file__)
    return np.load(os.path.join(module_path, "data", "mackey_glass_t17.npy"))
