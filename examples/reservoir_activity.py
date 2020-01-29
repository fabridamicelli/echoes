import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

from echoes import ESNGenerative
from echoes.datasets import load_mackeyglasst17
from echoes.plotting import set_mystyle, plot_reservoir_activity, plot_predicted_ts

set_mystyle()  # just aesthetics


# Load data and define train/test length
data = load_mackeyglasst17().reshape(-1, 1)
trainlen, testlen = 2000, 2000
totallen = trainlen + testlen

# Instantiate model
esn = ESNGenerative(
    n_inputs=1,
    n_outputs=1,
    n_reservoir=200,
    spectral_radius=1.25,
    teacher_forcing=True,
    leak_rate=.4,
    regression_params={
        "method": "pinv"
    },
    store_states_pred=True,   # store states for plotting
    random_seed=42,
).fit(None, data[: trainlen])  # fit the model

prediction = esn.predict(n_steps=testlen)

neurons_to_plot = sorted(np.random.randint(0, esn.n_reservoir, size=9))   # pick 9 neurons at random to plot

plot_reservoir_activity(esn,
                        neurons_to_plot,
                        pred=True,   # plot activity during prediction
                        end=500,
                        figsize=(12, 8))
