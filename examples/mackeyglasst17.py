"""
Mackey-Glass-t17
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

from echoes.esn import EchoStateNetwork
from echoes.datasets import load_mackeyglasst17

sns.set(context="notebook", style="whitegrid", font_scale=1.4,
        rc={'grid.linestyle': '--',
            'grid.linewidth': 0.8,})

# Instantiate the model: several parameter choices here are rather arbitrary and
# even not so conventional (e.g., spectral radius > 1), but this is just for the
# sake of the example. Many other constellations produce also satisfactory results,
# so feel free to play around with them.
esn = EchoStateNetwork(
    n_inputs=1,
    n_outputs=1,
    n_reservoir=200,
    spectral_radius=1.25,
    teacher_forcing=True,
    leak_rate=.4,
    regression_params={
        "method": "pinv"
    },
    random_seed=42,
    verbose=True
)

# Load data and define train/test length
data = load_mackeyglasst17().reshape(-1, 1)
trainlen, testlen = 2000, 2000
totallen = trainlen + testlen
# Fit the model. Inputs is None because we only have the target time series
esn.fit(None, data[:trainlen]);

# Input is None because we will use the generative mode, were only the feedback
# is needed to compute the next states and predict outputs
prediction = esn.predict(None, mode="generative", n_steps=testlen)

# Plot test
plt.figure(figsize=(22, 5))
plt.subplot(1, 4, (1, 3))
plt.title("test")
plt.plot(data[trainlen: totallen],
         color="steelblue",
         label="target system",
         linewidth=5.5)
plt.xlabel('time')

plt.plot(prediction,
         linestyle='--',
         color="orange",
         linewidth=2,
         label="generative ESN",)
plt.ylabel("oscillator")
plt.xlabel('time')
plt.legend(fontsize='small')
plt.show()
