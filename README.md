# echoes 
(work in progress!)

Python module for Echo State Networks (ESN).

High level API for machine learning with Echo State Networks.
The design attempts to follow the intuitions from standard packages, eg scikit-learn,
as much as possible. Documentation, type annotations and examples are all over the place.

As intended for research purposes as well, several typical benchmark tasks, datasets
and plotting functionalities are also included and straight forward to run, thus 
saving boiler plate code and allowing the user to quickly test the ESN.


## Installation
[Download](https://github.com/fabridamicelli/echoes/archive/master.zip) or clone repo like this:
```sh
git clone https://github.com/fabridamicelli/echoes
```

Then cd to directory and pip install:
```sh
cd echoes
pip install .
```
Recommendation: install the package in a separate virtual environment, e.g., created with [(mini)conda](https://conda.io/docs/user-guide/install/index.html).

## Examples
 - Mackey-Glass-t17 (generative ESN) [(notebook)](https://nbviewer.jupyter.org/github/fabridamicelli/echoes/blob/master/examples/notebooks/mackeyglasst17.ipynb)

```python

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

from echoes import ESNGenerative
from echoes.datasets import load_mackeyglasst17

sns.set(context="notebook", style="whitegrid", font_scale=1.4, 
        rc={'grid.linestyle': '--', 
            'grid.linewidth': 0.8,})

# Instantiate the model: several parameter choices here are rather arbitrary and 
# even not so conventional (e.g., spectral radius > 1), but this is just for the 
# sake of the example. Many other constellations produce also satisfactory results, 
# so feel free to play around with them.
esn = ESNGenerative(
    n_inputs=1,
    n_outputs=1,
    n_reservoir=200,
    spectral_radius=1.25,
    teacher_forcing=True,
    leak_rate=.4,
    regression_method="pinv",
    random_seed=42,
)

# Load data and define train/test length
data = load_mackeyglasst17().reshape(-1, 1)
trainlen, testlen = 2000, 2000
totallen = trainlen + testlen
# Fit the model. Inputs is None because we only have the target time series
esn.fit(None, data[:trainlen]);

# Input is None because we will use the generative mode, were only the feedback 
# is needed to compute the next states and predict outputs
prediction = esn.predict(n_steps=testlen)

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
```
![Alt Text](https://github.com/fabridamicelli/echoes/blob/master/examples/figs/mackeyglasst17.png)

---
 - sin-cos (predictive ESN) [(notebook)](https://nbviewer.jupyter.org/github/fabridamicelli/echoes/blob/master/examples/notebooks/sincos.ipynb)
                                          
```python
from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.metrics import mean_squared_error

from echoes import ESNPredictive

sns.set(context="notebook", style="whitegrid", font_scale=1.4, 
        rc={'grid.linestyle': '--', 
            'grid.linewidth': 0.8,})

# Prepare synthetic data 
traininglen, testlen = 500, 500
totallen = traininglen + testlen
x = np.linspace(0, 30*np.pi, totallen).reshape(-1,1)

inputs = np.sin(x)
outputs = np.cos(x)

inputs_train = inputs[: traininglen]
outputs_train = outputs[: traininglen]

inputs_test= inputs[traininglen:]
outputs_test = outputs[traininglen:]

esn = ESNPredictive(
    n_inputs=1,
    n_outputs=1,
    n_reservoir=20,
    spectral_radius=.95,
    leak_rate=.4,
    n_transient=100,
    teacher_forcing=False,
    regression_method="pinv",
    random_seed=42
).fit(inputs_train, outputs_train)

prediction_test = esn.predict(inputs_test)

# Plot test (discarding same initial transient as for training)
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, (1,2))
plt.plot(outputs_test[esn.n_transient:], label='target signal',
         color="steelblue", linewidth=5.5)
plt.plot(prediction_test[esn.n_transient:], label='predicted signal',
         linestyle='--', color="orange",  linewidth=2,)
plt.ylabel("oscillator")
plt.xlabel('time')
plt.legend(fontsize=("small"), loc=2)

```
![Alt Text](https://github.com/fabridamicelli/echoes/blob/master/examples/figs/sincos.png)

---
 - Memory capacity [(notebook)](https://nbviewer.jupyter.org/github/fabridamicelli/echoes/blob/master/examples/notebooks/memory_capacity.ipynb)

```python
import numpy as np

from echoes.tasks import MemoryCapacity
from echoes.plotting import plot_forgetting_curve, set_mystyle
set_mystyle() # make nicer plots, can be removed

# Echo state network parameters (after Jaeger)
n_reservoir = 20
W = np.random.choice([0, .47, -.47], p=[.8, .1, .1], size=(n_reservoir, n_reservoir))
W_in = np.random.choice([.1, -.1], p=[.5, .5], size=(n_reservoir, 2))

# Task parameters (after Jaeger)
inputs_func=np.random.uniform
inputs_params={"low":-.5, "high":.5, "size":200}
lags = [1, 2, 5, 10, 15, 20, 25, 30]

esn_params = dict(
    n_inputs=1,
    n_outputs=len(lags),  # automatically decided based on lags
    n_reservoir=20,
    W=W,
    W_in=W_in,
    spectral_radius=.9,
    bias=0,
    n_transient=100,
    regression_method="pinv",
    random_seed=42,
)

# Initialize the task object
mc = MemoryCapacity(
    inputs_func=inputs_func,
    inputs_params=inputs_params,
    esn_params=esn_params,
    lags=lags
).fit_predict()  # Run the task

plot_forgetting_curve(mc.lags, mc.forgetting_curve_)
```
![Alt Text](https://github.com/fabridamicelli/echoes/blob/master/examples/figs/memory_capacity.png)

---

 - Plot reservoir activity [(notebook)](https://nbviewer.jupyter.org/github/fabridamicelli/echoes/blob/master/examples/notebooks/reservoir_activity.ipynb)

```python
import numpy as np

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
    regression_method="pinv",
    store_states_pred=True,   # store states for plotting
    random_seed=42,
).fit(None, data[: trainlen])  # fit the model 

prediction = esn.predict(n_steps=testlen)

# Pick 9 neurons at random to plot
neurons_to_plot = sorted(np.random.randint(0, esn.n_reservoir, size=9))

plot_reservoir_activity(esn,
                        neurons_to_plot,
                        pred=True,   # plot activity during prediction
                        end=500,
                        figsize=(12, 8))
```
![Alt Text](https://github.com/fabridamicelli/echoes/blob/master/examples/figs/reservoir_activity.png)


## Requirements
### Dependencies
   - numpy
   - sklearn
   - pandas
   - matplotlib
   - seaborn
   - tests:
     - mypy
     - pytest 

The code has been tested with Python 3.7 on Ubuntu 18.04.

## Features
### ESN types
 - predictive
 - generative
### ESN hyperparameters
 - input scaling and shift
 - reservoir sparsity
 - leakage rate
 - input noise (regularization)
 - arbitrary:
   - activation function
   - bias 
   - activation_out function
   - reservoir matrix
   - input matrix
   - feedback matrix
 - fit only states (fit outgoing weights without inputs and bias)
 - regression parameters:
   - method: 
     - pinv
     - ridge (sklearn Ridge parameters available)
     - ridge-formula

### Plotting
 - True vs predicted time series.
 - Task related plots (e.g, forgetting curve).
 - Reservoir neurons activity.

### Tasks 
 - Memory capacity

### Model selection
 - Grid search

### Datasets
 - Mackey-Glass-t17 

### TODO
 - teacher scaling and shift
 - neuron models 
 - numba acceleration
 - add Tasks 
 - add Datasets
 - ensemble
 - visualization grid search

## Tests 
Run tests with 
```
make test
```

## References
  - [Reservoir computing approaches to recurrent neural network training, Mantas & Jaeger, 2009](https://www.sciencedirect.com/science/article/pii/S1574013709000173)
  - [A Practical Guide to Applying Echo State Networks, Mantas, 2012](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_36)
  - [Echo state network (scholarpedia), Jaeger](http://www.scholarpedia.org/article/Echo_state_network)
  - [Short Term Memory in Echo State Networks, Jaeger, 2001](http://publica.fraunhofer.de/eprints/urn_nbn_de_0011-b-731310.pdf)
