# echoes
Pythonic Echo State Networks (work in progress!)

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
 - Mackey-Glass-t17 (generative mode) [(notebook)](https://github.com/fabridamicelli/echoes/blob/master/examples/MackeyGlass-t17.ipynb)

```python

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
```
![Alt Text](https://github.com/fabridamicelli/echoes/blob/master/examples/mackeyglasst17.png)

---
 - sin-cos (predictive mode) [(notebook)](https://github.com/fabridamicelli/echoes/blob/master/examples/sin-cos.ipynb)

```python

from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.metrics import mean_squared_error

from echoes.esn import EchoStateNetwork

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

esn = EchoStateNetwork(
    n_inputs=1,
    n_outputs=1,
    n_reservoir=20,
    spectral_radius=.95,
    leak_rate=.4,
    n_transient=100,
    teacher_forcing=False,
    regression_params={
        "method": "pinv",
    },
    random_seed=42
).fit(inputs_train, outputs_train)

prediction_test = esn.predict(inputs_test, mode="predictive")

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
![Alt Text](https://github.com/fabridamicelli/echoes/blob/master/examples/sin-cos.png)


## Requirements
The code has been tested with Python 3.7 on Ubuntu 18.04.
   - numpy
   - sklearn
   - examples:
     - matplotlib
     - seaborn
   - tests:
     - mypy
     - pytest 


## Features
 - input scaling and shift
 - prediction modes: 
   - predictive
   - generative
 - reservoir sparsity
 - leakeage rate
 - input noise (regularization)
 - arbitrary:
   - activation function
   - bias 
   - readout function
   - reservoir matrix
   - input matrix
   - feedback matrix
   - dynamical regime
 - regression parameters:
   - method: 
     - pinv
     - ridge 
     - ridge-formula
   - sklearn Ridge solvers
 - fit only states (fit outgoing weights without inputs and bias)


TODO:
 - teacher scaling and shift
 - accelerate with numba?


## Tests 
Run tests with 
```
make test
```

## References
  - [Reservoir computing approaches to recurrent neural network training, Mantas & Jaeger, 2009](https://www.sciencedirect.com/science/article/pii/S1574013709000173)
  - [A Practical Guide to Applying Echo State Networks, Mantas, 2012](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_36)
  - [Echo state network (scholarpedia), Jaeger](http://www.scholarpedia.org/article/Echo_state_network)
