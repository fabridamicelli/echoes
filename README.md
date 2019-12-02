# echoes
Bioinspired, Pythonic Echo State Networks

## Requirements
   - numpy
   - sklearn

# TODO
## ESN implementation
   - add docstrings
   - add annotations

## Notes
   - weights explote with OLS fitting
   - regularization: noisy input to x(t), ridge or both noise
   - mind huge regularization effect of noisy input
## Features
   - linear regression options
   - sparsity? reservoir links to delete
   - arbitrary reservoir matrix
   - arbitrary dynamical regime
   - arbitrary activation function
   - arbitrary readout function
   - main parameters:
     - spectral radius, reservoir size
   - later on: accelerate with numba (collect states)
   - leakeage rate
   - input scaling and shifting
   - bias neuron - clear impact on prediction
   - prediction modes: continous/input-based


## Tasks
   - memory capacity
     - evolution of RMSE in (test) time
   - recall task
   - furthest accurate prediction (epsilon, steps on test set (thesis is pretty short)

## Tests - we need unit tests!
   - different types of input (generative, non generative mode)
   - input with multiple dimensions
   - output with multiple dimensions
   - assert shapes of arrays
   - with and without continuation
   - different activation functions
   - mypy:
	- utils functions


# Resources
  - pyESN: adaptation (pythonic, modular, features, etc.)
  - easyesn
  - nathaniel??

