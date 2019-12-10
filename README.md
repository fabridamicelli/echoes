# echoes
Pythonic Echo State Networks (work in progress!)

## Examples
   - [Mackey-Glass-t17](add figure and link to notebook)
   - [sin -> cos](add)
   - shifted sequences
   - frequency generator


## Requirements
   - numpy
   - sklearn
   - Tests:
     - mypy
     - pytest 

## Features
   - prediction modes: generative, predictive
   - sparsity of reservoir 
   - arbitrary activation function
   - leakeage rate
   - arbitrary bias 
   - regression parameters:
     - method: pinv, ridge, ridge-formula
     - solver: sklearn Ridge solvers
   - TODO
     - input scaling and shifting
     - arbitrary readout function
     - arbitrary reservoir matrix
     - arbitrary dynamical regime
     - accelerate with numba?
     - teacher scaling and shift
## Tasks
   - memory capacity
     - evolution of RMSE in (test) time
   - recall task
   - furthest accurate prediction (epsilon, steps on test set (thesis is pretty short)
   - sin -> cos
   - narma
   - perturbation decay in time
   - frequency generator

## Tests 
Run tests with 
```
make test
```

### TODO - we need more unit tests!
 - different types of input (generative, non generative mode)
 - input with multiple dimensions
 - output with multiple dimensions
 - assert shapes of arrays
 - with and without continuation
 - different activation functions
 - mypy:
   - esn
   - utils functions
 - arbitrary input matrices -> adjust spectral radius.
 - TODO: check: bias influences the states evolution?

# Resources
## Literature
  - Mantas
  - scholarpedia Herbert Jaeger
## Python implementations [](add links)
  - pyESN: adaptation (pythonic, modular, features, etc.)
  - easyesn
  - nathaniel
  - mantas
