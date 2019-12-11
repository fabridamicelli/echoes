# echoes
Pythonic Echo State Networks (work in progress!)

## Examples
 - [Mackey-Glass-t17 (generative mode)](https://github.com/fabridamicelli/echoes/blob/master/examples/MackeyGlass-t17.ipynb)
![Alt Text](https://github.com/fabridamicelli/echoes/blob/master/examples/mackeyglasst17.png)

---
 - [sin-cos (predictive mode)](https://github.com/fabridamicelli/echoes/blob/master/examples/sin-cos.ipynb)
![Alt Text](https://github.com/fabridamicelli/echoes/blob/master/examples/sin-cos.png)

## Requirements
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
