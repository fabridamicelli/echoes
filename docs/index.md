# echoes 
High level API for machine learning with Echo State Networks (ESN) – work in progress!.

Check out the examples for a quick start and [What are Echo State Networks?](https://fabridamicelli.github.io/echoes/tutorial/) section for a little intro about Echo State Networks.

The library is scikit-learn compatible, thus you can directly use sklearn utils, such as ```GridSearchCV```.
Moreover, it tries to stick to the intuitions that you might bring from using sklearn.
For example, models can be initialized without passing arguments (but kwargs are enforced if you choose to pass any); attributes generated during fitting are stored with trailing underscore; etc.


## Installation
You can install the package via pip
```sh
pip install echoes
```

## Citing
If you find echoes useful for a publication, then please use the following BibTeX to cite it:

```
@misc{echoes,
  author = {Damicelli, Fabrizio},
  title = {echoes: Echo State Networks with Python},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fabridamicelli/echoes}},
}
```

## Requirements
### Dependencies
 - numpy
 - numba
 - scikit-learn
 - pandas
 - matplotlib
 - seaborn 
 - tests: mypy, pytest 

The code has been tested with Python 3.7 on Ubuntu 16.04/18.04.

## Tests 
Run tests with 
```
make test
```
