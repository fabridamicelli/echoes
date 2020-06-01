# echoes 
(work in progress!)

Scikit-learn compatible, high level API for machine learning with Echo State Networks (ESN).

Check out the examples for a quick start.

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
 - sklearn
 - pandas
 - matplotlib
 - seaborn
 - tests: mypy, pytest 

The code has been tested with Python 3.7 on Ubuntu 16.04.

### Datasets
 - Mackey-Glass-t17 

## Tests 
Run tests with 
```
make test
```
