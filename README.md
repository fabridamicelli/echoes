# echoes 

- [x] Machine learning with Echo State Networks(ESN)
- [x] High level API
- [x] Follows `scikit-learn` design, eg `fit-predict` interface
- [x] `scikit-learn` compatible, eg tools like `GridSearchCV` work out of the box

Quick start: See [Documentation](https://fabridamicelli.github.io/echoes/) and examples.

## Installation
```bash
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

## Dependencies 
 - numpy
 - sklearn
 - pandas
 - matplotlib
 - seaborn
 - numba
 - tests:
   - mypy
   - pytest 

The code has been tested with Python 3.8.12 and 3.9.7 on Ubuntu 20.04.3 LTS.

## Tests 
Run tests with 
```bash
make test
```

## Contributing
Any feedback on bugs, suggestions, problems? Just open an issue, I'll very soon back to you.


## References
  - [Reservoir computing approaches to recurrent neural network training, Mantas & Jaeger, 2009](https://www.sciencedirect.com/science/article/pii/S1574013709000173)
  - [A Practical Guide to Applying Echo State Networks, Mantas, 2012](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_36)
  - [Echo state network (scholarpedia), Jaeger](http://www.scholarpedia.org/article/Echo_state_network)
  - [Short Term Memory in Echo State Networks, Jaeger, 2001](http://publica.fraunhofer.de/eprints/urn_nbn_de_0011-b-731310.pdf)
