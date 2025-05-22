## 1.0.0 (2025-05-22)
### Breaking changes
- Drop support for python 3.7 and 3.8
- Require `scikit-learn >= 1.2.0` because of deprecated parameter (see[this issue](https://github.com/scikit-learn/scikit-learn/pull/17772))
- Remove parameter `ridge_normalize` in all estimators to follow the scikit-learn deprecation 

### Refactor
- Improve test coverage and documentation.
- Fix type annotations
