## 1.0.2 (2025-06-04)
### Refactor
- Add sklearn tags and refactor `ESNRegressor` fit/predict methods to make it pass the `check_estimator` check for scikit-learn (v1.6) compatibility.
- Add test `check_estimator`

### Documentation
- Add notebook examples using scikit-learn `GridSearchCV` and `Pipeline`

## 1.0.1 (2025-05-22)
### Fix
- Add missing data

## 1.0.0 (2025-05-22)
### Breaking changes
- Drop support for python 3.7 and 3.8
- Require `scikit-learn >= 1.2.0` because of deprecated parameter (see[this issue](https://github.com/scikit-learn/scikit-learn/pull/17772))
- Remove parameter `ridge_normalize` in all estimators to follow the scikit-learn deprecation 

### Fix
- Move call to `activation_out` into the predict inner loop to properly apply activation on outputs

### Refactor
- Remove magic shift factor 0.5 in `update_states`
- Improve test coverage and documentation.
- Fix type annotations
- Adopt `pyproject.toml`
- Migrate to `uv`
