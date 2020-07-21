test:
	mypy echoes/esn/_base.py --ignore-missing-imports --no-strict-optional
	mypy echoes/esn/_regressor.py --ignore-missing-imports --no-strict-optional
	mypy echoes/esn/_generator.py --ignore-missing-imports --no-strict-optional
	mypy echoes/utils.py --ignore-missing-imports --no-strict-optional
	mypy echoes/datasets.py --ignore-missing-imports --no-strict-optional
	mypy echoes/plotting/_core.py --ignore-missing-imports --no-strict-optional
	python -m pytest tests

release:
	rm dist/*
	python setup.py sdist bdist_wheel
	twine upload dist/*
