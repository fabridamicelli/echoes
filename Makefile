test:
	python -m pytest -vv tests
	
lint:
	mypy echoes/esn/_base.py --ignore-missing-imports --no-strict-optional
	mypy echoes/esn/_regressor.py --ignore-missing-imports --no-strict-optional
	mypy echoes/esn/_generator.py --ignore-missing-imports --no-strict-optional
	mypy echoes/utils.py --ignore-missing-imports --no-strict-optional
	mypy echoes/datasets.py --ignore-missing-imports --no-strict-optional
	mypy echoes/plotting/_core.py --ignore-missing-imports --no-strict-optional

release:
	rm dist/*
	python setup.py sdist bdist_wheel
	twine upload dist/*

docs:
	mkdocs gh-deploy
