test:
	mypy echoes/esn/_base.py --ignore-missing-imports
	mypy echoes/esn/_regressor.py --ignore-missing-imports
	mypy echoes/esn/_generator.py --ignore-missing-imports
	mypy echoes/utils.py --ignore-missing-imports
	mypy echoes/datasets.py --ignore-missing-imports
	mypy echoes/plotting/_core.py --ignore-missing-imports
	python -m pytest tests
