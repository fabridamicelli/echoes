test:
	mypy echoes/esn.py --ignore-missing-imports
	mypy echoes/utils.py --ignore-missing-imports
	mypy echoes/datasets.py --ignore-missing-imports
	python -m pytest tests
