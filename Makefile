test:
	mypy echoes/esn/_base.py --ignore-missing-imports
	mypy echoes/esn/_predictive.py --ignore-missing-imports
	mypy echoes/esn/_generative.py --ignore-missing-imports
	mypy echoes/utils.py --ignore-missing-imports
	mypy echoes/datasets.py --ignore-missing-imports
	mypy echoes/plotting/_core.py --ignore-missing-imports
	mypy echoes/plotting/_memory_capacity.py --ignore-missing-imports
	mypy echoes/tasks/_memory_capacity.py --ignore-missing-imports
	python -m pytest tests
