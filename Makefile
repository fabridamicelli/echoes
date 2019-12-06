test:
	echo 'checking type hints'
	mypy esn.py --ignore-missing-imports
	mypy utils.py --ignore-missing-imports
	mypy datasets.py --ignore-missing-imports
