test:
	uv run python -m pytest -vv tests
	
lint:
	uv run mypy . --ignore-missing-imports --no-strict-optional

release:
	rm dist/*
	# python setup.py sdist bdist_wheel
	# twine upload dist/*
	uv build
	uv publish

docs:
	mkdocs gh-deploy
