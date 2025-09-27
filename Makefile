test:
	uv run python -m pytest -vv tests
	
lint:
	uv run mypy . --ignore-missing-imports --no-strict-optional

check-formatting:
	uv run black --check .

release:
	rm dist/*
	uv build
	uv publish

docs:
	mkdocs gh-deploy
