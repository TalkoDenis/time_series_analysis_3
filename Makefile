.PHONY: clean install test lint lint-fix test test-coverage test-total format format-fix show-resourses start-project mypy

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf *.egg-info .pytest_cache/ .ruff_cache/ coverage.xml htmlcov/ profile.html build/ dist/

install:
	uv pip sync --extra dev pyproject.toml

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format --check .

format-fix:
	uv run ruff format .

test:
	uv run pytest

test-coverage:
	uv run pytest --cov=stat_project --cov-report=xml:coverage.xml

test-total:

	uv run pytest --cov=stat_project

show-resourses:
	uv run scalene --html --outfile profile.html main.py

start-project:
	uv run python main.py

mypy:
	uv run mypy stat_project