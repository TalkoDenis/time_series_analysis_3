.PHONY: clean install test lint

clean:
	rm -rf dist/ build/ *.egg-info


install:
	uv sync


lint:
	uv run ruff check main.py


lint-fix:
	uv run ruff check --fix


test:
	uv run pytests