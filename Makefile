.PHONY: clean install test lint

clean:
	rm -rf dist/ build/ *.egg-info


install:
	uv sync


lint:
	uv run ruff check brain_games


lint-fix:
	uv run ruff check --fix