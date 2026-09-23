.PHONY: pipeline test

pipeline:
	uv run python -m demand_lab

test:
	uv run pytest -q
