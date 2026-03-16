SHELL := /bin/zsh
PROJECT_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CONDA ?= /Users/charles/miniconda3/bin/conda
ENV_NAME ?= conda_rag

.PHONY: help conda-create conda-update dev-install lint format test pre-commit-install pre-commit-run debug-ingest

help:
	@echo "Targets:"
	@echo "  conda-create        Create $(ENV_NAME) from environment.yml"
	@echo "  conda-update        Update $(ENV_NAME) from environment.yml"
	@echo "  dev-install         Editable install in the active interpreter"
	@echo "  lint                Run ruff check"
	@echo "  format              Run ruff format"
	@echo "  test                Run pytest"
	@echo "  pre-commit-install  Install git hooks"
	@echo "  pre-commit-run      Run pre-commit on all files"
	@echo "  debug-ingest        Run examples/debug_ingest.py in $(ENV_NAME)"

conda-create:
	$(CONDA) env create -f environment.yml

conda-update:
	$(CONDA) env update -n $(ENV_NAME) -f environment.yml --prune

dev-install:
	python -m pip install --upgrade pip
	python -m pip install -e '.[dev]'

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -q

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

debug-ingest:
	$(CONDA) run -n $(ENV_NAME) python examples/debug_ingest.py

