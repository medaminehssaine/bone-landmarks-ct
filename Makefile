PY ?= python

.PHONY: help data data-dev sanity test lint

help:
	@echo "data       full VerSe'20 download (~38 GB) + fracture grades"
	@echo "data-dev   masks/centroids of all splits + 3 training CTs (~0.9 GB)"
	@echo "sanity     centroid convention check, dataset overview, overlay figure"
	@echo "test lint  unit tests / ruff"

data:
	PYTHON=$(PY) bash scripts/download_verse.sh

data-dev:
	PYTHON=$(PY) bash scripts/download_verse.sh dev

sanity:
	$(PY) scripts/check_centroids.py
	$(PY) scripts/dataset_overview.py
	$(PY) scripts/make_overlays.py

test:
	$(PY) -m pytest -q

lint:
	ruff check . && ruff format --check .
