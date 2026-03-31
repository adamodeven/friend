PYTHON ?= python3

.PHONY: install dev test run migrate bootstrap

install:
	$(PYTHON) -m pip install -e .

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

migrate:
	alembic upgrade head

bootstrap:
	$(PYTHON) scripts/bootstrap_user.py

