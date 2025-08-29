PACKAGE_NAME  := descent-workflow

# Detect which conda package manager is available
CONDA_CMD := $(shell \
	if command -v micromamba >/dev/null 2>&1; then \
		echo "micromamba"; \
	elif command -v mamba >/dev/null 2>&1; then \
		echo "mamba"; \
	elif command -v conda >/dev/null 2>&1; then \
		echo "conda"; \
	else \
		echo "conda"; \
	fi)

CONDA_ENV_RUN := $(CONDA_CMD) run -a "" --name $(PACKAGE_NAME)

.PHONY: env lint format

env:
	$(CONDA_CMD) create     --name $(PACKAGE_NAME)
	$(CONDA_CMD) env update --name $(PACKAGE_NAME) --file environment.yaml
	$(CONDA_ENV_RUN) pre-commit install || true

lint:
	$(CONDA_ENV_RUN) ruff check --fix .

format:
	$(CONDA_ENV_RUN) ruff format .

type-check:
	$(CONDA_ENV_RUN) mypy --follow-imports=silent --ignore-missing-imports --strict workflow
