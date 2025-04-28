PACKAGE_NAME  := descent-workflow
CONDA_ENV_RUN := conda run --no-capture-output --name $(PACKAGE_NAME)

.PHONY: env lint format

env:
	mamba create     --name $(PACKAGE_NAME)
	mamba env update --name $(PACKAGE_NAME) --file environment.yaml
	$(CONDA_ENV_RUN) pre-commit install || true

lint:
	$(CONDA_ENV_RUN) ruff check --fix .

format:
	$(CONDA_ENV_RUN) ruff format .

type-check:
	$(CONDA_ENV_RUN) mypy --follow-imports=silent --ignore-missing-imports --strict workflow
