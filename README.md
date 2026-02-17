# descent-workflow
A [snakemake](https://snakemake.github.io/)-based workflow to fit force-fields with [descent](https://github.com/SimonBoothroyd/descent/tree/main).

## Setup

This project uses [pixi](https://pixi.sh) for environment management. Install pixi first:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Running the Workflow

Run the workflow with e.g.:

```bash
pixi run snakemake --cores all train --config workflow_config_path=configs/initial_fit_espaloma_linearised_harmonics.yaml
```

This will run the workflow with the config file `configs/initial_fit_espaloma_linearised_harmonics.yaml`. All main settings can be modified in this file, including specifying the functions to use to run different stages.

## Development

Install the pre-commit hook with

```bash
pixi run setup
```

```bash
# Format code
pixi run format

# Lint code
pixi run lint

# Type check
pixi run type-check

# Run all checks
pixi run check
```
