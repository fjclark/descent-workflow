# descent-workflow
A [snakemake](https://snakemake.github.io/)-based workflow to fit force-fields with [descent](https://github.com/SimonBoothroyd/descent/tree/main).

Install the environment with
```bash
make env
```

Then, run the workflow with
```bash
cd workflow
snakemake --cores all train --config workflow_config_path=configs/initial_fit_espaloma_linearised_harmonics.yaml
```
This will run the workflow with the config file `configs/initial_fit_espaloma_linearised_harmonics.yaml`. All main settings can be modified in this file, including specifying the functions to use to run different stages.
