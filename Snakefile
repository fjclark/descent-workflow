"""
Snakefile to control the overall fitting workflow. Run with e.g.
`snakemake --cores all train --configfile configs/initial_fit_espaloma_linearised_harmonics.yaml`
"""
import importlib
from pathlib import Path
from typing import Callable

from loguru import logger

from descent_workflow.filter import filter_and_cluster_espaloma
from descent_workflow.get_data import get_data_espaloma
from descent_workflow.get_data_byte_dance import get_data_byte_dance
from descent_workflow.models import WorkflowConfig
from descent_workflow.parameterise import create_torch_ff_and_top
from descent_workflow.train import train
from descent_workflow.utils import get_fn
from descent_workflow.generate_types import generate_bespoke_types, TypeGenConfig
from descent_workflow.benchmark import run_yammbs_benchmarking

# Load the configuration from a yaml file
if workflow.configfiles:
    if len(workflow.configfiles) > 1:
        raise ValueError(f"Multiple config files provided: {workflow.configfiles}. Please provide only one config file.")
    CONFIG_FILE = workflow.configfiles[0]
else:
    raise ValueError("workflow_config_path not found in SnakeMake config."
                     " Please provide the path to the workflow configuration file with e.g."
                     " `snakemake --cores all train --configfile configs/initial_fit_espaloma_linearised_harmonics.yaml`")

workflow_config = WorkflowConfig.from_file(CONFIG_FILE)
logger.info(f"Loaded workflow configuration {workflow_config}")


rule get_data:
    output:
        workflow_config.get_data_output_smiles
    run:
        get_data_fn = get_fn(workflow_config.get_data_fn)
        get_data_fn(workflow_config.data_dir)
        workflow_config.to_file(workflow_config.data_dir / "workflow_config.yaml")

rule parameterise:
    input:
        workflow_config.get_data_output_smiles
    output:
        workflow_config.torch_ffs_and_tops_path
    run:
        create_torch_ff_and_top(workflow_config)
        workflow_config.to_file(workflow_config.data_dir / "workflow_config.yaml")


rule filter_and_cluster:
    input:
        workflow_config.torch_ffs_and_tops_path
    output:
        directory(workflow_config.filtered_data_dir)
    run:
        filter_and_cluster_fn = get_fn(workflow_config.filter_and_cluster_fn)
        filter_and_cluster_fn(workflow_config)
        workflow_config.to_file(workflow_config.filtered_data_dir / "workflow_config.yaml")

rule generate_bespoke_types:
    input:
        workflow_config.filtered_data_dir,
        workflow_config.starting_force_field_path
    output:
        workflow_config.bespoke_types_ff_path
    run:
        if workflow_config.type_generation_config is None:
            logger.info("Type generation config is None, skipping bespoke type generation")
            # Create a symlink or copy to satisfy Snakemake dependencies
            from shutil import copy2
            workflow_config.type_gen_output_dir.mkdir(parents=True, exist_ok=True)
            copy2(workflow_config.starting_force_field_path, workflow_config.bespoke_types_ff_path)
        else:
            logger.info("Generating bespoke types from configuration")
            # Parse dict into TypeGenConfig for validation
            type_gen_config = TypeGenConfig(**workflow_config.type_generation_config)
            
            # Generate bespoke types
            data_dir = workflow_config.filtered_data_dir / "data-train"
            test_data_dir = workflow_config.filtered_data_dir / "data-test"
            generate_bespoke_types(
                config=type_gen_config,
                base_ff_path=workflow_config.starting_force_field_path,
                data_dir=data_dir,
                output_dir=workflow_config.type_gen_output_dir,
                test_data_dir=test_data_dir,
            )
            
        workflow_config.to_file(workflow_config.type_gen_output_dir / "workflow_config.yaml")

rule reparameterize_with_bespoke_types:
    input:
        workflow_config.filtered_data_dir,
        workflow_config.bespoke_types_ff_path,
        workflow_config.torch_ffs_and_tops_path
    output:
        workflow_config.final_torch_ffs_and_tops_path
    run:
        logger.info("Re-parameterizing filtered data with bespoke force field")
        # Get SMILES from filtered dataset
        import json
        from datasets import Dataset
        
        filtered_train_data = Dataset.load_from_disk(str(workflow_config.filtered_data_dir / "data-train"))
        filtered_test_data = Dataset.load_from_disk(str(workflow_config.filtered_data_dir / "data-test"))
        
        # Extract unique SMILES from filtered datasets
        all_smiles = set(filtered_train_data["smiles"]) | set(filtered_test_data["smiles"])
        unique_smiles_sorted = sorted(all_smiles)
        
        logger.info(f"Re-parameterizing {len(unique_smiles_sorted)} unique molecules with bespoke types")
        
        # Import the parameterize function
        from descent_workflow.parameterise import apply_parameters
        import torch
        
        # Re-parameterize with bespoke force field
        force_field, topologies = apply_parameters(
            unique_smiles_sorted,
            str(workflow_config.effective_ff_path),
            linearise_harm=workflow_config.linearise_harm,
            chunk_size=workflow_config.parameterise_chunk_size,
        )
        
        # Save the new torch force field and topologies
        output_path = Path(output[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save((force_field, topologies), output_path)
        
        logger.info(f"Saved re-parameterized force field to {output_path}")

rule train:
    input:
        workflow_config.filtered_data_dir,
        workflow_config.final_torch_ffs_and_tops_path
    output:
        protected(directory(workflow_config.fit_dir)),
        workflow_config.output_ff_path
    run:
        train(workflow_config)
        workflow_config.to_file(workflow_config.fit_dir / "workflow_config.yaml")

rule get_benchmarking_data:
    output:
        "benchmarking/benchmarking_input_data/filtered-industry-cached.json"
    run:
        get_sage_benchmarking_data(output_dir="benchmarking/benchmarking_input_data")

# rule get_benchmarking_data:
#     output:
#         "benchmarking/benchmarking_input_data2/OpenFF-Industry-Benchmark-Season-1-v1.1-filtered-charge-coverage-cache.json"
#     shell:
#         """
#         cd benchmarking/benchmarking_input_data
#         python download_dataset.py                                          \
#         --name      "OpenFF Industry Benchmark Season 1 v1.1"                      \
#         --type      "optimization"                                      \
#         --output    "OpenFF-Industry-Benchmark-Season-1-v1.1.json" \
#         --filter_output "OpenFF-Industry-Benchmark-Season-1-v1.1-intermediate.json"

#         python filter_dataset_parallel.py \
#         --input                         "OpenFF-Industry-Benchmark-Season-1-v1.1-intermediate.json"        \
#         --output                        "OpenFF-Industry-Benchmark-Season-1-v1.1-filtered-charge-coverage.json"         \
#         --charge-backend                "openeye"            \
#         --forcefield                    "openff_unconstrained-2.2.0.offxml" \
#         --n-workers                     300                     \
#         --worker-type                   "local"                 \
#         --batch-size                    10                      \
#         --memory                        30                       \
#         --walltime                      48                      \
#         --queue                         "free"                  \
#         --conda-environment             "descent-workflow-deepchem" \

#         python cache_dataset.py 32 \
#         """

rule run_industry_benchmark:
    input:
        workflow_config.output_ff_path
    output:
        protected(str(workflow_config.benchmarking_dir) + "/icrmsd.csv")
        # protected("benchmarking/output/openff_unconstrained-2.2.0/icrmsd.csv")
    run:
        run_yammbs_benchmarking(workflow_config)

rule plot_industry_benchmark:
    input:
        str(workflow_config.benchmarking_dir) + "/icrmsd.csv"
    run:
        plot_benchmark()

# rule run_biaryl_torsion_benchmark:
#     input:
#         workflow_config.output_ff_path
#     output:
#         protected("benchmarking/torsion_benchmarks/rowley_biaryl/oputput/torsions.png")
#     run:
#         run_torsion_benchmark(config=workflow_config,
#                               sqlite_file="benchmarking/torsion_benchmarks/rowley_biaryl/output/torsion-data.sqlite",
#                               output_dir="benchmarking/torsion_benchmarks/rowley_biaryl/output",
#                              )

rule get_alkane_torsion_data:
    output:
        "benchmarking/torsion_benchmarks/alkanes/input_data/qca-torsion-data.json"
    run:
        get_qca_torsion_data(
            dataset_name="OpenFF Alkane Torsion Drives v1.0",
            spec_name="default",
            output_dir="benchmarking/torsion_benchmarks/alkanes/input_data"
        )

rule get_byte_dance_torsion_data:
    output:
        "benchmarking/torsion_benchmarks/{dataset_name}/input_data/qca-torsion-data.json"
    params:
        dataset_name="{dataset_name}"
    run:
        get_data_byte_dance(
            dataset_name=params.dataset_name,
            output_dir=f"benchmarking/torsion_benchmarks/{params.dataset_name}/input_data"
        )

rule run_torsion_benchmark:
    input:
        "benchmarking/torsion_benchmarks/{benchmark}/input_data/qca-torsion-data.json",
        workflow_config.output_ff_path
    output:
        "benchmarking/torsion_benchmarks/{benchmark}/output/torsions.png"
    params:
        sqlite_file="benchmarking/torsion_benchmarks/{benchmark}/output/torsion-data.sqlite",
        torsion_data_json="benchmarking/torsion_benchmarks/{benchmark}/input_data/qca-torsion-data.json",
        output_dir="benchmarking/torsion_benchmarks/{benchmark}/output",
        output_metrics="benchmarking/torsion_benchmarks/{benchmark}/output/metrics.json",
        output_minimized="benchmarking/torsion_benchmarks/{benchmark}/output/minimized.json"
    shell:
        """
        mkdir -p {params.output_dir}
        yammbs_analyse_torsions \
            --qcarchive-torsion-data {params.torsion_data_json} \
            --database-file {params.sqlite_file} \
            --output-metrics {params.output_metrics} \
            --output-minimized {params.output_minimized} \
            --plot-dir {params.output_dir} \
            --base-force-fields openff-2.3.0 \
            --extra-force-fields {workflow_config.output_ff_path} \
            --method openmm_torsion_restrained
        """


    # run:
    #     run_torsion_benchmark(
    #         config=workflow_config,
    #         torsion_data_json=params.torsion_data_json,
    #         sqlite_file=params.sqlite_file,
    #         output_dir=params.output_dir,
    #     )

rule all:
    input:
        "benchmarking/torsion_benchmarks/rowley_biaryl/output/torsions.png",
        "benchmarking/torsion_benchmarks/torsionnet_500/output/torsions.png"
