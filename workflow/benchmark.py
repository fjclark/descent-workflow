"""Benchmark fit force fields. Many functions stolen from https://github.com/openforcefield/sage-2.2.1/tree/main"""

from pathlib import Path
import subprocess

# from yammbs.cached_result import CachedResultCollection,CachedResult
from openff.qcsubmit.results import OptimizationResultCollection
import logging
import sys
import json
import multiprocessing
import tqdm
import qcportal
from models import WorkflowConfig

import loguru

logger = loguru.logger


logging.getLogger('openff').setLevel(logging.ERROR)

SAGE22_INDUSTRY_BENCHMARK_JSON_URL = "https://raw.githubusercontent.com/openforcefield/sage-2.2.0/refs/heads/main/05_benchmark_forcefield/datasets/OpenFF-Industry-Benchmark-Season-1-v1.1-filtered-charge-coverage.json"


# From https://github.com/openforcefield/sage-2.2.1/blob/main/05_benchmark_forcefield/cache_dataset.py
def split_dataset_batch_for_cache(dataset,n=1):
    ds = dict(dataset)['entries']["https://api.qcarchive.molssi.org:443/"]
    n_keys = len(ds)
    group_size = int(n_keys/n)+1 

    split_datasets = []
    for group_idx in range(0,n):
        split_datasets.append(OptimizationResultCollection(entries={"https://api.qcarchive.molssi.org:443/": ds[group_idx*group_size:(group_idx+1)*group_size]}))

    return split_datasets


# From https://github.com/openforcefield/sage-2.2.1/blob/main/05_benchmark_forcefield/cache_dataset.py
# def cache_dataset(dataset,cache_file,n_procs=1,batch_size=500):
#     ds = OptimizationResultCollection.parse_file(dataset)
#     logger.info('Loaded dataset.',flush=True)
    
#     split_ds = split_dataset_batch_for_cache(ds,batch_size)
#     logger.info("Split dataset",flush=True)
#     logger.info(f"Number of entries in initial DS: {ds.n_results}",flush=True)
#     logger.info(f"Number of entries in split DS:   {sum([split_ds[i].n_results for i in range(0,len(split_ds))])}",flush=True)
#     logger.info(f'Size of each batch:              { [split_ds[i].n_results for i in range(0,len(split_ds))] }',flush=True)
    
    
#     logger.info('Starting cache',flush=True)
#     cache = []
#     with multiprocessing.Pool(n_procs) as pool:
#         for x in tqdm.tqdm(pool.imap(CachedResultCollection.from_qcsubmit_collection,split_ds),desc='Caching dataset',total = len(split_ds)):
#             cache.extend(x.inner) # "inner" is the internal list of cached results
#             #except qcportal.client_base.PortalRequestError:
#             #    logger.info("Error connecting to server")
#             #    split_ds.append(
    
#     logger.info('Done making cache',flush=True)
    
#     with open(cache_file,'w') as writefile:
#         jsondata = json.dumps(cache,default=CachedResult.to_dict,indent=2)
#         writefile.write(jsondata)#(cache.to_json(indent=2))

#     test = CachedResultCollection.from_json(cache_file)


def get_sage_benchmarking_data(output_dir: str | Path) -> None:
    """Get the benchmarking data used for Sage 2.2.1"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the industry benchmark json from the Sage 2.2.0 repo
    industry_benchmark_file = output_dir / "filtered-industry.json"
    if not industry_benchmark_file.exists():
        logger.info(f"Downloading {SAGE22_INDUSTRY_BENCHMARK_JSON_URL} to {industry_benchmark_file}")
        subprocess.run(
            [
                "curl",
                "-o",
                str(industry_benchmark_file),
                SAGE22_INDUSTRY_BENCHMARK_JSON_URL,
            ],
            check=True,
        )

    # Cache the dataset
    cached_industry_benchmark_file = output_dir / "filtered-industry-cached.json"
    if not cached_industry_benchmark_file.exists():
        logger.info(f"Creating cache for {industry_benchmark_file} at {cached_industry_benchmark_file}")
        # Use all available cores
        n_cores = multiprocessing.cpu_count()
        cache_dataset(
            dataset=industry_benchmark_file,
            cache_file=cached_industry_benchmark_file,
            n_procs=n_cores,
            batch_size=500,
        )


def run_yammbs_benchmarking(config: WorkflowConfig) -> None:
    """Run the yammbs benchmarking script with the given configuration."""

    # Set up directories
    output_dir = config.benchmarking_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the absolute path to required scripts/ files
    yammbs_benchmark_script_path = (Path(__file__).parent / "run_yammbs_script.py").absolute()
    ff_path = (Path(__file__).parent / config.output_ff_path).absolute()
    cached_dataset_path = (Path(__file__).parent / "benchmarking" / "benchmarking_input_data" / "filtered-industry-cached.json").absolute()

    # Run benchmark script with subprocess
    args = [
        "python",
        "-u",
        str(yammbs_benchmark_script_path),
        "--forcefield",
        str(ff_path),
        "--dataset",
        str(cached_dataset_path),
        "--sqlite-file",
        str(output_dir / "benchmark.sqlite"),
        "--out-dir",
        str(output_dir),
        "--procs",
        str(multiprocessing.cpu_count()),
    ]

    logger.info(f"Running benchmark with args: {args}")
    subprocess.run(args, check=True)
    logger.info(f"Benchmark complete. Results saved to {output_dir}", flush=True)

    # ff_path = (Path(__file__).parent / "output_ff/openff_unconstrained-2.2.0.offxml").absolute()
    # output_dir = (Path(__file__).parent / "benchmarking/output/openff_unconstrained-2.2.0").absolute()
    # cached_dataset_path = (Path(__file__).parent / "benchmarking" / "benchmarking_input_data" / "filtered-industry-cached.json").absolute()

    # # Also run the benchmark with openff2.2.0
    # args = [
    #     "python",
    #     "-u",
    #     str(yammbs_benchmark_script_path),
    #     "--forcefield",
    #     str(ff_path),
    #     "--dataset",
    #     str(cached_dataset_path),
    #     "--sqlite-file",
    #     str(output_dir / "benchmark.sqlite"),
    #     "--out-dir",
    #     str(output_dir),
    #     "--procs",
    #     str(multiprocessing.cpu_count()),
    # ]
    # logger.info(f"Running benchmark with openff2.2.0 with args: {args}")

    # subprocess.run(args, check=True)

    # logger.info(f"Benchmark complete. Results saved to {output_dir}", flush=True)

def run_torsion_benchmark(config: WorkflowConfig, sqlite_file: str | Path, torsion_data_json: str | Path, output_dir: str | Path) -> None:
    """Benchmark the force field on torsion data. Note that all ffs in the output ff dir will be benchmarked."""

    # Set up directories
    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the absolute path to all output ff files
    ff_files = list(Path(config.output_ff_dir).glob("*.offxml"))
    if not ff_files:
        raise ValueError(f"No force field files found in {config.output_ff_dir}")
    absolute_ff_files = [ff_file.absolute() for ff_file in ff_files]
    torsion_data_json = Path(torsion_data_json).absolute()
    sqlite_file = Path(sqlite_file).absolute()

    # Run benchmark script with subprocess
    args = [
        "yammbs_analyse_torsions",
        "--qcarchive-torsion-data",
        str(torsion_data_json),
        "--database-file",
        str(sqlite_file),
        "--plot-dir",
        str(output_dir),
    ]
    # Add ff files to the args
    for ff_file in absolute_ff_files:
        args.extend(["--extra-force-fields", str(ff_file)])

    logger.info(f"Running benchmark with args: {args}")
    res = subprocess.run(args, check=True, cwd=output_dir)
    logger.info(f"Benchmark complete. Results saved to {output_dir}", flush=True)