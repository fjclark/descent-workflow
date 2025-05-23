"""
This is almost identical to https://github.com/openforcefield/sage-2.2.1/blob/main/05_benchmark_forcefield/benchmark.py
This is included in a seperate script, rather than as a function in `benchmark.py` because the patterm:

```
from multiprocessing import freeze_support

def main():
    # Your code here

if __name__ == "__main__":
    freeze_support()
    main()
```
Is required for `multiprocessing` to work correctly. See https://github.com/openforcefield/yammbs/tree/main.
"""

import logging
import os
import time
import warnings

import click
from yammbs import MoleculeStore
from openff.qcsubmit.results import OptimizationResultCollection
from multiprocessing import freeze_support

# try to suppress stereo warnings - from lily's valence-fitting
# curate-dataset.py
logging.getLogger("openff").setLevel(logging.ERROR)

# suppress divide by zero in numpy.log
warnings.filterwarnings(
    "ignore", message="divide by zero", category=RuntimeWarning
)

import loguru

from yammbs.cached_result import CachedResultCollection

logger = loguru.logger

# this code is from Brent's benchmarking repo

@click.command()
@click.option("--forcefield", "-f", default="force-field.offxml")
@click.option("--dataset", "-d", default="datasets/industry.json")
@click.option("--sqlite-file", "-s", default="tmp.sqlite")
@click.option("--out-dir", "-o", default=".")
@click.option("--procs", "-p", default=16)
@click.option("--invalidate-cache", "-i", is_flag=True, default=False)
def main(forcefield, dataset, sqlite_file, out_dir, procs, invalidate_cache):
    if invalidate_cache and os.path.exists(sqlite_file):
        os.remove(sqlite_file)
    if os.path.exists(sqlite_file):
        logger.info(f"loading existing database from {sqlite_file}")
        store = MoleculeStore(sqlite_file)
    else:
        #logger.info(f"loading initial dataset from {dataset}")
        #opt = OptimizationResultCollection.parse_file(dataset)

        #logger.info(f"generating database, saving to {sqlite_file}")
        #store = MoleculeStore.from_qcsubmit_collection(opt, sqlite_file)
        logger.info(f"loading cached results from {dataset}",flush=True)
        cache = CachedResultCollection.from_json(dataset)

        store=MoleculeStore.from_cached_result_collection(cache,sqlite_file)

    logger.info("started optimizing store")
    start = time.time()
    store.optimize_mm(force_field=forcefield, n_processes=procs)
    logger.info(f"finished optimizing after {time.time() - start} sec")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    store.get_dde(forcefield,skip_check=True).to_csv(f"{out_dir}/dde.csv")
    store.get_rmsd(forcefield,skip_check=True).to_csv(f"{out_dir}/rmsd.csv")
    store.get_tfd(forcefield,skip_check=True).to_csv(f"{out_dir}/tfd.csv")
    store.get_internal_coordinate_rmsd(forcefield,skip_check=True).to_csv(f"{out_dir}/icrmsd.csv")



if __name__ == "__main__":
    freeze_support()
    main()

