"""This is almost identical to https://github.com/openforcefield/sage-2.2.1/blob/main/05_benchmark_forcefield/benchmark.py
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
from multiprocessing import freeze_support

import click
import loguru
import numpy
import yammbs._minimize
from yammbs import MoleculeStore
from yammbs.inputs import QCArchiveDataset

_minimize_openmm = yammbs._minimize._minimize_openmm


def _minimize_openmm_with_v_sites(mol, system, positions):
    """yammbs passes only atom positions, but force fields with virtual sites give
    systems with extra (trailing) v-site particles. Pad with placeholder positions
    (``computeVirtualSites`` sets the real ones) and return only the atom positions."""
    n_v_sites = system.getNumParticles() - len(positions)
    if n_v_sites == 0:
        return _minimize_openmm(mol, system, positions)
    padded = numpy.vstack([positions, numpy.zeros((n_v_sites, 3))])
    final_positions, final_energy = _minimize_openmm(mol, system, padded)
    return final_positions[: len(positions)], final_energy


# Patched at import time so that multiprocessing workers (forked or spawned, which
# re-import this module) also pick it up.
yammbs._minimize._minimize_openmm = _minimize_openmm_with_v_sites

# try to suppress stereo warnings - from lily's valence-fitting
# curate-dataset.py
logging.getLogger("openff").setLevel(logging.ERROR)

# suppress divide by zero in numpy.log
warnings.filterwarnings("ignore", message="divide by zero", category=RuntimeWarning)

logger = loguru.logger

# this code is from Brent's benchmarking repo


@click.command()
@click.option("--forcefield", "-f", default="force-field.offxml")
@click.option("--dataset", "-d", default="datasets/industry.json")
@click.option("--sqlite-file", "-s", default="tmp.sqlite")
@click.option("--out-dir", "-o", default=".")
@click.option("--procs", "-p", default=16)
@click.option("--invalidate-cache", "-i", is_flag=True, default=False)
def main(
    forcefield: str,
    dataset: str,
    sqlite_file: str,
    out_dir: str,
    procs: int,
    invalidate_cache: bool,
) -> None:
    if invalidate_cache and os.path.exists(sqlite_file):
        os.remove(sqlite_file)
    if os.path.exists(sqlite_file):
        logger.info(f"loading existing database from {sqlite_file}")
        store = MoleculeStore(sqlite_file)
    else:
        print(f"loading cached dataset from {dataset}", flush=True)
        with open(dataset) as inp:
            crc = QCArchiveDataset.model_validate_json(inp.read())
        store = MoleculeStore.from_qcarchive_dataset(crc, sqlite_file)

    logger.info("started optimizing store")
    start = time.time()
    store.optimize_mm(force_field=forcefield, n_processes=procs)
    logger.info(f"finished optimizing after {time.time() - start} sec")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    store.get_dde(forcefield, skip_check=True).to_csv(f"{out_dir}/dde.csv")
    store.get_rmsd(forcefield, skip_check=True).to_csv(f"{out_dir}/rmsd.csv")
    store.get_tfd(forcefield, skip_check=True).to_csv(f"{out_dir}/tfd.csv")
    store.get_internal_coordinate_rmsd(forcefield, skip_check=True).to_csv(
        f"{out_dir}/icrmsd.csv"
    )


if __name__ == "__main__":
    freeze_support()
    main()
