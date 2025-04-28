"""
Filter out any molecules which can't be parameterised.

Optionally, also cluster conformers by their RMSD. Some entries in `gen2-opt` have
~3000 very similar conformers.

Note: Mainly from: https://github.com/SimonBoothroyd/descent-ff/blob/main/energy-force/003-cluster-and-filter.py
"""

import functools
import multiprocessing

import datasets
import numpy
import openff.toolkit
import openff.units
import torch
import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina
from loguru import logger

import descent.targets.energy

from models import WorkflowConfig

from get_data import ESPALOMA_SOURCES

N_WORKERS = 28


def compute_best_rms(pairs: list[tuple[int, int]], mol: Chem.Mol) -> list[float]:
    # return rdMolAlign.GetBestRMS(Chem.Mol(mol), Chem.Mol(mol), *pair)
    atom_map = [(i, i) for i in range(mol.GetNumAtoms())]

    return [
        rdMolAlign.AlignMol(
            Chem.Mol(mol), Chem.Mol(mol), int(i), int(j), atomMap=atom_map
        )
        for i, j in pairs
    ]


def cluster_confs(
    entry: descent.targets.energy.Entry, pool: multiprocessing.Pool
) -> descent.targets.energy.Entry:
    try:
        smiles = entry["smiles"]

        energy_ref = entry["energy"]

        coords = entry["coords"].reshape(len(energy_ref), -1, 3).tolist()
        coords = [c * openff.units.unit.angstrom for c in coords]

        mol_openff = openff.toolkit.Molecule.from_mapped_smiles(
            smiles, allow_undefined_stereo=True
        )
        mol_openff._conformers = coords

        mol_rdkit: Chem.Mol = Chem.RemoveHs(mol_openff.to_rdkit())
        conf_ids = [conf.GetId() for conf in mol_rdkit.GetConformers()]

        conf_pairs = [(i, j) for i in range(len(conf_ids)) for j in range(i)]

        conf_pairs = numpy.array_split(numpy.array(conf_pairs), N_WORKERS)

        rms_fn = functools.partial(compute_best_rms, mol=mol_rdkit)

        dists = list(tqdm.tqdm(pool.imap(rms_fn, conf_pairs), total=len(conf_pairs)))
        dists = [d for dist in dists for d in dist]

        clusters = Butina.ClusterData(
            dists, len(conf_ids), 0.25, isDistData=True, reordering=True
        )
        cluster_ids = [cluster[0] for cluster in clusters]

        tqdm.tqdm.write(f"{smiles}: {len(conf_ids)} -> {len(cluster_ids)}")

        entry["energy"] = entry["energy"][cluster_ids]
        entry["coords"] = (
            entry["coords"].reshape(len(energy_ref), -1, 3)[cluster_ids, :, :].flatten()
        )
        entry["forces"] = (
            entry["forces"].reshape(len(energy_ref), -1, 3)[cluster_ids, :, :].flatten()
        )
    except BaseException as e:
        logger.info(f"failed to cluster {entry}: {e}", flush=True)

    return entry


def filter_and_cluster_espaloma(config: WorkflowConfig) -> None:
    for source in ESPALOMA_SOURCES:
        dataset = datasets.Dataset.load_from_disk(
            f"{config.data_dir}/data-raw/{source}"
        )
        unique_smiles = descent.targets.energy.extract_smiles(dataset)

        _, topologies = torch.load(config.torch_ffs_and_tops_path)
        topologies = {k: v for k, v in topologies.items() if k in unique_smiles}

        dataset_size = len(dataset)
        dataset = dataset.filter(lambda d: d["smiles"] in topologies)
        logger.info(f"Removed non-parameterisable: {dataset_size} -> {len(dataset)}")

        if source == "gen2-opt":
            with multiprocessing.Pool(N_WORKERS) as pool:
                cluster_fn = functools.partial(cluster_confs, pool=pool)
                dataset = dataset.map(cluster_fn, with_indices=False, batched=False)

        dataset.save_to_disk(config.filtered_data_dir / source)


def filter_spice2(config: WorkflowConfig) -> None:
    """Filter out any molecules which can't be parameterised."""

    sources = [config.data_dir / "data-train", config.data_dir / "data-test"]
    logger.info(f"Filtering {sources}")

    for source in sources:
        dataset = datasets.Dataset.load_from_disk(source)
        unique_smiles = descent.targets.energy.extract_smiles(dataset)

        _, topologies = torch.load(config.torch_ffs_and_tops_path)
        topologies = {k: v for k, v in topologies.items() if k in unique_smiles}

        dataset_size = len(dataset)
        dataset = dataset.filter(lambda d: d["smiles"] in topologies)
        logger.info(f"Removed non-parameterisable: {dataset_size} -> {len(dataset)}")

        dataset.save_to_disk(config.filtered_data_dir / source.name)
