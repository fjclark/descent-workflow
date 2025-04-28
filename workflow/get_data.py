"""Functions to obtain and process data for the workflow."""

import json
import pathlib
import typing

import descent.targets.energy
import dgl
import openff.toolkit
import openff.units
import openmm.unit
import torch
from tqdm import tqdm
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datasets
import deepchem as dc

import subprocess

from loguru import logger

HARTEE_TO_KCAL = (
    1.0 * openmm.unit.hartree * openmm.unit.AVOGADRO_CONSTANT_NA
).value_in_unit(openmm.unit.kilocalorie_per_mole)

BOHR_TO_ANGSTROM = (1.0 * openmm.unit.bohr).value_in_unit(openmm.unit.angstrom)

ESPALOMA_SOURCES = [
    "gen2-opt",
    "gen2-torsion",
    "spice-des-monomers",
    "spice-pubchem",
]

# This avoids Boron and Silicon as they're in 'SPICE PubChem Boron Silicon v1.0',
SPICE2_SOURCES = {
    "SPICE DES Monomers Single Points Dataset v1.1",
    "SPICE Dipeptides Single Points Dataset v1.3",
    "SPICE PubChem Set 1 Single Points Dataset v1.3",
    "SPICE PubChem Set 2 Single Points Dataset v1.3",
    "SPICE PubChem Set 3 Single Points Dataset v1.3",
    "SPICE PubChem Set 4 Single Points Dataset v1.3",
    "SPICE PubChem Set 5 Single Points Dataset v1.3",
    "SPICE PubChem Set 6 Single Points Dataset v1.3",
    "SPICE PubChem Set 7 Single Points Dataset v1.0",
    "SPICE PubChem Set 8 Single Points Dataset v1.0",
    "SPICE PubChem Set 9 Single Points Dataset v1.0",
    "SPICE PubChem Set 10 Single Points Dataset v1.0",
}


def download_espaloma_data(data_dir: pathlib.Path) -> None:
    """Download the ESPALOMA data from the Zenodo."""
    logger.info("Downloading ESPALOMA data from Zenodo. This may take a while...")

    cmds = [
        f"mkdir -p {data_dir}",
        f"curl -o {data_dir}/8150601.zip https://zenodo.org/api/records/8150601/files-archive",
        f"unzip {data_dir}/8150601.zip -d {data_dir}/8150601",
        f'for f in {data_dir}/8150601/*.tar.gz; do tar -zxvf "$f" -C {data_dir}/8150601; done',
        f"rm -r {data_dir}/8150601/*.tar.gz",
    ]

    for cmd in cmds:
        subprocess.run(
            cmd,
            check=True,
            shell=True,
        )


# From https://github.com/SimonBoothroyd/descent-ff/blob/main/energy-force/001-convert-espaloma-data.py
def process_entry_espaloma(root_dir: pathlib.Path) -> dict[str, typing.Any]:
    mol_dict = json.loads(json.loads((root_dir / "mol.json").read_text()))
    mol_dict["hierarchy_schemes"] = {}
    mol_dict["partial_charge_unit"] = mol_dict["partial_charges_unit"]
    del mol_dict["partial_charges_unit"]
    mol = openff.toolkit.Molecule.from_dict(mol_dict)

    graphs, extra = dgl.load_graphs(str(root_dir / "heterograph.bin"))
    assert len(graphs) == 1
    assert len(extra) == 0

    graph = graphs[0]

    energies = graph.ndata["u_qm"]["g"].flatten() * HARTEE_TO_KCAL

    forces = graph.ndata["u_qm_prime"]["n1"] * (HARTEE_TO_KCAL / BOHR_TO_ANGSTROM)
    forces = torch.swapaxes(forces, 0, 1)

    coords = graph.ndata["xyz"]["n1"] * BOHR_TO_ANGSTROM
    coords = torch.swapaxes(coords, 0, 1)

    return {
        "smiles": mol.to_smiles(mapped=True, isomeric=True),
        "coords": coords.flatten().tolist(),
        "energy": energies.flatten().tolist(),
        "forces": forces.flatten().tolist(),
    }


# Mainly from https://github.com/SimonBoothroyd/descent-ff/blob/main/energy-force/001-convert-espaloma-data.py
def process_dataset_espaloma(data_dir: pathlib.Path) -> None:
    root_dir = data_dir / "8150601"
    output_dir = data_dir / "data-raw"

    smiles_per_set = {}

    for source in ESPALOMA_SOURCES:
        source_dir = root_dir / source

        entries = [
            f for f in source_dir.glob("*") if f.is_dir() and not f.name.startswith(".")
        ]

        duplicate_dir = root_dir / "duplicated-isomeric-smiles-merge"

        entries_duplicate = list(
            duplicate_dir.glob(f"*/{source.replace('-opt', '')}/*")
        )
        entries_duplicate = [
            f for f in entries_duplicate if f.is_dir() and not f.name.startswith(".")
        ]
        entries.extend(entries_duplicate)

        logger.info(
            f"processing {len(entries)} entries from {source} "
            f"({len(entries_duplicate)} from duplicates)"
        )

        dataset = descent.targets.energy.create_dataset(
            [process_entry_espaloma(entry) for entry in tqdm(entries)]
        )
        dataset.save_to_disk(output_dir / source)

        unique_smiles = dataset.unique("smiles")
        lus = len(unique_smiles)
        tqdm.write(f"Found {len(dataset)} ({lus} unique) SMILES in {source}")

        smiles_per_set[source] = dataset.unique("smiles")

    with open(output_dir / "smiles.json", "w") as file:
        json.dump(smiles_per_set, file)


def get_data_espaloma(data_dir: pathlib.Path | str) -> None:
    data_dir = pathlib.Path(data_dir)
    logger.info("Getting data for ESPALOMA...")
    download_espaloma_data(data_dir)
    process_dataset_espaloma(data_dir)
    logger.info("Done getting data for ESPALOMA.")


def download_spice2_data(data_dir: pathlib.Path) -> None:
    """Download the SPICE data from the Zenodo."""
    logger.info("Downloading SPICE data from Zenodo. This may take a while...")

    output_file = data_dir / "SPICE-2.0.1.hdf5"
    if output_file.exists():
        logger.info(f"SPICE data already exists at {output_file}. Skipping download.")
        return

    cmds = [
        f"mkdir -p {data_dir}",
        f" wget -O {output_file} https://zenodo.org/record/10975225/files/SPICE-2.0.1.hdf5?download=1",
    ]

    for cmd in cmds:
        subprocess.run(
            cmd,
            check=True,
            shell=True,
        )


def process_dataset_spice2(data_dir: pathlib.Path) -> None:
    """Process the SPICE dataset and save it to disk (without filtering forces)."""

    output_dir = data_dir / "data-raw"

    with h5py.File(data_dir / "SPICE-2.0.1.hdf5") as spice:
        all_data = []
        all_smiles = set()

        for record in tqdm(spice.values(), desc="Extracting dataset", ncols=80):
            smiles = record["smiles"].asstr()[0]
            subset = record["subset"].asstr()[0]

            # Only extract the data if it's of the desired type
            if subset not in SPICE2_SOURCES:
                continue

            # extract the data
            all_smiles.add(smiles)
            n_conformers = record["conformations"].shape[0]
            assert len(record["dft_total_energy"]) == n_conformers
            energies = [
                record["dft_total_energy"][i] * HARTEE_TO_KCAL
                for i in range(n_conformers)
            ]
            coords = [
                record["conformations"][i] * BOHR_TO_ANGSTROM
                for i in range(n_conformers)
            ]
            forces = [
                record["dft_total_gradient"][i]
                * -1
                * (HARTEE_TO_KCAL / BOHR_TO_ANGSTROM)
                for i in range(n_conformers)
            ]
            all_data.append(
                {
                    "smiles": smiles,
                    "coords": coords,
                    "energy": energies,
                    "forces": forces,
                }
            )

        dataset = descent.targets.energy.create_dataset(all_data)
        dataset.save_to_disk(output_dir)
        unique_smiles = dataset.unique("smiles")
        logger.info(
            f"Found {len(dataset)} ({len(unique_smiles)} unique) SMILES in SPICE2"
        )
        with open(output_dir / "smiles.json", "w") as file:
            json.dump(list(unique_smiles), file)


def filter_spice2_dataset_by_forces(data_dir: pathlib.Path) -> None:
    """Filter the SPICE dataset by forces and save it to disk."""

    logger.info("Filtering SPICE dataset by forces...")

    input_dir = data_dir / "data-raw"
    output_dir = data_dir / "data-filtered-by-forces"

    dataset = datasets.load_from_disk(input_dir)
    data_df = dataset.to_pandas()

    def get_rms(array: np.ndarray) -> float:
        return np.sqrt(np.mean(array**2))

    data_df["rms_forces"] = data_df["forces"].apply(lambda x: get_rms(np.array(x)))

    # Plot the distribution of the RMS forces
    # Get the percentiles in increments of 5
    percentile_intervals = np.array([85, 90, 95, 97.5, 99])
    percentile_values = np.percentile(data_df["rms_forces"], percentile_intervals)

    # Create a dict of the percentiles
    percentile_dict = {
        interval: value
        for interval, value in zip(percentile_intervals, percentile_values, strict=True)
    }
    logger.info(f"Percentiles: {percentile_dict}")

    # Plot boxplot of the rmse forces
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=data_df["rms_forces"], ax=ax)

    for interval, value in percentile_dict.items():
        # Add a vertical line at the percentile
        ax.axvline(x=value, color="red", linestyle="--", alpha=0.5)
        # Write the percentile value
        ax.text(value, 0.4, f"{interval:.2f}", color="red", rotation=90, va="center")

    ax.set_xlabel("RMS Forces (kcal mol$^{-1}$ $\mathrm{\AA}^{-1})$")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of RMS Forces")
    fig.savefig(str(output_dir / "rms_forces.png"), dpi=300, bbox_inches="tight")

    # Get the data above the 95th percentile
    df_highest_95 = data_df[data_df["rms_forces"] > percentile_dict[95]]
    logger.info(f"Cutoff: {percentile_dict[95]:.2f} kcal/(mol Angstrom)")
    high_force_smiles = df_highest_95["smiles"].tolist()
    with open(output_dir / "high_force_smiles.json", "w") as file:
        json.dump(high_force_smiles, file)
    logger.info(f"Removed {len(df_highest_95)} entries with high forces")

    # Save a filtered dataset without the high forces
    filtered_dataset = dataset.filter(lambda x: x["smiles"] not in high_force_smiles)
    filtered_dataset.save_to_disk(output_dir)
    logger.info(
        f"Filtered dataset (containing {len(filtered_dataset)} entries) saved to {output_dir}"
    )

    # Save all of the smiles to a json file
    with open(output_dir / "smiles.json", "w") as file:
        json.dump(list(filtered_dataset.unique("smiles")), file)


def split_train_test_spice2(data_dir: pathlib.Path | str) -> None:
    """Split the SPICE2 dataset into training and testing sets."""
    data_dir = pathlib.Path(data_dir)
    logger.info("Splitting SPICE2 dataset into training and testing sets...")

    input_dir = data_dir / "data-filtered-by-forces"
    output_dirs = {
        "train": data_dir / "data-train",
        "test": data_dir / "data-test",
    }

    with open(input_dir / "smiles.json", "r") as file:
        smiles = json.load(file)
    input_dataset = datasets.load_from_disk(input_dir)

    Xs = np.zeros(len(smiles))
    dc_dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=smiles)
    maxminspliter = dc.splits.MaxMinSplitter()
    train_dataset, test_dataset = maxminspliter.train_test_split(
        dataset=dc_dataset,
        frac_train=0.95,
        train_dir=output_dirs["train"],
        test_dir=output_dirs["test"],
    )

    train_index, test_index = [], []
    for i, entry in enumerate(input_dataset):
        if entry["smiles"] in train_dataset.ids:
            train_index.append(i)
        elif entry["smiles"] in test_dataset.ids:
            test_index.append(i)
        else:
            raise RuntimeError("The smiles was not in training or testing")

    logger.info(
        f"Train: {len(train_index)}, Test: {len(test_index)}, Total: {len(input_dataset)}"
    )
    train_split = input_dataset.select(indices=train_index)
    train_split.save_to_disk(output_dirs["train"])
    test_split = input_dataset.select(indices=test_index)
    test_split.save_to_disk(output_dirs["test"])
    logger.info("Done splitting SPICE2 dataset into training and testing sets.")

    smiles_train_test_dict = {
        "train": train_split.unique("smiles"),
        "test": test_split.unique("smiles"),
    }

    # Save the smiles to a json file
    with open(data_dir / "smiles_test_train.json", "w") as file:
        json.dump(smiles_train_test_dict, file)
    logger.info(f"Saved train/test smiles to {data_dir / 'smiles_test_train.json'}")


def get_data_spice2_force_filtered(data_dir: pathlib.Path | str) -> None:
    data_dir = pathlib.Path(data_dir)
    logger.info("Getting data for SPICE...")
    # download_spice2_data(data_dir)
    # process_dataset_spice2(data_dir)
    # filter_spice2_dataset_by_forces(data_dir)
    split_train_test_spice2(data_dir)
    logger.info("Done getting data for SPICE.")
