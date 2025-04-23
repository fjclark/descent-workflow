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
