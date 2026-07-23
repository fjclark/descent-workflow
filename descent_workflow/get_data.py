"""Functions to obtain and process data for the workflow."""

import json
import pathlib
import subprocess
import typing
from typing import Any

import datasets
import deepchem as dc
import descent.targets.energy
import dgl
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import openff.toolkit
import openff.units
import openmm.unit
import seaborn as sns
import torch
from loguru import logger
from tqdm import tqdm

HARTEE_TO_KCAL = (1.0 * openmm.unit.hartree * openmm.unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    openmm.unit.kilocalorie_per_mole
)

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

        entries = [f for f in source_dir.glob("*") if f.is_dir() and not f.name.startswith(".")]

        duplicate_dir = root_dir / "duplicated-isomeric-smiles-merge"

        entries_duplicate = list(duplicate_dir.glob(f"*/{source.replace('-opt', '')}/*"))
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
            energies = [record["dft_total_energy"][i] * HARTEE_TO_KCAL for i in range(n_conformers)]
            coords = [record["conformations"][i] * BOHR_TO_ANGSTROM for i in range(n_conformers)]
            forces = [
                record["dft_total_gradient"][i] * -1 * (HARTEE_TO_KCAL / BOHR_TO_ANGSTROM)
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
        logger.info(f"Found {len(dataset)} ({len(unique_smiles)} unique) SMILES in SPICE2")
        with open(output_dir / "smiles.json", "w") as file:
            json.dump(list(unique_smiles), file)


def get_rms(array: npt.NDArray[np.floating[Any]]) -> float:
    """Root mean square of all elements of an array."""
    result: float = float(np.sqrt(np.mean(array**2)))
    return result


def filter_dataset_by_forces(
    dataset: datasets.Dataset,
    percentile: float = 95,
    out_dir: pathlib.Path | str | None = None,
    cutoff: float | None = None,
) -> tuple[datasets.Dataset, dict[str, Any]]:
    """Remove entries with non-finite or high RMS forces from an in-memory dataset.

    Mirrors :func:`filter_spice2_dataset_by_forces` (per-entry RMS force above a
    percentile cutoff) but operates on an in-memory ``datasets.Dataset`` and adds an
    explicit non-finite (NaN/Inf) guard on the ``coords``/``energy``/``forces`` values.
    The percentile filter alone does not catch these because ``NaN > cutoff`` is
    ``False``.

    Args:
        dataset: dataset with ``smiles``/``coords``/``energy``/``forces`` columns.
        percentile: RMS-force percentile above which entries are removed. Ignored when
            ``cutoff`` is provided.
        out_dir: if given, write ``force_filter_report.json`` and an RMS-force boxplot
            for inspection.
        cutoff: pre-computed RMS-force cutoff in kcal/(mol Angstrom). If ``None`` it is
            computed from ``percentile`` over the finite entries of this dataset. Pass a
            shared cutoff to filter several splits consistently.

    Returns:
        ``(filtered_dataset, report)``. ``report`` records the cutoff, the input/output
        counts and the removed / non-finite SMILES.
    """
    data_df = dataset.to_pandas()

    def is_finite_row(row: Any) -> bool:
        for col in ("coords", "energy", "forces"):
            if not np.all(np.isfinite(np.asarray(row[col], dtype=float))):
                return False
        return True

    finite_mask = data_df.apply(is_finite_row, axis=1)
    nonfinite_smiles = data_df.loc[~finite_mask, "smiles"].tolist()
    if nonfinite_smiles:
        logger.warning(
            f"Removing {len(nonfinite_smiles)} entries with non-finite "
            "coords/energy/forces"
        )

    finite_df = data_df[finite_mask].copy()
    finite_df["rms_forces"] = finite_df["forces"].apply(
        lambda x: get_rms(np.asarray(x, dtype=float))
    )

    if cutoff is None:
        cutoff = float(np.percentile(finite_df["rms_forces"], percentile))
    logger.info(
        f"RMS force cutoff ({percentile}th percentile): "
        f"{cutoff:.2f} kcal/(mol Angstrom)"
    )

    high_force_smiles = finite_df.loc[
        finite_df["rms_forces"] > cutoff, "smiles"
    ].tolist()
    logger.info(f"Removing {len(high_force_smiles)} entries with high RMS forces")

    remove_smiles = set(nonfinite_smiles) | set(high_force_smiles)
    filtered = dataset.filter(lambda x: x["smiles"] not in remove_smiles)

    report: dict[str, Any] = {
        "percentile": percentile,
        "cutoff_kcal_per_mol_angstrom": cutoff,
        "n_input": len(dataset),
        "n_nonfinite_removed": len(nonfinite_smiles),
        "n_high_force_removed": len(high_force_smiles),
        "n_output": len(filtered),
        "nonfinite_smiles": nonfinite_smiles,
        "high_force_smiles": high_force_smiles,
    }
    logger.info(
        f"Filtered dataset by forces: {report['n_input']} -> {report['n_output']} "
        f"entries ({report['n_nonfinite_removed']} non-finite, "
        f"{report['n_high_force_removed']} high-force)"
    )

    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "force_filter_report.json", "w") as file:
            json.dump(report, file, indent=2)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=finite_df["rms_forces"], ax=ax)
        ax.axvline(x=cutoff, color="red", linestyle="--", alpha=0.5)
        ax.text(cutoff, 0.4, f"{percentile}th", color="red", rotation=90, va="center")
        ax.set_xlabel(r"RMS Forces (kcal mol$^{-1}$ $\mathrm{\AA}^{-1})$")
        ax.set_title("Distribution of RMS Forces")
        fig.savefig(str(out_dir / "rms_forces.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return filtered, report


def _min_interatomic_dist(coords: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """Smallest pairwise atom distance per conformer.

    ``coords`` is ``(n_conf, n_atoms, 3)``; returns ``(n_conf,)``. Conformers are processed
    in memory-bounded chunks so large molecules with many conformers do not blow up the
    ``(n_conf, n_atoms, n_atoms)`` distance tensor.
    """
    n_conf, n_atoms, _ = coords.shape
    out = np.empty(n_conf, dtype=np.float64)
    idx = np.arange(n_atoms)
    chunk = max(1, int(2_000_000 // (n_atoms * n_atoms + 1)))
    for start in range(0, n_conf, chunk):
        block = coords[start : start + chunk]
        diff = block[:, :, None, :] - block[:, None, :, :]
        dist = np.sqrt((diff**2).sum(axis=3))
        dist[:, idx, idx] = np.inf
        out[start : start + chunk] = dist.min(axis=(1, 2))
    return out


def _filter_row_by_conformer_quality(
    row: dict[str, Any], *, max_atom_force: float, min_interatomic_dist: float
) -> dict[str, Any]:
    """Rewrite one molecule row, keeping only conformers that pass the quality gates.

    A conformer is kept iff its ``coords``/``energy``/``forces`` are all finite, its maximum
    per-atom force is ``<= max_atom_force`` (kcal/mol/Angstrom) and its smallest interatomic
    distance is ``>= min_interatomic_dist`` (Angstrom). Returns the rewritten flat
    ``coords``/``energy``/``forces`` plus ``_n_conf_before``/``_n_kept`` bookkeeping columns.
    """
    energy = np.asarray(row["energy"], dtype=np.float64)
    n_conf = int(energy.shape[0])
    coords = np.asarray(row["coords"], dtype=np.float64)
    n_atoms = coords.size // (3 * n_conf) if n_conf else 0
    if n_conf == 0 or n_atoms == 0:
        return {"coords": [], "energy": [], "forces": [], "_n_conf_before": n_conf, "_n_kept": 0}

    c = coords.reshape(n_conf, n_atoms, 3)
    f = np.asarray(row["forces"], dtype=np.float64).reshape(n_conf, n_atoms, 3)

    finite = (
        np.isfinite(c).all(axis=(1, 2))
        & np.isfinite(f).all(axis=(1, 2))
        & np.isfinite(energy)
    )
    max_force = np.linalg.norm(f, axis=2).max(axis=1)
    min_dist = _min_interatomic_dist(c)
    keep = finite & (max_force <= max_atom_force) & (min_dist >= min_interatomic_dist)

    return {
        "coords": c[keep].reshape(-1).astype(np.float32).tolist(),
        "energy": energy[keep].astype(np.float32).tolist(),
        "forces": f[keep].reshape(-1).astype(np.float32).tolist(),
        "_n_conf_before": n_conf,
        "_n_kept": int(keep.sum()),
    }


def _plot_conformer_quality(
    dataset: datasets.Dataset,
    max_atom_force: float,
    min_interatomic_dist: float,
    out_dir: pathlib.Path,
    max_conformers: int = 200_000,
) -> None:
    """Histogram per-conformer max atom force and min interatomic distance (subsampled)."""
    step = max(1, len(dataset) // 5000)
    maxf: list[float] = []
    mind: list[float] = []
    for i in range(0, len(dataset), step):
        row = dataset[i]
        energy = np.asarray(row["energy"], dtype=np.float64)
        n_conf = int(energy.shape[0])
        if n_conf == 0:
            continue
        coords = np.asarray(row["coords"], dtype=np.float64)
        n_atoms = coords.size // (3 * n_conf)
        if n_atoms < 2:
            continue
        c = coords.reshape(n_conf, n_atoms, 3)
        f = np.asarray(row["forces"], dtype=np.float64).reshape(n_conf, n_atoms, 3)
        maxf.extend(np.linalg.norm(f, axis=2).max(axis=1).tolist())
        mind.extend(_min_interatomic_dist(c).tolist())
        if len(maxf) >= max_conformers:
            break

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(np.clip(maxf, 0, 3 * max_atom_force), bins=100)
    axes[0].axvline(max_atom_force, color="red", linestyle="--", alpha=0.6)
    axes[0].set_xlabel(r"max atom force (kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$)")
    axes[0].set_ylabel("conformers")
    axes[0].set_title("Per-conformer max atom force")
    axes[1].hist(np.clip(mind, 0, 2.0), bins=100)
    axes[1].axvline(min_interatomic_dist, color="red", linestyle="--", alpha=0.6)
    axes[1].set_xlabel(r"min interatomic distance ($\mathrm{\AA}$)")
    axes[1].set_ylabel("conformers")
    axes[1].set_title("Per-conformer min interatomic distance")
    fig.tight_layout()
    fig.savefig(str(out_dir / "conformer_quality.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def filter_conformers_by_quality(
    dataset: datasets.Dataset,
    max_atom_force: float = 250.0,
    min_interatomic_dist: float = 0.9,
    min_conformers: int = 5,
    out_dir: pathlib.Path | str | None = None,
    num_proc: int | None = None,
) -> tuple[datasets.Dataset, dict[str, Any]]:
    """Drop individual off-equilibrium conformers a harmonic FF cannot represent.

    Operates *per conformer* (unlike :func:`filter_dataset_by_forces`, which drops whole
    molecules): for each one-molecule-per-row entry it keeps only the conformers whose
    ``coords``/``energy``/``forces`` are finite, whose maximum per-atom force is
    ``<= max_atom_force`` (kcal/mol/Angstrom) and whose smallest interatomic distance is
    ``>= min_interatomic_dist`` (Angstrom), rewriting the row's flat ``coords``/``energy``/
    ``forces`` arrays from the survivors. Molecules left with ``< min_conformers`` conformers
    are dropped entirely.

    The force gate is calibrated to the SPICE envelope (SPICE per-conformer max atom force
    p99 ~ 200 kcal/mol/Angstrom, vs a broad off-equilibrium tail to ~1000+ in the OMol25 GEOM
    data). The distance gate is a topology-free catch for pathologically compressed bonds (no
    two atoms in a real organic molecule sit < 0.9 Angstrom apart). Because only conformers
    are removed the SMILES set is essentially unchanged, so an existing parameterisation stays
    valid without re-running it.

    Args:
        dataset: one-molecule-per-row dataset with ``smiles``/``coords``/``energy``/``forces``.
        max_atom_force: keep conformers with max per-atom force at or below this.
        min_interatomic_dist: keep conformers with min interatomic distance at or above this.
        min_conformers: drop molecules left with fewer conformers than this.
        out_dir: if given, write ``conformer_quality_filter_report.json`` and a diagnostic plot.
        num_proc: processes for the per-row rewrite (passed to ``datasets.Dataset.map``).

    Returns:
        ``(filtered_dataset, report)``.
    """
    logger.info(
        f"Filtering conformers by quality: max_atom_force={max_atom_force} kcal/(mol Angstrom), "
        f"min_interatomic_dist={min_interatomic_dist} Angstrom, min_conformers={min_conformers}"
    )
    mapped = dataset.map(
        _filter_row_by_conformer_quality,
        fn_kwargs={
            "max_atom_force": max_atom_force,
            "min_interatomic_dist": min_interatomic_dist,
        },
        num_proc=num_proc,
        desc="Filtering conformers by quality",
    )

    n_conf_before_col = np.asarray(mapped["_n_conf_before"], dtype=np.int64)
    n_kept_col = np.asarray(mapped["_n_kept"], dtype=np.int64)
    keep_mol = n_kept_col >= min_conformers

    n_conf_before = int(n_conf_before_col.sum())
    n_conf_after_gate = int(n_kept_col.sum())
    n_conf_final = int(n_kept_col[keep_mol].sum())

    filtered = mapped.filter(
        lambda r: r["_n_kept"] >= min_conformers, desc="Dropping sparse molecules"
    ).remove_columns(["_n_conf_before", "_n_kept"])

    report: dict[str, Any] = {
        "max_atom_force_kcal_per_mol_angstrom": max_atom_force,
        "min_interatomic_dist_angstrom": min_interatomic_dist,
        "min_conformers": min_conformers,
        "n_molecules_in": len(dataset),
        "n_molecules_out": len(filtered),
        "n_molecules_dropped": int((~keep_mol).sum()),
        "n_conformers_in": n_conf_before,
        "n_conformers_passing_gates": n_conf_after_gate,
        "n_conformers_out": n_conf_final,
        "n_conformers_removed": n_conf_before - n_conf_final,
        "frac_conformers_removed": (
            (n_conf_before - n_conf_final) / n_conf_before if n_conf_before else 0.0
        ),
    }
    logger.info(
        f"Conformer quality filter: {report['n_molecules_in']:,} -> "
        f"{report['n_molecules_out']:,} molecules ({report['n_molecules_dropped']:,} dropped); "
        f"{n_conf_before:,} -> {n_conf_final:,} conformers "
        f"({report['frac_conformers_removed'] * 100:.1f}% removed)"
    )

    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "conformer_quality_filter_report.json", "w") as file:
            json.dump(report, file, indent=2)
        _plot_conformer_quality(dataset, max_atom_force, min_interatomic_dist, out_dir)

    return filtered, report


def filter_spice2_dataset_by_forces(data_dir: pathlib.Path) -> None:
    """Filter the SPICE dataset by forces and save it to disk."""
    logger.info("Filtering SPICE dataset by forces...")

    input_dir = data_dir / "data-raw"
    output_dir = data_dir / "data-filtered-by-forces"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(input_dir)
    data_df = dataset.to_pandas()

    data_df["rms_forces"] = data_df["forces"].apply(lambda x: get_rms(np.array(x)))

    # Plot the distribution of the RMS forces
    # Get the percentiles in increments of 5
    percentile_intervals = np.array([85, 90, 95, 97.5, 99])
    percentile_values = np.percentile(data_df["rms_forces"], percentile_intervals)

    # Create a dict of the percentiles
    percentile_dict = dict(zip(percentile_intervals, percentile_values, strict=True))
    logger.info(f"Percentiles: {percentile_dict}")

    # Plot boxplot of the rmse forces
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=data_df["rms_forces"], ax=ax)

    for interval, value in percentile_dict.items():
        # Add a vertical line at the percentile
        ax.axvline(x=value, color="red", linestyle="--", alpha=0.5)
        # Write the percentile value
        ax.text(value, 0.4, f"{interval:.2f}", color="red", rotation=90, va="center")

    ax.set_xlabel(r"RMS Forces (kcal mol$^{-1}$ $\mathrm{\AA}^{-1})$")
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

    logger.info(f"Train: {len(train_index)}, Test: {len(test_index)}, Total: {len(input_dataset)}")
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
    download_spice2_data(data_dir)
    process_dataset_spice2(data_dir)
    filter_spice2_dataset_by_forces(data_dir)
    split_train_test_spice2(data_dir)
    logger.info("Done getting data for SPICE.")


def split_train_test(
    data_dir: pathlib.Path | str,
    dataset: datasets.Dataset,
    frac_train: float = 0.95,
    seed: int = 42,
) -> None:
    """Randomly split a descent dataset into train/test at the molecule level.

    Splits by unique SMILES (not by row) so no molecule appears in both sets, then writes
    ``data_dir/data-train``, ``data_dir/data-test`` and ``data_dir/smiles_test_train.json``
    (``{"train": [...], "test": [...]}``) — the layout consumed by ``filter.filter_spice2``
    and ``parameterise.create_torch_ff_and_top``.
    """
    data_dir = pathlib.Path(data_dir)
    logger.info(f"Randomly splitting dataset ({len(dataset):,} rows) into train/test...")

    unique_smiles = list(dataset.unique("smiles"))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_smiles)
    n_train = int(round(frac_train * len(unique_smiles)))
    train_smiles = set(unique_smiles[:n_train])
    test_smiles = set(unique_smiles[n_train:])
    logger.info(
        f"Split {len(unique_smiles):,} molecules -> "
        f"{len(train_smiles):,} train / {len(test_smiles):,} test"
    )

    # Partition rows via the (cheap) smiles column rather than iterating full rows.
    smiles_col = dataset["smiles"]
    train_index = [i for i, s in enumerate(smiles_col) if s in train_smiles]
    test_index = [i for i, s in enumerate(smiles_col) if s in test_smiles]

    train_split = dataset.select(indices=train_index)
    test_split = dataset.select(indices=test_index)

    train_split.save_to_disk(str(data_dir / "data-train"))
    test_split.save_to_disk(str(data_dir / "data-test"))
    logger.info(
        f"Saved {len(train_split):,} train and {len(test_split):,} test rows to {data_dir}"
    )

    smiles_train_test_dict = {
        "train": train_split.unique("smiles"),
        "test": test_split.unique("smiles"),
    }
    with open(data_dir / "smiles_test_train.json", "w") as file:
        json.dump(smiles_train_test_dict, file)
    logger.info(f"Saved train/test smiles to {data_dir / 'smiles_test_train.json'}")


def get_data_omol25_combined(data_dir: pathlib.Path | str) -> None:
    """Download, reprocess and combine the OMol25 SPICE + GEOM datasets for fitting.

    Reproduces the ``tmp_geom`` pipeline from scratch: downloads the raw descent-format
    SPICE and GEOM HuggingFace datasets, fetches SPICE metadata, deduplicates each
    (stereo-collapsed, ``min_conformers=5``; SPICE also filtered to the
    ``pubchem_numbered+dipeptides`` source preset), concatenates them, and writes a random
    95/5 train/test split. Every expensive step skips when its output already exists, so
    the rule is restartable.
    """
    from . import omol25

    data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Getting combined OMol25 SPICE + GEOM data...")

    # 1. Download raw descent-format datasets (one conformer per row).
    spice_raw = omol25.download_descent_format_dataset("spice", data_dir / "descent_format_spice_hf")
    geom_raw = omol25.download_descent_format_dataset("geom", data_dir / "descent_format_geom_hf")

    # 2. SPICE metadata (for the source-collection preset). GEOM is a single collection.
    spice_metadata = omol25.fetch_metadata("spice", data_dir / "metadata_spice.parquet")

    # 3. Deduplicate each dataset into one-molecule-per-row descent datasets.
    spice_dedup = omol25.deduplicate(
        spice_raw,
        data_dir / "descent_format_spice_pubchem_dipeptides_nostereo_min5_dedup_hf",
        metadata_path=spice_metadata,
        keep=omol25.PRESETS["pubchem_numbered+dipeptides"],
        collapse_stereo=True,
        min_conformers=5,
    )
    geom_dedup = omol25.deduplicate(
        geom_raw,
        data_dir / "descent_format_geom_nostereo_min5_dedup_hf",
        collapse_stereo=True,
        min_conformers=5,
    )

    # 4. Combine (identical schemas) and 5. split train/test.
    logger.info("Combining SPICE + GEOM deduplicated datasets...")
    spice_ds = datasets.Dataset.load_from_disk(str(spice_dedup))
    geom_ds = datasets.Dataset.load_from_disk(str(geom_dedup))
    combined = datasets.concatenate_datasets([spice_ds, geom_ds])
    logger.info(
        f"Combined: {len(spice_ds):,} SPICE + {len(geom_ds):,} GEOM = {len(combined):,} molecules"
    )

    # 4b. Drop off-equilibrium conformers a harmonic FF cannot represent. The OMol25 GEOM
    # portion is genuine but aggressively off-equilibrium (bonds compressed to <0.9 Angstrom,
    # per-atom forces far above the SPICE envelope), which makes force-matching diverge to
    # NaN. Filter per conformer (keeping the good conformers of each molecule) rather than by
    # whole-molecule RMS force, which is a relative cut that leaves the distorted geometries in.
    combined, _ = filter_conformers_by_quality(combined, out_dir=data_dir)

    split_train_test(data_dir, combined)
    logger.info("Done getting combined OMol25 SPICE + GEOM data.")


def get_qca_torsion_data(
    dataset_name: str, output_dir: pathlib.Path | str, spec_name: str = "default"
) -> None:
    """Get the QCA torsion data in json format."""
    from openff.qcsubmit.results import TorsionDriveResultCollection
    from qcportal import PortalClient
    from yammbs.torsion.inputs import QCArchiveTorsionDataset

    logger.info(f"Getting QCA torsion data for {dataset_name}...")
    client = PortalClient("https://api.qcarchive.molssi.org:443", cache_dir=output_dir)

    torsion_dataset = TorsionDriveResultCollection.from_server(
        client=client,
        datasets=dataset_name,
        spec_name=spec_name,
    )

    dataset = QCArchiveTorsionDataset.from_qcsubmit_collection(torsion_dataset)

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset to json
    output_json_path = output_dir / "qca-torsion-data.json"
    output_json_path.write_text(dataset.model_dump_json())

    # Save README with provenance
    output_text_path = output_dir / "qca-torsion-data-readme.txt"
    output_text_path.write_text(f"Dataset {dataset_name} with spec {spec_name}.\n")
