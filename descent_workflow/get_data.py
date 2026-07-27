"""Functions to obtain and process data for the workflow."""

import json
import pathlib
import shutil
import subprocess
import typing
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from .models import WorkflowConfig

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


def get_data_espaloma(
    data_dir: pathlib.Path | str, config: "WorkflowConfig | None" = None
) -> None:
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


_DROP_KEYS = (
    "_n_dropped_nonfinite",
    "_n_dropped_force",
    "_n_dropped_rmsforce",
    "_n_dropped_energy",
    "_n_dropped_dist",
)


def _filter_row_by_conformer_quality(
    row: dict[str, Any],
    *,
    max_atom_force: float | None,
    max_rms_force: float | None,
    max_relative_energy: float | None,
    min_interatomic_dist: float,
) -> dict[str, Any]:
    """Rewrite one molecule row, keeping only conformers that pass the quality gates.

    A conformer is kept iff its ``coords``/``energy``/``forces`` are all finite and it passes
    every *enabled* gate: max per-atom force ``<= max_atom_force``, RMS force
    ``<= max_rms_force`` (both kcal/mol/Angstrom), relative energy (conformer energy minus this
    molecule's lowest finite conformer energy) ``<= max_relative_energy`` (kcal/mol), and min
    interatomic distance ``>= min_interatomic_dist`` (Angstrom). A gate whose threshold is
    ``None`` is skipped. Returns the rewritten flat ``coords``/``energy``/``forces`` plus
    ``_n_conf_before``/``_n_kept`` and per-criterion ``_n_dropped_*`` bookkeeping columns (drop
    counts are independent per gate, so they can overlap).
    """
    zero_drops = {k: 0 for k in _DROP_KEYS}
    energy = np.asarray(row["energy"], dtype=np.float64)
    n_conf = int(energy.shape[0])
    coords = np.asarray(row["coords"], dtype=np.float64)
    n_atoms = coords.size // (3 * n_conf) if n_conf else 0
    if n_conf == 0 or n_atoms == 0:
        return {
            "coords": [], "energy": [], "forces": [],
            "_n_conf_before": n_conf, "_n_kept": 0, **zero_drops,
        }

    c = coords.reshape(n_conf, n_atoms, 3)
    f = np.asarray(row["forces"], dtype=np.float64).reshape(n_conf, n_atoms, 3)

    finite = (
        np.isfinite(c).all(axis=(1, 2))
        & np.isfinite(f).all(axis=(1, 2))
        & np.isfinite(energy)
    )
    atom_force = np.linalg.norm(f, axis=2)
    max_force = atom_force.max(axis=1)
    rms_force = np.sqrt((atom_force**2).mean(axis=1))
    min_dist = _min_interatomic_dist(c)
    # Relative energy vs this molecule's lowest *finite* conformer.
    baseline = energy[finite].min() if finite.any() else 0.0
    rel_energy = energy - baseline

    keep = finite.copy()
    drops = dict(zero_drops)
    drops["_n_dropped_nonfinite"] = int((~finite).sum())
    if max_atom_force is not None:
        gate = max_force <= max_atom_force
        drops["_n_dropped_force"] = int((finite & ~gate).sum())
        keep &= gate
    if max_rms_force is not None:
        gate = rms_force <= max_rms_force
        drops["_n_dropped_rmsforce"] = int((finite & ~gate).sum())
        keep &= gate
    if max_relative_energy is not None:
        gate = rel_energy <= max_relative_energy
        drops["_n_dropped_energy"] = int((finite & ~gate).sum())
        keep &= gate
    gate = min_dist >= min_interatomic_dist
    drops["_n_dropped_dist"] = int((finite & ~gate).sum())
    keep &= gate

    return {
        "coords": c[keep].reshape(-1).astype(np.float32).tolist(),
        "energy": energy[keep].astype(np.float32).tolist(),
        "forces": f[keep].reshape(-1).astype(np.float32).tolist(),
        "_n_conf_before": n_conf,
        "_n_kept": int(keep.sum()),
        **drops,
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


def _conformer_stats(
    dataset: datasets.Dataset, n_target: int = 5000, max_conformers: int = 200_000
) -> dict[str, np.ndarray]:
    """Subsample conformers and return per-conformer stat arrays for the diagnostic plots.

    Returns arrays for ``max_force`` and ``rms_force`` (kcal/mol/Angstrom), ``rel_energy``
    (kcal/mol, vs each molecule's lowest finite conformer) and ``min_dist`` (Angstrom).
    """
    step = max(1, len(dataset) // n_target)
    maxf: list[float] = []
    rmsf: list[float] = []
    rele: list[float] = []
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
        atom_force = np.linalg.norm(f, axis=2)
        finite = np.isfinite(energy)
        baseline = energy[finite].min() if finite.any() else np.nan
        maxf.extend(atom_force.max(axis=1).tolist())
        rmsf.extend(np.sqrt((atom_force**2).mean(axis=1)).tolist())
        rele.extend((energy - baseline).tolist())
        mind.extend(_min_interatomic_dist(c).tolist())
        if len(maxf) >= max_conformers:
            break
    return {
        "max_force": np.asarray(maxf, dtype=np.float64),
        "rms_force": np.asarray(rmsf, dtype=np.float64),
        "rel_energy": np.asarray(rele, dtype=np.float64),
        "min_dist": np.asarray(mind, dtype=np.float64),
    }


def plot_conformer_distributions(
    datasets_by_source: dict[str, datasets.Dataset],
    thresholds: dict[str, float | None],
    out_dir: pathlib.Path | str,
    stage: str,
    reference_source: str = "SPICE",
) -> None:
    """Overlay per-conformer force/energy/geometry distributions, broken down by source.

    Draws one panel each for max atom force, RMS force, relative energy and min interatomic
    distance, overlaying every source in ``datasets_by_source``. Each panel marks the
    (source-agnostic) filter threshold as a solid red line and, for the force/energy panels,
    the ``reference_source`` (SPICE) p99 / p99.9 / max as dashed grey lines, so it is clear the
    aggressive cut sits beyond the SPICE envelope and which source the outliers come from.

    ``stage`` (e.g. ``"before"`` / ``"after"``) is used in the title and output filename.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {name: _conformer_stats(ds) for name, ds in datasets_by_source.items()}

    # (stat key, x-label, threshold key, is-an-upper-bound-gate)
    panels = [
        ("max_force", r"max atom force (kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$)", "max_atom_force", True),
        ("rms_force", r"RMS force (kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$)", "max_rms_force", True),
        ("rel_energy", r"relative energy (kcal mol$^{-1}$)", "max_relative_energy", True),
        ("min_dist", r"min interatomic distance ($\mathrm{\AA}$)", "min_interatomic_dist", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, (key, xlabel, thr_key, upper) in zip(axes, panels, strict=True):
        thr = thresholds.get(thr_key)
        finite_by_source = {
            name: s[key][np.isfinite(s[key])] for name, s in stats.items() if s[key].size
        }
        allvals = (
            np.concatenate(list(finite_by_source.values()))
            if finite_by_source
            else np.asarray([])
        )
        if allvals.size == 0:
            ax.set_title(key.replace("_", " "))
            continue

        if upper:
            hi = float(np.percentile(allvals, 99.5))
            if thr is not None:
                hi = max(hi, thr * 1.5)
            lo = max(0.0, float(allvals.min()))
            clip_range = (lo, hi if hi > lo else lo + 1.0)
        else:
            clip_range = (0.0, 2.0)

        for (name, vals), color in zip(finite_by_source.items(), colors, strict=False):
            ax.hist(
                np.clip(vals, *clip_range),
                bins=100,
                histtype="step",
                density=True,
                label=name,
                color=color,
            )
        if thr is not None:
            ax.axvline(thr, color="red", linestyle="-", alpha=0.9, label="threshold")
        ref = finite_by_source.get(reference_source)
        if upper and ref is not None and ref.size:
            for pct, ls in ((99, ":"), (99.9, "-."), (100, "--")):
                val = float(ref.max()) if pct >= 100 else float(np.percentile(ref, pct))
                ax.axvline(
                    val, color="grey", linestyle=ls, alpha=0.6,
                    label=f"{reference_source} p{pct:g}",
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_title(key.replace("_", " "))

    axes[0].legend(fontsize=8)
    fig.suptitle(f"Per-conformer distributions ({stage} filtering)")
    fig.tight_layout()
    fig.savefig(
        str(out_dir / f"conformer_distributions_{stage}.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def filter_conformers_by_quality(
    dataset: datasets.Dataset,
    max_atom_force: float | None = 250.0,
    min_interatomic_dist: float = 0.9,
    min_conformers: int = 5,
    max_rms_force: float | None = None,
    max_relative_energy: float | None = None,
    out_dir: pathlib.Path | str | None = None,
    num_proc: int | None = None,
) -> tuple[datasets.Dataset, dict[str, Any]]:
    """Drop individual off-equilibrium conformers a harmonic FF cannot represent.

    Operates *per conformer* (unlike :func:`filter_dataset_by_forces`, which drops whole
    molecules): for each one-molecule-per-row entry it keeps only the conformers whose
    ``coords``/``energy``/``forces`` are finite and which pass every *enabled* gate - max
    per-atom force ``<= max_atom_force``, RMS force ``<= max_rms_force`` (kcal/mol/Angstrom),
    relative energy (vs the molecule's lowest conformer) ``<= max_relative_energy`` (kcal/mol),
    and min interatomic distance ``>= min_interatomic_dist`` (Angstrom) - rewriting the row's
    flat ``coords``/``energy``/``forces`` from the survivors. Molecules left with
    ``< min_conformers`` conformers are dropped entirely. A gate whose threshold is ``None`` is
    skipped.

    The force/energy gates are calibrated to the SPICE envelope (SPICE per-conformer max atom
    force p99 ~ 200 kcal/mol/Angstrom, vs a broad off-equilibrium tail to ~1000+ in the OMol25
    GEOM data). The distance gate is a topology-free catch for pathologically compressed bonds.
    Because only conformers are removed the SMILES set is essentially unchanged, so an existing
    parameterisation stays valid without re-running it.

    Args:
        dataset: one-molecule-per-row dataset with ``smiles``/``coords``/``energy``/``forces``.
        max_atom_force: keep conformers with max per-atom force at or below this (``None`` disables).
        min_interatomic_dist: keep conformers with min interatomic distance at or above this.
        min_conformers: drop molecules left with fewer conformers than this.
        max_rms_force: keep conformers with RMS force at or below this (``None`` disables).
        max_relative_energy: keep conformers whose energy minus their molecule's lowest conformer
            energy is at or below this, kcal/mol (``None`` disables).
        out_dir: if given, write ``conformer_quality_filter_report.json`` and a diagnostic plot.
        num_proc: processes for the per-row rewrite (passed to ``datasets.Dataset.map``).

    Returns:
        ``(filtered_dataset, report)``.
    """
    logger.info(
        f"Filtering conformers by quality: max_atom_force={max_atom_force}, "
        f"max_rms_force={max_rms_force}, max_relative_energy={max_relative_energy}, "
        f"min_interatomic_dist={min_interatomic_dist} Angstrom, min_conformers={min_conformers}"
    )
    mapped = dataset.map(
        _filter_row_by_conformer_quality,
        fn_kwargs={
            "max_atom_force": max_atom_force,
            "max_rms_force": max_rms_force,
            "max_relative_energy": max_relative_energy,
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

    # Per-criterion drop counts, summed over all conformers (independent gates, may overlap).
    dropped_by_criterion = {
        k.replace("_n_dropped_", ""): int(np.asarray(mapped[k], dtype=np.int64).sum())
        for k in _DROP_KEYS
    }

    filtered = mapped.filter(
        lambda r: r["_n_kept"] >= min_conformers, desc="Dropping sparse molecules"
    ).remove_columns(["_n_conf_before", "_n_kept", *_DROP_KEYS])

    report: dict[str, Any] = {
        "max_atom_force_kcal_per_mol_angstrom": max_atom_force,
        "max_rms_force_kcal_per_mol_angstrom": max_rms_force,
        "max_relative_energy_kcal_per_mol": max_relative_energy,
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
        "conformers_dropped_by_criterion": dropped_by_criterion,
    }
    logger.info(
        f"Conformer quality filter: {report['n_molecules_in']:,} -> "
        f"{report['n_molecules_out']:,} molecules ({report['n_molecules_dropped']:,} dropped); "
        f"{n_conf_before:,} -> {n_conf_final:,} conformers "
        f"({report['frac_conformers_removed'] * 100:.1f}% removed)"
    )
    logger.info(f"  conformers dropped by criterion (may overlap): {dropped_by_criterion}")

    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "conformer_quality_filter_report.json", "w") as file:
            json.dump(report, file, indent=2)
        if max_atom_force is not None:
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


def get_data_spice2_force_filtered(
    data_dir: pathlib.Path | str, config: "WorkflowConfig | None" = None
) -> None:
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


# Pre-computed per-conformer SMILES-quality CSVs (structural checks), row-aligned by
# ``index`` to the raw descent-format HF datasets. Copied into the data dir at run time.
SMILES_QUALITY_CSV_SOURCE_DIR = pathlib.Path("tmp_geom/summary")
SMILES_QUALITY_CSV_NAMES = {
    "spice": "descent-format-spice_smiles_quality.csv",
    "geom": "descent-format-geom_smiles_quality.csv",
}


def _copy_smiles_quality_csv(dataset: str, data_dir: pathlib.Path) -> pathlib.Path | None:
    """Copy the ``dataset`` smiles-quality CSV into ``data_dir`` (skip if already there).

    Returns the destination path, or ``None`` if the source CSV is missing.
    """
    src = SMILES_QUALITY_CSV_SOURCE_DIR / SMILES_QUALITY_CSV_NAMES[dataset]
    dst = data_dir / SMILES_QUALITY_CSV_NAMES[dataset]
    if dst.exists():
        logger.info(f"  smiles-quality CSV for '{dataset}' already at {dst}. Skipping copy.")
        return dst
    if not src.exists():
        logger.warning(
            f"  smiles-quality CSV for '{dataset}' not found at {src}; structural filter "
            "will be skipped for this source."
        )
        return None
    logger.info(f"  copying smiles-quality CSV {src} -> {dst}")
    shutil.copy(src, dst)
    return dst


def _effective_conformer_filter(config: "WorkflowConfig | None") -> dict[str, Any]:
    """Resolve the force/energy conformer-filter thresholds from the config.

    Falls back to the historical lenient defaults (max_atom_force=250, min dist 0.9, min 5
    conformers, no RMS/energy gates) when the config does not specify a ``conformer_filter``.
    """
    cf = getattr(config, "conformer_filter", None) if config is not None else None
    if cf is None:
        return {
            "max_atom_force": 250.0,
            "max_rms_force": None,
            "max_relative_energy": None,
            "min_interatomic_dist": 0.9,
            "min_conformers": 5,
        }
    return {
        "max_atom_force": cf.max_atom_force,
        "max_rms_force": cf.max_rms_force,
        "max_relative_energy": cf.max_relative_energy,
        "min_interatomic_dist": cf.min_interatomic_dist,
        "min_conformers": cf.min_conformers,
    }


def get_data_omol25_combined(
    data_dir: pathlib.Path | str, config: "WorkflowConfig | None" = None
) -> None:
    """Download, reprocess and combine the OMol25 SPICE + GEOM datasets for fitting.

    Reproduces the ``tmp_geom`` pipeline from scratch: downloads the raw descent-format
    SPICE and GEOM HuggingFace datasets, fetches SPICE metadata, deduplicates each
    (stereo-collapsed, ``min_conformers=5``; SPICE also filtered to the
    ``pubchem_numbered+dipeptides`` source preset), concatenates them, and writes a random
    95/5 train/test split. Every expensive step skips when its output already exists, so
    the rule is restartable.

    Two source-agnostic filters are applied, driven by the workflow config:

    * a **structural SMILES-quality** filter (``config.smiles_quality_keep``) applied per
      source during deduplication using the pre-computed ``smiles_quality`` CSVs, and
    * an **aggressive per-conformer force/energy** filter (``config.conformer_filter``)
      applied once to the combined dataset, with before/after distribution plots written to
      ``data_dir/filter_plots``.
    """
    from . import omol25

    data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Getting combined OMol25 SPICE + GEOM data...")

    keep_classifications = getattr(config, "smiles_quality_keep", None) if config else None
    conf_filter = _effective_conformer_filter(config)
    logger.info(f"SMILES-quality keep classifications: {keep_classifications}")
    logger.info(f"Conformer force/energy filter thresholds: {conf_filter}")

    # 1. Download raw descent-format datasets (one conformer per row).
    spice_raw = omol25.download_descent_format_dataset("spice", data_dir / "descent_format_spice_hf")
    geom_raw = omol25.download_descent_format_dataset("geom", data_dir / "descent_format_geom_hf")

    # 2. SPICE metadata (for the source-collection preset). GEOM is a single collection.
    spice_metadata = omol25.fetch_metadata("spice", data_dir / "metadata_spice.parquet")

    # 2b. Copy the per-conformer structural-quality CSVs into the data dir (self-contained run).
    spice_quality_csv = geom_quality_csv = None
    if keep_classifications:
        spice_quality_csv = _copy_smiles_quality_csv("spice", data_dir)
        geom_quality_csv = _copy_smiles_quality_csv("geom", data_dir)

    # 3. Deduplicate each dataset into one-molecule-per-row descent datasets, applying the
    #    per-conformer structural (SMILES-quality) filter while conformers are still ungrouped.
    spice_dedup = omol25.deduplicate(
        spice_raw,
        data_dir / "descent_format_spice_pubchem_dipeptides_nostereo_min5_dedup_hf",
        metadata_path=spice_metadata,
        keep=omol25.PRESETS["pubchem_numbered+dipeptides"],
        collapse_stereo=True,
        min_conformers=5,
        quality_csv=spice_quality_csv,
        keep_classifications=keep_classifications,
    )
    geom_dedup = omol25.deduplicate(
        geom_raw,
        data_dir / "descent_format_geom_nostereo_min5_dedup_hf",
        collapse_stereo=True,
        min_conformers=5,
        quality_csv=geom_quality_csv,
        keep_classifications=keep_classifications,
    )

    # 4. Combine (identical schemas).
    logger.info("Combining SPICE + GEOM deduplicated datasets...")
    spice_ds = datasets.Dataset.load_from_disk(str(spice_dedup))
    geom_ds = datasets.Dataset.load_from_disk(str(geom_dedup))
    combined = datasets.concatenate_datasets([spice_ds, geom_ds])
    logger.info(
        f"Combined: {len(spice_ds):,} SPICE + {len(geom_ds):,} GEOM = {len(combined):,} molecules"
    )

    # 4b. Diagnostic plots + aggressive per-conformer force/energy filter. The OMol25 GEOM
    # portion is genuine but aggressively off-equilibrium (bonds compressed to <0.9 Angstrom,
    # per-atom forces / strain energies far above the SPICE envelope), which makes
    # force/energy-matching diverge. Filter per conformer (keeping the good conformers of each
    # molecule) with one source-agnostic threshold set, so any future source is covered too.
    plot_dir = data_dir / "filter_plots"
    plot_thresholds = {
        "max_atom_force": conf_filter["max_atom_force"],
        "max_rms_force": conf_filter["max_rms_force"],
        "max_relative_energy": conf_filter["max_relative_energy"],
        "min_interatomic_dist": conf_filter["min_interatomic_dist"],
    }
    logger.info("Plotting per-conformer distributions before force/energy filtering...")
    plot_conformer_distributions(
        {"SPICE": spice_ds, "GEOM": geom_ds}, plot_thresholds, plot_dir, stage="before"
    )

    combined, _ = filter_conformers_by_quality(
        combined,
        max_atom_force=conf_filter["max_atom_force"],
        max_rms_force=conf_filter["max_rms_force"],
        max_relative_energy=conf_filter["max_relative_energy"],
        min_interatomic_dist=conf_filter["min_interatomic_dist"],
        min_conformers=conf_filter["min_conformers"],
        out_dir=data_dir,
    )

    # Split the filtered combined data back by source (for the "after" plot only).
    spice_smiles = set(spice_ds["smiles"])
    geom_only_smiles = set(geom_ds["smiles"]) - spice_smiles
    after_spice = combined.filter(lambda r: r["smiles"] in spice_smiles)
    after_geom = combined.filter(lambda r: r["smiles"] in geom_only_smiles)
    logger.info("Plotting per-conformer distributions after force/energy filtering...")
    plot_conformer_distributions(
        {"SPICE": after_spice, "GEOM": after_geom}, plot_thresholds, plot_dir, stage="after"
    )

    # 5. Split train/test.
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
