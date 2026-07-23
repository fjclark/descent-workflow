"""Download and reprocess the OpenFF "descent-format" SPICE and GEOM datasets.

Named ``omol25`` after the original data source: the OpenFF
``openforcefield/descent-format-{spice,geom}`` HuggingFace datasets are recomputed
from OMol25.

This module ports the ad-hoc ``tmp_geom`` scripts (``download_geom.py``,
``fetch_metadata.py``, ``deduplicate_datasets.py``) into reusable library functions so
they can be driven from the Snakemake workflow.

The raw HF datasets store one *conformer* per row, so a molecule is spread over many rows
sharing the same mapped ``smiles``. :func:`deduplicate` merges those rows into one row per
molecule, concatenating the per-conformer data:

    energy : list[float]  length n_conformers
    coords : list[float]  length n_conformers * n_atoms * 3   (conformers concatenated)
    forces : list[float]  length n_conformers * n_atoms * 3

The merge is done by *recomputing offsets* over the (already correctly ordered) flat value
buffers, so the large coords/forces value buffers are never copied element-by-element.

Optionally, conformers can first be filtered by their ``source`` collection using an
aligned metadata parquet (see :func:`fetch_metadata`).
"""

from __future__ import annotations

import json
import pathlib
import re
import time
from collections import Counter
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger

LIST_COLS = ("coords", "energy", "forces")

# Named source-collection filters (predicate over the `collection` label).
PRESETS: dict[str, Callable[[str], bool]] = {
    # 10 numbered SPICE PubChem sets + dipeptides; excludes Boron/Silicon,
    # solvated sets, amino-acid ligand, DES, ion pairs.
    "pubchem_numbered+dipeptides": (
        lambda c: bool(re.match(r"SPICE_PubChem_Set_\d+", c)) or c.startswith("SPICE_Dipeptides")
    ),
}


# --------------------------------------------------------------------------------------
# Download (from tmp_geom/download_geom.py)
# --------------------------------------------------------------------------------------
def download_descent_format_dataset(name: str, out_dir: pathlib.Path | str) -> pathlib.Path:
    """Download a descent-format dataset from the HuggingFace Hub to local disk.

    Downloads ``openforcefield/descent-format-<name>`` (config ``descent_data``, split
    ``train``) and saves it to ``out_dir`` as an Arrow/HF dataset. Skips the download if
    ``out_dir`` already exists.
    """
    out_dir = pathlib.Path(out_dir)
    if out_dir.exists():
        logger.info(f"Descent-format dataset '{name}' already exists at {out_dir}. Skipping download.")
        return out_dir

    logger.info(f"Downloading descent-format-{name} from HuggingFace. This may take a while...")
    dataset = load_dataset(
        f"openforcefield/descent-format-{name}",
        "descent_data",
        split="train",
    )
    dataset.save_to_disk(str(out_dir))
    logger.info(f"Saved descent-format-{name} ({len(dataset):,} conformers) to {out_dir}")
    return out_dir


# --------------------------------------------------------------------------------------
# Metadata (from tmp_geom/fetch_metadata.py)
# --------------------------------------------------------------------------------------
def collection_of(source: str, dataset: str) -> str:
    """Derive the ``collection`` label from a granular ``source`` path.

    spice: ``spice/<COLLECTION>_spice_<mol>/...``  -> COLLECTION
    geom : ``geom_orca6/geom_<mol>/...``           -> geom_orca6  (single source)
    """
    parts = source.split("/")
    if dataset == "spice":
        return re.split(r"_spice_", parts[1])[0]
    return parts[0]  # geom: single top-level source


def fetch_metadata(dataset: str, out_path: pathlib.Path | str) -> pathlib.Path:
    """Download per-conformer metadata, derive ``collection``, and cache as parquet.

    Reads all ``metadata/*.parquet`` shards from ``openforcefield/descent-format-<dataset>``
    (columns ``smiles``, ``source``), derives a ``collection`` label per conformer, and
    writes a ``(smiles, source, collection)`` parquet to ``out_path``. This parquet is
    per-conformer and row-aligned with the raw ``_hf`` data. Skips if ``out_path`` exists.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    out_path = pathlib.Path(out_path)
    if out_path.exists():
        logger.info(f"Metadata for '{dataset}' already exists at {out_path}. Skipping fetch.")
        return out_path

    logger.info(f"Downloading {dataset} metadata ...")
    repo = f"openforcefield/descent-format-{dataset}"
    files = sorted(
        f for f in list_repo_files(repo, repo_type="dataset")
        if f.startswith("metadata/") and f.endswith(".parquet")
    )
    smiles_chunks, source_chunks = [], []
    for i, f in enumerate(files):
        path = hf_hub_download(repo, f, repo_type="dataset")
        t = pq.read_table(path, columns=["smiles", "source"])
        smiles_chunks.append(t.column("smiles").combine_chunks())
        source_chunks.append(t.column("source").combine_chunks())
        logger.info(f"  [{dataset}] shard {i + 1}/{len(files)}: {t.num_rows:,} rows")
    # keep chunked: concatenating millions of long SMILES overflows int32 string offsets.
    smiles = pa.chunked_array(smiles_chunks)
    source = pa.chunked_array(source_chunks)
    coll = pa.array([collection_of(s, dataset) for s in source.to_pylist()])
    tab = pa.table({"smiles": smiles, "source": source, "collection": coll})

    pq.write_table(tab, str(out_path))
    _report_collections(dataset, tab)
    logger.info(f"  cached metadata -> {out_path}")
    return out_path


def _report_collections(dataset: str, tab: pa.Table) -> None:
    """Log conformer + unique-molecule counts per source collection."""
    colls = tab.column("collection").to_pylist()
    smiles = tab.column("smiles").to_pylist()
    conf = Counter(colls)
    mols: dict[str, set] = {}
    for c, s in zip(colls, smiles, strict=True):
        mols.setdefault(c, set()).add(s)
    logger.info(
        f"{dataset.upper()} ({tab.num_rows:,} conformers, "
        f"{len(set(smiles)):,} unique molecules, {len(conf)} collections)"
    )
    for c, n in conf.most_common():
        logger.info(f"  {n:>12,} conformers {len(mols[c]):>11,} molecules  {c}")


# --------------------------------------------------------------------------------------
# Deduplicate (from tmp_geom/deduplicate_datasets.py)
# --------------------------------------------------------------------------------------
def _build_keep_mask(
    metadata_path: pathlib.Path | str,
    raw_table: pa.Table,
    n_rows: int,
    keep: Callable[[str], bool],
) -> pa.Array:
    """Boolean mask over raw rows whose source collection passes ``keep``.

    The metadata parquet is per-conformer and row-aligned with the data; we assert
    alignment on ``smiles`` before trusting it.
    """
    md = pq.read_table(str(metadata_path), columns=["smiles", "collection"])
    assert md.num_rows == n_rows, "metadata row count != data row count"
    a = md.column("smiles").combine_chunks()
    b = raw_table.column("smiles").combine_chunks()
    assert pc.all(pc.equal(a, b)).as_py(), "metadata not row-aligned with data"
    coll = md.column("collection")
    keep_names = sorted(c for c in coll.unique().to_pylist() if keep(c))
    mask = pc.is_in(coll, value_set=pa.array(keep_names))
    n_keep = pc.sum(pc.cast(mask, pa.int64())).as_py()
    logger.info(
        f"  source filter: keeping {len(keep_names)} collections, "
        f"{n_keep:,}/{n_rows:,} conformers"
    )
    for k in keep_names:
        logger.info(f"    + {k}")
    return mask


def _regroup_list_column(col: pa.ChunkedArray, group_starts: np.ndarray, n_rows: int) -> pa.Array:
    """Merge per-row list values into per-group lists via new offsets only."""
    la = col.combine_chunks()  # single ListArray, values in sorted-row order
    old_off = la.offsets.to_numpy()  # int32/int64, length n_rows + 1
    # value index at the start of each group, plus the final end.
    bounds = np.append(group_starts, n_rows)
    new_off = old_off[bounds].astype(old_off.dtype, copy=False)
    offsets = pa.array(new_off, type=la.offsets.type)
    # values buffer is shared (zero-copy); only the offsets are new.
    return type(la).from_arrays(offsets, la.values)


def _strip_stereo(smiles_col: pa.ChunkedArray) -> pa.ChunkedArray:
    """Remove stereo descriptors (@, /, \\) from mapped SMILES.

    SPICE/GEOM perceive stereochemistry per-conformer, so conformers of one molecule can
    land on different stereoisomeric mapped-SMILES. Byte-identical stripped strings keep
    the same atom-map ordering, so the flat coords/forces stay compatible and can be
    merged safely. (Molecules written with genuinely different atom orderings are
    intentionally NOT merged, as that would require reindexing coordinates.)
    """
    s = smiles_col
    for ch in ("@", "/", "\\"):
        s = pc.replace_substring(s, pattern=ch, replacement="")
    return s


def _multi_molecule_mask(smiles_col: pa.Array) -> pa.Array:
    """Boolean keep-mask dropping multi-molecule (``.``-joined) SMILES.

    Chunked parameterisation cannot handle the rigid constraints / virtual sites that
    OpenFF assigns to multi-component records (e.g. solvated or ion-pair entries, which
    pull in water/ions), so these are dropped here.
    """
    return pc.invert(pc.match_substring(smiles_col, "."))


def _threshold_stats(grp_sizes: np.ndarray, keep_mol: np.ndarray, min_conformers: int) -> dict:
    """Stats for dropping molecules with fewer than ``min_conformers`` conformers."""
    removed = grp_sizes[~keep_mol]
    n_mol = int(grp_sizes.size)
    n_conf = int(grp_sizes.sum())
    n_mol_rm = int(removed.size)
    n_conf_rm = int(removed.sum())
    # counts of removed molecules by their (below-threshold) conformer count
    hist = (
        {int(k): int(v) for k, v in zip(*np.unique(removed, return_counts=True), strict=True)}
        if removed.size
        else {}
    )
    return {
        "min_conformers": int(min_conformers),
        "molecules_before": n_mol,
        "molecules_kept": n_mol - n_mol_rm,
        "molecules_removed": n_mol_rm,
        "molecules_removed_frac": n_mol_rm / n_mol if n_mol else 0.0,
        "conformers_before": n_conf,
        "conformers_kept": n_conf - n_conf_rm,
        "conformers_removed": n_conf_rm,
        "conformers_removed_frac": n_conf_rm / n_conf if n_conf else 0.0,
        "removed_histogram": hist,  # {conformers_per_removed_molecule: count}
    }


def deduplicate(
    path: pathlib.Path | str,
    out_path: pathlib.Path | str,
    metadata_path: pathlib.Path | str | None = None,
    keep: Callable[[str], bool] | None = None,
    collapse_stereo: bool = False,
    min_conformers: int = 1,
) -> pathlib.Path:
    """Reprocess a raw descent-format dataset so each unique SMILES is a single row.

    Sorts the table by ``smiles`` so every molecule's rows are contiguous, then builds the
    merged list columns by recomputing offsets over the flat value buffers. Optionally
    filters conformers by source collection (``keep`` + ``metadata_path``), collapses
    stereochemistry, and drops molecules with fewer than ``min_conformers`` conformers.

    Skips (and returns) if ``out_path`` already exists.
    """
    path, out_path = pathlib.Path(path), pathlib.Path(out_path)
    if out_path.exists():
        logger.info(f"Deduplicated dataset already exists at {out_path}. Skipping.")
        return out_path

    logger.info(f"Deduplicating {path} -> {out_path}")
    ds = load_from_disk(str(path))
    n_rows = len(ds)
    raw = ds.data.table
    logger.info(f"  input rows (conformers): {n_rows:,}")

    # --- optional source-collection filter (per-conformer) ---
    mask = None
    if keep is not None:
        if metadata_path is None:
            raise ValueError("metadata_path is required with keep")
        mask = _build_keep_mask(metadata_path, raw, n_rows, keep)

    # Widen to 64-bit offsets up front: for big datasets the concatenated buffers overflow
    # 32-bit offsets during take/sort even though the final per-molecule totals fit. Both
    # the float lists (large_list) and the long SMILES strings (large_string) need this.
    # We cast everything back at the end.
    t0 = time.time()
    smiles_col = raw.column("smiles")
    if collapse_stereo:
        smiles_col = _strip_stereo(smiles_col)
        logger.info("  collapsing stereochemistry (grouping by stereo-stripped SMILES)")
    tab = pa.table(
        {
            "smiles": pc.cast(smiles_col, pa.large_string()),
            **{n: pc.cast(raw.column(n), pa.large_list(pa.float32())) for n in LIST_COLS},
        }
    )
    if mask is not None:
        tab = tab.filter(mask)
        n_rows = tab.num_rows
        logger.info(f"  filtered to {n_rows:,} conformers")

    # --- sort the whole table by smiles (molecules become contiguous) ---
    sort_idx = pc.sort_indices(tab.column("smiles"))
    tab = tab.take(sort_idx)
    logger.info(f"  cast + sorted by smiles in {time.time() - t0:.1f}s")

    # --- group boundaries: first row index of each distinct smiles ---
    sm = tab.column("smiles").combine_chunks()
    changed = pc.not_equal(sm.slice(1), sm.slice(0, len(sm) - 1))
    change_pos = np.nonzero(changed.to_numpy(zero_copy_only=False))[0] + 1
    group_starts = np.concatenate([[0], change_pos]).astype(np.int64)
    n_groups = len(group_starts)
    logger.info(f"  unique molecules: {n_groups:,}")

    # --- build merged columns in the canonical descent Entry order ---
    # (descent.targets.energy.DATA_SCHEMA: id, smiles, coords, box_vectors, energy,
    #  forces). The raw inputs have no id / box_vectors, so emit nulls.
    t0 = time.time()
    null_id = pa.array([None] * n_groups, type=pa.string())
    null_box = pa.array([None] * n_groups, type=pa.list_(pa.float32()))
    # regroup with int64 offsets, then cast back to descent's list(float32)
    # (per-molecule totals fit int32).
    small = pa.list_(pa.float32())
    merged = {
        n: pc.cast(_regroup_list_column(tab.column(n), group_starts, n_rows), small)
        for n in LIST_COLS
    }
    out_table = pa.table(
        {
            "id": null_id,
            "smiles": pc.cast(sm.take(pa.array(group_starts)), pa.string()),
            "coords": merged["coords"],
            "box_vectors": null_box,
            "energy": merged["energy"],
            "forces": merged["forces"],
        }
    )
    logger.info(f"  merged {len(LIST_COLS)} list columns in {time.time() - t0:.1f}s")

    # --- sanity checks: no data lost, no duplicate smiles ---
    assert out_table.num_rows == n_groups
    assert pc.count_distinct(out_table.column("smiles")).as_py() == n_groups, "duplicate smiles remain"
    for name in LIST_COLS:
        n_in = len(tab.column(name).combine_chunks().values)
        n_out = len(out_table.column(name).combine_chunks().values)
        assert n_in == n_out, f"{name}: value count changed {n_in} -> {n_out}"
    # spot check: energies per molecule == rows per molecule
    grp_sizes = np.diff(np.append(group_starts, n_rows))
    e_lens = pc.list_value_length(out_table.column("energy")).to_numpy()
    assert np.array_equal(grp_sizes, e_lens), "energy grouping mismatch"
    logger.info(
        f"  checks passed (conformers/mol: min={grp_sizes.min()} "
        f"mean={grp_sizes.mean():.2f} max={grp_sizes.max()})"
    )

    # --- drop multi-molecule (`.`-joined) records ---
    keep_single = _multi_molecule_mask(out_table.column("smiles"))
    n_multi = pc.sum(pc.cast(pc.invert(keep_single), pa.int64())).as_py()
    if n_multi:
        out_table = out_table.filter(keep_single)
        grp_sizes = grp_sizes[keep_single.to_numpy(zero_copy_only=False)]
        logger.info(f"  dropped {n_multi:,} multi-molecule SMILES; {out_table.num_rows:,} molecules remain")

    # --- optional minimum-conformers-per-molecule filter ---
    if min_conformers > 1:
        keep_mol = grp_sizes >= min_conformers
        stats = _threshold_stats(grp_sizes, keep_mol, min_conformers)
        out_table = out_table.filter(pa.array(keep_mol))
        grp_sizes = grp_sizes[keep_mol]
        stats_path = pathlib.Path(f"{out_path}.filterstats.json")
        with open(stats_path, "w") as fh:
            json.dump(stats, fh, indent=2)
        logger.info(
            f"  min_conformers={min_conformers}: removed "
            f"{stats['molecules_removed']:,}/{stats['molecules_before']:,} molecules "
            f"({stats['molecules_removed_frac']:.1%}) and "
            f"{stats['conformers_removed']:,}/{stats['conformers_before']:,} conformers "
            f"({stats['conformers_removed_frac']:.1%}); kept "
            f"{stats['molecules_kept']:,} molecules"
        )
        logger.info(f"  removed-molecule conformer counts: {stats['removed_histogram']}")
        logger.info(f"  stats -> {stats_path}")

    t0 = time.time()
    # Set the torch format (mirroring descent.targets.energy.create_dataset) so that
    # indexing returns tensors, not Python lists. The format is persisted in state.json
    # and restored on load, and propagates through concatenate/filter/split downstream.
    ds = Dataset(out_table)
    ds.set_format("torch")
    ds.save_to_disk(str(out_path))
    logger.info(f"  saved -> {out_path} in {time.time() - t0:.1f}s")
    return out_path
