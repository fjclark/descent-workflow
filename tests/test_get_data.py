"""Tests for the force / non-finite dataset filter used to give OMol25 parity with SPICE."""

import descent.targets.energy as energy
import torch

from descent_workflow.get_data import filter_dataset_by_forces, get_rms


def _entry(smiles: str, fval: float, nan: bool = False, inf: bool = False) -> dict:
    """Build a two-conformer entry with a constant force magnitude ``fval``."""
    n_conf, n_at = 2, 3
    coords = torch.zeros(n_conf, n_at, 3)
    energies = torch.zeros(n_conf)
    forces = torch.full((n_conf, n_at, 3), float(fval))
    if nan:
        forces[0, 0, 0] = float("nan")
    if inf:
        forces[0, 0, 0] = float("inf")
    return {"smiles": smiles, "coords": coords, "energy": energies, "forces": forces}


def _dataset() -> "energy.datasets.Dataset":
    entries = [
        _entry("[H:1][C:2]([H:3])([H:4])[H:5]", 1.0),
        _entry("[H:1][O:2][H:3]", 1.2),
        _entry("[C:1]#[N:2]", 1.1),
        _entry("[H:1][N:2]([H:3])[H:4]", 50.0),  # high-force outlier
        _entry("[Cl:1][Cl:2]", 1.0, nan=True),   # non-finite (NaN)
        _entry("[F:1][F:2]", 1.0, inf=True),     # non-finite (Inf)
    ]
    return energy.create_dataset(entries)


def test_get_rms():
    assert get_rms(torch.full((4,), 3.0).numpy()) == 3.0


def test_filter_removes_nonfinite_and_high_force():
    ds = _dataset()
    filtered, report = filter_dataset_by_forces(ds, percentile=95)

    kept = set(filtered["smiles"])
    # Both non-finite entries dropped.
    assert "[Cl:1][Cl:2]" not in kept
    assert "[F:1][F:2]" not in kept
    # The single high-force outlier dropped.
    assert "[H:1][N:2]([H:3])[H:4]" not in kept
    # Ordinary entries retained.
    assert "[H:1][O:2][H:3]" in kept

    assert report["n_nonfinite_removed"] == 2
    assert report["n_high_force_removed"] == 1
    assert report["n_output"] == len(filtered) == 3
    assert set(report["nonfinite_smiles"]) == {"[Cl:1][Cl:2]", "[F:1][F:2]"}
    assert report["high_force_smiles"] == ["[H:1][N:2]([H:3])[H:4]"]


def test_shared_cutoff_applied_to_a_split():
    """A pre-computed cutoff lets several splits be filtered consistently."""
    ds = _dataset()
    _, combined_report = filter_dataset_by_forces(ds, percentile=95)
    cutoff = combined_report["cutoff_kcal_per_mol_angstrom"]

    filtered, report = filter_dataset_by_forces(ds, cutoff=cutoff)
    assert report["cutoff_kcal_per_mol_angstrom"] == cutoff
    assert "[H:1][N:2]([H:3])[H:4]" not in set(filtered["smiles"])
