"""Regression tests for chunked parameterisation.

These prove that ``apply_parameters`` processing molecules in small chunks and merging
the per-chunk results produces output equivalent to converting every molecule in a
single ``smee.converters.convert_interchange`` call (the monolithic path it replaced).

Comparisons are made *by parameter key* rather than by row/column position, so they are
robust to the (physically irrelevant) parameter-ordering differences that chunked
merging can introduce.
"""

import pathlib

import pytest
import smee
import smee.converters
import torch

from descent_workflow.parameterise import (
    apply_parameters,
    build_interchange,
    linearise_harmonics_force_field,
    linearise_harmonics_topology,
)

FF_PATH = str(
    pathlib.Path(__file__).resolve().parents[1]
    / "input_ff"
    / "openff_unconstrained-2.3.0.offxml"
)

# The first seven entries are improper-free alkanes/alcohols, so with ``chunk_size=7``
# the leading chunk contains no improper torsions even though later molecules do. This
# exercises the empty-valence-map synthesis path in ``_FFAccumulator.finalize``.
_PLAIN_SMILES = [
    "C", "CC", "CCC", "CCCC", "CCCCC", "CC(C)C", "CCO",  # chunk 0: no impropers
    "c1ccccc1", "NC=O", "CC(=O)O", "CC(=O)N", "c1ccncc1",
    "OCC", "CCN", "CC#N", "CCOC", "c1ccc(O)cc1", "CC(=O)C",
    "ClCCl", "FCF", "CCS", "O=C(O)c1ccccc1", "Cc1ccccc1",
    "CCCCO", "CC(C)O", "CN(C)C", "c1ccc2ccccc2c1", "C1CCCCC1",
    "C1CCNCC1", "OCCO",
]

CHUNK_SIZE = 7


def _mapped(smiles: str) -> str:
    from openff.toolkit import Molecule

    return Molecule.from_smiles(smiles).to_smiles(mapped=True)


@pytest.fixture(scope="module")
def smiles() -> list[str]:
    mapped = [_mapped(s) for s in _PLAIN_SMILES]
    # An invalid mapped SMILES that ``build_interchange`` fails on. Both paths must drop
    # it consistently.
    mapped.insert(10, "this-is-not-a-valid-mapped-smiles")
    return mapped


def _monolithic(
    smiles: list[str],
) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    interchanges = [build_interchange(s, (FF_PATH,)) for s in smiles]
    kept_smiles = [s for s, i in zip(smiles, interchanges) if i is not None]
    kept = [i for i in interchanges if i is not None]
    force_field, topologies = smee.converters.convert_interchange(kept)
    return force_field, dict(zip(kept_smiles, topologies))


def _assert_ff_params_equal(
    ff_a: smee.TensorForceField, ff_b: smee.TensorForceField
) -> None:
    types_a = {p.type for p in ff_a.potentials}
    types_b = {p.type for p in ff_b.potentials}
    assert types_a == types_b, f"potential types differ: {types_a} vs {types_b}"

    for p_type in types_a:
        pa = ff_a.potentials_by_type[p_type]
        pb = ff_b.potentials_by_type[p_type]
        assert pa.parameter_cols == pb.parameter_cols
        rows_a = {k: r for k, r in zip(pa.parameter_keys, pa.parameters)}
        rows_b = {k: r for k, r in zip(pb.parameter_keys, pb.parameters)}
        assert set(rows_a) == set(rows_b), f"parameter keys differ for {p_type}"
        for key in rows_a:
            assert torch.allclose(
                rows_a[key], rows_b[key]
            ), f"parameter values differ for {p_type} / {key}"


def _dense_by_key(pmap, keys, ordered_keys):
    """Densify an assignment matrix and reorder its columns onto ``ordered_keys``."""
    dense = pmap.assignment_matrix.to_dense()
    key_to_col = {k: i for i, k in enumerate(keys)}
    out = torch.zeros((dense.shape[0], len(ordered_keys)), dtype=dense.dtype)
    for j, key in enumerate(ordered_keys):
        if key in key_to_col:
            out[:, j] = dense[:, key_to_col[key]]
    return out


def _assert_topologies_equal(ff_a, tops_a, ff_b, tops_b) -> None:
    assert set(tops_a) == set(tops_b), "topology SMILES sets differ"

    for smi in tops_a:
        ta, tb = tops_a[smi], tops_b[smi]
        assert set(ta.parameters) == set(
            tb.parameters
        ), f"handler types differ for {smi}"

        for p_type, ma in ta.parameters.items():
            mb = tb.parameters[p_type]
            keys_a = ff_a.potentials_by_type[p_type].parameter_keys
            keys_b = ff_b.potentials_by_type[p_type].parameter_keys
            ordered = list(dict.fromkeys([*keys_a, *keys_b]))

            da = _dense_by_key(ma, keys_a, ordered)
            db = _dense_by_key(mb, keys_b, ordered)
            assert da.shape == db.shape, f"assignment shape differs for {smi}/{p_type}"
            assert torch.allclose(
                da, db
            ), f"assignment matrices differ for {smi}/{p_type}"

            if isinstance(ma, smee.ValenceParameterMap):
                assert isinstance(mb, smee.ValenceParameterMap)
                if ma.particle_idxs.numel() == 0:
                    assert mb.particle_idxs.numel() == 0
                else:
                    assert torch.equal(
                        ma.particle_idxs, mb.particle_idxs
                    ), f"particle idxs differ for {smi}/{p_type}"
            else:
                assert torch.equal(ma.exclusions, mb.exclusions)
                assert torch.equal(ma.exclusion_scale_idxs, mb.exclusion_scale_idxs)


def test_chunked_matches_monolithic(smiles):
    ff_mono, tops_mono = _monolithic(smiles)
    ff_chunk, tops_chunk = apply_parameters(smiles, FF_PATH, chunk_size=CHUNK_SIZE)

    # The invalid SMILES must be dropped, and at least one improper-bearing molecule
    # must be present so the empty-map synthesis path is genuinely exercised.
    assert "this-is-not-a-valid-mapped-smiles" not in tops_chunk
    assert "ImproperTorsions" in ff_chunk.potentials_by_type

    _assert_ff_params_equal(ff_mono, ff_chunk)
    _assert_topologies_equal(ff_mono, tops_mono, ff_chunk, tops_chunk)


def test_chunked_matches_monolithic_linearised(smiles):
    ff_mono, tops_mono = _monolithic(smiles)
    ff_mono_lin = linearise_harmonics_force_field(ff_mono, "cpu")
    tops_mono_lin = {
        s: linearise_harmonics_topology(t, "cpu") for s, t in tops_mono.items()
    }

    ff_chunk, tops_chunk = apply_parameters(
        smiles, FF_PATH, chunk_size=CHUNK_SIZE, linearise_harm=True
    )

    assert "LinearBonds" in ff_chunk.potentials_by_type
    assert "LinearAngles" in ff_chunk.potentials_by_type

    _assert_ff_params_equal(ff_mono_lin, ff_chunk)
    _assert_topologies_equal(ff_mono_lin, tops_mono_lin, ff_chunk, tops_chunk)
