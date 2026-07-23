"""Parameterise a dataset with a force field and save to a torch file, optionally
linearising terms. This borrows heavily from Brent's, Josh's, and Thomas's code:

https://github.com/SimonBoothroyd/descent-ff/blob/main/energy-force/002-parameterize.py
https://github.com/ntBre/descent-ff/blob/main/energy-force/parameterize.py
https://github.com/jthorton/SPICE-SMEE/
"""

import copy
import functools
import gc
import json
import math
import multiprocessing
from typing import Callable, Literal

import numpy as np
import openff.interchange
import openff.toolkit
import smee
import smee.converters
import torch
import tqdm
from loguru import logger
from .models import WorkflowConfig
from openff.interchange import Interchange
from openff.units import unit as off_unit

_ANGSTROM = off_unit.angstrom
_RADIANS = off_unit.radians
_KCAL_PER_MOL_ANGSQ = off_unit.kilocalories_per_mole / off_unit.angstrom**2
_KCAL_PER_MOL_RADSQ = off_unit.kilocalories_per_mole / off_unit.radians**2


def _compute_linear_harmonic_params(
    k: float,
    eq_value: float,
    compute_lower_bound: Callable[[float], float],
    compute_upper_bound: Callable[[float], float],
) -> tuple[float, float, float, float]:
    """Compute linearized harmonic parameters from standard parameters.

    This generic function distributes a force constant across two bounds,
    inversely proportional to the distance from each bound.

    Args:
        k: Force constant (e.g., kcal/mol/Å² or kcal/mol/rad²)
        eq_value: Equilibrium value (e.g., bond length or angle)
        compute_lower_bound: Function that takes eq_value and returns
            lower bound
        compute_upper_bound: Function that takes eq_value and returns
            upper bound

    Returns:
        Tuple of (k1, k2, eq1, eq2) where:
        - k1, k2: Distributed force constants
        - eq1, eq2: Lower and upper equilibrium value bounds

    """
    eq1 = compute_lower_bound(eq_value)
    eq2 = compute_upper_bound(eq_value)
    d = eq2 - eq1
    # Distribute force constant inversely proportional to distance from bounds
    k1 = k * (eq2 - eq_value) / d
    k2 = k * (eq_value - eq1) / d
    return k1, k2, eq1, eq2


def _linearize_bond_parameters(
    potential: smee.TensorPotential, device_type: str
) -> smee.TensorPotential:
    """Linearize bond potential parameters.

    Converts standard harmonic bond parameters (k, length) to linearized
    form (k1, k2, b1, b2) where the equilibrium bond length range is
    [0.5*length, 1.5*length].
    """
    new_potential = copy.deepcopy(potential)
    new_potential.type = "LinearBonds"
    new_potential.fn = "(k1+k2)/2*(r-(k1*length1+k2*length2)/(k1+k2))**2"
    new_potential.parameter_cols = ("k1", "k2", "b1", "b2")

    # Get dtype from the first parameter
    dtype = potential.parameters.dtype

    new_params = [
        _compute_linear_harmonic_params(
            param[0].item(),
            param[1].item(),
            lambda b: b - 0.4,  # Lower bound: current length - 0.4 Å
            lambda b: b + 0.4,  # Upper bound: current length + 0.4 Å
        )
        for param in potential.parameters
    ]

    new_potential.parameters = torch.tensor(
        new_params, dtype=dtype, requires_grad=False, device=device_type
    )
    new_potential.parameter_units = (
        _KCAL_PER_MOL_ANGSQ,
        _KCAL_PER_MOL_ANGSQ,
        _ANGSTROM,
        _ANGSTROM,
    )
    return new_potential


def _linearize_angle_parameters(
    potential: smee.TensorPotential, device_type: str
) -> smee.TensorPotential:
    """Linearize angle potential parameters.

    Converts standard harmonic angle parameters (k, angle) to linearized form
    (k1, k2, angle1, angle2) where the equilibrium angle range is [0, π].
    """
    new_potential = copy.deepcopy(potential)
    new_potential.type = "LinearAngles"
    new_potential.fn = "(k1+k2)/2*(r-(k1*angle1+k2*angle2)/(k1+k2))**2"
    new_potential.parameter_cols = ("k1", "k2", "angle1", "angle2")

    # Get dtype from the first parameter
    dtype = potential.parameters.dtype

    new_params = [
        _compute_linear_harmonic_params(
            param[0].item(),
            param[1].item(),
            lambda a: max(0.0, a - math.pi / 3),  # Lower bound: max(0, angle - π/3)
            lambda a: min(math.pi, a + math.pi / 3),  # Upper bound: min(π, angle + π/3)
        )
        for param in potential.parameters
    ]

    new_potential.parameters = torch.tensor(
        new_params, dtype=dtype, requires_grad=False, device=device_type
    )
    new_potential.parameter_units = (
        _KCAL_PER_MOL_RADSQ,
        _KCAL_PER_MOL_RADSQ,
        _RADIANS,
        _RADIANS,
    )
    return new_potential


def linearise_harmonics_force_field(
    ff: smee.TensorForceField, device_type: str
) -> smee.TensorForceField:
    """Linearize the harmonic potential parameters in the forcefield.

    This converts Bonds and Angles potentials to their linearized forms
    (LinearBonds and LinearAngles) for more robust optimization.
    """
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []

    for potential in ff.potentials:
        if potential.type in {"Bonds", "UreyBradleys"}:
            ff_copy.potentials.append(_linearize_bond_parameters(potential, device_type))
        elif potential.type == "Angles":
            ff_copy.potentials.append(_linearize_angle_parameters(potential, device_type))
        else:
            ff_copy.potentials.append(potential)

    return ff_copy


def linearise_harmonics_topology(
    topology: smee.TensorTopology, device_type: Literal["cpu", "cuda"]
) -> smee.TensorTopology:
    """Linearize harmonic potential parameters in the topology.

    This updates the topology to use LinearBonds and LinearAngles
    parameter maps instead of Bonds and Angles.
    """
    topology_copy = topology.to(device_type)
    if "Bonds" in topology_copy.parameters:
        topology_copy.parameters["LinearBonds"] = copy.deepcopy(topology_copy.parameters["Bonds"])
        del topology_copy.parameters["Bonds"]
    if "UreyBradleys" in topology_copy.parameters:
        topology_copy.parameters["LinearUreyBradleys"] = copy.deepcopy(
            topology_copy.parameters["UreyBradleys"]
        )
        del topology_copy.parameters["Bonds"]
    if "Angles" in topology_copy.parameters:
        topology_copy.parameters["LinearAngles"] = copy.deepcopy(topology_copy.parameters["Angles"])
        del topology_copy.parameters["Angles"]
    return topology_copy


# From https://github.com/thomasjamespope/bespokefit_smee/tree/main
def linearize_harmonics(ff: smee.TensorForceField, device_type: str) -> smee.TensorForceField:
    """Linearize the harmonic potential parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        # if potential.type in {"Bonds", "UreyBradleys"}:
        if potential.type in {"Bonds"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "Linear" + potential.type
            new_potential.fn = "(k1+k2)/2*(r-(k1*length1+k2*length2)/(k1+k2))**2"
            new_potential.parameter_cols = ("k1", "k2", "b1", "b2")
            new_params = []
            for param in potential.parameters:
                k = param[0].item()
                b = param[1].item()
                dt = param.dtype
                b1 = 1.5
                b2 = 6.0
                # b1 = b * 0.9
                # b2 = b * 1.1
                d = b2 - b1
                k1 = k * (b2 - b) / d
                k2 = k * (b - b1) / d
                new_params.append([k1, k2, b1, b2])
            new_potential.parameters = torch.tensor(
                new_params, dtype=dt, requires_grad=False, device=device_type
            )
            new_potential.parameter_units = (
                _KCAL_PER_MOL_ANGSQ,
                _KCAL_PER_MOL_ANGSQ,
                _ANGSTROM,
                _ANGSTROM,
            )
            ff_copy.potentials.append(new_potential)
        elif potential.type in {"Angles"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearAngles"
            new_potential.fn = "(k1+k2)/2*(r-(k1*angle1+k2*angle2)/(k1+k2))**2"
            new_potential.parameter_cols = ("k1", "k2", "angle1", "angle2")
            new_params = []
            for param in potential.parameters:
                k = param[0].item()
                a = param[1].item()
                dt = param.dtype
                # a1 = a * 0.9
                # a2 = a * 1.1
                a1 = 0.0
                a2 = np.pi
                d = a2 - a1
                k1 = k * (a2 - a) / d
                k2 = k * (a - a1) / d
                new_params.append([k1, k2, a1, a2])
            new_potential.parameters = torch.tensor(
                new_params, dtype=dt, requires_grad=False, device=device_type
            )
            new_potential.parameter_units = (
                _KCAL_PER_MOL_RADSQ,
                _KCAL_PER_MOL_RADSQ,
                _RADIANS,
                _RADIANS,
            )
            ff_copy.potentials.append(new_potential)
        else:
            ff_copy.potentials.append(potential)
    return ff_copy


def build_interchange(
    smiles: str, force_field_paths: tuple[str, ...]
) -> openff.interchange.Interchange | None:
    try:
        return openff.interchange.Interchange.from_smirnoff(
            openff.toolkit.ForceField(*force_field_paths, load_plugins=True),
            openff.toolkit.Molecule.from_mapped_smiles(
                smiles, allow_undefined_stereo=True
            ).to_topology(),
        )
    except BaseException as e:
        logger.error(f"failed to parameterize {smiles}: {e}")
        return None


# The number of particle-index columns for each valence handler type. Used to
# synthesise empty parameter maps for topologies belonging to a chunk that happened to
# contain no interactions of a type that is present globally (see ``_FFAccumulator``).
_VALENCE_N_COLS: dict[str, int] = {
    "Bonds": 2,
    "UreyBradleys": 2,
    "Angles": 3,
    "ProperTorsions": 4,
    "ImproperTorsions": 4,
}


def _reindex_assignment_matrix(
    matrix: torch.Tensor, local_to_global: torch.Tensor, n_global: int
) -> torch.Tensor:
    """Remap the parameter (column) indices of a sparse assignment matrix from a
    chunk-local parameter ordering to a global one and set the declared number of
    parameter columns to ``n_global``.

    The row dimension (interactions for valence maps, particles for nonbonded maps) is
    left unchanged.
    """
    matrix = matrix.coalesce()
    indices = matrix.indices()
    new_indices = torch.stack([indices[0], local_to_global[indices[1]]])
    return torch.sparse_coo_tensor(
        new_indices, matrix.values(), (matrix.shape[0], n_global)
    ).coalesce()


def _rebuild_parameter_map(
    pmap: smee.ParameterMap, new_matrix: torch.Tensor
) -> smee.ParameterMap:
    """Return a copy of ``pmap`` with its assignment matrix replaced by ``new_matrix``,
    preserving the map subclass and its other fields."""
    if isinstance(pmap, smee.ValenceParameterMap):
        return smee.ValenceParameterMap(pmap.particle_idxs, new_matrix)
    if isinstance(pmap, smee.NonbondedParameterMap):
        # exclusions / exclusion_scale_idxs index into the potential's ``attributes``
        # (the 1-n scaling factors), which are identical across chunks for the same
        # force field, so they need no remapping.
        return smee.NonbondedParameterMap(
            new_matrix, pmap.exclusions, pmap.exclusion_scale_idxs
        )
    raise NotImplementedError(f"unsupported parameter map type {type(pmap)}")


def _reindex_parameter_map(
    pmap: smee.ParameterMap, local_to_global: torch.Tensor, n_global: int
) -> smee.ParameterMap:
    """Return a copy of ``pmap`` whose assignment-matrix columns are remapped to the
    global parameter ordering with width ``n_global``."""
    new_matrix = _reindex_assignment_matrix(
        pmap.assignment_matrix, local_to_global, n_global
    )
    return _rebuild_parameter_map(pmap, new_matrix)


class _FFAccumulator:
    """Incrementally merge the ``(TensorForceField, list[TensorTopology])`` results of
    per-chunk :func:`smee.converters.convert_interchange` calls into a single force
    field and topology dict with a globally consistent parameter ordering.

    Only the light-weight tensor topologies are held across chunks; the heavy OpenFF
    ``Interchange`` objects live for the duration of a single chunk only. Iterating
    chunks in the same order as a monolithic conversion and appending newly-seen
    parameter keys in first-seen order reproduces the monolithic output.
    """

    def __init__(self) -> None:
        # Per potential type, the metadata shared by every chunk (everything except the
        # ``parameters`` tensor and ``parameter_keys``), captured from the first chunk
        # that contains the type.
        self._meta: dict[str, dict] = {}
        # Global first-seen parameter key ordering per potential type.
        self._keys: dict[str, list] = {}
        self._key_to_idx: dict[str, dict] = {}
        # Accumulated per-key parameter rows (1-D tensors), stacked at ``finalize``.
        self._param_rows: dict[str, list[torch.Tensor]] = {}
        # Potential types in first-seen order (defines the final ``potentials`` order).
        self._type_order: list[str] = []
        # Merged topologies and their SMILES, in insertion order.
        self._topologies: list[smee.TensorTopology] = []
        self._smiles: list[str] = []

    @staticmethod
    def _capture_meta(potential: smee.TensorPotential) -> dict:
        return {
            "fn": potential.fn,
            "parameter_cols": potential.parameter_cols,
            "parameter_units": potential.parameter_units,
            "attributes": (
                None if potential.attributes is None else potential.attributes.clone()
            ),
            "attribute_cols": potential.attribute_cols,
            "attribute_units": potential.attribute_units,
            "exceptions": potential.exceptions,
        }

    def _assert_meta_matches(
        self, p_type: str, potential: smee.TensorPotential
    ) -> None:
        meta = self._meta[p_type]
        assert potential.fn == meta["fn"], f"inconsistent fn for {p_type} across chunks"
        assert (
            potential.parameter_cols == meta["parameter_cols"]
        ), f"inconsistent parameter_cols for {p_type} across chunks"
        assert (
            potential.parameter_units == meta["parameter_units"]
        ), f"inconsistent parameter_units for {p_type} across chunks"
        assert (
            potential.attribute_cols == meta["attribute_cols"]
        ), f"inconsistent attribute_cols for {p_type} across chunks"
        assert (
            potential.attribute_units == meta["attribute_units"]
        ), f"inconsistent attribute_units for {p_type} across chunks"
        if meta["attributes"] is None:
            assert (
                potential.attributes is None
            ), f"inconsistent attributes for {p_type} across chunks"
        else:
            assert potential.attributes is not None and torch.allclose(
                potential.attributes, meta["attributes"]
            ), f"inconsistent attributes for {p_type} across chunks"

    def add_chunk(
        self,
        force_field: smee.TensorForceField,
        topologies: list[smee.TensorTopology],
        smiles: list[str],
    ) -> None:
        assert (
            force_field.v_sites is None
        ), "chunked parameterisation does not support force fields with virtual sites"

        local_to_global: dict[str, torch.Tensor] = {}

        for potential in force_field.potentials:
            assert potential.exceptions is None, (
                "chunked parameterisation does not support parameter exceptions "
                f"(found on {potential.type})"
            )

            p_type = potential.type

            if p_type not in self._meta:
                self._meta[p_type] = self._capture_meta(potential)
                self._keys[p_type] = []
                self._key_to_idx[p_type] = {}
                self._param_rows[p_type] = []
                self._type_order.append(p_type)
            else:
                self._assert_meta_matches(p_type, potential)

            keys = self._keys[p_type]
            key_to_idx = self._key_to_idx[p_type]
            rows = self._param_rows[p_type]

            mapping = torch.empty(len(potential.parameter_keys), dtype=torch.int64)
            for i, key in enumerate(potential.parameter_keys):
                if key not in key_to_idx:
                    key_to_idx[key] = len(keys)
                    keys.append(key)
                    rows.append(potential.parameters[i].clone())
                mapping[i] = key_to_idx[key]
            local_to_global[p_type] = mapping

        assert len(topologies) == len(smiles), "mismatched topologies and smiles"

        for topology, smi in zip(topologies, smiles, strict=True):
            assert (
                topology.v_sites is None and topology.constraints is None
            ), "chunked parameterisation does not support v-sites or constraints"
            topology.parameters = {
                p_type: _reindex_parameter_map(
                    pmap, local_to_global[p_type], len(self._keys[p_type])
                )
                for p_type, pmap in topology.parameters.items()
            }
            self._topologies.append(topology)
            self._smiles.append(smi)

    def _empty_valence_map(
        self, p_type: str, n_global: int
    ) -> smee.ValenceParameterMap:
        assert p_type in _VALENCE_N_COLS, (
            f"topology is missing the non-valence potential {p_type!r}; cannot "
            "synthesise an empty parameter map for it"
        )
        # Match the representation smee produces for a handler with no interactions:
        # a 1-D empty ``particle_idxs`` and an all-zero float64 assignment matrix.
        empty_matrix = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.int64),
            torch.zeros(0, dtype=torch.float64),
            (0, n_global),
        ).coalesce()
        return smee.ValenceParameterMap(torch.tensor([]), empty_matrix)

    def finalize(
        self,
    ) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
        n_global = {p_type: len(self._keys[p_type]) for p_type in self._type_order}

        potentials = [
            smee.TensorPotential(
                type=p_type,
                fn=self._meta[p_type]["fn"],
                parameters=torch.stack(self._param_rows[p_type]),
                parameter_keys=self._keys[p_type],
                parameter_cols=self._meta[p_type]["parameter_cols"],
                parameter_units=self._meta[p_type]["parameter_units"],
                attributes=self._meta[p_type]["attributes"],
                attribute_cols=self._meta[p_type]["attribute_cols"],
                attribute_units=self._meta[p_type]["attribute_units"],
                exceptions=self._meta[p_type]["exceptions"],
            )
            for p_type in self._type_order
        ]

        # Column indices are already global; only the declared width may still be stale
        # (it was set to the running global count when the topology's chunk was merged).
        for topology in self._topologies:
            new_params = {}
            for p_type in self._type_order:
                if p_type in topology.parameters:
                    pmap = topology.parameters[p_type]
                    width = pmap.assignment_matrix.shape[1]
                    new_params[p_type] = _reindex_parameter_map(
                        pmap, torch.arange(width, dtype=torch.int64), n_global[p_type]
                    )
                else:
                    new_params[p_type] = self._empty_valence_map(
                        p_type, n_global[p_type]
                    )
            topology.parameters = new_params

        force_field = smee.TensorForceField(potentials)
        topologies = dict(zip(self._smiles, self._topologies, strict=True))
        return force_field, topologies


def apply_parameters(
    unique_smiles: list[str],
    *force_field_paths: str,
    linearise_harm: bool = False,
    chunk_size: int = 5000,
) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    """Parameterise ``unique_smiles`` with the given force field(s).

    To keep peak memory bounded for large datasets, the SMILES are processed in chunks
    of ``chunk_size``: only one chunk's worth of heavy OpenFF ``Interchange`` objects is
    resident at a time. The per-chunk conversions are merged into a single force field
    and topology dict with a globally consistent parameter ordering, producing output
    equivalent to converting every molecule in a single call.
    """
    build_interchange_fn = functools.partial(
        build_interchange, force_field_paths=force_field_paths
    )
    chunk_size = max(1, chunk_size)
    accumulator = _FFAccumulator()

    n_molecules = len(unique_smiles)
    n_chunks = math.ceil(n_molecules / chunk_size)

    with multiprocessing.get_context("spawn").Pool() as pool:
        chunk_bar = tqdm.tqdm(
            range(0, n_molecules, chunk_size),
            total=n_chunks,
            desc="parameterising chunks",
            unit="chunk",
            position=0,
        )
        for chunk_idx, start in enumerate(chunk_bar):
            chunk_smiles = unique_smiles[start : start + chunk_size]
            # Inner, per-molecule bar so progress is visible while a (large) chunk of
            # interchanges is being built, rather than only ticking once per chunk.
            interchanges = list(
                tqdm.tqdm(
                    pool.imap(build_interchange_fn, chunk_smiles),
                    total=len(chunk_smiles),
                    desc=f"  building interchanges (chunk {chunk_idx + 1}/{n_chunks})",
                    unit="mol",
                    position=1,
                    leave=False,
                )
            )

            kept_smiles = [
                s
                for s, i in zip(chunk_smiles, interchanges, strict=True)
                if i is not None
            ]
            kept_interchanges: list[Interchange] = [
                i for i in interchanges if i is not None
            ]
            del interchanges

            if not kept_interchanges:
                continue

            force_field_chunk, topologies_chunk = smee.converters.convert_interchange(
                kept_interchanges
            )
            # Free this chunk's Interchanges before building the next chunk.
            del kept_interchanges
            gc.collect()

            accumulator.add_chunk(force_field_chunk, topologies_chunk, kept_smiles)
            del force_field_chunk, topologies_chunk

    force_field, topologies = accumulator.finalize()

    if linearise_harm:
        force_field = linearise_harmonics_force_field(force_field, device_type="cpu")
        topologies = {
            smiles: linearise_harmonics_topology(topology, device_type="cpu")
            for smiles, topology in topologies.items()
        }

    return force_field, topologies


def create_torch_ff_and_top(config: WorkflowConfig) -> None:
    """Save a pytorch version of a force field and training topologies to
    ``torch_path``.

    Topologies are loaded from ``smiles_path``, which should be a JSON file
    containing a list of SMILES.
    """
    smiles_per_source: dict[str, list[str]] = json.loads(config.get_data_output_smiles.read_text())

    unique_smiles_set = set()

    for source, smiles in smiles_per_source.items():
        print(f"{source}: {len(smiles)}")
        unique_smiles_set.update(smiles)

    logger.info(f"N smiles={len(unique_smiles_set)}")

    unique_smiles_sorted = sorted(unique_smiles_set)

    logger.info(f"Parameterising. Linearise_harm={config.linearise_harm}")
    force_field, topologies = apply_parameters(
        unique_smiles_sorted,
        *[str(config.starting_force_field_path)],
        linearise_harm=config.linearise_harm,
        chunk_size=config.parameterise_chunk_size,
    )

    torch_path = config.torch_ffs_and_tops_path
    torch_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((force_field, topologies), torch_path)

    logger.info("Torch force field and topologies saved successfully.")
    logger.info(f"Saved to {torch_path}")
