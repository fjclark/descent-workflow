"""Parameterise a dataset with a force field and save to a torch file, optionally
linearising terms. This borrows heavily from Brent's, Josh's, and Thomas's code:

https://github.com/SimonBoothroyd/descent-ff/blob/main/energy-force/002-parameterize.py
https://github.com/ntBre/descent-ff/blob/main/energy-force/parameterize.py
https://github.com/jthorton/SPICE-SMEE/
"""

import copy
import functools
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


def apply_parameters(
    unique_smiles: list[str], *force_field_paths: str, linearise_harm: bool = False
) -> tuple[smee.TensorForceField, dict[str, smee.TensorTopology]]:
    build_interchange_fn = functools.partial(build_interchange, force_field_paths=force_field_paths)

    with multiprocessing.get_context("spawn").Pool() as pool:
        interchanges = list(
            pool.imap(
                build_interchange_fn,
                tqdm.tqdm(
                    unique_smiles,
                    total=len(unique_smiles),
                    desc="building interchanges",
                ),
            )
        )

    # Filter out None interchanges
    unique_smiles_filtered: list[str] = [
        s for s, i in zip(unique_smiles, interchanges, strict=True) if i is not None
    ]
    interchanges_filtered: list[Interchange] = [
        i for s, i in zip(unique_smiles, interchanges, strict=True) if i is not None
    ]

    force_field, topologies = smee.converters.convert_interchange(interchanges_filtered)

    if linearise_harm:
        force_field = linearise_harmonics_force_field(force_field, device_type="cpu")
        topologies = [
            linearise_harmonics_topology(topology, device_type="cpu") for topology in topologies
        ]

    return force_field, dict(zip(unique_smiles_filtered, topologies, strict=True))


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
    )

    torch_path = config.torch_ffs_and_tops_path
    torch_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((force_field, topologies), torch_path)

    logger.info("Torch force field and topologies saved successfully.")
    logger.info(f"Saved to {torch_path}")
