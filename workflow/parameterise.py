"""
Parameterise a dataset with a force field and save to a torch file, optionally
linearising terms. This borrows heavily from Brent's, Josh's, and Thomas's code:

https://github.com/SimonBoothroyd/descent-ff/blob/main/energy-force/002-parameterize.py
https://github.com/ntBre/descent-ff/blob/main/energy-force/parameterize.py
https://github.com/jthorton/SPICE-SMEE/
"""

import functools
import json
import multiprocessing

import openff.interchange
import openff.toolkit
import smee
import smee.converters
import torch
import tqdm
from openff.units import unit as off_unit
import copy
import numpy as np

from models import WorkflowConfig

from loguru import logger

_ANGSTROM = off_unit.angstrom
_RADIANS = off_unit.radians
_KCAL_PER_MOL_ANGSQ = off_unit.kilocalories_per_mole / off_unit.angstrom**2
_KCAL_PER_MOL_RADSQ = off_unit.kilocalories_per_mole / off_unit.radians**2


# From https://github.com/thomasjamespope/bespokefit_smee/tree/main
def linearize_harmonics(
    ff: smee.TensorForceField, device_type: str
) -> smee.TensorForceField:
    """Linearize the harmonic potential parameters in the forcefield for more robust optimization"""
    ff_copy = copy.deepcopy(ff)
    ff_copy.potentials = []
    for potential in ff.potentials:
        if potential.type in {"Bonds"}:
            new_potential = copy.deepcopy(potential)
            new_potential.type = "LinearBonds"
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
            openff.toolkit.ForceField(*force_field_paths),
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
    build_interchange_fn = functools.partial(
        build_interchange, force_field_paths=force_field_paths
    )

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

    unique_smiles, interchanges = zip(
        *[(s, i) for s, i in zip(unique_smiles, interchanges) if i is not None]
    )

    force_field, topologies = smee.converters.convert_interchange(interchanges)

    if linearise_harm:
        force_field = linearize_harmonics(force_field, device_type="cpu")
        for topology in topologies:
            topology.force_field = force_field
            topology.parameters["LinearBonds"] = copy.deepcopy(
                topology.parameters["Bonds"]
            )
            topology.parameters["LinearAngles"] = copy.deepcopy(
                topology.parameters["Angles"]
            )

    return force_field, {
        smiles: topology for smiles, topology in zip(unique_smiles, topologies)
    }


def create_torch_ff_and_top(config: WorkflowConfig) -> None:
    """Save a pytorch version of a force field and training topologies to
    ``torch_path``.

    Topologies are loaded from ``smiles_path``, which should be a JSON file
    containing a list of SMILES.
    """
    smiles_per_source: dict[str, list[str]] = json.loads(
        config.get_data_output_smiles.read_text()
    )

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
