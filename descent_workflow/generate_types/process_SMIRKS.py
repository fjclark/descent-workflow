"""
Simple SMIRKS pattern generation for molecular mechanics force fields.

This module provides functions to generate SMIRKS patterns at different specificity
levels for force field parameter assignment. It follows a functional approach with
simple, composable functions rather than complex factory patterns.

Functions are registered using decorators to create two registries:
- atom_fn_map: Registry of atom SMIRKS generation functions
- bond_fn_map: Registry of bond SMIRKS generation functions

These registries are used by the configuration system to look up functions by name.

Functions
---------
get_atom_descriptors
    Extract atomic properties for SMIRKS generation.
get_bond_descriptors
    Extract bond properties for SMIRKS generation.
get_atom_smirks_standard
    Standard atom SMIRKS: [#NXM:ID]
get_atom_smirks_with_ring_info
    Atom SMIRKS with ring membership: [#NXM;rN:ID]
get_atom_smirks_terminal_wildcard
    Terminal atoms as wildcards: [*:ID]
get_atom_smirks_terminal_h_no_h
    Terminal atoms as H or not-H: [#1:ID] or [!#1:ID]
get_atom_smirks_terminal_h_no_h_ring_info
    Terminal atoms as H/not-H with ring info for non-terminals
get_bond_smirks_standard
    Standard bond SMIRKS with explicit bond types
get_bond_smirks_with_ring_info
    Bond SMIRKS with ring membership
get_bond_smirks_non_central_wildcard
    Wildcard non-central bonds
get_bond_smirks_wildcard
    All bonds as wildcards
add_types_to_ff
    Integrate component parameters into OpenFF force fields.
"""

from copy import deepcopy
from typing import Callable
from rdkit import Chem
from tqdm import tqdm
from loguru import logger

from openff.toolkit import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterType

from .molecular_classes import MMComponent


# Registry for SMIRKS generation functions
atom_fn_map: dict[str, Callable] = {}
bond_fn_map: dict[str, Callable] = {}


def register_atom_smirks(name: str) -> Callable:
    """Decorator to register an atom SMIRKS generation function."""

    def decorator(func: Callable) -> Callable:
        atom_fn_map[name] = func
        return func

    return decorator


def register_bond_smirks(name: str) -> Callable:
    """Decorator to register a bond SMIRKS generation function."""

    def decorator(func: Callable) -> Callable:
        bond_fn_map[name] = func
        return func

    return decorator


def get_atom_descriptors(at_idx: int, mol: Chem.Mol) -> dict[str, str]:
    """
    Extract atomic properties for SMIRKS pattern generation.

    Parameters
    ----------
    at_idx : int
        Atom index in the molecule.
    mol : rdkit.Chem.Mol
        RDKit molecule object with MDL aromaticity model applied.

    Returns
    -------
    dict[str, str]
        Atomic descriptors:
        - 'atomic_num': "#6", "#7", etc.
        - 'degree': "X1", "X2", "X3", "X4"
        - 'charge': "+0", "+1", "-1", etc.
        - 'ring_size': ";r6" or ";!r3;!r4;!r5;!r6;!r7;!r8"
        - 'aromaticity': ";a" or ";A"
    """
    # Find ring sizes (3-8) containing this atom
    ring_sizes = []
    for ring in mol.GetRingInfo().AtomRings():
        if at_idx in ring:
            ring_size = len(ring)
            if 3 <= ring_size <= 8:
                ring_sizes.append(ring_size)

    atom = mol.GetAtomWithIdx(at_idx)

    descriptors = {
        "atomic_num": f"#{atom.GetAtomicNum()}",
        "degree": f"X{atom.GetDegree()}",
        "charge": str(atom.GetFormalCharge()),
        "ring_size": (
            f";r{min(ring_sizes)}"  # RDKit's r<n> matches the smallest ring only
            if ring_sizes
            else ";!r3;!r4;!r5;!r6;!r7;!r8"
        ),
        "aromaticity": ";a" if atom.GetIsAromatic() else ";A",
    }

    return descriptors


def get_bond_descriptors(atom_idxs: tuple[int, int], mol: Chem.Mol) -> dict[str, str]:
    """
    Extract bond properties for SMIRKS pattern generation.

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of bonded atoms.
    mol : rdkit.Chem.Mol
        RDKit molecule object.

    Returns
    -------
    dict[str, str]
        Bond descriptors:
        - 'bond_smarts': "-", "=", "#", ":", or "~"
        - 'ring_smarts': ";@" (in ring) or ";!@" (not in ring)
    """
    bond = mol.GetBondBetweenAtoms(*atom_idxs)
    if bond is None:
        raise ValueError(f"No bond found between atoms {atom_idxs}")

    TYPE_TO_SMARTS = {
        Chem.BondType.SINGLE: "-",
        Chem.BondType.DOUBLE: "=",
        Chem.BondType.TRIPLE: "#",
        Chem.BondType.AROMATIC: ":",
    }

    bond_smarts = TYPE_TO_SMARTS.get(bond.GetBondType(), "~")
    ring_smarts = ";@" if bond.IsInRing() else ";!@"

    return {"bond_smarts": bond_smarts, "ring_smarts": ring_smarts}


# ============================================================================
# Atom SMIRKS Generation Functions
# ============================================================================


@register_atom_smirks("STANDARD")
def get_atom_smirks_standard(
    at_idx: int, at_id: int, mol: Chem.Mol, terminal_idxs: tuple[int, int]
) -> str:
    """
    Generate standard atom SMIRKS: [#NXM:ID].

    Parameters
    ----------
    at_idx : int
        Atom index in molecule.
    at_id : int
        Position in component (0-based).
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    terminal_idxs : tuple[int, int]
        Indices of terminal atoms (ignored in standard mode).

    Returns
    -------
    str
        SMIRKS atom pattern, e.g., "[#6X4:1]"
    """
    ds = get_atom_descriptors(at_idx, mol)
    return f"[{ds['atomic_num']}{ds['degree']}:{at_id + 1}]"


@register_atom_smirks("WITH_RING_INFO")
def get_atom_smirks_with_ring_info(
    at_idx: int, at_id: int, mol: Chem.Mol, terminal_idxs: tuple[int, int]
) -> str:
    """
    Generate atom SMIRKS with ring information: [#NXM;rK:ID].

    Parameters
    ----------
    at_idx : int
        Atom index in molecule.
    at_id : int
        Position in component (0-based).
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    terminal_idxs : tuple[int, int]
        Indices of terminal atoms (ignored in this mode).

    Returns
    -------
    str
        SMIRKS atom pattern, e.g., "[#6X3;r6:1]"
    """
    ds = get_atom_descriptors(at_idx, mol)
    return f"[{ds['atomic_num']}{ds['degree']}{ds['ring_size']}:{at_id + 1}]"


@register_atom_smirks("TERMINAL_WILDCARD")
def get_atom_smirks_terminal_wildcard(
    at_idx: int, at_id: int, mol: Chem.Mol, terminal_idxs: tuple[int, int]
) -> str:
    """
    Generate atom SMIRKS with terminal wildcards.

    Terminal atoms: [*:ID]
    Non-terminal: [#NXM:ID]

    Parameters
    ----------
    at_idx : int
        Atom index in molecule.
    at_id : int
        Position in component (0-based).
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    terminal_idxs : tuple[int, int]
        Indices indicating which positions are terminal.

    Returns
    -------
    str
        SMIRKS atom pattern
    """
    ds = get_atom_descriptors(at_idx, mol)

    if at_id in terminal_idxs:
        return f"[*:{at_id + 1}]"
    else:
        return f"[{ds['atomic_num']}{ds['degree']}:{at_id + 1}]"


@register_atom_smirks("TERMINAL_H_NO_H")
def get_atom_smirks_terminal_h_no_h(
    at_idx: int, at_id: int, mol: Chem.Mol, terminal_idxs: tuple[int, int]
) -> str:
    """
    Generate atom SMIRKS distinguishing H vs non-H for terminals.

    Terminal atoms: [#1:ID] or [!#1:ID]
    Non-terminal: [#NXM:ID]

    Parameters
    ----------
    at_idx : int
        Atom index in molecule.
    at_id : int
        Position in component (0-based).
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    terminal_idxs : tuple[int, int]
        Indices indicating which positions are terminal.

    Returns
    -------
    str
        SMIRKS atom pattern
    """
    ds = get_atom_descriptors(at_idx, mol)

    if at_id in terminal_idxs:
        atomic_num = ds["atomic_num"] if ds["atomic_num"] == "#1" else "!#1"
        return f"[{atomic_num}:{at_id + 1}]"
    else:
        return f"[{ds['atomic_num']}{ds['degree']}:{at_id + 1}]"


@register_atom_smirks("TERMINAL_H_NO_H_RING_INFO")
def get_atom_smirks_terminal_h_no_h_ring_info(
    at_idx: int, at_id: int, mol: Chem.Mol, terminal_idxs: tuple[int, int]
) -> str:
    """
    Generate atom SMIRKS with H/non-H terminals and ring info for non-terminals.

    Terminal atoms: [#1:ID] or [!#1:ID]
    Non-terminal: [#NXM;rK:ID]

    Parameters
    ----------
    at_idx : int
        Atom index in molecule.
    at_id : int
        Position in component (0-based).
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    terminal_idxs : tuple[int, int]
        Indices indicating which positions are terminal.

    Returns
    -------
    str
        SMIRKS atom pattern
    """
    ds = get_atom_descriptors(at_idx, mol)

    if at_id in terminal_idxs:
        atomic_num = ds["atomic_num"] if ds["atomic_num"] == "#1" else "!#1"
        return f"[{atomic_num}:{at_id + 1}]"
    else:
        return f"[{ds['atomic_num']}{ds['degree']}{ds['ring_size']}:{at_id + 1}]"


# ============================================================================
# Bond SMIRKS Generation Functions
# ============================================================================


@register_bond_smirks("STANDARD")
def get_bond_smirks_standard(atom_idxs: tuple[int, int], central_bond: bool, mol: Chem.Mol) -> str:
    """
    Generate standard bond SMIRKS with explicit bond types.

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of bonded atoms.
    central_bond : bool
        Whether this is the central bond (ignored in standard mode).
    mol : rdkit.Chem.Mol
        RDKit molecule object.

    Returns
    -------
    str
        Bond SMIRKS pattern: "-", "=", "#", or ":"
    """
    return get_bond_descriptors(atom_idxs, mol)["bond_smarts"]


@register_bond_smirks("WITH_RING_INFO")
def get_bond_smirks_with_ring_info(
    atom_idxs: tuple[int, int], central_bond: bool, mol: Chem.Mol
) -> str:
    """
    Generate bond SMIRKS with ring membership information.

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of bonded atoms.
    central_bond : bool
        Whether this is the central bond (ignored in this mode).
    mol : rdkit.Chem.Mol
        RDKit molecule object.

    Returns
    -------
    str
        Bond SMIRKS pattern: "-;@", "-;!@", etc.
    """
    bond_ds = get_bond_descriptors(atom_idxs, mol)
    return bond_ds["bond_smarts"] + bond_ds["ring_smarts"]


@register_bond_smirks("NON_CENTRAL_WILDCARD")
def get_bond_smirks_non_central_wildcard(
    atom_idxs: tuple[int, int], central_bond: bool, mol: Chem.Mol
) -> str:
    """
    Generate bond SMIRKS with wildcards for non-central bonds.

    Central bonds: explicit type ("-", "=", etc.)
    Non-central: wildcard ("~")

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of bonded atoms.
    central_bond : bool
        Whether this is the central bond in the component.
    mol : rdkit.Chem.Mol
        RDKit molecule object.

    Returns
    -------
    str
        Bond SMIRKS pattern
    """
    if central_bond:
        return get_bond_descriptors(atom_idxs, mol)["bond_smarts"]
    else:
        return "~"


@register_bond_smirks("WILDCARD")
def get_bond_smirks_wildcard(atom_idxs: tuple[int, int], central_bond: bool, mol: Chem.Mol) -> str:
    """
    Generate wildcard bond SMIRKS for all bonds.

    Parameters
    ----------
    atom_idxs : tuple[int, int]
        Indices of bonded atoms (ignored).
    central_bond : bool
        Whether this is the central bond (ignored).
    mol : rdkit.Chem.Mol
        RDKit molecule object (ignored).

    Returns
    -------
    str
        Bond SMIRKS pattern: "~"
    """
    return "~"


# ============================================================================
# Force Field Assembly
# ============================================================================


def add_types_to_ff(
    ff: ForceField,
    component_types: dict[int, dict[str, list[MMComponent]]],
    component_class: type[MMComponent],
    extra_parameters: list[ParameterType] | None = None,
    n_workers: int | None = None,  # Ignored, kept for compatibility
) -> ForceField:
    """
    Add component parameters to a force field.

    This implementation is simpler and more memory-efficient than the multiprocessing
    version. It processes parameters sequentially with a progress bar.

    Parameters
    ----------
    ff : openff.toolkit.ForceField
        Base force field to extend.
    component_types : dict[int, dict[str, list[MMComponent]]]
        Component organization: {specificity_level: {smirks: [components]}}.
    component_class : type[MMComponent]
        Component type (Bond, Angle, ProperTorsion, ImproperTorsion).
    extra_parameters : list[ParameterType], optional
        Additional parameters to append.
    n_workers : int, optional
        Ignored for compatibility with old API.

    Returns
    -------
    openff.toolkit.ForceField
        New force field with added parameters.
    """
    ff_copy = deepcopy(ff)
    handler = component_class.handler_class(version=component_class.handler_version)

    # Write the lowest specificity level first
    for specificity_num, components_by_type in sorted(
        component_types.items(), key=lambda item: item[0]
    ):
        logger.info(f"Adding parameters for specificity {specificity_num}...")

        # Sort by population (most common first) for better organization
        sorted_items = sorted(
            components_by_type.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )

        for i, (smirks, components) in tqdm(
            enumerate(sorted_items),
            total=len(components_by_type),
            desc=f"Adding params (spec {specificity_num})",
            unit="param",
        ):
            parameter = component_class.get_parameter(smirks, specificity_num, components, i, ff)
            handler.parameters.append(parameter)

    # Add any extra parameters at the end
    if extra_parameters:
        logger.info(f"Adding {len(extra_parameters)} extra parameters...")
        for parameter in extra_parameters:
            handler.parameters.append(parameter)

    tag_name = component_class.handler_class._TAGNAME
    if tag_name is not None:
        ff_copy.deregister_parameter_handler(tag_name)
    ff_copy.register_parameter_handler(handler)

    return ff_copy
