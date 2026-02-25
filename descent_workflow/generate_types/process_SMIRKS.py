"""
Simple SMIRKS pattern generation for molecular mechanics force fields.

This module provides functions to generate SMIRKS patterns at different specificity
levels for force field parameter assignment. It follows a functional approach with
simple, composable functions rather than complex factory patterns.

Functions are registered using decorators to create two registries:
- atom_fn_map: Registry of atom SMIRKS generation functions
- bond_fn_map: Registry of bond SMIRKS generation functions

These registries are used by the configuration system to look up functions by name.

Shell-Based Detail Control for Outer-Sphere Atoms
--------------------------------------------------
When outer-sphere atoms are included in SMIRKS patterns, their level of detail
is controlled automatically based on their distance from core atoms using shell
indicators (enum classes). Both atoms and bonds use the same hierarchical numbering:

AtomShell (hierarchical levels):
- CORE_CENTRAL (-1): Central atom in 4-atom terms
- CORE_TERMINAL (0): Terminal atoms in components
- OUTER_1 (1): First-shell outer atoms (distance 1) - Full detail (element, degree, ring info)
- OUTER_2 (2): Second-shell outer atoms (distance 2) - Reduced detail (element only)

BondShell (hierarchical levels):
- CORE_CENTRAL (-1): Central bond in 4-atom terms
- CORE (0): Bonds between core atoms
- OUTER_1 (1): Bonds to first-shell atoms - Full detail (bond type + ring info)
- OUTER_2 (2): Bonds to second-shell atoms - Minimal detail (bond type only)

Example SMIRKS with varying outer-sphere detail:
    [#6X4:1(-;!@[#1X1;!r3;!r4;!r5;!r6;!r7;!r8])(-[#1])]-[#6X4:2]
           └─────────────────────────────────────┘ └──────┘
           Distance-1 (OUTER_1): Full detail      Distance-2 (OUTER_2): Element only

This hierarchical detail control allows specificity levels to naturally
fall back to less detailed patterns when rare SMIRKS patterns are demoted
during the hierarchical organization phase.

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

Outer-sphere atom functions (outer_atom_fn_map)
    Applied to unindexed neighbour atoms. Signature: (at_idx, mol, shell).
    - STANDARD: [#NXM] — element + degree
    - WITH_RING_INFO: [#NXM;rK] — element + degree + ring (default)
    - WILDCARD: [*]
    - H_NO_H: [#1] or [!#1]

Outer-sphere bond functions (outer_bond_fn_map)
    Applied to bonds connecting core atoms to neighbours. Signature: (outer_idx, core_idx, mol, shell).
    - STANDARD: explicit bond type (-, =, #, :)
    - WITH_RING_INFO: bond type + ring info (default)
    - WILDCARD: ~

add_types_to_ff
    Integrate component parameters into OpenFF force fields.

Outer-sphere expansion
    Controlled entirely by ``outer_sphere_distance`` on :class:`~.molecular_classes.SpecificityLevel`.
    Any atom/bond SMIRKS function can be combined with a non-None ``outer_sphere_distance``
    to include first- or second-shell neighbour atoms in the pattern.
"""

from copy import deepcopy
from enum import IntEnum
from typing import Callable
from rdkit import Chem
from tqdm import tqdm
from loguru import logger

from openff.toolkit import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterType

from .molecular_classes import MMComponent


# Shell indicators for controlling specificity of outer-sphere atoms
# Atoms and bonds are controlled separately for flexibility


class AtomShell(IntEnum):
    """Shell level indicators for atoms in components.

    Hierarchical position of atoms from innermost to outermost:
    - CORE_CENTRAL (-1): Central atom in 4-atom terms (torsion/improper index 1)
    - CORE_TERMINAL (0): Terminal atoms (end atoms in bonds/angles, terminal in torsions)
    - OUTER_1 (1): First-shell outer atoms (distance 1 from core)
    - OUTER_2 (2): Second-shell outer atoms (distance 2 from core)
    """

    CORE_CENTRAL = -1
    CORE_TERMINAL = 0
    OUTER_1 = 1
    OUTER_2 = 2


class BondShell(IntEnum):
    """Shell level indicators for bonds in components.

    Hierarchical position of bonds from innermost to outermost:
    - CORE_CENTRAL (-1): Central bond in 4-atom terms (torsion/improper)
    - CORE (0): Bonds between core atoms
    - OUTER_1 (1): Bonds connecting core to first-shell outer atoms
    - OUTER_2 (2): Bonds connecting core to second-shell outer atoms
    """

    CORE_CENTRAL = -1
    CORE = 0
    OUTER_1 = 1
    OUTER_2 = 2


# Registry for core SMIRKS generation functions
atom_fn_map: dict[str, Callable] = {}
bond_fn_map: dict[str, Callable] = {}

# Registry for outer-sphere SMIRKS generation functions
# Outer atom fn signature: (at_idx: int, mol: Chem.Mol, shell: AtomShell) -> str
# Outer bond fn signature: (outer_idx: int, core_idx: int, mol: Chem.Mol, shell: BondShell) -> str
outer_atom_fn_map: dict[str, Callable] = {}
outer_bond_fn_map: dict[str, Callable] = {}


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


def get_outer_atom_smirks(
    at_idx: int,
    mol: Chem.Mol,
    include_ring_info: bool = True,
    shell: AtomShell = AtomShell.OUTER_1,
) -> str:
    """
    Generate SMIRKS pattern for an outer-sphere atom (unindexed).

    Outer-sphere atoms are atoms bonded to core atoms but are not themselves
    labeled in the SMIRKS pattern. They are used to provide chemical context.
    Detail level is controlled by the shell parameter.

    Parameters
    ----------
    at_idx : int
        Atom index in molecule.
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    include_ring_info : bool, optional
        If True, include ring information. Default True. Ignored for AtomShell.OUTER_2
        atoms when detail is low.
    shell : AtomShell, optional
        Shell indicator controlling detail level:
        - AtomShell.OUTER_1: Full detail (element, degree, ring info)
        - AtomShell.OUTER_2: Reduced detail (just element and degree, no ring)
        Default AtomShell.OUTER_1.

    Returns
    -------
    str
        SMIRKS pattern for outer atom (unindexed), e.g., "[#6X4]" or "[#6X4;r6]"
    """
    ds = get_atom_descriptors(at_idx, mol)

    # Control detail based on shell
    if shell == AtomShell.OUTER_2:
        # Second shell: minimal detail
        pattern = f"[{ds['atomic_num']}]"
    elif shell == AtomShell.OUTER_1 and include_ring_info:
        # First shell with ring info
        pattern = f"[{ds['atomic_num']}{ds['degree']}{ds['ring_size']}]"
    else:
        # First shell without ring info, or default
        pattern = f"[{ds['atomic_num']}{ds['degree']}]"

    return pattern


def get_outer_bond_smirks_to_core(
    outer_idx: int,
    core_idx: int,
    mol: Chem.Mol,
    include_ring_info: bool = True,
    shell: BondShell = BondShell.OUTER_1,
) -> str:
    """
    Generate bond SMIRKS for bond between outer-sphere and core atom (unindexed).

    Detail level is controlled by the shell parameter to allow different levels
    of specificity for bonds at different distances from the core.

    Parameters
    ----------
    outer_idx : int
        Index of outer-sphere atom.
    core_idx : int
        Index of core atom.
    mol : rdkit.Chem.Mol
        RDKit molecule object.
    include_ring_info : bool, optional
        If True, include ring information when appropriate. Default True.
        Ignored for BondShell.OUTER_2 atoms.
    shell : BondShell, optional
        Shell indicator controlling detail level:
        - BondShell.OUTER_1: Full detail (bond type + ring info if include_ring_info)
        - BondShell.OUTER_2: Minimal detail (just bond type, no ring info)
        Default BondShell.OUTER_1.

    Returns
    -------
    str
        Bond SMIRKS pattern (unindexed), e.g., "-" or "=" or "-;@"
    """
    bond = mol.GetBondBetweenAtoms(outer_idx, core_idx)
    if bond is None:
        raise ValueError(f"No bond found between atoms {outer_idx} and {core_idx}")

    TYPE_TO_SMARTS = {
        Chem.BondType.SINGLE: "-",
        Chem.BondType.DOUBLE: "=",
        Chem.BondType.TRIPLE: "#",
        Chem.BondType.AROMATIC: ":",
    }

    bond_smarts = TYPE_TO_SMARTS.get(bond.GetBondType(), "~")

    # Control detail based on shell
    if shell == BondShell.OUTER_2 or not include_ring_info:
        # Second shell or minimal detail: just bond type
        return bond_smarts
    else:
        # First shell with ring info
        ring_smarts = ";@" if bond.IsInRing() else ";!@"
        return bond_smarts + ring_smarts


def _register_outer_atom_smirks(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        outer_atom_fn_map[name] = func
        return func

    return decorator


def _register_outer_bond_smirks(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        outer_bond_fn_map[name] = func
        return func

    return decorator


# ── Outer-sphere atom functions ───────────────────────────────────────────────


@_register_outer_atom_smirks("STANDARD")
def get_outer_atom_smirks_standard(at_idx: int, mol: Chem.Mol, shell: AtomShell) -> str:
    """Outer atom: element + degree, no ring info. E.g. ``[#6X4]``."""
    ds = get_atom_descriptors(at_idx, mol)
    if shell == AtomShell.OUTER_2:
        return f"[{ds['atomic_num']}]"
    return f"[{ds['atomic_num']}{ds['degree']}]"


@_register_outer_atom_smirks("WITH_RING_INFO")
def get_outer_atom_smirks_with_ring_info(at_idx: int, mol: Chem.Mol, shell: AtomShell) -> str:
    """Outer atom: element + degree + ring info (default). E.g. ``[#6X4;!r3]``."""
    return get_outer_atom_smirks(at_idx, mol, include_ring_info=True, shell=shell)


@_register_outer_atom_smirks("WILDCARD")
def get_outer_atom_smirks_wildcard(at_idx: int, mol: Chem.Mol, shell: AtomShell) -> str:
    """Outer atom: fully generic ``[*]``."""
    return "[*]"


@_register_outer_atom_smirks("H_NO_H")
def get_outer_atom_smirks_h_no_h(at_idx: int, mol: Chem.Mol, shell: AtomShell) -> str:
    """Outer atom: ``[#1]`` for hydrogen, ``[!#1]`` for everything else."""
    ds = get_atom_descriptors(at_idx, mol)
    return "[#1]" if ds["atomic_num"] == "#1" else "[!#1]"


# ── Outer-sphere bond functions ───────────────────────────────────────────────


@_register_outer_bond_smirks("STANDARD")
def get_outer_bond_smirks_standard(
    outer_idx: int, core_idx: int, mol: Chem.Mol, shell: BondShell
) -> str:
    """Outer bond: explicit bond type, no ring info. E.g. ``-``, ``=``."""
    return get_outer_bond_smirks_to_core(
        outer_idx, core_idx, mol, include_ring_info=False, shell=shell
    )


@_register_outer_bond_smirks("WITH_RING_INFO")
def get_outer_bond_smirks_with_ring_info(
    outer_idx: int, core_idx: int, mol: Chem.Mol, shell: BondShell
) -> str:
    """Outer bond: explicit bond type + ring info (default). E.g. ``-;!@``."""
    return get_outer_bond_smirks_to_core(
        outer_idx, core_idx, mol, include_ring_info=True, shell=shell
    )


@_register_outer_bond_smirks("WILDCARD")
def get_outer_bond_smirks_wildcard(
    outer_idx: int, core_idx: int, mol: Chem.Mol, shell: BondShell
) -> str:
    """Outer bond: fully generic ``~``."""
    return "~"


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


def _generate_parameter_worker(
    args: tuple[str, int, list[MMComponent], int, ForceField, type[MMComponent]],
) -> tuple[int, ParameterType]:
    """Generate a single parameter for multiprocessing or threading."""
    smirks, specificity_num, components, index, ff, component_class = args
    parameter = component_class.get_parameter(smirks, specificity_num, components, index, ff)
    return index, parameter


def add_types_to_ff(
    ff: ForceField,
    component_types: dict[int, dict[str, list[MMComponent]]],
    component_class: type[MMComponent],
    extra_parameters: list[ParameterType] | None = None,
    n_workers: int | None = None,
) -> ForceField:
    """
    Add component parameters to a force field.

    Parameters are generated in parallel using multiprocessing by default, with a
    threaded fallback when multiprocessing is disabled or fails. This keeps memory
    usage reasonable while accelerating parameter creation.

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
        Number of worker processes/threads. Defaults to a conservative value.

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
