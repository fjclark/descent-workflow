"""
Molecular Mechanics Component Classes for Force Field Parameter Generation.

This module provides classes and utilities for representing molecular mechanics (MM)
components such as bonds, angles, and torsions, with functionality to generate
SMIRKS patterns and force field parameters.

Key Components
--------------
MMComponent : Abstract base class
    Base class for all molecular mechanics components with shared functionality
    for SMIRKS generation and parameter assignment.

Bond, Angle, ProperTorsion, ImproperTorsion : Concrete MM component classes
    Specific implementations for different component types with their respective
    parameter handlers and physics.

Examples
--------
Creating and analyzing bond components:

>>> from openff.toolkit import Molecule
>>> mol = Molecule.from_smiles("CCO")
>>> bond_indices = (0, 1)  # C-C bond
>>> bond = Bond(indices=bond_indices, mol=mol, rdkit_mol=mol.to_rdkit())
>>> print(f"Bond: {bond.indices}")
Bond: (0, 1)

Generating SMIRKS patterns for different specificity levels:

>>> from process_SMIRKS import SMIRKSFactory
>>> factory = SMIRKSFactory()
>>> specificity = factory.create_specificity_level("RingAware")
>>> smirks = bond.get_smirks(specificity)
>>> print(f"SMIRKS: {smirks}")
SMIRKS: [#6X4:1]-[#6X4:2]

Creating force field parameters from component groups:

>>> bonds = [Bond(...), Bond(...)]  # Multiple bond components
>>> base_ff = ForceField("openff-2.0.0.offxml")
>>> parameter = Bond.get_parameter(
...     smirks="[#6:1]-[#6:2]",
...     specificity_num=0,
...     components=bonds,
...     index=0,
...     base_ff=base_ff
... )
>>> print(f"Parameter length: {parameter.length}")
Parameter length: 1.52 angstrom
"""

from typing import Optional, Callable, cast
from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
from rdkit import Chem

from openff.units import unit as off_unit
from openff.toolkit import Molecule, ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import (
    ParameterHandler,
    ParameterType,
    ParameterList,
    BondHandler,
    AngleHandler,
    ProperTorsionHandler,
    ImproperTorsionHandler,
)

from .outer_sphere import OuterSphereAtoms

SAGE_230 = ForceField("openff_unconstrained-2.3.0.offxml")

# ---------------------------------------------------------------------------
# Module-level unit constants — constructed ONCE, reused everywhere.
# Unit arithmetic with openff-units Quantity objects is expensive; creating
# them on every get_parameter() call (which may run thousands of times) is a
# significant source of overhead.
# ---------------------------------------------------------------------------
_KCAL_PER_MOL_PER_ANG2 = off_unit.kilocalorie_per_mole / off_unit.angstroms**2
_KCAL_PER_MOL_PER_RAD2 = off_unit.kilocalorie_per_mole / off_unit.radian**2
_DEGREES = off_unit.degrees
_ANGSTROMS = off_unit.angstroms
_RADIANS = off_unit.radians

# Zero-valued Quantity sentinels used as list elements for torsion defaults.
_ZERO_K_TORSION = 0 * _KCAL_PER_MOL_PER_RAD2
_ZERO_DEGREES = 0 * _DEGREES
_180_DEGREES = 180 * _DEGREES

# Immutable default lists shared across all ProperTorsion / ImproperTorsion
# parameter creation calls.  We copy them on use so the handler cannot mutate
# the originals.
_PROPER_DEFAULT_K = [_ZERO_K_TORSION] * 4
_PROPER_DEFAULT_PHASE = [_ZERO_DEGREES] * 4
_PROPER_DEFAULT_PERIODICITY = [1, 2, 3, 4]
_PROPER_DEFAULT_IDIVF = [1.0] * 4

_IMPROPER_DEFAULT_K = [_ZERO_K_TORSION]
_IMPROPER_DEFAULT_PHASE = [_180_DEGREES]
_IMPROPER_DEFAULT_PERIODICITY = [2]
_IMPROPER_DEFAULT_IDIVF = [1.0]

SpecificityLevel = namedtuple(
    "SpecificityLevel",
    [
        "name",
        "get_atom_smirks",  # (at_idx, at_id, mol, terminal_idxs) -> str  — indexed core atoms
        "get_bond_smirks",  # (atom_idxs, central_bond, mol) -> str       — core bonds
        "outer_sphere_distance",  # int | None  — None = no outer sphere
        "get_outer_atom_smirks",  # (at_idx, mol, shell) -> str                 — unindexed outer atoms
        "get_outer_bond_smirks",  # (outer_idx, core_idx, mol, shell) -> str    — outer bonds
    ],
    defaults=[
        None,
        None,
        None,
    ],  # outer_sphere_distance, get_outer_atom_smirks, get_outer_bond_smirks
)


def get_bond_idxs(mol: Molecule) -> set[tuple[int, int]]:
    """
    Extract bond atom indices from a molecule.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The input molecule to analyze.

    Returns
    -------
    set[tuple[int, int]]
        Set of tuples containing the atom indices of each bond,
        with indices sorted within each tuple to ensure consistency.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> bonds = get_bond_idxs(mol)
    >>> print(bonds)
    {(0, 1), (1, 2)}
    """
    return cast(
        set[tuple[int, int]],
        {tuple(sorted((a.molecule_atom_index for a in bond.atoms))) for bond in mol.bonds},
    )


def get_angle_idxs(mol: Molecule) -> set[tuple[int, int, int]]:
    """
    Extract angle atom indices from a molecule.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The input molecule to analyze.

    Returns
    -------
    set[tuple[int, int, int]]
        Set of tuples containing the atom indices of each angle in the order
        (terminal, central, terminal).

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> angles = get_angle_idxs(mol)
    >>> print(angles)
    {(0, 1, 2)}
    """
    return cast(
        set[tuple[int, int, int]],
        {tuple(a.molecule_atom_index for a in angle) for angle in mol.angles},
    )


def get_proper_torsion_idxs(mol: Molecule) -> set[tuple[int, int, int, int]]:
    """
    Extract proper torsion atom indices from a molecule.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The input molecule to analyze.

    Returns
    -------
    set[tuple[int, int, int, int]]
        Set of tuples containing the atom indices of each proper torsion
        in the order (atom1, atom2, atom3, atom4) where atom2-atom3
        is the central bond.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCCO")
    >>> torsions = get_proper_torsion_idxs(mol)
    >>> print(torsions)
    {(0, 1, 2, 3)}
    """
    return cast(
        set[tuple[int, int, int, int]],
        {tuple(a.molecule_atom_index for a in torsion) for torsion in mol.propers},
    )


def get_improper_torsion_idxs(mol: Molecule) -> set[tuple[int, int, int, int]]:
    """
    Extract improper torsion atom indices from a molecule.

    Parameters
    ----------
    mol : openff.toolkit.Molecule
        The input molecule to analyze.

    Returns
    -------
    set[tuple[int, int, int, int]]
        Set of tuples containing the atom indices of each improper torsion.
        The third atom (index 2) is always the central atom.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("C(C)(C)C")  # tetrahedral carbon
    >>> impropers = get_improper_torsion_idxs(mol)
    """
    # For the moment, find the impropers assigned by sage2.3.0 and just use those.
    return set(SAGE_230.label_molecules(mol.to_topology())[0]["ImproperTorsions"].keys())


class MMComponent(ABC):
    """
    Abstract base class for molecular mechanics components.

    Represents force field components such as bonds, angles, and torsions with
    functionality to generate SMIRKS patterns and create force field parameters.

    Parameters
    ----------
    indices : tuple[int, ...]
        Atom indices defining the component (e.g., (0, 1) for a bond).
    mol : openff.toolkit.Molecule
        The OpenFF Molecule object containing this component.
    rdkit_mol : rdkit.Chem.Mol
        The RDKit representation of the molecule for chemical analysis.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in the component (must be defined by subclasses).
    handler_class : type[ParameterHandler]
        SMIRNOFF parameter handler class (must be defined by subclasses).
    handler_version : float
        Version of the parameter handler (must be defined by subclasses).
    parameter_type : type[ParameterType]
        SMIRNOFF parameter type class (must be defined by subclasses).
    getter_fn : Callable[[Molecule], set[tuple[int, ...]]]
        Function to extract components from molecules (must be defined by subclasses).

    Notes
    -----
    This is an abstract base class that enforces subclasses to define required
    class attributes. The class uses __slots__ for memory efficiency.

    All MMComponent instances automatically generate mapped SMILES strings
    during initialization and provide methods for SMIRKS pattern generation
    and parameter creation.

    Examples
    --------
    Subclasses must define all required class attributes:

    >>> class Bond(MMComponent):
    ...     n_atoms = 2
    ...     handler_class = BondHandler
    ...     handler_version = 0.3
    ...     parameter_type = BondHandler.BondType
    ...     getter_fn = get_bond_idxs

    Creating a component instance:

    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> bond = Bond(indices=(0, 1), mol=mol, rdkit_mol=mol.to_rdkit())
    >>> print(bond.mapped_smiles)
    [C:1][C:2][O:3]
    """

    # __slots__ = ["mapped_smiles", "indices", "mol", "rdkit_mol", "n_atoms", "handler_class", "parameter_type", "getter_fn"]
    __slots__ = [
        "mapped_smiles",
        "indices",
        "mol",
        "rdkit_mol",
        "n_atoms",
        "outer_sphere_indices",
    ]

    mapped_smiles: str
    indices: tuple[int, ...]
    mol: Molecule
    rdkit_mol: Chem.Mol
    n_atoms: int  # Subclass must define number of atoms
    outer_sphere_indices: Optional[OuterSphereAtoms]  # Optional outer-sphere atom tracking
    handler_class: type[ParameterHandler]  # Subclass must define handler class
    handler_version: float
    parameter_type: type[ParameterType]  # Subclass must define parameter type
    getter_fn: Callable[[Molecule], set[tuple[int, ...]]]  # Subclass must define getter function

    def __init_subclass__(cls, **kwargs):
        """
        Enforce that required class attributes are defined in subclasses.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to parent class.

        Raises
        ------
        TypeError
            If any required class attribute is missing from the subclass.
        """
        super().__init_subclass__(**kwargs)

        required_attrs = [
            "n_atoms",
            "handler_class",
            "handler_version",
            "parameter_type",
            "getter_fn",
        ]

        for attr in required_attrs:
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")

    def __init__(
        self,
        indices: tuple[int, ...],
        mol: Molecule,
        rdkit_mol: Chem.Mol,
        outer_sphere_indices: Optional[OuterSphereAtoms] = None,
    ):
        """
        Initialize the molecular mechanics component.

        Parameters
        ----------
        indices : tuple[int, ...]
            Atom indices defining the component (core atoms).
        mol : openff.toolkit.Molecule
            OpenFF Molecule object.
        rdkit_mol : rdkit.Chem.Mol
            RDKit molecule representation.
        outer_sphere_indices : Optional[OuterSphereAtoms]
            Optional outer-sphere atom information. If provided, atoms bonded
            to core atoms will be tracked for SMIRKS generation.
        """
        self.indices = indices
        self.mol = mol
        self.rdkit_mol = rdkit_mol
        self.outer_sphere_indices = outer_sphere_indices

        # Apply MDL aromaticity model
        Chem.SetAromaticity(self.rdkit_mol, Chem.AromaticityModel.AROMATICITY_MDL)
        self.mapped_smiles = mol.to_smiles(mapped=True)

    @property
    def central_bond_index(self) -> Optional[int]:
        """
        Get the index of the central bond in the component.

        Returns
        -------
        Optional[int]
            Index of the central bond for 4-atom components (torsions),
            None for other component types.
        """
        if self.n_atoms == 4:
            return 1
        return None

    @property
    def terminal_atom_indices(self) -> tuple[int, int]:
        """
        Get the indices of terminal atoms in the component.

        Returns
        -------
        tuple[int, int]
            Indices of the first and last atoms in the core component.
        """
        return (0, self.n_atoms - 1)

    @property
    def all_indices(self) -> tuple[int, ...]:
        """
        Get all atom indices including core and outer-sphere atoms.

        Returns
        -------
        tuple[int, ...]
            Sorted tuple of all atom indices (core + outer). If no outer atoms,
            returns just the core indices.
        """
        if self.outer_sphere_indices is None or not self.outer_sphere_indices.has_outer_atoms:
            return self.indices
        return self.outer_sphere_indices.all_indices

    def outer_atoms_at_distance(self, distance: int) -> tuple[int, ...]:
        """
        Get outer-sphere atoms at a specific distance from core atoms.

        Parameters
        ----------
        distance : int
            Distance in bonds from core atoms.

        Returns
        -------
        tuple[int, ...]
            Atom indices at the specified distance, or empty tuple if none
            or if outer-sphere information not available.
        """
        if self.outer_sphere_indices is None:
            return ()
        return self.outer_sphere_indices.get_atoms_at_distance(distance)

    def _construct_smirks(self, atoms: list[str], bonds: list[str]) -> str:
        """
        Construct SMIRKS representation from atom and bond patterns.

        Parameters
        ----------
        atoms : list[str]
            List of atom SMIRKS patterns.
        bonds : list[str]
            List of bond SMIRKS patterns.

        Returns
        -------
        str
            Complete SMIRKS pattern for the component.

        Raises
        ------
        AssertionError
            If the number of atoms or bonds is inconsistent with component type.
        """
        # Make sure that the lengths of atoms and bonds are consistent
        assert len(atoms) == self.n_atoms, f"Expected {self.n_atoms} atoms, got {len(atoms)}"
        assert len(bonds) == self.n_atoms - 1, (
            f"Expected {self.n_atoms - 1} bonds, got {len(bonds)}"
        )

        smirks = atoms[0]
        for bond, atom in zip(bonds, atoms[1:], strict=True):
            smirks += f"{bond}{atom}"

        return smirks

    def _construct_smirks_with_outer_sphere(
        self,
        atoms: list[str],
        bonds: list[str],
        outer_atoms_by_idx: dict[int, list[str]],
    ) -> str:
        """
        Construct SMIRKS representation including outer-sphere atoms.

        Outer-sphere atoms are attached to core atoms using parentheses syntax
        appended *after* the closing bracket of each indexed atom token.
        For example: [#6X4:1](-[#1])(-[#1])-[#6X4:2] represents a C-C bond
        where the first carbon is bonded to two hydrogens.

        Parameters
        ----------
        atoms : list[str]
            List of core atom SMIRKS patterns (indexed), e.g. ``["[#6X4:1]", "[#6X4:2]"]``.
        bonds : list[str]
            List of bond SMIRKS patterns between core atoms.
        outer_atoms_by_idx : dict[int, list[str]]
            Mapping from core atom position (0-based) to a list of branch strings.
            Each branch is already formatted as ``"(bond_smarts[atom_smarts])"``.
            Example: ``{0: "(-;!@[#1X1;!r3])", 1: "(-[#8X2])"}``.

        Returns
        -------
        str
            Complete SMIRKS pattern including outer-sphere atoms.
        """
        # Validate input
        assert len(atoms) == self.n_atoms, f"Expected {self.n_atoms} atoms, got {len(atoms)}"
        assert len(bonds) == self.n_atoms - 1, (
            f"Expected {self.n_atoms - 1} bonds, got {len(bonds)}"
        )

        # Append outer-sphere branches *after* each indexed atom token (correct SMIRKS syntax).
        # Correct: [#6X4:1](-[#1])  Wrong: [#6X4:1(-[#1])]
        smirks_atoms = [
            atom + "".join(outer_atoms_by_idx.get(i, [])) for i, atom in enumerate(atoms)
        ]

        # Construct full SMIRKS
        smirks = smirks_atoms[0]
        for bond, atom in zip(bonds, smirks_atoms[1:], strict=True):
            smirks += f"{bond}{atom}"

        return smirks

    def _build_outer_patterns(
        self,
        core_idxs_ordered: tuple[int, ...],
        get_outer_atom_smirks: Callable,
        get_outer_bond_smirks: Callable,
        max_distance: int = 1,
    ) -> dict[int, list[str]]:
        """
        Build outer-sphere branch patterns for a given ordering of core atoms.

        Returns a mapping from core-atom *position* (in ``core_idxs_ordered``) to
        a list of branch strings, each formatted as
        ``"(bond_smarts[atom_smarts])"``.

        Parameters
        ----------
        core_idxs_ordered : tuple[int, ...]
            Core atom indices in the desired traversal order (forward or reversed).
        get_outer_atom_smirks : Callable
            Function to generate outer-sphere unindexed atom SMIRKS.
        get_outer_bond_smirks : Callable
            Function to generate outer-sphere unindexed bond SMIRKS.
        max_distance : int, optional
            Maximum shell distance to include (1 or 2). Atoms at distances
            greater than this are ignored even if present in
            ``self.outer_sphere_indices``. Default is 1.

        Returns
        -------
        dict[int, list[str]]
            ``{position: ["(-[#1])", ...], ...}``
        """
        from .process_SMIRKS import AtomShell, BondShell

        outer_atoms_by_idx: dict[int, list[str]] = {}

        for core_pos, core_idx in enumerate(core_idxs_ordered):
            outer_patterns: list[str] = []

            available_distances = [
                d
                for d in sorted(self.outer_sphere_indices.outer_by_distance.keys())  # type: ignore[union-attr]
                if d <= max_distance
            ]
            for distance in available_distances:
                shell_atom = AtomShell.OUTER_1 if distance == 1 else AtomShell.OUTER_2
                shell_bond = BondShell.OUTER_1 if distance == 1 else BondShell.OUTER_2

                for outer_idx in self.outer_sphere_indices.get_atoms_at_distance(distance):  # type: ignore[union-attr]
                    # Only include outer atoms actually bonded to this core atom
                    if self.rdkit_mol.GetBondBetweenAtoms(core_idx, outer_idx) is None:
                        continue

                    outer_atom_pattern = get_outer_atom_smirks(
                        outer_idx, self.rdkit_mol, shell=shell_atom
                    )
                    outer_bond_pattern = get_outer_bond_smirks(
                        outer_idx, core_idx, self.rdkit_mol, shell=shell_bond
                    )
                    outer_patterns.append(f"({outer_bond_pattern}{outer_atom_pattern})")

            if outer_patterns:
                outer_atoms_by_idx[core_pos] = outer_patterns

        return outer_atoms_by_idx

    def get_smirks(self, specificity_level: SpecificityLevel) -> str:
        """
        Generate SMIRKS representation for the component.

        Parameters
        ----------
        specificity_level : SpecificityLevel
            Defines the level of detail for atom and bond descriptions.
            If ``specificity_level.outer_sphere_distance`` is an integer
            (1 or 2), outer-sphere atoms up to that shell are included as
            unindexed branch atoms appended after each indexed atom token.
            If ``None`` (the default), no outer-sphere expansion is performed
            even when ``outer_sphere_indices`` is populated on the component.

        Returns
        -------
        str
            SMIRKS pattern representing this component. Returns the
            lexicographically smallest pattern to ensure consistency
            regardless of atom ordering.
        """
        outer_distance: int | None = specificity_level.outer_sphere_distance
        has_outer_sphere = (
            outer_distance is not None
            and self.outer_sphere_indices is not None
            and self.outer_sphere_indices.has_outer_atoms
        )

        idxs = self.indices
        n = self.n_atoms

        atoms_fwd = [
            specificity_level.get_atom_smirks(
                at_idx, at_id, self.rdkit_mol, self.terminal_atom_indices
            )
            for at_idx, at_id in zip(idxs, range(n), strict=True)
        ]
        atoms_bwd = [
            specificity_level.get_atom_smirks(
                at_idx, at_id, self.rdkit_mol, self.terminal_atom_indices
            )
            for at_idx, at_id in zip(reversed(idxs), range(n), strict=True)
        ]

        bonds_fwd = [
            specificity_level.get_bond_smirks(
                (idxs[j], idxs[j + 1]),
                central_bond=(j == self.central_bond_index),
                mol=self.rdkit_mol,
            )
            for j in range(n - 1)
        ]

        if has_outer_sphere:
            outer_fwd = self._build_outer_patterns(
                tuple(idxs),
                get_outer_atom_smirks=specificity_level.get_outer_atom_smirks,
                get_outer_bond_smirks=specificity_level.get_outer_bond_smirks,
                max_distance=outer_distance,
            )
            outer_bwd = self._build_outer_patterns(
                tuple(reversed(idxs)),
                get_outer_atom_smirks=specificity_level.get_outer_atom_smirks,
                get_outer_bond_smirks=specificity_level.get_outer_bond_smirks,
                max_distance=outer_distance,
            )

            smirks_fwd = self._construct_smirks_with_outer_sphere(atoms_fwd, bonds_fwd, outer_fwd)
            smirks_bwd = self._construct_smirks_with_outer_sphere(
                atoms_bwd, list(reversed(bonds_fwd)), outer_bwd
            )
        else:
            smirks_fwd = self._construct_smirks(atoms_fwd, bonds_fwd)
            smirks_bwd = self._construct_smirks(atoms_bwd, list(reversed(bonds_fwd)))

        # Return the lexicographically smallest SMIRKS representation to ensure
        # that the order of atoms does not affect the representation.
        return min(smirks_fwd, smirks_bwd)

    def matches_smirks(self, smirks: str) -> bool:
        """
        Check if component matches a given SMIRKS pattern.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern to test against.

        Returns
        -------
        bool
            True if this component matches the SMIRKS pattern, False otherwise.
        """
        indices_list = self.mol.chemical_environment_matches(smirks)

        if not indices_list:
            return False

        for indices in indices_list:
            if self.indices == indices:
                return True

        return False

    def __str__(self):
        """Return string representation of the component."""
        return f"{self.mapped_smiles} ({self.indices})"

    @classmethod
    @abstractmethod
    def get_parameter(
        cls,
        smirks: str,
        specificity_num: int,
        components: list["MMComponent"],
        index: int,
        base_ff: ForceField,
    ) -> ParameterType:
        """
        Generate force field parameter for component type.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern defining the parameter scope.
        specificity_num : int
            Specificity level number for parameter identification.
        components : list[MMComponent]
            List of components to derive parameter values from.
        index : int
            Parameter index for identification.
        base_ff : openff.toolkit.ForceField
            Base force field to derive initial parameter values.

        Returns
        -------
        ParameterType
            Force field parameter object for this component type.
        """
        ...


class Bond(MMComponent):
    """
    Molecular mechanics bond component.

    Represents a covalent bond between two atoms with associated force field
    parameters for bond stretching.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in a bond (always 2).
    handler_class : type[BondHandler]
        SMIRNOFF bond parameter handler.
    handler_version : float
        Version 0.4 of the bond handler.
    parameter_type : type[BondHandler.BondType]
        SMIRNOFF bond parameter type.
    getter_fn : Callable[[Molecule], set[tuple[int, ...]]]
        Function to extract bond indices from molecules.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> bond = Bond(indices=(0, 1), mol=mol, rdkit_mol=mol.to_rdkit())
    >>> bond.n_atoms
    2
    """

    n_atoms: int = 2
    handler_class: type[ParameterHandler] = BondHandler
    handler_version: float = 0.4
    parameter_type: type[ParameterType] = BondHandler.BondType
    getter_fn: Callable[[Molecule], set[tuple[int, ...]]] = cast(
        Callable[[Molecule], set[tuple[int, ...]]], get_bond_idxs
    )

    @classmethod
    def get_parameter(
        cls,
        smirks: str,
        specificity_num: int,
        components: list["MMComponent"],
        index: int,
        base_ff: ForceField,
    ) -> BondHandler.BondType:
        """
        Generate bond parameter from component statistics.

        Creates a bond parameter by averaging force constants and equilibrium
        bond lengths from similar bonds in a base force field.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern defining the bond type.
        specificity_num : int
            Specificity level for parameter identification.
        components : list[MMComponent]
            List of bond components to derive parameter values from.
        index : int
            Parameter index for identification.
        base_ff : openff.toolkit.ForceField
            Base force field to extract reference parameters.

        Returns
        -------
        BondHandler.BondType
            Bond parameter with averaged force constant and equilibrium length.
        """
        # assert all(isinstance(c, Bond) for c in components), f"All components must be Bond instances but got {[type(c) for c in components]}"

        base_ff_parameters = get_parameters_for_components(components, base_ff)
        # Reuse module-level unit constants — avoids repeated unit arithmetic.
        mean_k = (
            np.mean([p.k.m_as(_KCAL_PER_MOL_PER_ANG2) for p in base_ff_parameters])
            * _KCAL_PER_MOL_PER_ANG2
        )
        mean_length = np.mean([p.length.m_as(_ANGSTROMS) for p in base_ff_parameters]) * _ANGSTROMS

        parameter = BondHandler.BondType(
            smirks=smirks,
            k=mean_k,
            length=mean_length,
            id=f"specificity={specificity_num} index={index} count={len(components)}",
        )
        return parameter

    def get_bond_length(self):
        """
        Calculate bond lengths from all molecular conformers.

        Computes the Euclidean distance between the two atoms defining this bond
        for each available conformer in the RDKit molecule. This provides geometric
        information that can be used for parameter fitting or validation.

        Returns
        -------
        list[float] or None
            List of bond lengths in Angstroms, one for each conformer.
            Returns None if no conformers are available.

        Examples
        --------
        >>> from openff.toolkit import Molecule
        >>> mol = Molecule.from_smiles("CCO")
        >>> mol.generate_conformers(n_conformers=2)
        >>> bond = Bond(indices=(0, 1), mol=mol, rdkit_mol=mol.to_rdkit())
        >>> lengths = bond.get_bond_length()
        >>> len(lengths)
        2
        >>> all(1.4 < length < 1.6 for length in lengths)  # Typical C-C bond
        True

        Notes
        -----
        The calculation uses 3D coordinates from RDKit conformers. If multiple
        conformers are available, each will yield a separate bond length value
        reflecting the molecular flexibility or different optimization states.
        """
        n_confs = self.rdkit_mol.GetNumConformers()
        if n_confs == 0:
            return None
        lengths = []
        for i in range(n_confs):
            conf = self.rdkit_mol.GetConformer(i)
            pos1 = conf.GetAtomPosition(self.indices[0])
            pos2 = conf.GetAtomPosition(self.indices[1])
            length = np.linalg.norm(
                np.array([pos1.x, pos1.y, pos1.z]) - np.array([pos2.x, pos2.y, pos2.z])
            )
            lengths.append(length)
        return lengths


class Angle(MMComponent):
    """
    Molecular mechanics angle component.

    Represents a bond angle between three atoms with associated force field
    parameters for angle bending.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in an angle (always 3).
    handler_class : type[AngleHandler]
        SMIRNOFF angle parameter handler.
    handler_version : float
        Version 0.3 of the angle handler.
    parameter_type : type[AngleHandler.AngleType]
        SMIRNOFF angle parameter type.
    getter_fn : Callable[[Molecule], set[tuple[int, ...]]]
        Function to extract angle indices from molecules.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> angle = Angle(indices=(0, 1, 2), mol=mol, rdkit_mol=mol.to_rdkit())
    >>> angle.n_atoms
    3
    """

    n_atoms: int = 3
    handler_class: type[ParameterHandler] = AngleHandler
    handler_version: float = 0.3
    parameter_type: type[ParameterType] = AngleHandler.AngleType
    getter_fn: Callable[[Molecule], set[tuple[int, ...]]] = cast(
        Callable[[Molecule], set[tuple[int, ...]]], get_angle_idxs
    )

    @classmethod
    def get_parameter(
        cls,
        smirks: str,
        specificity_num: int,
        components: list["MMComponent"],
        index: int,
        base_ff: ForceField,
    ) -> AngleHandler.AngleType:
        """
        Generate angle parameter from component statistics.

        Creates an angle parameter by averaging force constants and equilibrium
        angles from similar angles in a base force field.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern defining the angle type.
        specificity_num : int
            Specificity level for parameter identification.
        components : list[MMComponent]
            List of angle components to derive parameter values from.
        index : int
            Parameter index for identification.
        base_ff : openff.toolkit.ForceField
            Base force field to extract reference parameters.

        Returns
        -------
        AngleHandler.AngleType
            Angle parameter with averaged force constant and equilibrium angle.
        """
        # assert all(isinstance(c, Angle) for c in components), f"All components must be Angle instances but got {[type(c) for c in components]}"

        base_ff_parameters = get_parameters_for_components(components, base_ff)
        # Reuse module-level unit constants — avoids repeated unit arithmetic.
        mean_k = (
            np.mean([p.k.m_as(_KCAL_PER_MOL_PER_RAD2) for p in base_ff_parameters])
            * _KCAL_PER_MOL_PER_RAD2
        )
        mean_angle = np.mean([p.angle.m_as(_DEGREES) for p in base_ff_parameters]) * _DEGREES

        parameter = AngleHandler.AngleType(
            smirks=smirks,
            k=mean_k,
            angle=mean_angle,
            id=f"specificity={specificity_num} index={index} count={len(components)}",
        )
        return parameter

    def get_angle(self):
        """
        Calculate angle values from all molecular conformers.

        Computes the bond angle between the three atoms defining this angle
        for each available conformer in the RDKit molecule. The angle is measured
        between vectors from the central atom (second atom) to the two terminal atoms.

        Returns
        -------
        list[float] or None
            List of angle values in degrees, one for each conformer.
            Each angle is between 0 and 180 degrees. Returns None if no conformers
            are available.

        Examples
        --------
        >>> from openff.toolkit import Molecule
        >>> mol = Molecule.from_smiles("CCO")
        >>> mol.generate_conformers(n_conformers=2)
        >>> angle = Angle(indices=(0, 1, 2), mol=mol, rdkit_mol=mol.to_rdkit())
        >>> angles = angle.get_angle()
        >>> len(angles)
        2
        >>> all(100 < angle < 120 for angle in angles)  # Typical C-C-O angle
        True

        Notes
        -----
        The calculation uses 3D coordinates from RDKit conformers and applies
        the standard vector dot product formula for angle calculation:

        .. math::
            \\theta = \\arccos\\left(\\frac{\\mathbf{v_1} \\cdot \\mathbf{v_2}}{|\\mathbf{v_1}| |\\mathbf{v_2}|}\\right)

        where v1 and v2 are vectors from the central atom to the terminal atoms.
        The result is clipped to [-1, 1] to avoid numerical errors in arccos.
        """
        n_confs = self.rdkit_mol.GetNumConformers()
        if n_confs == 0:
            return None
        angles = []
        for i in range(n_confs):
            conf = self.rdkit_mol.GetConformer(i)
            pos1 = conf.GetAtomPosition(self.indices[0])
            pos2 = conf.GetAtomPosition(self.indices[1])
            pos3 = conf.GetAtomPosition(self.indices[2])
            v1 = np.array([pos1.x, pos1.y, pos1.z]) - np.array([pos2.x, pos2.y, pos2.z])
            v2 = np.array([pos3.x, pos3.y, pos3.z]) - np.array([pos2.x, pos2.y, pos2.z])
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angles.append(np.degrees(angle_rad))
        return angles


class ProperTorsion(MMComponent):
    """
    Molecular mechanics proper torsion component.

    Represents a proper dihedral angle between four atoms with associated
    force field parameters for torsional rotation about the central bond.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in a proper torsion (always 4).
    handler_class : type[ProperTorsionHandler]
        SMIRNOFF proper torsion parameter handler.
    handler_version : float
        Version 0.4 of the proper torsion handler.
    parameter_type : type[ProperTorsionHandler.ProperTorsionType]
        SMIRNOFF proper torsion parameter type.
    getter_fn : Callable[[Molecule], set[tuple[int, ...]]]
        Function to extract proper torsion indices from molecules.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCCO")
    >>> torsion = ProperTorsion(indices=(0, 1, 2, 3), mol=mol, rdkit_mol=mol.to_rdkit())
    >>> torsion.n_atoms
    4
    >>> torsion.central_bond_index
    1
    """

    n_atoms: int = 4
    handler_class: type[ParameterHandler] = ProperTorsionHandler
    handler_version: float = 0.4
    parameter_type: type[ParameterType] = ProperTorsionHandler.ProperTorsionType
    getter_fn: Callable[[Molecule], set[tuple[int, ...]]] = cast(
        Callable[[Molecule], set[tuple[int, ...]]], get_proper_torsion_idxs
    )

    @classmethod
    def get_parameter(
        cls,
        smirks: str,
        specificity_num: int,
        components: list["MMComponent"],
        index: int,
        base_ff: ForceField,
    ) -> ProperTorsionHandler.ProperTorsionType:
        """
        Generate proper torsion parameter with default values.

        Creates a proper torsion parameter with default force constants and
        phase angles. Currently uses zero force constants as placeholders.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern defining the torsion type.
        specificity_num : int
            Specificity level for parameter identification.
        components : list[MMComponent]
            List of torsion components (used for counting).
        index : int
            Parameter index for identification.
        base_ff : openff.toolkit.ForceField
            Base force field (not currently used for torsions).

        Returns
        -------
        ProperTorsionHandler.ProperTorsionType
            Proper torsion parameter with default values for 4 periodicities.
        """
        # assert all(isinstance(c, ProperTorsion) for c in components), "All components must be ProperTorsion instances"
        #
        # Use module-level cached default lists. We pass copies so the handler
        # cannot mutate the shared originals.
        parameter = ProperTorsionHandler.ProperTorsionType(
            smirks=smirks,
            k=list(_PROPER_DEFAULT_K),
            phase=list(_PROPER_DEFAULT_PHASE),
            periodicity=list(_PROPER_DEFAULT_PERIODICITY),
            idivf=list(_PROPER_DEFAULT_IDIVF),
            id=f"specificity={specificity_num} index={index} count={len(components)}",
        )
        return parameter


class ImproperTorsion(MMComponent):
    """
    Molecular mechanics improper torsion component.

    Represents an improper dihedral angle used to maintain planarity or
    tetrahedral geometry. The third atom is always the central atom.

    Attributes
    ----------
    n_atoms : int
        Number of atoms in an improper torsion (always 4).
    handler_class : type[ImproperTorsionHandler]
        SMIRNOFF improper torsion parameter handler.
    handler_version : float
        Version 0.3 of the improper torsion handler.
    parameter_type : type[ImproperTorsionHandler.ImproperTorsionType]
        SMIRNOFF improper torsion parameter type.
    getter_fn : Callable[[Molecule], set[tuple[int, ...]]]
        Function to extract improper torsion indices from molecules.

    Notes
    -----
    Improper torsions have special symmetry considerations where the third
    atom is always treated as the central atom, affecting SMIRKS generation.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("C(C)(C)C")  # tetrahedral carbon
    >>> improper = ImproperTorsion(indices=(0, 1, 2, 3), mol=mol, rdkit_mol=mol.to_rdkit())
    >>> improper.n_atoms
    4
    """

    n_atoms: int = 4
    handler_class: type[ParameterHandler] = ImproperTorsionHandler
    handler_version: float = 0.3
    parameter_type: type[ParameterType] = ImproperTorsionHandler.ImproperTorsionType
    getter_fn: Callable[[Molecule], set[tuple[int, ...]]] = cast(
        Callable[[Molecule], set[tuple[int, ...]]], get_improper_torsion_idxs
    )

    # Need to override some methods as impropers are not symmetrical in that
    # the third atom is always the central atom.
    def _construct_smirks(self, atoms: list[str], bonds: list[str]) -> str:
        """
        Construct SMIRKS representation for improper torsions.

        Improper torsions use a special branched SMIRKS syntax where the
        third atom is always central: [a1][b1][a2]([b2][a3])[b3][a4]

        Parameters
        ----------
        atoms : list[str]
            List of atom SMIRKS patterns (length 4).
        bonds : list[str]
            List of bond SMIRKS patterns (length 3).

        Returns
        -------
        str
            Branched SMIRKS pattern for the improper torsion.

        Raises
        ------
        AssertionError
            If the number of atoms or bonds is incorrect.
        """
        # Make sure that the lengths of atoms and bonds are consistent
        assert len(atoms) == self.n_atoms, f"Expected {self.n_atoms} atoms, got {len(atoms)}"
        assert len(bonds) == self.n_atoms - 1, (
            f"Expected {self.n_atoms - 1} bonds, got {len(bonds)}"
        )

        return f"{atoms[0]}{bonds[0]}{atoms[1]}({bonds[1]}{atoms[2]}){bonds[2]}{atoms[3]}"

    def get_smirks(self, specificity_level: SpecificityLevel) -> str:
        """
        Generate SMIRKS representation for improper torsions.

        Unlike proper torsions, improper torsions are not symmetric due to
        the central atom constraint, so no reverse ordering is considered.

        Parameters
        ----------
        specificity_level : SpecificityLevel
            Defines the level of detail for atom and bond descriptions.

        Returns
        -------
        str
            Branched SMIRKS pattern with the third atom as central.
        """
        idxs = self.indices
        n = self.n_atoms

        atoms = [
            specificity_level.get_atom_smirks(
                at_idx, at_id, self.rdkit_mol, self.terminal_atom_indices
            )
            for at_idx, at_id in zip(idxs, range(n), strict=True)
        ]

        # Third atom is always central
        bond_atom_numbers = {(0, 1), (1, 2), (1, 3)}
        bonds = [
            specificity_level.get_bond_smirks(
                (idxs[i], idxs[j]), central_bond=(j == 2), mol=self.rdkit_mol
            )
            for i, j in bond_atom_numbers
        ]
        return self._construct_smirks(atoms, bonds)

    @classmethod
    def get_parameter(
        cls,
        smirks: str,
        specificity_num: int,
        components: list["MMComponent"],
        index: int,
        base_ff: ForceField,
    ) -> ImproperTorsionHandler.ImproperTorsionType:
        """
        Generate improper torsion parameter with default values.

        Creates an improper torsion parameter with default force constant
        and phase angle for maintaining molecular geometry.

        Parameters
        ----------
        smirks : str
            SMIRKS pattern defining the improper torsion type.
        specificity_num : int
            Specificity level for parameter identification.
        components : list[MMComponent]
            List of improper torsion components (used for counting).
        index : int
            Parameter index for identification.
        base_ff : openff.toolkit.ForceField
            Base force field (not currently used for impropers).

        Returns
        -------
        ImproperTorsionHandler.ImproperTorsionType
            Improper torsion parameter with default values.
        """
        # assert all(isinstance(c, ImproperTorsion) for c in components), "All components must be ImproperTorsion instances"
        #
        # Use module-level cached default lists. We pass copies so the handler
        # cannot mutate the shared originals.
        parameter = ImproperTorsionHandler.ImproperTorsionType(
            smirks=smirks,
            k=list(_IMPROPER_DEFAULT_K),
            phase=list(_IMPROPER_DEFAULT_PHASE),
            periodicity=list(_IMPROPER_DEFAULT_PERIODICITY),
            idivf=list(_IMPROPER_DEFAULT_IDIVF),
            id=f"specificity={specificity_num} index={index} count={len(components)}",
        )
        return parameter


# Cache for labeling results to avoid re-labeling the same molecules
_LABELING_CACHE: dict[tuple, dict] = {}


def get_parameters_for_components(
    components: list[MMComponent], forcefield: ForceField, max_samples: int = 10
) -> ParameterList:
    """
    Extract force field parameters for a list of MM components.

    Retrieves parameters from a force field by labeling molecules containing
    the components and extracting the relevant parameter values.

    Parameters
    ----------
    components : list[MMComponent]
        List of molecular mechanics components to get parameters for.
    forcefield : openff.toolkit.ForceField
        Force field to extract parameters from.
    max_samples : int, default 10
        Maximum number of components to sample for parameter extraction.
        Used to limit computation time for large component lists.

    Returns
    -------
    ParameterList
        List of parameter objects extracted from the force field.

    Notes
    -----
    For efficiency, if the component list exceeds max_samples, a random
    subset is chosen for parameter extraction. This provides a representative
    sample while keeping computation time reasonable.

    Molecule labeling results are cached to avoid redundant computation
    when the same molecule appears in multiple component lists.

    Examples
    --------
    >>> bonds = [Bond(...), Bond(...)]  # List of bond components
    >>> ff = ForceField("openff-2.0.0.offxml")
    >>> params = get_parameters_for_components(bonds, ff, max_samples=5)
    """
    parameters = []
    subsampled_components = (
        components
        if len(components) <= max_samples
        else np.random.choice(components, max_samples, replace=False)
    )
    for c in subsampled_components:
        cache_key = (c.mapped_smiles, id(forcefield))

        # Reuse cached labeling if available
        if cache_key not in _LABELING_CACHE:
            _LABELING_CACHE[cache_key] = forcefield.label_molecules(c.mol.to_topology())[0]

        assigned_parameters = _LABELING_CACHE[cache_key]
        tag_name = c.handler_class._TAGNAME
        if tag_name is not None:
            parameters.append(assigned_parameters[tag_name][c.indices])
    return cast(ParameterList, parameters)
