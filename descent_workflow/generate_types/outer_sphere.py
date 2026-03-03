"""
Utilities for handling outer-sphere atoms in SMIRKS generation.

This module provides data structures and functions to support including atoms
bonded to core potential atoms in SMIRKS patterns. Outer-sphere atoms capture
broader chemical context for improved SMIRKS specificity.

Key Classes
-----------
OuterSphereAtoms : Data structure tracking core and neighboring atoms by distance.
ComponentWithOuterSphere : Wrapper for components including outer-sphere information.

Key Functions
-------------
get_atom_neighbors_by_distance : Extract atoms bonded to a given atom at specific distances.
build_outer_sphere_indices : Build complete outer-sphere atom map for a component.
"""

from dataclasses import dataclass, field

from openff.toolkit import Molecule


@dataclass
class OuterSphereAtoms:
    """
    Data structure tracking core atoms and their outer-sphere neighbors.

    This class organizes atoms by their distance from core atoms in the molecule.
    Used to maintain both core atom indices and neighboring atom indices at
    distances 1 and 2 bonds away.

    Attributes
    ----------
    core_indices : tuple[int, ...]
        Indices of the core atoms in the potential (e.g., bonded atoms in a bond,
        central atoms in an angle). Immutable tuple.
    outer_by_distance : dict[int, tuple[int, ...]]
        Mapping of distance (in bonds) to outer-sphere atom indices at that distance.
        Keys are 1, 2, etc. Values are tuples of atom indices.
        Example: {1: (5, 6, 7), 2: (8, 9)}.
    n_atoms_total : int
        Total number of atoms in the structure including core and all outer atoms.

    Examples
    --------
    >>> outer = OuterSphereAtoms(
    ...     core_indices=(0, 1),
    ...     outer_by_distance={1: (2, 3, 4)},
    ...     n_atoms_total=5
    ... )
    >>> print(outer.all_indices)
    (0, 1, 2, 3, 4)
    """

    core_indices: tuple[int, ...] = ()
    outer_by_distance: dict[int, tuple[int, ...]] = field(default_factory=dict)
    n_atoms_total: int = 0

    @property
    def all_indices(self) -> tuple[int, ...]:
        """
        Get all indices (core + outer atoms at all distances).

        Returns
        -------
        tuple[int, ...]
            Sorted tuple of all atom indices.
        """
        all_atoms = set(self.core_indices)
        for outer_atoms in self.outer_by_distance.values():
            all_atoms.update(outer_atoms)
        return tuple(sorted(all_atoms))

    @property
    def has_outer_atoms(self) -> bool:
        """Check if any outer-sphere atoms are present."""
        return bool(self.outer_by_distance)

    def get_atoms_at_distance(self, distance: int) -> tuple[int, ...]:
        """
        Get outer-sphere atoms at a specific distance.

        Parameters
        ----------
        distance : int
            Distance in bonds from core atoms.

        Returns
        -------
        tuple[int, ...]
            Atom indices at the specified distance, or empty tuple if none.
        """
        return self.outer_by_distance.get(distance, ())


def get_atom_neighbors_by_distance(
    atom_idx: int,
    mol: Molecule,
    max_distance: int = 1,
) -> dict[int, set[int]]:
    """
    Get all atoms bonded to a given atom at specified distances.

    Uses breadth-first search to find atoms at exact distances from the target atom.
    Does not include the target atom itself.

    Parameters
    ----------
    atom_idx : int
        Index of the reference atom.
    mol : openff.toolkit.Molecule
        The molecule to analyze.
    max_distance : int
        Maximum distance (in bonds) to search. Default is 1 (direct neighbors).

    Returns
    -------
    dict[int, set[int]]
        Mapping from distance to set of atom indices at that distance.
        Example: {1: {5, 6, 7}, 2: {8, 9, 10}}.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CC(C)O")
    >>> neighbors = get_atom_neighbors_by_distance(1, mol, max_distance=2)
    >>> print(neighbors[1])  # Direct neighbors of carbon at index 1
    {0, 2, 3}
    """
    neighbors_by_dist = {}

    # Initialize: direct neighbors are distance 1
    graph = mol.to_networkx()
    direct_neighbors = set(graph.neighbors(atom_idx))
    if direct_neighbors:
        neighbors_by_dist[1] = direct_neighbors

    # BFS for distances > 1
    for distance in range(2, max_distance + 1):
        next_shell = set()
        current_shell = neighbors_by_dist.get(distance - 1, set())

        for neighbor in current_shell:
            for next_neighbor in graph.neighbors(neighbor):
                # Exclude atoms already found at closer distances
                if next_neighbor != atom_idx and not any(
                    next_neighbor in neighbors_by_dist.get(d, set()) for d in range(1, distance)
                ):
                    next_shell.add(next_neighbor)

        if next_shell:
            neighbors_by_dist[distance] = next_shell

    return neighbors_by_dist


def build_outer_sphere_indices(
    core_indices: tuple[int, ...],
    mol: Molecule,
    max_distance: int = 1,
) -> OuterSphereAtoms:
    """
    Build outer-sphere atom map for a component's core atoms.

    Collects all atoms bonded to core atoms up to the specified distance,
    organized by distance from the core. Excludes atoms that are themselves core atoms.

    Parameters
    ----------
    core_indices : tuple[int, ...]
        Indices of core atoms (e.g., bonded atoms in a bond).
    mol : openff.toolkit.Molecule
        The molecule to analyze.
    max_distance : int
        Maximum distance (in bonds) from core atoms to include. Default is 1.

    Returns
    -------
    OuterSphereAtoms
        Data structure with organized core and outer atom indices.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CC(C)O")
    >>> outer = build_outer_sphere_indices((1,), mol, max_distance=1)
    >>> print(outer.core_indices)
    (1,)
    >>> print(outer.outer_by_distance)
    {1: (0, 2, 3)}
    """
    outer_by_distance = {}
    core_set = set(core_indices)

    for distance in range(1, max_distance + 1):
        outer_at_distance = set()

        for core_atom in core_indices:
            neighbors = get_atom_neighbors_by_distance(core_atom, mol, max_distance=distance)
            atoms_at_dist = neighbors.get(distance, set())

            # Exclude core atoms
            outer_at_distance.update(atoms_at_dist - core_set)

        if outer_at_distance:
            outer_by_distance[distance] = tuple(sorted(outer_at_distance))

    return OuterSphereAtoms(
        core_indices=core_indices,
        outer_by_distance=outer_by_distance,
        n_atoms_total=len(mol.atoms),
    )
