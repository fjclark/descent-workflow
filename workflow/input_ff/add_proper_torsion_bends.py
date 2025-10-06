"""Add ProperTorsion-Bend terms to a supplied OpenFF force field."""

from copy import deepcopy
from pathlib import Path

import typer
from openff.toolkit import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import ProperTorsionType
from openff.units import unit as off_unit
from smirnoff_plugins.handlers.valence import ProperTorsionBendHandler
import re
from tqdm import tqdm


def reverse_smirks(smirks):
    """
    Reverses a SMIRKS pattern by reversing the order of labeled atoms.
    Assumes all atoms are in [] brackets and everything else is a bond.

    Args:
        smirks: A SMIRKS pattern string with labeled atoms (e.g., :1, :2, etc.)

    Returns:
        A new SMIRKS pattern with atoms reordered so labels appear in original order

    Example:
        Input:  [*:1]-[#7X3:2]-!@[#6X3:3](=[#8,#16,#7:4])-[#6,#1]
        Output: [#6,#1]-[#8,#16,#7:1]=[#6X3:2]-!@[#7X3:3]-[*:4]
    """

    # Split into atoms (in brackets) and bonds (everything else)
    parts = re.split(r"(\[[^\]]+\])", smirks)

    # Filter out empty strings
    parts = [p for p in parts if p]

    # Separate atoms and bonds
    atoms = []
    bonds = []

    for i, part in enumerate(parts):
        if part.startswith("["):
            atoms.append(part)
        else:
            bonds.append(part)

    # Extract labels from atoms
    def get_label(atom_str):
        match = re.search(r":(\d+)", atom_str)
        return int(match.group(1)) if match else None

    # Change label in atom
    def set_label(atom_str, new_label):
        return re.sub(r":\d+", f":{new_label}", atom_str)

    # Find all labeled atoms and create reverse mapping
    labeled_atoms = [(i, get_label(atom)) for i, atom in enumerate(atoms)]
    labeled_atoms = [(i, lbl) for i, lbl in labeled_atoms if lbl is not None]

    if not labeled_atoms:
        return smirks

    # Create mapping: old_label -> new_label (reversed)
    sorted_labels = sorted([lbl for _, lbl in labeled_atoms])
    reversed_labels = list(reversed(sorted_labels))
    reverse_map = {old: new for old, new in zip(sorted_labels, reversed_labels)}

    # Apply new labels to atoms
    new_atoms = []
    for atom in atoms:
        label = get_label(atom)
        if label is not None:
            new_atoms.append(set_label(atom, reverse_map[label]))
        else:
            new_atoms.append(atom)

    # Reverse the order of atoms
    new_atoms = list(reversed(new_atoms))

    # Reverse the order of bonds
    new_bonds = list(reversed(bonds))

    # Reconstruct SMIRKS by interleaving atoms and bonds
    result = []
    for i in range(len(new_atoms)):
        result.append(new_atoms[i])
        if i < len(new_bonds):
            result.append(new_bonds[i])

    return "".join(result)


def proper_torsion_smirks_to_proper_torsion_bend_smirks(smirks: str) -> list[str]:
    """
    Convert a proper torsion SMIRKS to a proper torsion-bend SMIRKS by:

    1. Checking if the SMIRKS is symmetrical. If it is, return a list with the original SMIRKS.
    2. If the SMIRKS is not symmetrical, return a list with the original SMIRKS and the reversed SMIRKS.
    """
    # Parse the SMIRKS with the OpenFF toolkit
    from openff.toolkit.topology import Molecule

    reversed_smirks = reverse_smirks(smirks)
    if smirks == reversed_smirks:
        return [smirks]
    return [smirks, reversed_smirks]


def get_proper_torsion_bend_parameters(
    proper_torsion: ProperTorsionType,
    angle0: off_unit.Quantity = 109.5 * off_unit.degree,
) -> list[ProperTorsionBendHandler.ProperTorsionBendType]:
    """
    Get the ProperTorsion-Bend parameters for a given proper torsion parameter.

    Parameters
    ----------
    proper_torsion : ProperTorsionType
        The proper torsion parameter to convert to ProperTorsion-Bend parameters.
    angle0 : off_unit.Quantity, optional
        The equilibrium angle for the ProperTorsion-Bend term, in degrees.

    Returns
    -------
    list[ProperTorsionBendHandler.ProperTorsionBendType]
        A list of ProperTorsion-Bend parameters.
    """
    proper_torsion_bends = []
    all_smirks = proper_torsion_smirks_to_proper_torsion_bend_smirks(
        proper_torsion.smirks
    )

    for i, smirks in enumerate(all_smirks):
        proper_torsion_bends.append(
            ProperTorsionBendHandler.ProperTorsionBendType(
                smirks=smirks,
                angle0=angle0,
                periodicity=proper_torsion.periodicity,
                phase=proper_torsion.phase,
                k=proper_torsion.k,
                id=f"{proper_torsion.id}_bend_{i}",
            )
        )
    return proper_torsion_bends


def add_proper_torsion_bend_terms(forcefield: ForceField) -> ForceField:
    """
    Add ProperTorsion-Bend terms to the force field, initialising the equilibrium angles
    to 109.5 degrees and copying the periodicity, phase, and force constants from the
    corresponding ProperTorsion terms.
    """
    new_ff = deepcopy(forcefield)
    proper_torsion_handler = new_ff.get_parameter_handler("ProperTorsions")
    proper_torsion_bend_handler = new_ff.get_parameter_handler("ProperTorsionBends")
    for parameter in tqdm(
        proper_torsion_handler.parameters, desc="Adding ProperTorsion-Bend terms"
    ):
        proper_torsion_bends = get_proper_torsion_bend_parameters(parameter)
        for ptb in proper_torsion_bends:
            proper_torsion_bend_handler.parameters.append(ptb)

    return new_ff


def main(input_ff_path: Path, output_ff_path: Path):
    """
    Add ProperTorsion-Bend terms to a supplied OpenFF force field.
    """
    forcefield = ForceField(str(input_ff_path), load_plugins=True)
    new_forcefield = add_proper_torsion_bend_terms(forcefield)
    new_forcefield.to_file(str(output_ff_path))


if __name__ == "__main__":
    typer.run(main)
