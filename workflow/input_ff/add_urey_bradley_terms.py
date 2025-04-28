"""Add Urey-Bradley terms to a supplied OpenFF force field."""

from openff.toolkit import ForceField
from copy import deepcopy
from openff.toolkit.typing.engines.smirnoff.parameters import BondType, AngleType
from openff.units import unit as off_unit
import typer
from pathlib import Path

def angle_smirks_to_urey_bradley_smirks(smirks: str) -> str:
    """
    Convert an angle SMIRKS to a Urey-Bradley SMIRKS by removing the ':2' label and converting the
    ':3' label to a '2' label.
    """
    # Remove the ':2' label
    smirks = smirks.replace(":2", "")
    # Convert the ':3' label to a '2' label
    smirks = smirks.replace(":3", ":2")
    return smirks

def get_urey_bradley_bond_parameter(angle: AngleType) -> BondType:
    """
    Get the Urey-Bradley bond parameter for a given angle parameter.
    """
    return BondType(
        smirks=angle_smirks_to_urey_bradley_smirks(angle.smirks),
        k= 0.0 * off_unit.kilocalories_per_mole / off_unit.angstrom ** 2,
        length= 2.0 * off_unit.angstrom,
        id=f"{angle.id}_urey_bradley",
    )

def add_urey_bradley_terms(forcefield: ForceField) -> ForceField:
    """
    Add Urey-Bradley terms to the force field, initialising the force constants
    to 0 and distances to 2 Angstroms.
    """
    new_ff = deepcopy(forcefield)
    angle_handler = new_ff.get_parameter_handler("Angles")
    bond_handler = new_ff.get_parameter_handler("Bonds")
    for parameter in angle_handler.parameters:
        ub_bond = get_urey_bradley_bond_parameter(parameter)
        bond_handler.parameters.append(ub_bond)

    return new_ff

def main(input_ff_path: Path, output_ff_path: Path):
    """
    Add Urey-Bradley terms to a supplied OpenFF force field.
    """
    forcefield = ForceField(str(input_ff_path))
    new_forcefield = add_urey_bradley_terms(forcefield)
    new_forcefield.to_file(str(output_ff_path))

if __name__ == "__main__":
    typer.run(main)