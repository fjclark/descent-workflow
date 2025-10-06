"""Functionality for file format interconversion, e.g. between .pt and .offxml force fields."""

from copy import deepcopy
from pathlib import Path

import loguru
import torch
from models import WorkflowConfig
from openff.toolkit import ForceField
from openff.units import unit as off_unit
from smee import TensorForceField

logger = loguru.logger


# Mainly written by Josh Horton
def pt_ff_to_off_ff(
    base_force_field: ForceField, tensor_force_field: TensorForceField
) -> ForceField:
    """Convert the FF from pt to OFF ForceField format."""

    # Copy the base force field to avoid modifying it
    base_force_field = deepcopy(base_force_field)

    for potential in tensor_force_field.potentials:
        potential_type = str(potential.type)

        parameter_names = potential.parameter_cols
        parameter_units = potential.parameter_units

        if potential_type in ["Bonds", "Angles", "UreyBradleys"]:
            handler = base_force_field.get_parameter_handler(potential_type)
            for i in range(len(potential.parameters)):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = potential.parameters[i].detach().cpu()
                for j, (p, unit) in enumerate(zip(parameter_names, parameter_units)):
                    setattr(ff_parameter, p, opt_parameters[j] * unit)

        if potential_type in ["LinearBonds", "LinearAngles", "LinearUreyBradleys"]:
            handler = base_force_field.get_parameter_handler(
                potential_type.replace("Linear", "")
            )
            for i in range(len(potential.parameters)):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_linear_parameters = potential.parameters[i].detach().cpu()
                # Convert linear parameters back to harmonic parameters
                k1, k2 = (
                    opt_linear_parameters[0].item(),
                    opt_linear_parameters[1].item(),
                )
                b1, b2 = (
                    opt_linear_parameters[2].item(),
                    opt_linear_parameters[3].item(),
                )
                k = k1 + k2
                b = (b1 * k1 + b2 * k2) / k
                logger.info(f"Converting {smirks} from linear to harmonic")
                logger.info(f"Parameter names: {parameter_names}")
                if potential_type in ["LinearBonds", "LinearUreyBradleys"]:
                    ff_parameter.k = k * parameter_units[0]
                    ff_parameter.length = b * parameter_units[2]
                elif potential_type == "LinearAngles":
                    # Convert to kcal mol-1 deg -2 (from kcal mol-1 rad -2)
                    ff_parameter.k = k * parameter_units[0]
                    angle = (b * parameter_units[2]).to(off_unit.degree)
                    # Reflect the angle if it is negative
                    if angle < 0 * off_unit.degree:
                        angle = -angle
                    # Ensure the angle is within 0 to 360 degrees
                    elif angle > 360 * off_unit.degree:
                        angle = angle - (angle // (360 * off_unit.degree)) * (
                            360 * off_unit.degree
                        )
                    # Reflect the angle about 180 degrees if it is greater than 180
                    if angle > 180 * off_unit.degree:
                        angle = 360 * off_unit.degree - angle
                    ff_parameter.angle = angle

        elif potential_type in ["ProperTorsions"]:
            handler = base_force_field.get_parameter_handler(potential_type)
            # we need to collect the k values into a list accross the entries
            collection_data = {}
            for i in range(len(potential.parameters)):
                smirks = potential.parameter_keys[i].id
                if smirks not in collection_data:
                    collection_data[smirks] = {}
                opt_parameters = potential.parameters[i].detach().cpu()
                # find k and the perodicity
                k_index = parameter_names.index("k")
                k = opt_parameters[k_index] * parameter_units[k_index]
                p = int(opt_parameters[parameter_names.index("periodicity")])
                collection_data[smirks][p] = k
            # now update the force field
            for smirks, tor_data in collection_data.items():
                ff_parameter = handler[smirks]
                k_s = [tor_data[p] for p in ff_parameter.periodicity]
                logger.info(ff_parameter.periodicity)
                logger.info(k_s)
                logger.info(smirks)
                ff_parameter.k = k_s

        elif potential_type in ["ProperTorsionBends"]:
            handler = base_force_field.get_parameter_handler(potential_type)
            # we need to collect the k values into a list accross the entries
            collection_data = {}
            for i in range(len(potential.parameters)):
                smirks = potential.parameter_keys[i].id
                if smirks not in collection_data:
                    collection_data[smirks] = {}
                opt_parameters = potential.parameters[i].detach().cpu()
                # find k and the perodicity
                k_index = parameter_names.index("k")
                k = opt_parameters[k_index] * parameter_units[k_index]
                p = int(opt_parameters[parameter_names.index("periodicity")])
                collection_data[smirks][p] = {}
                collection_data[smirks][p]["k"] = k
                # Extract the angle0 value from the shared parameters
                all_angle0 = (
                    potential.shared_parameter_mapping @ potential.shared_parameters
                )
                angle0_index = potential.shared_parameter_cols.index("angle0")
                angle0 = (
                    all_angle0[i][angle0_index]
                    * potential.shared_parameter_units[angle0_index]
                )
                collection_data[smirks][p]["angle0"] = angle0.to(off_unit.degree)

            # now update the force field
            for smirks, tor_data in collection_data.items():
                ff_parameter = handler[smirks]
                k_s = [tor_data[p]["k"] for p in ff_parameter.periodicity]
                angle0 = [tor_data[p]["angle0"] for p in ff_parameter.periodicity]
                assert len(set(angle0)) == 1, "All angle0 values must be the same"
                logger.info(ff_parameter.periodicity)
                logger.info(k_s)
                logger.info(smirks)
                ff_parameter.k = k_s
                ff_parameter.angle0 = angle0[0]

        elif potential_type in ["ImproperTorsions"]:
            handler = base_force_field.get_parameter_handler(potential_type)
            # we only fit the v2 terms for impropers so convert to list and set
            for i in range(len(potential.parameters)):
                smirks = potential.parameter_keys[i].id
                opt_parameters = potential.parameters[i].detach().cpu()
                k_index = parameter_names.index("k")
                ff_parameter = handler[smirks]
                ff_parameter.k = [opt_parameters[k_index] * parameter_units[k_index]]

        if potential_type in ["Electrostatics", "vdW"]:
            logger.warning(
                f"Only setting the attributes (and not the parameters) for {potential_type} potentials."
            )
            handler = base_force_field.get_parameter_handler(potential_type)
            for attr_name, attr_unit, attr_val in zip(
                potential.attribute_cols,
                potential.attribute_units,
                potential.attributes,
                strict=True,
            ):
                # Convert scale_14 to scale14
                attr_name = attr_name.replace("_", "")
                setattr(handler, attr_name, attr_val * attr_unit)

    return base_force_field


def pt_file_to_offxml(
    base_force_field: str | Path,
    output: str | Path,
    tensor_force_field_path: str | Path,
) -> None:
    """Convert the FF from pt to offxml format."""

    logger.info(
        f"Converting {tensor_force_field_path} to OFF format with base {base_force_field}"
    )

    tensor_ff = torch.load(tensor_force_field_path)
    base_ff = ForceField(
        base_force_field, load_plugins=True, allow_cosmetic_attributes=True
    )

    ff = pt_ff_to_off_ff(base_ff, tensor_ff)
    ff.to_file(output)

    logger.info(f"Saved offxml force field to {output}")


def pt_file_to_offxml_with_description(config: WorkflowConfig) -> None:
    """Convert the FF from pt to offxml format with a description."""

    if not config.output_ff_dir.exists():
        config.output_ff_dir.mkdir(parents=True, exist_ok=True)
    pt_file_to_offxml(
        base_force_field=config.starting_force_field_path,
        tensor_force_field_path=config.final_torch_ff_path,
        output=config.output_ff_path,
    )

    # Save a description of the force field
    description_name = config.output_ff_name.replace(".offxml", ".txt")
    description_path = config.output_ff_dir / description_name
    with open(description_path, "w") as f:
        f.write(config.experiment_description)
