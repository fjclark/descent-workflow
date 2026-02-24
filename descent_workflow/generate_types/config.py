"""
Configuration for bespoke type generation.

This module provides configuration classes for generating force field parameters
with bespoke SMIRKS patterns based on training data. The configuration controls
specificity levels, population cutoffs, and integration of linear force fields.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator

from openff.toolkit import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import ParameterType

from .molecular_classes import (
    Bond,
    Angle,
    ProperTorsion,
    ImproperTorsion,
    MMComponent,
    SpecificityLevel,
)
from . import process_SMIRKS


class TypeGenConfig(BaseModel):
    """
    Configuration for bespoke molecular mechanics type generation.

    This class defines how to generate SMIRKS patterns at different specificity
    levels for bonds, angles, proper torsions, and improper torsions. It also
    specifies paths to linear force field files for filtering unwanted patterns
    and adding extra parameters.

    Attributes
    ----------
    component_types : list[str]
        List of component types to generate. Valid options: "Bond", "Angle",
        "ProperTorsion", "ImproperTorsion". Default includes all types.
    bond_specificities : dict[str, dict[str, str]]
        Configuration for bond SMIRKS specificity levels. Keys are level names,
        values are dicts with "atom_smirks" and "bond_smirks" function names.
    angle_specificities : dict[str, dict[str, str]]
        Configuration for angle SMIRKS specificity levels.
    torsion_specificities : dict[str, dict[str, str]]
        Configuration for proper torsion SMIRKS specificity levels.
    improper_specificities : dict[str, dict[str, str]]
        Configuration for improper torsion SMIRKS specificity levels.
    cutoff_population : int
        Minimum number of component instances required to maintain a specific
        SMIRKS pattern. Patterns below this threshold are generalized.
    unwanted_smirks_paths : dict[str, Optional[Path]]
        Paths to force field files containing SMIRKS patterns to exclude.
        Keys are component type names (e.g., "ProperTorsion").
    extra_parameters_paths : dict[str, Optional[Path]]
        Paths to force field files containing additional parameters to include.
        Keys are component type names (e.g., "Angle").
    n_workers : Optional[int]
        Number of parallel workers. If None, uses all available CPU cores.

    Available SMIRKS Functions
    ---------------------------
    Atom SMIRKS functions:
        - STANDARD: [#NXM:ID]
        - WITH_RING_INFO: [#NXM;rK:ID]
        - TERMINAL_WILDCARD: [*:ID] for terminals, [#NXM:ID] for central
        - TERMINAL_H_NO_H: [#1:ID] or [!#1:ID] for terminals
        - TERMINAL_H_NO_H_RING_INFO: TERMINAL_H_NO_H + ring info for central

    Bond SMIRKS functions:
        - STANDARD: Explicit bond types (-, =, #, :)
        - WITH_RING_INFO: Bond types with ring info (e.g., -;@)
        - NON_CENTRAL_WILDCARD: ~ for non-central, explicit for central
        - WILDCARD: ~ for all bonds

    Examples
    --------
    >>> config = TypeGenConfig(
    ...     component_types=["Angle"],
    ...     angle_specificities={
    ...         "TerminalWildcard": {
    ...             "atom_smirks": "TERMINAL_WILDCARD",
    ...             "bond_smirks": "NON_CENTRAL_WILDCARD"
    ...         },
    ...         "Standard": {
    ...             "atom_smirks": "STANDARD",
    ...             "bond_smirks": "STANDARD"
    ...         }
    ...     },
    ...     cutoff_population=10
    ... )
    """

    component_types: list[str] = Field(
        default=["Bond", "Angle", "ProperTorsion", "ImproperTorsion"],
        description="Component types to generate SMIRKS patterns for.",
    )

    bond_specificities: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "Standard": {
                "atom_smirks": "STANDARD",
                "bond_smirks": "STANDARD",
            }
        },
        description="Bond SMIRKS configurations by specificity level name.",
    )

    angle_specificities: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "TerminalWildcard": {
                "atom_smirks": "TERMINAL_WILDCARD",
                "bond_smirks": "NON_CENTRAL_WILDCARD",
            },
            "TerminalHnoH": {
                "atom_smirks": "TERMINAL_H_NO_H",
                "bond_smirks": "NON_CENTRAL_WILDCARD",
            },
            "Standard": {
                "atom_smirks": "STANDARD",
                "bond_smirks": "STANDARD",
            },
        },
        description="Angle SMIRKS configurations by specificity level name.",
    )

    torsion_specificities: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "TerminalWildcard": {
                "atom_smirks": "TERMINAL_WILDCARD",
                "bond_smirks": "NON_CENTRAL_WILDCARD",
            },
            "TerminalHnoH": {
                "atom_smirks": "TERMINAL_H_NO_H",
                "bond_smirks": "NON_CENTRAL_WILDCARD",
            },
            "Standard": {
                "atom_smirks": "STANDARD",
                "bond_smirks": "STANDARD",
            },
        },
        description="Proper torsion SMIRKS configurations by specificity level name.",
    )

    improper_specificities: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "TerminalWildcard": {
                "atom_smirks": "TERMINAL_WILDCARD",
                "bond_smirks": "NON_CENTRAL_WILDCARD",
            },
            "TerminalHnoH": {
                "atom_smirks": "TERMINAL_H_NO_H",
                "bond_smirks": "NON_CENTRAL_WILDCARD",
            },
            "Standard": {
                "atom_smirks": "STANDARD",
                "bond_smirks": "STANDARD",
            },
        },
        description="Improper torsion SMIRKS configurations by specificity level name.",
    )

    cutoff_population: int = Field(
        default=10,
        ge=1,
        description="Minimum component count to maintain specificity level.",
    )

    unwanted_smirks_paths: dict[str, Optional[Path]] = Field(
        default_factory=dict,
        description="Paths to force fields with SMIRKS patterns to exclude by component type.",
    )

    extra_parameters_paths: dict[str, Optional[Path]] = Field(
        default_factory=dict,
        description="Paths to force fields with extra parameters to include by component type.",
    )

    coverage_smiles_paths: dict[str, Optional[Path]] = Field(
        default_factory=dict,
        description="Paths to CSV files containing SMILES lists for coverage checking. Keys are dataset names (e.g., 'validation', 'test_set_3'). Each CSV file should have a column named 'smiles', 'SMILES', 'smi', or 'SMI', or the first column will be used.",
    )

    n_workers: Optional[int] = Field(
        default=None,
        description="Number of parallel workers. None uses all CPU cores.",
    )

    model_config = {"populate_by_name": True}

    @validator("component_types")
    def validate_component_types(cls, v: list[str]) -> list[str]:
        """Validate component type names."""
        valid_types = {"Bond", "Angle", "ProperTorsion", "ImproperTorsion"}
        for component_type in v:
            if component_type not in valid_types:
                raise ValueError(
                    f"Invalid component type '{component_type}'. "
                    f"Must be one of: {valid_types}"
                )
        return v

    @validator(
        "unwanted_smirks_paths", "extra_parameters_paths", "coverage_smiles_paths"
    )
    def validate_paths_exist(
        cls, v: dict[str, Optional[Path]]
    ) -> dict[str, Optional[Path]]:
        """Validate that specified paths exist."""
        for component_type, path in v.items():
            if path is not None and not path.exists():
                raise ValueError(
                    f"Specified path for {component_type} does not exist: {path}"
                )
        return v

    def get_component_class(self, component_type: str) -> type[MMComponent]:
        """
        Get the MMComponent class for a component type name.

        Parameters
        ----------
        component_type : str
            Component type name (e.g., "Bond", "Angle").

        Returns
        -------
        type[MMComponent]
            The corresponding component class.

        Raises
        ------
        ValueError
            If component type is not recognized.
        """
        component_map = {
            "Bond": Bond,
            "Angle": Angle,
            "ProperTorsion": ProperTorsion,
            "ImproperTorsion": ImproperTorsion,
        }
        if component_type not in component_map:
            raise ValueError(f"Unknown component type: {component_type}")
        return component_map[component_type]

    def get_specificity_config(self, component_type: str) -> dict[str, dict[str, str]]:
        """
        Get specificity configuration for a component type.

        Parameters
        ----------
        component_type : str
            Component type name.

        Returns
        -------
        dict[str, dict[str, str]]
            Specificity level configurations mapping names to
            {"atom_smirks": ..., "bond_smirks": ...} dicts.
        """
        config_map = {
            "Bond": self.bond_specificities,
            "Angle": self.angle_specificities,
            "ProperTorsion": self.torsion_specificities,
            "ImproperTorsion": self.improper_specificities,
        }
        return config_map.get(component_type, {})

    def get_unwanted_smirks(self, component_type: str) -> list[str] | None:
        """
        Load unwanted SMIRKS patterns from force field file.

        Parameters
        ----------
        component_type : str
            Component type name.

        Returns
        -------
        list[str] | None
            List of SMIRKS patterns to exclude, or None if no file specified.
        """
        if (path := self.unwanted_smirks_paths.get(component_type)) is None:
            return None

        ff = ForceField(str(path))
        component_class = self.get_component_class(component_type)
        handler_name = component_class.handler_class._TAGNAME
        if handler_name is None:
            return None
        handler = ff.get_parameter_handler(handler_name)
        return [param.smirks for param in handler.parameters]

    def get_extra_parameters(self, component_type: str) -> list[ParameterType] | None:
        """
        Load extra parameters from force field file.

        Parameters
        ----------
        component_type : str
            Component type name.

        Returns
        -------
        list[ParameterType] | None
            List of extra parameters to include.
        """
        if (path := self.extra_parameters_paths.get(component_type)) is None:
            return None

        ff = ForceField(str(path))
        component_class = self.get_component_class(component_type)
        handler_name = component_class.handler_class._TAGNAME
        handler = ff.get_parameter_handler(handler_name)
        return list(handler.parameters)

    def build_specificity_levels(
        self, component_type: str
    ) -> dict[int, SpecificityLevel]:
        """
        Build SpecificityLevel objects for a component type.

        This directly maps config SMIRKS function names to the actual functions,
        using the decorator-based registries from process_SMIRKS.

        Parameters
        ----------
        component_type : str
            Component type name.

        Returns
        -------
        dict[int, SpecificityLevel]
            Mapping from specificity level number to SpecificityLevel object.
            Higher numbers indicate more specific patterns.
        """
        specificity_config = self.get_specificity_config(component_type)

        # Build specificity levels from config using registries
        specificity_levels = {}
        for i, (name, config) in enumerate(specificity_config.items()):
            atom_fn = process_SMIRKS.atom_fn_map[config["atom_smirks"]]
            bond_fn = process_SMIRKS.bond_fn_map[config["bond_smirks"]]

            specificity_levels[i] = SpecificityLevel(
                name=f"{i}:{name}",
                get_atom_smirks=atom_fn,
                get_bond_smirks=bond_fn,
            )

        return specificity_levels
