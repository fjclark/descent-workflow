"""
Functionality for generating bespoke molecular mechanics types based on training data.

This module provides tools for extracting molecular mechanics components (bonds, angles,
torsions) from training datasets, organizing them by specificity levels, and assembling
them into force field parameters with bespoke SMIRKS patterns.

Main Entry Point
----------------
generate_bespoke_types : function
    Orchestrates the complete type generation workflow from dataset to force field.

Configuration
-------------
TypeGenConfig : class
    Configuration for specificity levels, population cutoffs, and linear FF integration.

Key Modules
-----------
molecular_classes : Component classes (Bond, Angle, ProperTorsion, ImproperTorsion)
process_mmcomponents : Extract and organize components from datasets
process_SMIRKS : Generate SMIRKS patterns with configurable specificity
coverage : Analyze force field coverage on molecular datasets
orchestrate : Main workflow coordination and checkpointing

Examples
--------
>>> from descent_workflow.generate_types import generate_bespoke_types, TypeGenConfig
>>> from pathlib import Path
>>>
>>> config = TypeGenConfig(
...     component_types=["Bond", "Angle", "ProperTorsion"],
...     cutoff_population=10,
... )
>>> ff_path = generate_bespoke_types(
...     config=config,
...     base_ff_path=Path("input_ff/sage-2.2.0.offxml"),
...     data_dir=Path("data/spice2/data-filtered-sage/data-train"),
...     output_dir=Path("data/spice2/generated_types"),
... )

Big thanks to @jaclark5 for organizing the original notebook into these modules
https://github.com/openforcefield/back-to-school-jen/tree/main/4_make_offxmls/from_finlay
"""

from .config import TypeGenConfig
from .orchestrate import generate_bespoke_types

__all__ = ["generate_bespoke_types", "TypeGenConfig"]
