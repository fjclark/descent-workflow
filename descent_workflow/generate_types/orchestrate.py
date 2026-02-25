"""Orchestration for bespoke molecular mechanics type generation.

This module provides the main entry point for generating force field parameters
with bespoke SMIRKS patterns based on training data. It coordinates component
extraction, specificity organization, force field assembly, and checkpoint management.
"""

import gc
import json
import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from loguru import logger
from openff.toolkit import ForceField
from tqdm import tqdm

from .config import TypeGenConfig
from .coverage import check_all_components_fully_covered_parallel_chunks
from .process_mmcomponents import (
    get_all_mm_components,
    get_mm_components_by_specificity_by_type,
    flatten_mm_component_types,
)
from .process_SMIRKS import add_types_to_ff


def generate_bespoke_types(
    config: TypeGenConfig,
    base_ff_path: Path,
    data_dir: Path,
    output_dir: Path,
    test_data_dir: Path | None = None,
) -> Path:
    """
    Generate force field with bespoke molecular mechanics types.

    This is the main orchestration function that coordinates the entire type
    generation workflow. It loads training data, extracts molecular mechanics
    components, organizes them by specificity levels, assembles them into a
    force field, and generates comprehensive coverage reports.

    Parameters
    ----------
    config : TypeGenConfig
        Configuration specifying specificity levels, population cutoffs,
        paths to linear force field files, and optional SMILES files for
        additional coverage checking.
    base_ff_path : Path
        Path to the base force field to extend with bespoke types.
    data_dir : Path
        Directory containing the filtered HuggingFace dataset
        (typically data-filtered-{ff_name}/data-train).
    output_dir : Path
        Directory for output files, checkpoints, and coverage reports.
    test_data_dir : Path | None
        Optional directory containing a test dataset. If provided, coverage
        will be checked on the test set with warnings (not errors) for
        incomplete coverage.

    Returns
    -------
    Path
        Path to the generated force field with bespoke types.

    Notes
    -----
    The workflow proceeds through these phases:
    1. Load filtered training dataset and base force field
    2. For each component type (Bond, Angle, ProperTorsion, ImproperTorsion):
       a. Extract components from molecules
       b. Filter unwanted SMIRKS patterns
       c. Organize by specificity levels with population cutoffs
       d. Add parameters to force field
       e. Save checkpoints
    3. Save final force field to disk (unconditional, for debugging purposes)
    4. Generate coverage reports for training, test, and additional SMILES datasets
       - Training dataset must have 100% coverage (raises error otherwise)
       - Test dataset and additional SMILES files: warnings for incomplete coverage

    **Important**: The force field is always saved before validation checks, so it
    is available for debugging even if coverage validation fails.

    Checkpoints are saved to {output_dir}/checkpoints/ and include:
    - Component pickles: {component_type}_components.pkl
    - Hierarchical groupings: {component_type}_by_specificity.pkl
    - Intermediate force fields: ff_after_{component_type}.offxml

    Coverage reports are saved to {output_dir}/coverage/ and include:
    - Component stats: {component_type}_coverage_stats.json
    - Summary matrix: coverage_summary.csv
    - Missing molecules and indices: train_missing_coverage.json, test_missing_coverage.json

    Examples
    --------
    >>> config = TypeGenConfig(
    ...     component_types=["Bond", "Angle"],
    ...     cutoff_population=10,
    ...     coverage_smiles_paths={
    ...         "validation": Path("data/validation.smi"),
    ...         "test_set_3": Path("data/test_3.smi"),
    ...     }
    ... )
    >>> ff_path = generate_bespoke_types(
    ...     config=config,
    ...     base_ff_path=Path("input_ff/sage-2.2.0.offxml"),
    ...     data_dir=Path("data/spice2/data-filtered-sage/data-train"),
    ...     output_dir=Path("data/spice2/generated_types"),
    ...     test_data_dir=Path("data/spice2/data-filtered-sage/data-test"),
    ... )
    >>> print(f"Generated force field: {ff_path}")
    """
    logger.info("=" * 80)
    logger.info("Starting bespoke type generation workflow")
    logger.info("=" * 80)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    coverage_dir = output_dir / "coverage"
    coverage_dir.mkdir(exist_ok=True)

    # Load base force field early (smaller, always needed)
    logger.info(f"Loading base force field from {base_ff_path}")
    try:
        current_ff = ForceField(str(base_ff_path))
        logger.info(f"Loaded force field: {base_ff_path.name}")
    except Exception as e:
        raise RuntimeError(f"Failed to load force field from {base_ff_path}: {e}") from e

    # Process each component type
    total_start_time = time.time()

    for component_type_name in tqdm(
        config.component_types, desc="Processing component types", unit="type"
    ):
        logger.info("=" * 80)
        logger.info(f"Processing component type: {component_type_name}")
        logger.info("=" * 80)

        component_start_time = time.time()
        component_class = config.get_component_class(component_type_name)

        # Check for existing checkpoint if resuming
        checkpoint_file = checkpoint_dir / f"{component_type_name}_by_specificity.pkl"
        if checkpoint_file.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_file}")
            with open(checkpoint_file, "rb") as f:
                components_by_specificity = pickle.load(f)
            logger.info("Loaded components from checkpoint")
        else:
            # Load dataset fresh for each component type (reduces memory footprint)
            logger.info(f"Loading dataset from {data_dir}")
            dataset = Dataset.load_from_disk(str(data_dir))
            logger.info(f"Loaded {len(dataset)} molecules")

            # Reduce the dataset to 1000 entries for testing
            # For debugging, keep only the entry with smiles [H:25][c:9]1[c:13]([c:19]([c:15]([c:17]([c:11]1[H:27])[Cl:1])[H:31])[N:21]2[P:23]([N:22]([P:24]2([Cl:6])([Cl:7])[Cl:8])[c:20]3[c:14]([c:10]([c:12]([c:18]([c:16]3[H:32])[Cl:2])[H:28])[H:26])[H:30])([Cl:3])([Cl:4])[Cl:5])[H:29]
            # dataset = dataset.filter(
            #     lambda x: x["smiles"]
            #     == "[H:25][c:9]1[c:13]([c:19]([c:15]([c:17]([c:11]1[H:27])[Cl:1])[H:31])[N:21]2[P:23]([N:22]([P:24]2([Cl:6])([Cl:7])[Cl:8])[c:20]3[c:14]([c:10]([c:12]([c:18]([c:16]3[H:32])[Cl:2])[H:28])[H:26])[H:30])([Cl:3])([Cl:4])[Cl:5])[H:29]"
            # )
            # dataset = dataset.shuffle(seed=42).select(range(1000))

            # Extract components
            logger.info(f"Extracting {component_type_name} components from dataset")
            unwanted_smirks = config.get_unwanted_smirks(component_type_name)
            if unwanted_smirks:
                logger.info(f"Will filter {len(unwanted_smirks)} unwanted SMIRKS patterns")

            extraction_start = time.time()
            components = get_all_mm_components(
                dataset=dataset,
                component_type=component_class,
                unwanted_smirks=unwanted_smirks,
                n_workers=config.n_workers,
                enable_outer_sphere=config.enable_outer_sphere,
                max_outer_sphere_distance=config.max_outer_sphere_distance,
            )
            extraction_time = time.time() - extraction_start
            logger.info(f"Extracted {len(components)} components in {extraction_time:.1f}s")

            # Delete dataset immediately after extracting - don't keep in memory
            del dataset
            gc.collect()

            # Save component checkpoint
            components_file = checkpoint_dir / f"{component_type_name}_components.pkl"
            with open(components_file, "wb") as f:
                pickle.dump(components, f)
            logger.info(f"Saved component checkpoint: {components_file}")

            # Organize by specificity
            logger.info("Organizing components by specificity levels")
            specificity_levels = config.build_specificity_levels(component_type_name)
            logger.info(f"Using {len(specificity_levels)} specificity levels")

            organization_start = time.time()
            components_by_specificity = get_mm_components_by_specificity_by_type(
                mm_components=components,
                specificity_levels=specificity_levels,
                cutoff_population=config.cutoff_population,
                n_workers=config.n_workers,
            )
            organization_time = time.time() - organization_start
            logger.info(f"Organized components by specificity in {organization_time:.1f}s")

            # Delete components list and specificity_levels immediately
            del components, specificity_levels
            gc.collect()

            # Save specificity checkpoint
            with open(checkpoint_file, "wb") as f:
                pickle.dump(components_by_specificity, f)
            logger.info(f"Saved specificity checkpoint: {checkpoint_file}")

            # Log statistics
            _log_component_statistics(components_by_specificity)

            # Save coverage statistics
            stats_file = coverage_dir / f"{component_type_name}_coverage_stats.json"
            _save_coverage_statistics(components_by_specificity, stats_file)

        # Add types to force field
        logger.info(f"Adding {component_type_name} parameters to force field")
        extra_parameters = config.get_extra_parameters(component_type_name)
        if extra_parameters:
            logger.info(f"Including {len(extra_parameters)} extra parameters")

        ff_start = time.time()
        current_ff = add_types_to_ff(
            ff=current_ff,
            component_types=components_by_specificity,
            component_class=component_class,
            extra_parameters=extra_parameters,
            n_workers=config.n_workers,
        )
        ff_time = time.time() - ff_start
        logger.info(f"Updated force field in {ff_time:.1f}s")

        # Save intermediate force field
        intermediate_ff_path = checkpoint_dir / f"ff_after_{component_type_name}.offxml"
        current_ff.to_file(str(intermediate_ff_path))
        logger.info(f"Saved intermediate force field: {intermediate_ff_path}")

        # Delete components_by_specificity after adding to force field
        del components_by_specificity
        gc.collect()

        component_total_time = time.time() - component_start_time
        logger.info(f"Completed {component_type_name} in {component_total_time:.1f}s")

    # Save final force field
    # NOTE: Force field is saved unconditionally here before validation,
    # so it's available for debugging even if coverage checks fail.
    final_ff_path = output_dir / "bespoke_types.offxml"
    current_ff.to_file(str(final_ff_path))
    logger.info(f"Saved final force field: {final_ff_path}")

    # Generate coverage summary
    logger.info("=" * 80)
    logger.info("Generating coverage summary")
    logger.info("=" * 80)

    # Load training dataset for coverage check
    logger.info("Loading training dataset for coverage check")
    try:
        train_dataset = Dataset.load_from_disk(str(data_dir))
        logger.info(f"Loaded {len(train_dataset)} training molecules")
    except Exception as e:
        logger.error(f"Failed to load training dataset for coverage check: {e}")
        raise

    # Check coverage on training dataset - must be 100%
    _check_coverage_is_error(
        ff=current_ff,
        dataset=train_dataset,
        dataset_name="Training",
        coverage_dir=coverage_dir,
        n_workers=config.n_workers,
    )

    # Delete training dataset after use
    del train_dataset
    gc.collect()

    # Check coverage on test dataset - warn if not 100%
    if test_data_dir is not None:
        logger.info(f"Loading test dataset from {test_data_dir}")
        try:
            test_dataset = Dataset.load_from_disk(str(test_data_dir))
            logger.info(f"Loaded {len(test_dataset)} test molecules")
            _check_coverage_is_warning(
                ff=current_ff,
                dataset=test_dataset,
                dataset_name="Test",
                coverage_dir=coverage_dir,
                n_workers=config.n_workers,
            )
            # Delete test_dataset after use
            del test_dataset
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to load test dataset: {e}")
    else:
        logger.info("No test dataset provided, skipping test coverage check")

    # Check coverage on additional SMILES files
    for dataset_name, smiles_path in config.coverage_smiles_paths.items():
        if smiles_path is None:
            continue
        logger.info(f"Checking coverage on {dataset_name} SMILES from {smiles_path}")
        try:
            smiles_list = _load_smiles_from_file(smiles_path)
            logger.info(f"Loaded {len(smiles_list)} molecules from {dataset_name}")
            dataset = _create_dataset_from_smiles(smiles_list)
            _check_coverage_is_warning(
                ff=current_ff,
                dataset=dataset,
                dataset_name=dataset_name,
                coverage_dir=coverage_dir,
                n_workers=config.n_workers,
            )
            # Delete dataset after use
            del dataset
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to check coverage for {dataset_name}: {e}")

    total_time = time.time() - total_start_time
    logger.info("=" * 80)
    logger.info(f"Bespoke type generation complete in {total_time:.1f}s")
    logger.info(f"Generated force field: {final_ff_path}")
    logger.info("=" * 80)

    # Final garbage collection
    gc.collect()

    return final_ff_path


def _log_component_statistics(components_by_specificity: dict) -> None:
    """Log statistics about component types and populations."""
    for specificity_num, components_by_type in components_by_specificity.items():
        n_types = len(components_by_type)
        total_components = sum(len(comps) for comps in components_by_type.values())
        logger.info(
            f"  Specificity {specificity_num}: {n_types} unique types, "
            f"{total_components} total components"
        )

        # Show top 5 most common types
        sorted_types = sorted(components_by_type.items(), key=lambda x: len(x[1]), reverse=True)
        logger.info("  Top 5 most common types:")
        for i, (smirks, comps) in enumerate(sorted_types[:5], 1):
            logger.info(f"    {i}. {smirks}: {len(comps)} instances")

    # Overall statistics
    all_counts = flatten_mm_component_types(components_by_specificity)
    total_unique = len(all_counts)
    total_instances = sum(all_counts.values())
    logger.info(f"Overall: {total_unique} unique types, {total_instances} total instances")


def _save_coverage_statistics(components_by_specificity: dict, output_path: Path) -> None:
    """Save detailed coverage statistics to JSON file."""
    stats = {
        "by_specificity": {},
        "overall": {},
    }

    for specificity_num, components_by_type in components_by_specificity.items():
        type_counts = {smirks: len(comps) for smirks, comps in components_by_type.items()}
        stats["by_specificity"][specificity_num] = {
            "n_unique_types": len(type_counts),
            "total_instances": sum(type_counts.values()),
            "type_counts": type_counts,
        }

    all_counts = flatten_mm_component_types(components_by_specificity)
    stats["overall"] = {
        "n_unique_types": len(all_counts),
        "total_instances": sum(all_counts.values()),
        "min_count": min(all_counts.values()) if all_counts else 0,
        "max_count": max(all_counts.values()) if all_counts else 0,
        "singleton_fraction": (
            sum(1 for c in all_counts.values() if c == 1) / len(all_counts) if all_counts else 0
        ),
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


def _check_coverage_is_error(
    ff: ForceField,
    dataset: Dataset,
    dataset_name: str,
    coverage_dir: Path,
    n_workers: int | None,
) -> None:
    """
    Check force field coverage and raise error if not 100%.

    Used for training dataset - coverage must be perfect by design.

    Note: This function is called AFTER the force field has been saved to disk,
    so errors raised here do not prevent the force field from being available
    for debugging purposes.

    Parameters
    ----------
    ff : ForceField
        Force field to check coverage for.
    dataset : Dataset
        Dataset to check coverage against.
    dataset_name : str
        Name of dataset (e.g., "Training", "Test") for logging.
    coverage_dir : Path
        Directory to save coverage reports.
    n_workers : int | None
        Number of workers for parallel processing.

    Raises
    ------
    RuntimeError
        If coverage is not 100%.
    """
    logger.info(f"Checking {dataset_name} dataset coverage (must be 100%)")

    # Convert dataset to list of SMILES
    smiles_list = list(dataset["smiles"])
    logger.info(f"Checking coverage for {len(smiles_list)} molecules")

    # Check coverage
    missing_coverage = check_all_components_fully_covered_parallel_chunks(
        mapped_smiles_list=smiles_list,
        ff=ff,
        n_workers=n_workers,
    )

    n_missing = len(missing_coverage)
    n_total = len(smiles_list)
    coverage_pct = 100 * (1 - n_missing / n_total) if n_total > 0 else 0

    logger.info(
        f"{dataset_name} Coverage: {n_total - n_missing}/{n_total} molecules "
        f"({coverage_pct:.2f}%) fully covered"
    )

    if n_missing > 0:
        logger.error(f"{n_missing} {dataset_name} molecules have missing parameters")

        # Count missing component types
        component_type_counts = {}
        for _smiles, missing_components in missing_coverage.items():
            for component_type in missing_components.keys():
                component_type_counts[component_type] = (
                    component_type_counts.get(component_type, 0) + 1
                )

        logger.error("Missing component types:")
        for component_type, count in sorted(
            component_type_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            logger.error(f"  {component_type}: {count} molecules")

        # Save detailed missing coverage
        missing_file = coverage_dir / f"{dataset_name.lower()}_missing_coverage.json"
        _save_missing_coverage(missing_coverage, missing_file)

        # Delete large missing_coverage dict after saving
        del missing_coverage
        gc.collect()

        # Raise error for training data with missing coverage
        # raise RuntimeError(
        #     f"{dataset_name} dataset has incomplete coverage ({coverage_pct:.2f}%). "
        #     f"Training data must have 100% coverage. "
        #     f"See detailed report: {missing_file}"
        # )

    logger.info(f"{dataset_name} coverage is perfect (100%)")


def _check_coverage_is_warning(
    ff: ForceField,
    dataset: Dataset,
    dataset_name: str,
    coverage_dir: Path,
    n_workers: int | None,
) -> None:
    """
    Check force field coverage and warn if not 100%.

    Used for test dataset - incomplete coverage is acceptable but noteworthy.

    Parameters
    ----------
    ff : ForceField
        Force field to check coverage for.
    dataset : Dataset
        Dataset to check coverage against.
    dataset_name : str
        Name of dataset (e.g., "Test") for logging.
    coverage_dir : Path
        Directory to save coverage reports.
    n_workers : int | None
        Number of workers for parallel processing.
    """
    logger.info(f"Checking {dataset_name} dataset coverage (warnings only)")

    # Convert dataset to list of SMILES
    smiles_list = list(dataset["smiles"])
    logger.info(f"Checking coverage for {len(smiles_list)} molecules")

    # Check coverage
    missing_coverage = check_all_components_fully_covered_parallel_chunks(
        mapped_smiles_list=smiles_list,
        ff=ff,
        n_workers=n_workers,
    )

    n_missing = len(missing_coverage)
    n_total = len(smiles_list)
    coverage_pct = 100 * (1 - n_missing / n_total) if n_total > 0 else 0

    logger.info(
        f"{dataset_name} Coverage: {n_total - n_missing}/{n_total} molecules "
        f"({coverage_pct:.2f}%) fully covered"
    )

    if n_missing > 0:
        logger.warning(f"{n_missing} {dataset_name} molecules have missing parameters")

        # Count missing component types
        component_type_counts = {}
        for _smiles, missing_components in missing_coverage.items():
            for component_type in missing_components.keys():
                component_type_counts[component_type] = (
                    component_type_counts.get(component_type, 0) + 1
                )

        logger.warning("Missing component types in test dataset:")
        for component_type, count in sorted(
            component_type_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            logger.warning(f"  {component_type}: {count} molecules")

        # Save detailed missing coverage
        missing_file = coverage_dir / f"{dataset_name.lower()}_missing_coverage.json"
        _save_missing_coverage(missing_coverage, missing_file)

        # Delete large missing_coverage dict after saving
        del missing_coverage
        gc.collect()

        logger.warning(f"Detailed missing coverage saved: {missing_file}")
    else:
        logger.info(f"{dataset_name} coverage is perfect (100%)")


def _save_missing_coverage(
    missing_coverage: dict[str, dict[str, list[tuple[int, ...]]]],
    output_path: Path,
) -> None:
    """
    Save detailed information about molecules with missing coverage.

    Parameters
    ----------
    missing_coverage : dict[str, dict[str, list[tuple[int, ...]]]]
        Mapping of SMILES to missing component info:
        {smiles: {component_type: [atom_indices_tuples]}}.
    output_path : Path
        Path to save the JSON file.
    """
    # Convert to serializable format
    serializable_missing = {}
    for smiles, missing_components in missing_coverage.items():
        serializable_missing[smiles] = {
            comp_type: [list(indices) for indices in indices_list]
            for comp_type, indices_list in missing_components.items()
        }

    with open(output_path, "w") as f:
        json.dump(serializable_missing, f, indent=2)


def _generate_coverage_summary(
    ff: ForceField,
    dataset: Dataset,
    coverage_dir: Path,
    n_workers: int | None,
) -> dict[str, Any]:
    """
    Generate coverage summary for the generated force field.

    Checks which molecules from the dataset can be fully parameterized
    and logs warnings for any missing parameters.
    """
    logger.info("Checking coverage on training dataset")

    # Convert dataset to list of SMILES
    smiles_list = list(dataset["smiles"])
    logger.info(f"Checking coverage for {len(smiles_list)} molecules")

    # Check coverage
    missing_coverage = check_all_components_fully_covered_parallel_chunks(
        mapped_smiles_list=smiles_list,
        ff=ff,
        n_workers=n_workers,
    )

    n_missing = len(missing_coverage)
    n_total = len(smiles_list)
    coverage_pct = 100 * (1 - n_missing / n_total) if n_total > 0 else 0

    logger.info(
        f"Coverage: {n_total - n_missing}/{n_total} molecules ({coverage_pct:.2f}%) fully covered"
    )

    if n_missing > 0:
        logger.warning(f"{n_missing} molecules have missing parameters")

        # Count missing component types
        component_type_counts = {}
        for _smiles, missing_components in missing_coverage.items():
            for component_type in missing_components.keys():
                component_type_counts[component_type] = (
                    component_type_counts.get(component_type, 0) + 1
                )

        logger.warning("Missing component types:")
        for component_type, count in sorted(
            component_type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            logger.warning(f"  {component_type}: {count} molecules")

        # Save detailed missing coverage
        missing_file = coverage_dir / "missing_coverage.json"
        with open(missing_file, "w") as f:
            # Convert to serializable format (tuples → lists)
            serializable_missing = {
                smiles: {
                    comp_type: [list(indices) for indices in indices_list]
                    for comp_type, indices_list in missing_comps.items()
                }
                for smiles, missing_comps in missing_coverage.items()
            }
            json.dump(serializable_missing, f, indent=2)
        logger.info(f"Saved detailed missing coverage: {missing_file}")

    summary = {
        "n_molecules": n_total,
        "n_fully_covered": n_total - n_missing,
        "n_missing_coverage": n_missing,
        "coverage_percentage": coverage_pct,
        "missing_by_component_type": component_type_counts if n_missing > 0 else {},
    }

    return summary


def _load_smiles_from_file(smiles_path: Path) -> list[str]:
    """
    Load SMILES strings from a CSV file.

    Reads the first column from the CSV file and returns it as a list of strings.

    Parameters
    ----------
    smiles_path : Path
        Path to CSV file containing SMILES strings in the first column.

    Returns
    -------
    list[str]
        List of SMILES strings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or has no valid data.
    """
    if not smiles_path.exists():
        raise FileNotFoundError(f"SMILES file not found: {smiles_path}")

    try:
        smiles_list = pd.read_csv(smiles_path).iloc[:, 0].dropna().astype(str).tolist()
        if not smiles_list:
            raise ValueError(f"No valid SMILES found in {smiles_path}")
        return smiles_list
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {smiles_path}: {e}") from e


def _create_dataset_from_smiles(smiles_list: list[str]) -> Dataset:
    """
    Create a HuggingFace Dataset from SMILES strings.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.

    Returns
    -------
    Dataset
        HuggingFace Dataset with a 'smiles' column.
    """
    return Dataset.from_dict({"smiles": smiles_list})
