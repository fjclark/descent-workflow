# Bespoke Type Generation

This module generates force field parameters with bespoke SMIRKS patterns based on training data. It analyzes the molecular mechanics components (bonds, angles, torsions) present in your dataset and creates specific parameter types at configurable specificity levels.

## Overview

The type generation workflow:

1. **Extract Components**: Identifies all molecular mechanics components (bonds, angles, proper/improper torsions) in the filtered training dataset
2. **Filter Patterns**: Removes unwanted SMIRKS patterns if specified
3. **Hierarchical Organization**: Groups components by specificity levels with population cutoffs
4. **Force Field Assembly**: Adds generated parameters to base force field
5. **Coverage Analysis**: Reports statistics and identifies missing parameter coverage

## Usage

### Enable Type Generation

Add a `type_generation_config` section to your workflow configuration YAML:

```yaml
type_generation_config:
  component_types:
    - "Bond"
    - "Angle"
    - "ProperTorsion"
    - "ImproperTorsion"
  
  cutoff_population: 10
  
  # Named specificity configurations for each component
  bond_specificities:
    Standard:
      atom_include_ring_info: false
      bond_specificity: "STANDARD"
  
  angle_specificities:
    TerminalWildcard:  # Wildcards on terminal atoms
      atom_terminal_behavior: "WILDCARD"
      bond_specificity: "NON_CENTRAL_WILDCARD"
    Standard:  # Explicit atoms everywhere
      atom_include_ring_info: false
      bond_specificity: "STANDARD"
  
  torsion_specificities:
    TerminalWildcard:
      atom_terminal_behavior: "WILDCARD"
      bond_specificity: "NON_CENTRAL_WILDCARD"
    Standard:
      atom_include_ring_info: false
      bond_specificity: "STANDARD"
  
  improper_specificities:
    Standard:
      atom_include_ring_info: false
      bond_specificity: "STANDARD"
```

See `configs/example_with_type_generation.yaml` for a complete example.

### Disable Type Generation

Either omit the `type_generation_config` section or set it to `null`:

```yaml
type_generation_config: null
```

The workflow will use the starting force field directly without generating bespoke types.

## Configuration Options

### Required Fields

- **`component_types`**: List of component types to generate (`"Bond"`, `"Angle"`, `"ProperTorsion"`, `"ImproperTorsion"`)
- **`cutoff_population`**: Minimum number of instances required to create a parameter type
- **Specificity dicts**: Named configurations for each component type:
  - `bond_specificities`: Dict mapping names to SMIRKSFactory configs
  - `angle_specificities`: Dict mapping names to SMIRKSFactory configs
  - `torsion_specificities`: Dict mapping names to SMIRKSFactory configs
  - `improper_specificities`: Dict mapping names to SMIRKSFactory configs

### Specificity Configuration Options

Each specificity configuration dict accepts:

**Atom Configuration:**
- `atom_include_ring_info` (bool): Include ring membership info in patterns
- `atom_terminal_behavior` (str): How to handle terminal atoms
  - `"STANDARD"`: Explicit atom types (e.g., `[#6X4:1]`)
  - `"WILDCARD"`: Wildcard on terminals (e.g., `[*:1]`)
  - `"H_NO_H"`: Distinguish H vs non-H (e.g., `[#1:1]` / `[!#1:1]`)
- `atom_bonded_behavior` (str): How to include bonded atoms (advanced)

**Bond Configuration:**
- `bond_include_ring_info` (bool): Include ring membership for bonds
- `bond_specificity` (str): Bond pattern detail level
  - `"STANDARD"`: Explicit bond types (e.g., `-`, `=`)
  - `"NON_CENTRAL_WILDCARD"`: Wildcards on non-central bonds
  - `"WILDCARD"`: All bonds as wildcards (`~`)

### Optional Fields

- **`unwanted_smirks_paths`**: Dict mapping component types to force field paths with patterns to exclude
- **`extra_parameters_paths`**: Dict mapping component types to force field paths with extra parameters
- **`n_workers`**: Number of parallel workers (None = use all cores)
- **`resume`**: Resume from checkpoints if True

## Specificity Levels

Specificity is controlled by the configuration dicts you provide. The order of keys determines the hierarchy (earlier = less specific, processed last as fallback):

### Example Progression

For angles with this configuration:
```yaml
angle_specificities:
  TerminalWildcard:  # Processed last (least specific fallback)
    atom_terminal_behavior: "WILDCARD"
  TerminalHnoH:      # Processed second (mid-level)
    atom_terminal_behavior: "H_NO_H"
  Standard:          # Processed first (most specific)
    atom_include_ring_info: false
```

This creates patterns like:
- **Standard**: `[#6X4:1]-[#6X3:2]-[#6X4:3]` (explicit atoms/bonds everywhere)
- **TerminalHnoH**: `[!#1:1]-[#6X3:2]-[!#1:3]` (H vs non-H on terminals)
- **TerminalWildcard**: `[*:1]-[#6X3:2]-[*:3]` (wildcards on terminals)

The algorithm tries the most specific patterns first, falling back to more general patterns for components with lower populations.

## Workflow Integration

When type generation is enabled, the workflow proceeds as:

1. **get_data**: Load molecular SMILES
2. **parameterise**: Create initial torch force fields with starting force field
3. **filter_and_cluster**: Filter training data
4. **generate_bespoke_types**: Analyze filtered data and create bespoke force field
5. **prepare_for_training**: Re-parameterize with bespoke force field
6. **train**: Train force field parameters

## Output Files

Type generation creates several outputs in `{data_dir}/generated_types/`:

### Force Field
- `bespoke_types.offxml`: Final force field with bespoke types

### Checkpoints (in `checkpoints/`)
Enable resumption and debugging:
- `{component}_components.pkl`: Extracted components for each type
- `{component}_by_specificity.pkl`: Hierarchical groupings at each level
- `ff_after_{component}.offxml`: Force field after adding each component type

### Coverage Reports (in `coverage/`)
Diagnostics and statistics:
- `{component}_coverage_stats.json`: Detailed statistics per specificity level
- `coverage_summary.csv`: Summary matrix of all components
- `missing_coverage_{component}.txt`: Details of parameters without coverage

## Module Structure

- **`config.py`**: `TypeGenConfig` dataclass for configuration validation
- **`orchestrate.py`**: Main `generate_bespoke_types()` function
- **`molecular_classes.py`**: Data classes for molecular mechanics components
- **`process_mmcomponents.py`**: Component extraction and organization
- **`process_SMIRKS.py`**: SMIRKS pattern generation and force field manipulation
- **`coverage.py`**: Coverage analysis and reporting

## Examples

### Basic Configuration

Generate types for all component types with standard specificity:

```yaml
type_generation_config:
  component_types: ["Bond", "Angle", "ProperTorsion", "ImproperTorsion"]
  cutoff_population: 10
  
  bond_specificities:
    Standard:
      atom_include_ring_info: false
      bond_specificity: "STANDARD"
  
  angle_specificities:
    Standard:
      atom_include_ring_info: false
      bond_specificity: "STANDARD"
  
  torsion_specificities:
    Standard:
      atom_include_ring_info: false
      bond_specificity: "STANDARD"
  
  improper_specificities:
    Standard:
      atom_include_ring_info: false
      bond_specificity: "STANDARD"
```

### Advanced Configuration

Include multiple specificity levels and unwanted pattern filtering:

```yaml
type_generation_config:
  component_types: ["Bond", "Angle", "ProperTorsion"]
  cutoff_population: 15
  
  bond_specificities:
    Standard:
      atom_include_ring_info: false
      bond_specificity: "STANDARD"
  
  angle_specificities:
    TerminalWildcard:
      atom_terminal_behavior: "WILDCARD"
      bond_specificity: "NON_CENTRAL_WILDCARD"
    TerminalHnoH:
      atom_terminal_behavior: "H_NO_H"
      bond_specificity: "NON_CENTRAL_WILDCARD"
    Standard:
      atom_include_ring_info: true
      bond_specificity: "STANDARD"
 
  torsion_specificities:
    TerminalWildcard:
      atom_terminal_behavior: "WILDCARD"
      bond_specificity: "NON_CENTRAL_WILDCARD"
    Standard:
      atom_include_ring_info: true
      bond_specificity: "STANDARD"
  
  # Exclude patterns from linear torsions force field
  unwanted_smirks_paths:
    ProperTorsion: "output_ff/trained_linear_torsions.offxml"
```

### Torsions Only with Ring Info

Generate only proper torsion types with ring awareness:

```yaml
type_generation_config:
  component_types: ["ProperTorsion"]
  cutoff_population: 5
  
  torsion_specificities:
    TerminalWildcard:
      atom_terminal_behavior: "WILDCARD"
      bond_specificity: "NON_CENTRAL_WILDCARD"
    RingAware:
      atom_include_ring_info: true
      bond_include_ring_info: true
      bond_specificity: "STANDARD"
    HighlySpecific:
      atom_include_ring_info: true
      atom_bonded_behavior: "CENTRAL_EXPLICIT_ATOMS"
      bond_include_ring_info: true
      bond_specificity: "STANDARD"
```

## Troubleshooting

### Type Generation Fails

Check the logs in `{data_dir}/generated_types/`. Common issues:
- Missing RDKit or OpenFF Toolkit dependencies
- Invalid SMIRKS patterns in unwanted_smirks file
- Insufficient population at higher specificity levels

### Coverage Warnings

If you see warnings about missing coverage:
- Lower specificity levels or cutoff_population
- Add missing patterns to extra_parameters
- Check that base force field has necessary parameter handlers

### Re-parameterization Slow

Re-parameterization after type generation can be slow for large datasets. This is expected as it must regenerate force field parameters for all filtered molecules with the new types.

## Implementation Notes

- Type generation uses multiprocessing with spawn context for HPC compatibility
- Checkpoints enable resumption if the workflow is interrupted
- Coverage reports are warnings only - they don't fail the workflow
- The effective_ff_path property automatically selects bespoke or starting FF downstream
