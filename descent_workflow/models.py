"""Pydantic models. These will be stored as, and read from yaml files."""

from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional
from descent.train import AttributeConfig, ParameterConfig

import yaml
from pydantic import BaseModel, Field, validator, model_validator

__version__ = version("descent-workflow")


class WorkflowConfig(BaseModel):
    """Configuration for the workflow."""

    version: str = Field(
        description="Version of the workflow config. Must match major.minor version of descent_workflow."
    )
    experiment_name: str = Field(default="", description="Name of the experiment.")
    experiment_description: str = Field(default="", description="Description of the experiment.")
    data_dir: Path = Field(
        default=Path("data/espaloma"), description="Directory where the data is stored."
    )
    get_data_fn: str = Field(
        default="get_data.get_data_espaloma", description="Function to get the data."
    )
    get_data_output_smiles: Path = Field(
        default=Path("data/espaloma/data-raw/smiles.json"),
        description="Output that snakemake will look for.",
    )

    starting_force_field_path: Path = Field(
        default=Path("input_ff/lj-sage-2-2-msm-0-expanded-torsions.offxml"),
        description="Path to the starting force field.",
    )

    filter_and_cluster_fn: str = Field(
        default="filter.filter_and_cluster_espaloma",
        description="Function to filter and cluster the data.",
    )

    batch_size: int = Field(default=500, description="Batch size for training.")

    minibatch_size: int = Field(default=256, description="Minibatch size for training.")

    n_epochs: int = Field(default=1000, description="Number of epochs for training.")

    learning_rate: float = Field(default=0.01, description="Learning rate for training.")

    energy_weight: float = Field(default=1.0, description="Weight for the energy loss.")

    force_weight: float = Field(default=1.0, description="Weight for the force loss.")

    torsion_weight: float = Field(default=0.0, description="Weight for the torsion regularization.")

    torsion_reg: str = Field(default="l1", description="Regularization for the torsion loss.")

    attributes: dict[str, AttributeConfig] = Field(
        default_factory=dict, description="Trainable attributes for the force field."
    )

    parameters: dict[str, ParameterConfig] = Field(
        default_factory=lambda: {
            "LinearBonds": ParameterConfig(
                cols=["k1", "k2"],
                scales={"k1": 0.0028, "k2": 0.028},
                limits={"k1": [None, None], "k2": [None, None]},
            ),
            "LinearAngles": ParameterConfig(
                cols=["k1", "k2"],
                scales={"k1": 0.0115, "k2": 0.0115},
                limits={"k1": [None, None], "k2": [None, None]},
            ),
            "ProperTorsions": ParameterConfig(cols=["k"], scales={"k": 8.72}),
            "ImproperTorsions": ParameterConfig(
                cols=["k"],
                scales={"k": 2.03},
            ),
        },
        description="Trainable parameters for the force field.",
    )
    type_generation_protocol_name: Optional[str] = Field(
        default=None,
        description=(
            "Name of the type generation protocol to use. Only required if type_generation_config is provided. "
            "This is used to name the type generation output directory and checkpoints."
        ),
    )

    type_generation_config: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Configuration for bespoke type generation. If None, type generation "
            "is skipped and the workflow uses the starting force field directly. "
            "If provided, will be validated as a TypeGenConfig when used."
        ),
    )

    model_config = {
        "populate_by_name": True,  # So we automatically get Paths from strings
    }

    @validator("parameters")
    def check_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Make sure that if we have LinearBonds, we also have LinearAngles."""
        if "LinearBonds" in v and "LinearAngles" not in v:
            raise ValueError("If you have LinearBonds, you must also have LinearAngles.")
        if "LinearAngles" in v and "LinearBonds" not in v:
            raise ValueError("If you have LinearAngles, you must also have LinearBonds.")
        return v

    @model_validator(mode="after")
    def check_type_generation_fields(self) -> "WorkflowConfig":
        """Validate that type_generation_protocol_name and type_generation_config are both provided or both None."""
        protocol_name = self.type_generation_protocol_name
        config = self.type_generation_config

        if (protocol_name is None) != (config is None):
            raise ValueError(
                "You must provide both type_generation_protocol_name and type_generation_config, or neither."
            )
        return self

    @validator("version")
    def check_version(cls, v: str) -> str:
        """Validate that config version matches workflow major.minor version."""
        # Parse config version
        try:
            config_parts = v.split(".")
            if len(config_parts) < 2:
                raise ValueError(
                    f"Config version '{v}' must be in format 'major.minor' or 'major.minor.patch'"
                )
            config_major = int(config_parts[0])
            config_minor = int(config_parts[1])
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Config version '{v}' must be in format 'major.minor' or 'major.minor.patch'"
            ) from e

        # Parse workflow version
        workflow_version = __version__.split("+")[0]  # Remove any build metadata
        try:
            workflow_parts = workflow_version.split(".")
            if len(workflow_parts) < 2:
                # Development version, skip validation
                return v
            workflow_major = int(workflow_parts[0])
            workflow_minor = int(workflow_parts[1])
        except (ValueError, IndexError):
            # Can't parse workflow version, skip validation
            return v

        # Check major.minor match
        if config_major != workflow_major or config_minor != workflow_minor:
            raise ValueError(
                f"Config version {config_major}.{config_minor} does not match "
                f"workflow version {workflow_major}.{workflow_minor}. "
                f"Please update your config file to use version {workflow_major}.{workflow_minor}.x"
            )

        return v

    @property
    def input_ff_name(self) -> str:
        # Remove the offxml extension from the force field path
        return self.starting_force_field_path.stem

    @property
    def torch_ffs_and_tops_path(self) -> Path:
        return self.data_dir / f"{self.input_ff_name}_ff_and_tops.pt"

    @property
    def filtered_data_dir(self) -> Path:
        return self.data_dir / f"data-filtered-{self.input_ff_name}"

    @property
    def output_ff_name(self) -> str:
        return f"{self.experiment_name}.offxml"

    @property
    def output_ff_dir(self) -> Path:
        return Path("output_ff")

    @property
    def output_ff_path(self) -> Path:
        return self.output_ff_dir / self.output_ff_name

    @property
    def benchmarking_dir(self) -> Path:
        return Path("benchmarking/output") / self.experiment_name

    @property
    def output_torch_ff_name(self) -> str:
        return f"{self.experiment_name}.pt"

    @property
    def fit_dir(self) -> Path:
        return Path("fits") / self.experiment_name

    @property
    def final_torch_ff_path(self) -> Path:
        return self.fit_dir / self.output_torch_ff_name

    @property
    def linearise_harm(self) -> bool:
        """Whether to linearise the harmonic terms in the force field."""
        # We have validated to ensure that if we have LinearBonds, we also have LinearAngles
        return "LinearBonds" in self.parameters

    @property
    def type_gen_output_dir(self) -> Path:
        """Directory for type generation outputs, checkpoints, and coverage reports."""
        return (
            self.data_dir
            / f"type_generation_output_{self.input_ff_name}_{self.type_generation_protocol_name}"
        )

    @property
    def bespoke_types_ff_path(self) -> Path:
        """Path to the generated force field with bespoke types."""
        return self.type_gen_output_dir / "bespoke_types.offxml"

    @property
    def type_gen_checkpoint_dir(self) -> Path:
        """Directory for type generation checkpoints."""
        return self.type_gen_output_dir / "checkpoints"

    @property
    def effective_ff_path(self) -> Path:
        """
        Path to the effective force field to use for parameterization.

        Returns the bespoke types force field if type generation is enabled,
        otherwise returns the starting force field path.
        """
        if self.type_generation_config is not None:
            return self.bespoke_types_ff_path
        return self.starting_force_field_path

    @property
    def final_torch_ffs_and_tops_path(self) -> Path:
        """
        Get the torch force field and topologies path to use for training.

        Returns the re-parameterized force field with bespoke types if enabled,
        otherwise returns the original parameterized force field.
        """
        if self.type_generation_config is not None:
            # Re-parameterized with bespoke types
            return self.type_gen_output_dir / "final_ff_and_tops.pt"
        else:
            # Original parameterization without bespoke types
            return self.torch_ffs_and_tops_path

    @classmethod
    def from_file(cls, filename: str | Path) -> "WorkflowConfig":
        """Load the configuration from a YAML file."""
        with open(filename, "r") as f:
            data = yaml.safe_load(f)
            return cls(**data)

    def to_file(self, filename: str | Path) -> None:
        """Save the configuration to a YAML file with nice formatting."""
        with open(filename, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)
