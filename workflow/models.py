"""Pydantic models. These will be stored as, and read from yaml files."""

from pydantic import BaseModel, Field, validator
import yaml
from pathlib import Path


class WorkflowConfig(BaseModel):
    """
    Configuration for the workflow.
    """

    experiment_name: str = Field(default="", description="Name of the experiment.")
    experiment_description: str = Field(
        default="", description="Description of the experiment."
    )
    data_dir: Path = Field(
        default="data/espaloma", description="Directory where the data is stored."
    )
    get_data_fn: str = Field(
        default="get_data.get_data_espaloma", description="Function to get the data."
    )
    get_data_output_smiles: Path = Field(
        default="data/espaloma/data-raw/smiles.json",
        description="Output that snakemake will look for.",
    )

    starting_force_field_path: Path = Field(
        default="input_ff/lj-sage-2-2-msm-0-expanded-torsions.offxml",
        description="Path to the starting force field.",
    )

    filter_and_cluster_fn: str = Field(
        default="filter.filter_and_cluster_espaloma",
        description="Function to filter and cluster the data.",
    )

    batch_size: int = Field(default=500, description="Batch size for training.")

    n_epochs: int = Field(default=1000, description="Number of epochs for training.")

    learning_rate: float = Field(
        default=0.01, description="Learning rate for training."
    )

    energy_weight: float = Field(default=1.0, description="Weight for the energy loss.")

    force_weight: float = Field(default=1.0, description="Weight for the force loss.")

    torsion_weight: float = Field(
        default=0.0, description="Weight for the torsion regularization."
    )

    torsion_reg: str = Field(
        default="l1", description="Regularization for the torsion loss."
    )

    attributes: dict = Field(
        default_factory=dict, description="Trainable attributes for the force field."
    )

    parameters: dict = Field(
        default_factory=lambda: {
            "LinearBonds": {
                "cols": ["k1", "k2"],
                "scales": {"k1": 0.0028, "k2": 0.028},
                "limits": {"k1": [None, None], "k2": [None, None]},
            },
            "LinearAngles": {
                "cols": ["k1", "k2"],
                "scales": {"k1": 0.0115, "k2": 0.0115},
                "limits": {"k1": [None, None], "k2": [None, None]},
            },
            "ProperTorsions": {"cols": ["k"], "scales": {"k": 8.72}},
            "ImproperTorsions": {"cols": ["k"], "scales": {"k": 2.03}},
        },
        description="Trainable parameters for the force field.",
    )

    model_config = {
        "populate_by_name": True,  # So we automatically get Paths from strings
    }

    @validator("parameters")
    def check_parameters(cls, v):
        """Make sure that if we have LinearBonds, we also have LinearAngles."""
        if "LinearBonds" in v and "LinearAngles" not in v:
            raise ValueError(
                "If you have LinearBonds, you must also have LinearAngles."
            )
        if "LinearAngles" in v and "LinearBonds" not in v:
            raise ValueError(
                "If you have LinearAngles, you must also have LinearBonds."
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

    @classmethod
    def from_file(cls, filename: str | Path) -> "WorkflowConfig":
        """
        Load the configuration from a YAML file.
        """
        with open(filename, "r") as f:
            data = yaml.safe_load(f)
            return cls(**data)

    def to_file(self, filename: str | Path):
        """
        Save the configuration to a YAML file with nice formatting.
        """
        with open(filename, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)
