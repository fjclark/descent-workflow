"""Train the force field. Mainly from https://github.com/jthorton/SPICE-SMEE/"""

import datasets
import tensorboardX
import torch

import descent.optim
import descent.targets.energy
import descent.utils.loss
import descent.utils.reporting
import descent.train
import math
import more_itertools
import pprint
from pathlib import Path
from tbparse import SummaryReader
import matplotlib.pyplot as plt
import smee

from loguru import logger

from models import WorkflowConfig


def write_metrics(
    epoch: int,
    loss: torch.Tensor,
    loss_energy: torch.Tensor,
    loss_forces: torch.Tensor,
    prior_k_torsions: torch.Tensor,
    loss_test: torch.Tensor,
    loss_test_energy: torch.Tensor,
    loss_test_forces: torch.Tensor,
    prior_k_torsions_test: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
)-> None:
    logger.info(
        f"epoch={epoch} loss_train={loss.detach().item():.6f}, loss_test={loss_test.detach().item():.6f}",
        flush=True,
    )

    writer.add_scalar("loss", loss.detach().item(), epoch)
    writer.add_scalar("loss_energy", loss_energy.detach().item(), epoch)
    writer.add_scalar("loss_forces", loss_forces.detach().item(), epoch)
    writer.add_scalar("prior_k_torsions", prior_k_torsions.detach().item(), epoch)

    writer.add_scalar("loss_test", loss_test.detach().item(), epoch)
    writer.add_scalar("loss_test_energy", loss_test_energy.detach().item(), epoch)
    writer.add_scalar("loss_test_forces", loss_test_forces.detach().item(), epoch)
    writer.add_scalar(
        "prior_k_torsions_test", prior_k_torsions_test.detach().item(), epoch
    )

    writer.add_scalar("rmse_energy", math.sqrt(loss_energy.detach().item()), epoch)
    writer.add_scalar("rmse_forces", math.sqrt(loss_forces.detach().item()), epoch)

    writer.add_scalar(
        "rmse_test_energy", math.sqrt(loss_test_energy.detach().item()), epoch
    )
    writer.add_scalar(
        "rmse_test_forces", math.sqrt(loss_test_forces.detach().item()), epoch
    )

    writer.flush()


def get_datasets(config: WorkflowConfig) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Get the training and test datasets."""
    test_dataset_names = [
        k
        for k in config.filtered_data_dir.iterdir()
        if k.is_dir() and "test" in k.name.lower()
    ]
    train_dataset_names = [
        k
        for k in config.filtered_data_dir.iterdir()
        if k.is_dir() and k not in test_dataset_names
    ]

    train_dataset = datasets.concatenate_datasets(
        [datasets.Dataset.load_from_disk(source) for source in train_dataset_names]
    )

    test_dataset = datasets.concatenate_datasets(
        [datasets.Dataset.load_from_disk(source) for source in test_dataset_names]
    )

    return train_dataset, test_dataset


def get_param_and_attr_configs(
    config: WorkflowConfig,
) -> tuple[descent.train.ParameterConfig, descent.train.AttributeConfig]:
    """Prepare parameter and attribute configurations."""
    # try:
    #     if config.parameters["Angles"]["limits"]["angle"][-1].lower() == "pi":
    #         config.parameters["Angles"]["limits"]["angle"][-1] = math.pi
    # except KeyError:
    #     pass

    parameters = {
        k: descent.train.ParameterConfig(**v) for k, v in config.parameters.items()
    }
    attributes = {
        k: descent.train.AttributeConfig(**v) for k, v in config.attributes.items()
    }
    return parameters, attributes


def setup_experiment_dir(config: WorkflowConfig) -> Path:
    """Set up the experiment directory."""
    experiment_dir = config.fit_dir
    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment directory {experiment_dir} already exists. Please remove it or choose a different experiment name."
        )
    experiment_dir.mkdir(parents=True)
    return experiment_dir


def write_hparams(writer: tensorboardX.SummaryWriter, config: WorkflowConfig) -> None:
    """Write hyperparameters to TensorBoard."""
    for v in tensorboardX.writer.hparams(
        {
            "optimizer": "Adam",
            "lr": config.learning_rate,
            "energy_weight": config.energy_weight,
            "force_weight": config.force_weight,
            "torsion_weight": config.torsion_weight,
        },
        {},
    ):
        writer.file_writer.add_summary(v)


def get_initial_torsions(force_field: smee.TensorForceField) -> torch.Tensor:
    """Get initial torsion values for regularization."""
    k_col_torsion = force_field.potentials_by_type[
        "ProperTorsions"
    ].parameter_cols.index("k")
    return (
        force_field.potentials_by_type["ProperTorsions"]
        .parameters[:, k_col_torsion]
        .detach()
    )


def get_losses(
    config: WorkflowConfig,
    trainable: descent.train.Trainable,
    x: torch.Tensor,
    dataset: datasets.Dataset,
    topologies: dict[str, smee.TensorTopology],
    initial_torsions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute losses for the current epoch."""
    ff = trainable.to_force_field(x)

    total_loss, energy_loss, force_loss, grad = (
        torch.zeros(size=(1,), device=x.device.type),
        torch.zeros(size=(1,), device=x.device.type),
        torch.zeros(size=(1,), device=x.device.type),
        None,
    )

    for batch_ids in more_itertools.batched(
        [i for i in range(len(dataset))], config.batch_size
    ):
        batch = dataset.select(indices=batch_ids)
        true_batch_size = len(dataset)

        cuda_batch = prepare_cuda_batch(batch)

        e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
            cuda_batch, ff, topologies, "mean"
        )

        batch_loss_energy = ((e_pred - e_ref) ** 2).sum() / true_batch_size
        batch_loss_force = ((f_pred - f_ref) ** 2).sum() / true_batch_size

        batch_loss = (
            config.energy_weight * batch_loss_energy
            + config.force_weight * batch_loss_force
        )

        (batch_grad,) = torch.autograd.grad(batch_loss, x, create_graph=True)
        batch_grad = batch_grad.detach()
        grad = batch_grad if grad is None else grad + batch_grad

        total_loss += batch_loss.detach()
        energy_loss += batch_loss_energy.detach()
        force_loss += batch_loss_force.detach()

    torsion_prior = compute_torsion_prior(config, ff, initial_torsions, x, grad)
    if config.torsion_weight > 0.0:
        total_loss += torsion_prior.detach()

    x.grad = grad

    return total_loss, energy_loss, force_loss, torsion_prior


def prepare_cuda_batch(batch: list) -> list:
    """Prepare a batch for CUDA."""
    cuda_batch = []
    for entry in batch:
        for key, value in entry.items():
            if key in ["coords", "energy", "forces"]:
                entry[key] = value.to("cuda")
            else:
                entry[key] = value
        cuda_batch.append(entry)
    return cuda_batch


def compute_torsion_prior(
    config: WorkflowConfig,
    ff: smee.TensorForceField,
    initial_torsions: torch.Tensor,
    x: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    """Compute torsion prior and update gradient."""
    if config.torsion_weight > 0.0:
        k_col_torsion = ff.potentials_by_type["ProperTorsions"].parameter_cols.index(
            "k"
        )
        torsion_prior = (
            ff.potentials_by_type["ProperTorsions"].parameters[:, k_col_torsion]
            - initial_torsions
        ).square().sum() * config.torsion_weight
        (torsion_grad,) = torch.autograd.grad(torsion_prior, x, create_graph=False)
        grad += torsion_grad.detach()
    else:
        torsion_prior = torch.tensor([0.0], requires_grad=True)
    return torsion_prior


def plot_loss(configs: list[WorkflowConfig], output_path: Path) -> None:
    """Plot the training and test total, force, and energy loss."""

    dfs = {
        config.experiment_name: SummaryReader(config.fit_dir).scalars
        for config in configs
    }

    # Three plots on one level
    with plt.style.context("ggplot"):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        scalar_names = {
            "Total Loss": {"Train": "loss", "Test": "loss_test"},
            "Force Loss": {"Train": "loss_forces", "Test": "loss_test_forces"},
            "Energy Loss": {"Train": "loss_energy", "Test": "loss_test_energy"},
        }

        for i, (title, scalars) in enumerate(scalar_names.items()):
            for experiment_name, df in dfs.items():
                for label, scalar in scalars.items():
                    df_filtered = df[df["tag"] == scalar]
                    linestyle = "-" if label == "Train" else "--"
                    axs[i].plot(
                        df_filtered["step"],
                        df_filtered["value"],
                        label=f"{experiment_name} {label}",
                        alpha=0.8,
                        linestyle=linestyle,
                    )

            axs[i].set_title(title)
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel("Loss")
            if i == 2:
                axs[i].legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=300)
        plt.close(fig)


def train(config: WorkflowConfig) -> None:
    """Use batching to fit to the SPICE dataset on a single GPU!"""

    force_field, topologies = torch.load(config.torch_ffs_and_tops_path)
    dataset_train, dataset_test = get_datasets(config)
    parameters, attributes = get_param_and_attr_configs(config)
    force_field = force_field.to("cuda")
    topologies = {
        smiles: topology.to("cuda") for smiles, topology in topologies.items()
    }

    logger.info(f"Training with {len(dataset_train)} entries")
    logger.info("Parameters: " + pprint.pformat(parameters))
    logger.info("Attributes: " + pprint.pformat(attributes))

    trainable = descent.train.Trainable(
        force_field=force_field, parameters=parameters, attributes=attributes
    )

    experiment_dir = setup_experiment_dir(config)
    config.to_file(experiment_dir / "workflow_config.yaml")

    x = trainable.to_values().to("cuda")

    with tensorboardX.SummaryWriter(str(experiment_dir)) as writer:
        optimizer = torch.optim.Adam([x], lr=config.learning_rate, amsgrad=True)
        write_hparams(writer, config)

        initial_torsions = get_initial_torsions(force_field)

        for i in range(config.n_epochs):
            losses: dict[str, list[torch.Tensor]] = {"train": [], "test": []}

            for dataset_name, dataset in {
                "train": dataset_train,
                "test": dataset_test,
            }.items():
                losses[dataset_name].extend(
                    get_losses(
                        config,
                        trainable,
                        x,
                        dataset,
                        topologies,
                        initial_torsions,
                    )
                )
                if dataset_name == "train":
                    optimizer.step()

                optimizer.zero_grad()

            write_metrics(
                i,
                *losses["train"],
                *losses["test"],
                writer,
            )

            plot_loss(
                [config],
                config.fit_dir / "losses.png",
            )

            if i % 10 == 0:
                torch.save(
                    trainable.to_force_field(x),
                    experiment_dir / f"force-field-epoch-{i}.pt",
                )
                plot_loss(
                    [config],
                    config.fit_dir / "losses.png",
                )

    torch.save(trainable.to_force_field(x), config.final_torch_ff_path)
