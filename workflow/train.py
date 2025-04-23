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

from loguru import logger

from models import WorkflowConfig
from get_data import ESPALOMA_SOURCES


def write_metrics(
    epoch: int,
    loss: torch.Tensor,
    loss_energy: torch.Tensor,
    loss_forces: torch.Tensor,
    prior_k_torsions: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
):
    logger.info(f"epoch={epoch} loss={loss.detach().item():.6f}", flush=True)

    writer.add_scalar("loss", loss.detach().item(), epoch)
    writer.add_scalar("loss_energy", loss_energy.detach().item(), epoch)
    writer.add_scalar("loss_forces", loss_forces.detach().item(), epoch)
    writer.add_scalar("prior_k_torsions", prior_k_torsions.detach().item(), epoch)

    writer.add_scalar("rmse_energy", math.sqrt(loss_energy.detach().item()), epoch)
    writer.add_scalar("rmse_forces", math.sqrt(loss_forces.detach().item()), epoch)
    writer.flush()


def train(config: WorkflowConfig):
    """Use batching to fit to the SPICE dataset on a single GPU!"""

    force_field, topologies = torch.load(config.torch_ffs_and_tops_path)
    dataset = datasets.concatenate_datasets(
        [
            datasets.Dataset.load_from_disk(config.filtered_data_dir / source)
            for source in ESPALOMA_SOURCES
        ]
    )

    # convert to cuda
    force_field = force_field.to("cuda")
    topologies = {
        smiles: topology.to("cuda") for smiles, topology in topologies.items()
    }

    # edit the Angles config to be PI
    try:
        if config.parameters["Angles"]["limits"]["angle"][-1].lower() == "pi":
            config.parameters["Angles"]["limits"]["angle"][-1] = math.pi
    except KeyError:
        pass

    parameters = {
        k: descent.train.ParameterConfig(**v) for k, v in config.parameters.items()
    }
    attributes = {
        k: descent.train.AttributeConfig(**v) for k, v in config.attributes.items()
    }

    logger.info(f"Training with {len(dataset)} entries")
    logger.info(pprint.pformat(parameters))
    logger.info(pprint.pformat(attributes))

    trainable = descent.train.Trainable(
        force_field=force_field, parameters=parameters, attributes=attributes
    )

    experiment_dir = config.fit_dir
    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment directory {experiment_dir} already exists. Please remove it or choose a different experiment name."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # write the config to the file
    config.to_file(experiment_dir / "workflow_config.yaml")

    x = trainable.to_values().to("cuda")

    with tensorboardX.SummaryWriter(str(experiment_dir)) as writer:
        optimizer = torch.optim.Adam([x], lr=config.learning_rate, amsgrad=True)

        # write hparams
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

        # grab the initial torsions to reg against if requested
        k_col_torsion = force_field.potentials_by_type[
            "ProperTorsions"
        ].parameter_cols.index("k")
        initial_torsions = (
            force_field.potentials_by_type["ProperTorsions"]
            .parameters[:, k_col_torsion]
            .detach()
        )

        # main fitting loop
        for i in range(config.n_epochs):
            ff = trainable.to_force_field(x)

            # set up tensors to hold the gradients
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

                # Convert batch to cuda
                cuda_batch = []
                for entry in batch:
                    for key, value in entry.items():
                        if key in ["coords", "energy", "forces"]:
                            entry[key] = value.to("cuda")
                        else:
                            entry[key] = value
                    cuda_batch.append(entry)

                e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                    cuda_batch, ff, topologies, "mean"
                )
                # L2 loss
                batch_loss_energy = ((e_pred - e_ref) ** 2).sum() / true_batch_size
                batch_loss_force = ((f_pred - f_ref) ** 2).sum() / true_batch_size

                # Equal sum of L2 loss on energies and forces
                batch_loss = (
                    config.energy_weight * batch_loss_energy
                    + config.force_weight * batch_loss_force
                )

                (batch_grad,) = torch.autograd.grad(batch_loss, x, create_graph=True)
                batch_grad = batch_grad.detach()
                if grad is None:
                    grad = batch_grad
                else:
                    grad += batch_grad

                # keep sum of squares to report MSE at the end
                total_loss += batch_loss.detach()
                energy_loss += batch_loss_energy.detach()
                force_loss += batch_loss_force.detach()

            if config.torsion_weight > 0.0:
                # after all batches work out the distance from the initial torsion values and times by the weight.
                torsion_prior = (
                    ff.potentials_by_type["ProperTorsions"].parameters[:, k_col_torsion]
                    - initial_torsions
                ).square().sum() * config.torsion_weight
                (torsion_grad,) = torch.autograd.grad(
                    torsion_prior, x, create_graph=False
                )
                # add the torsion gradient to the total gradient
                grad += torsion_grad.detach()
                # add the torsion loss to the total loss
                total_loss += torsion_prior.detach()

            else:
                torsion_prior = torch.tensor([0.0], requires_grad=True)

            # move the grad to the right place
            x.grad = grad

            write_metrics(
                epoch=i,
                loss=total_loss,
                loss_energy=energy_loss,
                loss_forces=force_loss,
                prior_k_torsions=torsion_prior,
                writer=writer,
            )

            optimizer.step()
            optimizer.zero_grad()

            if i % 10 == 0:
                torch.save(
                    trainable.to_force_field(x),
                    experiment_dir / f"force-field-epoch-{i}.pt",
                )

    torch.save(trainable.to_force_field(x), config.final_torch_ff_path)
