"""Train the force field. Mainly from https://github.com/jthorton/SPICE-SMEE/"""

import json
import math
import pprint
import shutil
from pathlib import Path
from typing import Any, Iterator

import datasets
import descent.optim
import descent.targets.energy
import descent.train
import descent.utils.loss
import descent.utils.reporting
import matplotlib.pyplot as plt
import more_itertools
import smee
import tensorboardX
import torch
from .convert_ff import pt_file_to_offxml_with_description
from loguru import logger
from .models import WorkflowConfig
from openff.interchange.models import PotentialKey
from tbparse import SummaryReader


def write_train_metrics(
    step: float,
    loss: torch.Tensor,
    loss_energy: torch.Tensor,
    loss_forces: torch.Tensor,
    prior_k_torsions: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
) -> None:
    logger.info(f"step={step} loss_train={loss.detach().item():.6f}", flush=True)

    writer.add_scalar("loss", loss.detach().item(), step)
    writer.add_scalar("loss_energy", loss_energy.detach().item(), step)
    writer.add_scalar("loss_forces", loss_forces.detach().item(), step)
    writer.add_scalar("prior_k_torsions", prior_k_torsions.detach().item(), step)

    writer.add_scalar("rmse_energy", math.sqrt(loss_energy.detach().item()), step)
    writer.add_scalar("rmse_forces", math.sqrt(loss_forces.detach().item()), step)

    writer.flush()


def write_test_metrics(
    step: float,
    loss_test: torch.Tensor,
    loss_test_energy: torch.Tensor,
    loss_test_forces: torch.Tensor,
    prior_k_torsions_test: torch.Tensor,
    writer: tensorboardX.SummaryWriter,
) -> None:
    logger.info(f"step={step} loss_test={loss_test.detach().item():.6f}", flush=True)

    writer.add_scalar("loss_test", loss_test.detach().item(), step)
    writer.add_scalar("loss_test_energy", loss_test_energy.detach().item(), step)
    writer.add_scalar("loss_test_forces", loss_test_forces.detach().item(), step)
    writer.add_scalar("prior_k_torsions_test", prior_k_torsions_test.detach().item(), step)

    writer.add_scalar("rmse_test_energy", math.sqrt(loss_test_energy.detach().item()), step)
    writer.add_scalar("rmse_test_forces", math.sqrt(loss_test_forces.detach().item()), step)

    writer.flush()


def get_datasets(config: WorkflowConfig) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Get the training and test datasets.

    Concatenates the ``*train*`` / ``*test*`` split directories under
    ``config.filtered_data_dir``. Backup directories (``*backup*``), temporary directories
    (``*.tmp``) and hidden directories are skipped, so in-place retrofits (e.g.
    ``scripts/filter_processed_omol25_by_conformer_quality.py``) that leave
    ``*.prefilter-backup`` / ``*.preconfqualfilter-backup`` copies next to the splits are not
    silently concatenated back into training.
    """

    def is_split_dir(path: Path) -> bool:
        name = path.name.lower()
        return (
            path.is_dir()
            and not path.name.startswith(".")
            and "backup" not in name
            and not name.endswith(".tmp")
        )

    split_dirs = [k for k in config.filtered_data_dir.iterdir() if is_split_dir(k)]
    test_dataset_names = [k for k in split_dirs if "test" in k.name.lower()]
    train_dataset_names = [k for k in split_dirs if k not in test_dataset_names]
    logger.info(
        f"Train split dirs: {[k.name for k in train_dataset_names]}; "
        f"test split dirs: {[k.name for k in test_dataset_names]}"
    )

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
    parameters = {}
    for k, v in config.parameters.items():
        if "include" in v:
            v["include"] = [PotentialKey(id=key_id) for key_id in v["include"]]  # type: ignore[call-arg]
        if "exclude" in v:
            v["exclude"] = [PotentialKey(id=key_id) for key_id in v["exclude"]]  # type: ignore[call-arg]  # type: ignore[call-arg]

        parameters[k] = descent.train.ParameterConfig(**v)

    attributes = {k: descent.train.AttributeConfig(**v) for k, v in config.attributes.items()}
    return parameters, attributes


def setup_experiment_dir(config: WorkflowConfig) -> Path:
    """Set up the experiment directory."""
    experiment_dir: Path = config.fit_dir
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
    k_col_torsion = force_field.potentials_by_type["ProperTorsions"].parameter_cols.index("k")
    return force_field.potentials_by_type["ProperTorsions"].parameters[:, k_col_torsion].detach()


def save_nonfinite_entries(
    batch: list[Any],
    ff: smee.TensorForceField,
    topologies: dict[str, smee.TensorTopology],
    out_dir: Path,
    max_saved: int = 50,
) -> list[dict[str, Any]]:
    """Isolate, log and save batch entries that give non-finite energy/force differences.

    ``predict`` concatenates the whole batch, so a per-entry re-run is needed to attribute
    a NaN/Inf to a specific molecule. For each entry we check ``torch.isfinite`` separately
    on the input data (coords/energy/forces) and on the predictions (e_pred/f_pred), which
    distinguishes a bad source datum from a diverged force field. Offending entries are
    logged and saved to ``out_dir`` (a HF dataset plus a ``nonfinite_entries.json``
    summary) for inspection.

    Returns the list of records (one per offending entry).
    """
    records: list[dict[str, Any]] = []

    for idx, entry in enumerate(batch):
        smiles = entry["smiles"]
        reasons: list[str] = []

        for key in ("coords", "energy", "forces"):
            if not torch.isfinite(entry[key]).all():
                reasons.append(f"input:{key}")

        try:
            e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                [entry], ff, topologies, "mean"
            )
            for name, tensor in (
                ("e_ref", e_ref),
                ("e_pred", e_pred),
                ("f_ref", f_ref),
                ("f_pred", f_pred),
            ):
                if not torch.isfinite(tensor).all():
                    reasons.append(f"pred:{name}")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"pred:exception:{type(exc).__name__}")

        if reasons:
            records.append({"index": idx, "smiles": smiles, "reasons": reasons})
            logger.error(f"Non-finite entry: smiles={smiles} reasons={reasons}")

    input_side = sum(1 for r in records if any(x.startswith("input:") for x in r["reasons"]))
    if records and input_side == 0:
        logger.error(
            "All non-finite entries are prediction-side; the force field has likely "
            "diverged (exploding gradients) rather than a single bad datum."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "nonfinite_entries.json", "w") as file:
        json.dump(records, file, indent=2)

    if records:
        saved = records[:max_saved]
        entry_dicts = [
            {
                "smiles": batch[r["index"]]["smiles"],
                "coords": batch[r["index"]]["coords"].detach().cpu(),
                "energy": batch[r["index"]]["energy"].detach().cpu(),
                "forces": batch[r["index"]]["forces"].detach().cpu(),
            }
            for r in saved
        ]
        try:
            descent.targets.energy.create_dataset(entry_dicts).save_to_disk(
                str(out_dir / "dataset")
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Could not save non-finite entries as a HF dataset: {exc}")
        logger.error(
            f"Saved {len(saved)} non-finite entries (of {len(records)}) to {out_dir}"
        )

    return records


def save_outlier_entries(
    records: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    out_dir: Path,
    max_saved: int = 200,
) -> None:
    """Persist entries dropped by the loss outlier filter, for later inspection.

    A lightweight identifier record (epoch/tag/smiles/RMSEs) for *every* drop is appended to
    ``out_dir/log.jsonl`` so the full history is inspectable without a file explosion. The full
    structures (coords/energy/forces) of up to ``max_saved`` *unique* dropped SMILES are saved to a
    HF dataset at ``out_dir/dataset`` so the offending molecules can be re-run and diagnosed
    (a molecule dropped every step is stored once).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "log.jsonl", "a") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    saved_path = out_dir / "saved_smiles.json"
    saved_smiles: set[str] = (
        set(json.loads(saved_path.read_text())) if saved_path.exists() else set()
    )
    if len(saved_smiles) >= max_saved:
        return

    new_entries: list[dict[str, Any]] = []
    new_smiles: set[str] = set()
    for entry in entries:
        smiles = entry["smiles"]
        if smiles in saved_smiles or smiles in new_smiles:
            continue
        new_entries.append(entry)
        new_smiles.add(smiles)
        if len(saved_smiles) + len(new_smiles) >= max_saved:
            break

    if not new_entries:
        return

    dataset_path = out_dir / "dataset"
    tmp_path = out_dir / "dataset.tmp"
    try:
        new_dataset = descent.targets.energy.create_dataset(new_entries)
        if dataset_path.exists():
            existing = datasets.Dataset.load_from_disk(str(dataset_path))
            combined = datasets.concatenate_datasets([existing, new_dataset])
        else:
            combined = new_dataset
        # save_to_disk cannot overwrite a directory it is currently reading from, so write to a
        # temp dir and swap it in.
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        combined.save_to_disk(str(tmp_path))
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        tmp_path.rename(dataset_path)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Could not save outlier entries as a HF dataset: {exc}")
        return

    saved_smiles |= new_smiles
    saved_path.write_text(json.dumps(sorted(saved_smiles)))


def get_losses(
    config: WorkflowConfig,
    trainable: descent.train.Trainable,
    x: torch.Tensor,
    dataset: datasets.Dataset,
    topologies: dict[str, smee.TensorTopology],
    initial_torsions: torch.Tensor,
    epoch: int = 0,
    tag: str = "train",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute losses for the current epoch."""
    ff = trainable.to_force_field(x)

    total_loss, energy_loss, force_loss, grad = (
        torch.zeros(size=(1,), device=x.device.type),
        torch.zeros(size=(1,), device=x.device.type),
        torch.zeros(size=(1,), device=x.device.type),
        None,
    )

    filt = config.loss_outlier_filter
    dropped_records: list[dict[str, Any]] = []
    dropped_entries: list[dict[str, Any]] = []

    for batch in dataset_batch_iterator(dataset, config.batch_size):
        true_batch_size = len(dataset)

        cuda_batch = prepare_cuda_batch(batch)

        # A single batched predict is used whether or not the filter is on. predict already loops
        # per molecule internally (distinct topologies can't be tensor-batched), so filtering does
        # not add prediction cost; we just slice the residuals per entry to build a keep-mask.
        e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
            cuda_batch, ff, topologies, "mean"
        )
        e_res_sq = (e_pred - e_ref) ** 2  # per-conformer squared error, shape (n_confs_total,)
        f_res_sq = ((f_pred - f_ref) ** 2).sum(dim=-1)  # per-atom squared force error, (n_rows,)

        if filt is None:
            batch_loss_energy = e_res_sq.sum() / true_batch_size
            batch_loss_force = f_res_sq.sum() / true_batch_size
        else:
            # Boolean keep-masks (detached); False = drop that entry's contributions from the loss.
            e_keep = torch.ones_like(e_res_sq, dtype=torch.bool)
            f_keep = torch.ones_like(f_res_sq, dtype=torch.bool)

            e_off = f_off = 0
            for entry in cuda_batch:
                n_conf = len(entry["energy"])
                n_rows = entry["forces"].numel() // 3

                # With predict(normalize=True), an entry's segment sum-of-squares equals its
                # mean-squared error in physical units, so the sqrt is the physical per-entry RMSE
                # (kcal/mol and kcal/mol/A).
                e_rmse = float(e_res_sq[e_off : e_off + n_conf].detach().sum().sqrt())
                f_rmse = float(f_res_sq[f_off : f_off + n_rows].detach().sum().sqrt())

                # NaN comparisons are always False, so non-finite entries are NOT dropped here;
                # they flow into batch_loss and trip the non-finite guard below (we surface NaNs
                # rather than hide them).
                is_outlier = (
                    filt.max_energy_rmse is not None and e_rmse > filt.max_energy_rmse
                ) or (filt.max_force_rmse is not None and f_rmse > filt.max_force_rmse)

                if is_outlier:
                    e_keep[e_off : e_off + n_conf] = False
                    f_keep[f_off : f_off + n_rows] = False
                    logger.warning(
                        f"Dropping outlier entry ({tag}, epoch {epoch}): "
                        f"smiles={entry['smiles']} e_rmse={e_rmse:.3f} kcal/mol "
                        f"f_rmse={f_rmse:.3f} kcal/mol/A"
                    )
                    dropped_records.append(
                        {
                            "epoch": epoch,
                            "tag": tag,
                            "smiles": entry["smiles"],
                            "e_rmse": e_rmse,
                            "f_rmse": f_rmse,
                        }
                    )
                    dropped_entries.append(
                        {
                            "smiles": entry["smiles"],
                            "coords": entry["coords"].detach().cpu(),
                            "energy": entry["energy"].detach().cpu(),
                            "forces": entry["forces"].detach().cpu(),
                        }
                    )

                e_off += n_conf
                f_off += n_rows

            batch_loss_energy = (e_res_sq * e_keep).sum() / true_batch_size
            batch_loss_force = (f_res_sq * f_keep).sum() / true_batch_size

        batch_loss = (
            config.energy_weight * batch_loss_energy + config.force_weight * batch_loss_force
        )

        if not torch.isfinite(batch_loss):
            out_dir = config.fit_dir / "nonfinite_entries" / f"epoch{epoch}_{tag}"
            logger.error(
                f"Non-finite loss ({tag}, epoch {epoch}); isolating offending entries "
                f"-> {out_dir}"
            )
            save_nonfinite_entries(cuda_batch, ff, topologies, out_dir)
            raise RuntimeError(
                f"Non-finite loss encountered ({tag}, epoch {epoch}). Offending entries "
                f"saved to {out_dir} for inspection."
            )

        # Dropped entries are masked to zero, so batch_loss still depends on x (grad is just zero
        # for their parameters); the autograd call is always well-defined.
        (batch_grad,) = torch.autograd.grad(batch_loss, x, create_graph=True)
        batch_grad = batch_grad.detach()
        grad = batch_grad if grad is None else grad + batch_grad

        total_loss += batch_loss.detach()
        energy_loss += batch_loss_energy.detach()
        force_loss += batch_loss_force.detach()

    if dropped_records:
        logger.warning(
            f"Dropped {len(dropped_records)} outlier entries from the loss ({tag}, epoch {epoch})."
        )
        save_outlier_entries(
            dropped_records,
            dropped_entries,
            config.fit_dir / "outlier_entries",
            max_saved=filt.max_saved,  # type: ignore[union-attr]
        )

    torsion_prior = compute_torsion_prior(config, ff, initial_torsions, x, grad)  # type: ignore[arg-type]
    if config.torsion_weight > 0.0:
        total_loss += torsion_prior.detach()

    x.grad = grad

    return total_loss, energy_loss, force_loss, torsion_prior


def dataset_batch_iterator(
    dataset: datasets.Dataset, batch_size: int, shuffle: bool = True
) -> Iterator[datasets.Dataset]:
    """Yield batches of data from the dataset."""
    if shuffle:
        dataset = dataset.shuffle()

    for batch_ids in more_itertools.batched(list(range(len(dataset))), batch_size):
        yield dataset.select(indices=batch_ids)


def prepare_cuda_batch(batch: list[Any]) -> list[Any]:
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
        # k_col_torsion = ff.potentials_by_type["ProperTorsions"].parameter_cols.index(
        #     "k"
        # )
        # torsion_prior = (
        #     ff.potentials_by_type["ProperTorsions"].parameters[:, k_col_torsion]
        #     - initial_torsions
        # ).square().sum() * config.torsion_weight
        # (torsion_grad,) = torch.autograd.grad(torsion_prior, x, create_graph=False)
        # grad += torsion_grad.detach()
        k_col_torsion = ff.potentials_by_type["ImproperTorsions"].parameter_cols.index("k")
        # Regularise above 10 kcal mol-1
        torsion_prior = (
            torch.clamp(
                ff.potentials_by_type["ImproperTorsions"].parameters[:, k_col_torsion] - 10.0,
                min=0.0,
            )
            .square()
            .sum()
            * config.torsion_weight
        )
        (torsion_grad,) = torch.autograd.grad(torsion_prior, x, create_graph=False)
        grad += torsion_grad.detach()
    else:
        torsion_prior = torch.tensor([0.0], requires_grad=True)
    result: torch.Tensor = torsion_prior
    return result


def plot_loss(configs: list[WorkflowConfig], output_path: Path) -> None:
    """Plot the training and test total, force, and energy loss."""
    dfs = {config.experiment_name: SummaryReader(config.fit_dir).scalars for config in configs}

    # Three plots on one level
    with plt.style.context("ggplot"):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
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
            axs[i].set_xlabel("Batch")
            axs[i].set_ylabel("Loss")
            if i == 2:
                axs[i].legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))

        fig.savefig(str(output_path), dpi=900)
        plt.close(fig)


def train(config: WorkflowConfig) -> None:
    """Use batching to fit to the SPICE dataset on a single GPU!"""
    force_field, topologies = torch.load(config.final_torch_ffs_and_tops_path)
    dataset_train, dataset_test = get_datasets(config)

    # Filter out the test dataset to only include molecules present in the topologies
    test_smiles = set(dataset_test["smiles"])
    topology_smiles = set(topologies.keys())
    filtered_test_smiles = test_smiles.intersection(topology_smiles)

    # Temporary hacks to deal with a bit of missing test/ train coverage
    # TODO: Fix the pipeline and remove this
    initial_len_test, initial_len_train = len(dataset_test), len(dataset_train)
    dataset_test = dataset_test.filter(lambda x: x["smiles"] in filtered_test_smiles)
    dataset_train = dataset_train.filter(lambda x: x["smiles"] in topology_smiles)
    final_len_test, final_len_train = len(dataset_test), len(dataset_train)
    logger.info(
        f"Filtered test dataset from {initial_len_test} to {final_len_test} entries; "
        f"Filtered train dataset from {initial_len_train} to {final_len_train} entries."
    )

    # Build the (optionally subsampled) test set used for during-training evaluation once, with a
    # fixed seed, so the test loss is comparable across steps.
    if config.test_subset_size is not None and len(dataset_test) > config.test_subset_size:
        dataset_test_eval = dataset_test.shuffle(seed=config.test_subset_seed).select(
            range(config.test_subset_size)
        )
        logger.info(
            f"Evaluating on a fixed test subset of {len(dataset_test_eval)} of "
            f"{len(dataset_test)} entries (every {config.test_eval_interval} step(s))."
        )
    else:
        dataset_test_eval = dataset_test
        logger.info(
            f"Evaluating on the full test set of {len(dataset_test_eval)} entries "
            f"(every {config.test_eval_interval} step(s))."
        )

    parameters, attributes = config.parameters, config.attributes
    force_field = force_field.to("cuda")
    topologies = {smiles: topology.to("cuda") for smiles, topology in topologies.items()}

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
            n_minibatches = math.ceil(len(dataset_train) / config.minibatch_size)
            for j, minibatch in enumerate(
                dataset_batch_iterator(dataset_train, config.minibatch_size)
            ):
                logger.info(f"Epoch {i}, minibatch {j} of {n_minibatches}")

                step = i * n_minibatches + j

                # Train pass: compute the loss/gradient and take an optimizer step.
                train_losses = get_losses(
                    config,
                    trainable,
                    x,
                    minibatch,
                    topologies,
                    initial_torsions,
                    epoch=i,
                    tag=f"train_mb{j}",
                )
                optimizer.step()
                optimizer.zero_grad()
                write_train_metrics(step, *train_losses, writer)  # type: ignore[call-arg]

                # Test pass: evaluate the (subsampled) test set only every N steps, purely for
                # logging. Its gradient is discarded.
                if step % config.test_eval_interval == 0:
                    test_losses = get_losses(
                        config,
                        trainable,
                        x,
                        dataset_test_eval,
                        topologies,
                        initial_torsions,
                        epoch=i,
                        tag=f"test_mb{j}",
                    )
                    optimizer.zero_grad()
                    write_test_metrics(step, *test_losses, writer)  # type: ignore[call-arg]

                    plot_loss(
                        [config],
                        config.fit_dir / "losses.png",
                    )

            if i % 1 == 0:
                torch.save(
                    trainable.to_force_field(x),
                    experiment_dir / f"force-field-epoch-{i}.pt",
                )

    # Save in pt and offxml format, saving in the output ff directory
    torch.save(trainable.to_force_field(x), config.final_torch_ff_path)
    pt_file_to_offxml_with_description(config)
    logger.info(f"Saved force field to {config.final_torch_ff_path} and {config.output_ff_path}")
