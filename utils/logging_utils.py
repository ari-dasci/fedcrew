"""Logging utilities for experiments."""

import os
from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from config import INSTRUMENTATION_VERSION, ExperimentConfig


@dataclass
class LoggerState:
    """Container for logger state."""

    writer: Optional[SummaryWriter]
    run: Optional[wandb.sdk.wandb_run.Run]
    device: str


def setup_logging(config: ExperimentConfig) -> LoggerState:
    """Initialize logging infrastructure.

    Args:
        config: Experiment configuration.

    Returns:
        LoggerState containing writer, wandb run, and device.
    """
    # Check CUDA availability
    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda"

    # Initialize tensorboard writer
    writer = (
        SummaryWriter(config.get_summary_writer_filename())
        if not config.no_log
        else None
    )

    # Initialize wandb
    run = None
    if not config.no_log:
        wandb.login()
        run = wandb.init(
            project="crp_aggregation",
            name=config.get_wandb_run_name(),
            group=config.wandb_group,
            tags=config.get_wandb_tags(),
            config={
                "instrumentation_version": INSTRUMENTATION_VERSION,
                "dataset": config.dataset,
                "clients": config.clients,
                "fedcrew": config.fedcrew,
                "epochs": config.epochs,
                "batchsize": config.batchsize,
                "samples": config.samples,
                "fedprox": config.fedprox,
                "fednova": config.fednova,
                "rounds": config.rounds,
                "l1": config.l1,
                "l2": config.l2,
                "alpha": config.alpha,
                "l2_fc": config.l2_fc,
                "seed": config.seed,
                "causal_mode": config.causal_mode,
                "causal_crp": config.causal_crp,
                "causal_logits": config.causal_logits,
                "anchor_selection": config.anchor_selection,
                "anchor_seed": config.anchor_seed,
                "wandb_group": config.wandb_group,
                "moon": config.moon,
                "moon_mu": config.moon_mu,
                "moon_tau": config.moon_tau,
                "aggregator": config.get_aggregator_name(),
            },
        )

    print(f"Running options: {config}")

    return LoggerState(writer=writer, run=run, device=device)


def log_client_metrics(
    logger: LoggerState,
    client_metrics: list,
    round_number: int,
) -> None:
    """Log client-level metrics to tensorboard and wandb.

    Args:
        logger: Logger state containing writer and wandb run.
        client_metrics: List of (loss, accuracy, confusion_matrix) tuples.
        round_number: Current federation round.
    """
    if logger.writer is None or logger.run is None:
        return

    losses = [loss for loss, _, _ in client_metrics]
    accs = [acc for _, acc, _ in client_metrics]

    if losses:
        avg_loss = sum(losses) / len(losses)
        median_loss = np.median(losses)
        max_loss = max(losses)
        min_loss = min(losses)

        logger.writer.add_scalar("Average Client Loss", avg_loss, round_number)
        logger.writer.add_scalar("Median Client Loss", median_loss, round_number)
        logger.writer.add_scalar("Max Client Loss", max_loss, round_number)
        logger.writer.add_scalar("Min Client Loss", min_loss, round_number)

        logger.run.log(
            {
                "Client/Average Loss": avg_loss,
                "Client/Median Loss": median_loss,
                "Client/Max Loss": max_loss,
                "Client/Min Loss": min_loss,
            },
            step=round_number,
        )

    if accs:
        avg_acc = sum(accs) / len(accs)
        median_acc = np.median(accs)
        max_acc = max(accs)
        min_acc = min(accs)

        logger.writer.add_scalar("Average Client Accuracy", avg_acc, round_number)
        logger.writer.add_scalar("Median Client Accuracy", median_acc, round_number)
        logger.writer.add_scalar("Max Client Accuracy", max_acc, round_number)
        logger.writer.add_scalar("Min Client Accuracy", min_acc, round_number)

        logger.run.log(
            {
                "Client/Average Accuracy": avg_acc,
                "Client/Median Accuracy": median_acc,
                "Client/Max Accuracy": max_acc,
                "Client/Min Accuracy": min_acc,
            },
            step=round_number,
        )


def log_server_metrics(
    logger: LoggerState,
    loss: float,
    acc: float,
    confusion_matrix: np.ndarray,
    round_number: int,
) -> None:
    """Log server-level metrics to tensorboard and wandb.

    Args:
        logger: Logger state containing writer and wandb run.
        loss: Test loss.
        acc: Test accuracy.
        confusion_matrix: Confusion matrix as numpy array.
        round_number: Current federation round.
    """
    if logger.writer is None or logger.run is None:
        return

    logger.writer.add_scalar("Loss", loss, round_number)
    logger.writer.add_scalar("Accuracy", acc, round_number)

    logger.run.log(
        {"Server/Loss": loss, "Server/Accuracy": acc},
        step=round_number,
    )

    # Log confusion matrix visualization
    fig = plt.figure()
    plt.imshow(confusion_matrix)
    plt.show()
    logger.writer.add_figure("confusion_matrix", fig, round_number)
    logger.run.log(
        {"Server/Confusion_Matrix": wandb.Image(fig)},
        step=round_number,
    )
    plt.close(fig)


def log_alignment_metrics(
    logger: LoggerState,
    alignment: dict,
    round_number: int,
) -> None:
    """Log CKA / classifier-head-divergence alignment metrics to tensorboard/wandb.

    Args:
        logger: Logger state containing writer and wandb run.
        alignment: Dict with "cka_mean"/"fc_divergence_mean" scalars (see
            `utils.alignment_utils.compute_client_alignment`).
        round_number: Current federation round.
    """
    if logger.writer is None or logger.run is None:
        return

    cka_mean = float(alignment["cka_mean"])
    fc_divergence_mean = float(alignment["fc_divergence_mean"])

    logger.writer.add_scalar("Alignment/CKA_Mean", cka_mean, round_number)
    logger.writer.add_scalar(
        "Alignment/FC_Divergence_Mean", fc_divergence_mean, round_number
    )

    logger.run.log(
        {
            "Alignment/CKA_Mean": cka_mean,
            "Alignment/FC_Divergence_Mean": fc_divergence_mean,
        },
        step=round_number,
    )


def log_samples(
    logger: LoggerState,
    samples: list,
    round_number: int,
) -> None:
    """Log sample images to tensorboard and wandb.

    Args:
        logger: Logger state containing writer and wandb run.
        samples: List of sample tensors/arrays.
        round_number: Current federation round.
    """
    if logger.writer is None or logger.run is None:
        return

    for i, sample in enumerate(samples):
        logger.writer.add_image(f"sample/{i}", sample, 0)
        logger.run.log(
            {f"Samples/Sample_{i}": wandb.Image(sample)},
            step=round_number,
        )


def log_crp_heatmap(
    logger: LoggerState,
    img,
    client_id: Union[int, str],
    sample_id: int,
    round_number: int,
) -> None:
    """Log CRP heatmap to tensorboard and wandb.

    Args:
        logger: Logger state containing writer and wandb run.
        img: PIL image of the heatmap.
        client_id: Client identifier.
        sample_id: Sample identifier.
        round_number: Current federation round.
    """
    if logger.writer is None or logger.run is None:
        return

    from torchvision import transforms

    transform = transforms.PILToTensor()
    tensor_img = transform(img)

    logger.writer.add_image(
        f"crp/client_{client_id}/image_{sample_id}",
        tensor_img,
        round_number,
    )
    logger.run.log(
        {f"CRP/Client_{client_id}/Image_{sample_id}": wandb.Image(img)},
        step=round_number,
    )


def save_predictions(
    predictions: dict,
    config: ExperimentConfig,
    round_number: int,
) -> None:
    """Save final-round per-sample predictions/logits to disk.

    Unlike the other `log_*` functions in this module, this is not gated on
    `logger.writer`/`logger.run` being set up (i.e. not skipped by `--no_log`)
    since it persists the scientific artifact needed for later per-class
    metric computation, not a monitoring convenience.

    Args:
        predictions: Dict with "logits", "preds", "targets" tensors.
        config: Experiment configuration.
        round_number: Current federation round.
    """
    path = config.get_predictions_path(round_number)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(predictions, path)


def finish_logging(logger: LoggerState) -> None:
    """Clean up logging resources.

    Args:
        logger: Logger state containing writer and wandb run.
    """
    if logger.run is not None:
        logger.run.finish()
    if logger.writer is not None:
        logger.writer.close()
