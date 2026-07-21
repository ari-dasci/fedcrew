"""Configuration module for experiment settings."""

from dataclasses import dataclass, field
from typing import List, Optional
import argparse
import warnings

# Bumped whenever the logged/saved artifact schema changes (new config fields,
# new checkpoint/prediction/CRP-map instrumentation, new aggregators, ...) so
# WandB runs from before/after such a change can be bulk-filtered by tag
# instead of parsing run names.
INSTRUMENTATION_VERSION = "fgcs-revision-v1"


@dataclass
class ExperimentConfig:
    """Configuration for a federated learning experiment.

    Attributes:
        dataset: Dataset to use (cifar_10, celeba, etc.)
        clients: Number of clients per round
        fedcrew: Whether to use FedCrew weighted aggregation
        lognum: Number for tensorboard log directory
        epochs: Number of epochs to train clients
        batchsize: Batch size for training
        clipgradients: Whether to clip gradients
        samples: Number of samples per class to select for CRP
        no_log: If True, disables logging
        fedprox: FedProx regularization factor (0.0 to disable)
        fednova: Whether to use FedNova
        rounds: Number of federation rounds
        l1: L1 regularization factor
        l2: L2 regularization factor (weight decay)
        alpha: Threshold for counting sample as correct in CRP
        l2_fc: L2 regularization factor for FC layer only
        seed: Optional random seed for reproducibility (None means no seed)
    """

    dataset: str
    clients: int
    fedcrew: bool
    lognum: int
    epochs: int
    batchsize: int
    clipgradients: bool
    samples: int
    no_log: bool
    fedprox: float
    fednova: bool
    rounds: int
    l1: float
    l2: float
    alpha: float
    l2_fc: float
    seed: Optional[int] = None
    causal_mode: str = "full"
    causal_crp: bool = True
    causal_logits: bool = True
    anchor_selection: str = "first"
    anchor_seed: Optional[int] = None
    wandb_group: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    predictions_dir: str = "predictions"
    crp_maps_dir: str = "crp_maps"
    alignment_dir: str = "alignment"
    save_final_artifacts: bool = True
    moon: bool = False
    moon_mu: float = 1.0
    moon_tau: float = 0.5
    wandb_tags: List[str] = field(default_factory=list)

    @property
    def causal(self) -> bool:
        """Backward compatibility property. Use fedcrew instead."""
        return self.fedcrew

    def get_aggregator_name(self) -> str:
        """Get the name of the aggregation method being used."""
        if self.fedcrew:
            return "fedcrew"
        elif self.fednova:
            return "fednova"
        elif self.moon:
            return "moon"
        elif self.fedprox > 0.0:
            return "fedprox"
        else:
            return "fedavg"

    def get_summary_writer_filename(self) -> str:
        """Generate tensorboard log directory path."""
        parts = [
            "fedcrew" if self.fedcrew else "avg",
            f"samples{self.samples}",
            "l1" if self.l1 > 0.0 else "",
            "l2" if self.l2 > 0.0 else "",
            f"lognum{self.lognum}" if self.lognum > 0 else "",
            "sgd" if self.epochs == 1 else "",
            f"alpha{self.alpha}",
            "l2_fc" if self.l2_fc > 0.0 else "",
            "seeded" if self.seed is not None else "",
            self.causal_mode if self.fedcrew else "",
            self.anchor_selection if self.fedcrew else "",
            f"anchor{self.anchor_seed}" if self.anchor_seed is not None else "",
            "moon" if self.moon else "",
            f"moon_mu{self.moon_mu}" if self.moon and self.moon_mu != 1.0 else "",
            f"moon_tau{self.moon_tau}" if self.moon and self.moon_tau != 0.5 else "",
        ]
        return f"runs/{self.dataset}/" + ".".join([part for part in parts if part])

    def get_wandb_run_name(self) -> str:
        """Generate Weights & Biases run name."""
        parts = [
            self.get_aggregator_name(),
            f"samples{self.samples}",
            "l1" if self.l1 > 0.0 else "",
            "l2" if self.l2 > 0.0 else "",
            f"lognum{self.lognum}" if self.lognum > 0 else "",
            f"client_epochs{self.epochs}",
            f"alpha{self.alpha}",
            "l2_fc" if self.l2_fc > 0.0 else "",
            "seeded" if self.seed is not None else "",
            self.causal_mode if self.fedcrew else "",
            self.anchor_selection if self.fedcrew else "",
            f"anchor{self.anchor_seed}" if self.anchor_seed is not None else "",
            f"moon_mu{self.moon_mu}" if self.moon and self.moon_mu != 1.0 else "",
            f"moon_tau{self.moon_tau}" if self.moon and self.moon_tau != 0.5 else "",
        ]
        return ".".join([part for part in parts if part])

    def get_wandb_tags(self) -> List[str]:
        """WandB tags for this run: the instrumentation-revision tag (so old vs.
        new runs can be bulk-filtered in one query) plus any user-supplied tags."""
        return [INSTRUMENTATION_VERSION, *self.wandb_tags]

    def get_checkpoint_path(self, round_number: int) -> str:
        """Path for saving the final global model checkpoint."""
        return (
            f"{self.checkpoint_dir}/{self.dataset}/{self.get_wandb_run_name()}"
            f"/round{round_number}.pt"
        )

    def get_predictions_path(self, round_number: int) -> str:
        """Path for saving final-round per-sample predictions/logits."""
        return (
            f"{self.predictions_dir}/{self.dataset}/{self.get_wandb_run_name()}"
            f"/round{round_number}.pt"
        )

    def get_crp_map_path(
        self, round_number: int, client_id: object, sample_id: int
    ) -> str:
        """Path for saving a raw final-round CRP relevance map tensor."""
        return (
            f"{self.crp_maps_dir}/{self.dataset}/{self.get_wandb_run_name()}"
            f"/round{round_number}/client_{client_id}/sample{sample_id}.pt"
        )

    def get_alignment_path(self, round_number: int) -> str:
        """Path for saving raw final-round CKA/fc-divergence matrices."""
        return (
            f"{self.alignment_dir}/{self.dataset}/{self.get_wandb_run_name()}"
            f"/round{round_number}.pt"
        )


def _create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the experiment."""
    parser = argparse.ArgumentParser(
        description="Conditional Relevance Propagation for Federated Learning (FedCrew)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "cifar_10",
            "cifar_10_non_iid",
            "celeba",
            "celeba_a",
            "celeba_m",
            "mnist_non_iid",
        ],
        default="cifar_10",
        help="Dataset to use",
    )
    parser.add_argument(
        "--clients", type=int, default=100, help="Number of clients per round"
    )
    parser.add_argument(
        "--fedcrew",
        action="store_true",
        help="Whether to use FedCrew weighted aggregation",
    )
    # Backward compatibility: --causal is deprecated
    parser.add_argument(
        "--causal",
        action="store_true",
        help=argparse.SUPPRESS,  # Hidden from help, for backward compatibility
    )
    parser.add_argument(
        "--causal-mode",
        choices=["full", "crp_only", "logits_only", "uniform"],
        default="full",
        help="Select which causal components to enable. 'uniform' aggregates "
        "the head with equal per-client weight (no CRP/logits scoring at all), "
        "for the 4-way ablation's uniform-aggregation baseline",
    )
    parser.add_argument(
        "--causal-crp",
        action="store_true",
        help="Enable CRP relevance scoring",
    )
    parser.add_argument(
        "--causal-logits",
        action="store_true",
        help="Enable logits-based DS scoring",
    )
    parser.add_argument(
        "--anchor-selection",
        choices=["first", "random"],
        default="first",
        help="How to choose the anchor samples",
    )
    parser.add_argument(
        "--anchor-seed",
        type=int,
        default=None,
        help="Seed used when anchor selection is random",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="W&B group name for related runs",
    )
    parser.add_argument(
        "--lognum",
        type=int,
        default=0,
        help="Number of logs to keep in the tensorboard",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=64,
        help="Batch size to use for training on clients",
    )
    parser.add_argument(
        "--clipgradients", action="store_true", help="Whether to clip gradients"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of samples per class to select",
    )
    parser.add_argument(
        "--no_log", action="store_true", help="If activated, no logs will be saved"
    )
    parser.add_argument(
        "--fedprox",
        type=float,
        default=0.0,
        help="FedProx regularization factor, set to 0.0 to disable",
    )
    parser.add_argument("--fednova", action="store_true", help="Whether to use FedNova")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument(
        "--l1", type=float, default=0.0, help="L1 regularization factor"
    )
    parser.add_argument(
        "--l2", type=float, default=0.0, help="L2 regularization factor"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Threshold for counting a sample as correct",
    )
    parser.add_argument(
        "--l2_fc",
        type=float,
        default=0.0,
        help="L2 regularization factor for fc layer (proximal term for fc only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (omit to disable seeding)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save final-round model checkpoints",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="predictions",
        help="Directory to save final-round per-sample predictions/logits",
    )
    parser.add_argument(
        "--crp-maps-dir",
        type=str,
        default="crp_maps",
        help="Directory to save final-round raw CRP relevance maps",
    )
    parser.add_argument(
        "--alignment-dir",
        type=str,
        default="alignment",
        help="Directory to save CKA/fc-divergence alignment matrices",
    )
    parser.add_argument(
        "--no-final-artifacts",
        action="store_true",
        help="Skip saving the final-round checkpoint/predictions/CRP maps",
    )
    parser.add_argument(
        "--moon",
        action="store_true",
        help="Whether to use MOON's model-contrastive client-side regularization",
    )
    parser.add_argument(
        "--moon-mu",
        type=float,
        default=1.0,
        help="MOON contrastive loss weight (mu)",
    )
    parser.add_argument(
        "--moon-tau",
        type=float,
        default=0.5,
        help="MOON contrastive loss temperature (tau)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=[],
        help="Extra WandB tags for this run (e.g. a sweep name), on top of the "
        "automatic instrumentation-revision tag",
    )
    return parser


def parse_args() -> ExperimentConfig:
    """Parse command line arguments and create ExperimentConfig.

    Returns:
        ExperimentConfig: Configuration object with all experiment settings.
    """
    parser = _create_parser()
    args = parser.parse_args()

    # Handle backward compatibility for --causal flag
    if args.causal and not args.fedcrew:
        warnings.warn(
            "The --causal flag is deprecated. Use --fedcrew instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args.fedcrew = True

    causal_crp = args.causal_crp
    causal_logits = args.causal_logits

    if args.causal_mode == "crp_only":
        causal_crp = True
        causal_logits = False
    elif args.causal_mode == "logits_only":
        causal_crp = False
        causal_logits = True
    elif args.causal_mode == "uniform":
        causal_crp = False
        causal_logits = False
    elif not causal_crp and not causal_logits:
        # causal_mode == "full" (default) with no explicit component flags
        causal_crp = True
        causal_logits = True

    return ExperimentConfig(
        dataset=args.dataset,
        clients=args.clients,
        fedcrew=args.fedcrew,
        lognum=args.lognum,
        epochs=args.epochs,
        batchsize=args.batchsize,
        clipgradients=args.clipgradients,
        samples=args.samples,
        no_log=args.no_log,
        fedprox=args.fedprox,
        fednova=args.fednova,
        rounds=args.rounds,
        l1=args.l1,
        l2=args.l2,
        alpha=args.alpha,
        l2_fc=args.l2_fc,
        seed=args.seed,
        causal_mode=args.causal_mode,
        causal_crp=causal_crp,
        causal_logits=causal_logits,
        anchor_selection=args.anchor_selection,
        anchor_seed=args.anchor_seed,
        wandb_group=args.wandb_group,
        checkpoint_dir=args.checkpoint_dir,
        predictions_dir=args.predictions_dir,
        crp_maps_dir=args.crp_maps_dir,
        alignment_dir=args.alignment_dir,
        save_final_artifacts=not args.no_final_artifacts,
        moon=args.moon,
        moon_mu=args.moon_mu,
        moon_tau=args.moon_tau,
        wandb_tags=args.wandb_tags,
    )
