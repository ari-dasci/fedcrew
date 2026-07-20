"""FedCrew: Federated Learning with Conditional Relevance Propagation."""

from flex.data import Dataset, FedDataset
from flex.pool import FlexPool, fed_avg, weighted_fed_avg
from tqdm import tqdm

from config import ExperimentConfig, parse_args
from datasets import get_dataset
from utils.crp_utils import (
    compute_features_weights_per_client,
    create_server_heatmap,
)
from utils.checkpoint_utils import save_checkpoint
from utils.data_utils import select_subsample_server_data
from utils.flex_boilerplate import (
    build_server_model,
    clean_up_models,
    copy_server_model_to_clients,
    fedcrew_weighted_average,
    get_clients_weights,
    set_aggregated_weights_to_server,
)
from utils.fednova import get_fednova_iters, obtain_fednova_weights
from utils.logging_utils import (
    finish_logging,
    log_client_metrics,
    log_server_metrics,
    LoggerState,
    save_predictions,
    setup_logging,
)
from utils.seed_utils import seed_everything
from utils.training import obtain_metrics, obtain_metrics_with_predictions, train

# Backward compatibility: also import old name


def train_pool(
    pool: FlexPool,
    config: ExperimentConfig,
    logger: LoggerState,
    flex_dataset: FedDataset,
    test_data: Dataset,
    must_have_indices: list,
) -> None:
    """Run the federated learning training loop.

    Args:
        pool: FlexPool with clients and servers.
        config: Experiment configuration.
        logger: Logger state for tensorboard/wandb.
        flex_dataset: Federated dataset.
        test_data: Test dataset.
        must_have_indices: Indices of clients that must participate in every round.
    """
    clients = pool.clients

    subsamples, new_test_set = pool.servers.map(
        select_subsample_server_data,
        config=config,
        logger=logger,
        round_number=0,
        k=config.samples,
    )[0]

    pool._data["server"] = new_test_set

    must_have_clients = clients.select(lambda id, _: id in must_have_indices)
    other_clients = clients.select(lambda id, _: id not in must_have_indices)

    AGG = (
        fedcrew_weighted_average
        if config.fedcrew
        else (weighted_fed_avg if config.fednova else fed_avg)
    )

    for round_number in tqdm(range(config.rounds)):
        is_final_round = round_number == config.rounds - 1
        selected_clients = other_clients.select(config.clients - len(must_have_clients))

        pool.servers.map(copy_server_model_to_clients, selected_clients)
        pool.servers.map(copy_server_model_to_clients, must_have_clients)

        selected_clients.map(train, config=config, device=logger.device)
        must_have_clients.map(train, config=config, device=logger.device)

        # Log client metrics every 5 rounds
        if (round_number + 1) % 5 == 0 and logger.writer:
            client_metrics = selected_clients.map(
                obtain_metrics, config=config, device=logger.device
            ) + must_have_clients.map(
                obtain_metrics, config=config, device=logger.device
            )
            log_client_metrics(logger, client_metrics, round_number)

        pool.aggregators.map(get_clients_weights, selected_clients)
        pool.aggregators.map(get_clients_weights, must_have_clients)

        ponderation_list = (
            obtain_fednova_weights(
                selected_clients.map(get_fednova_iters)
                + must_have_clients.map(get_fednova_iters)
            )
            if config.fednova
            else []
        )

        selected_clients.map(clean_up_models)

        ponderation_tensor = (
            pool.servers.map(
                compute_features_weights_per_client,
                subsamples=subsamples,
                config=config,
                device=logger.device,
            )[0]
            if config.fedcrew
            else None
        )

        # Aggregate weights
        if config.fedcrew:
            pool.aggregators.map(AGG, ponderation_tensor=ponderation_tensor)
        elif config.fednova:
            pool.aggregators.map(AGG, ponderation=ponderation_list)
        else:
            pool.aggregators.map(AGG)

        pool.aggregators.map(set_aggregated_weights_to_server, pool.servers)

        if is_final_round and config.save_final_artifacts:
            pool.servers.map(save_checkpoint, config=config, round_number=round_number)

        # Create server heatmap every 5 rounds (after first round), and always
        # at the final round for the fully-instrumented re-run.
        if ((round_number + 1) % 5 == 0 and round_number > 0) or is_final_round:
            pool.servers.map(
                create_server_heatmap,
                subsample_dataset=subsamples,
                config=config,
                logger=logger,
                round_number=round_number,
                device=logger.device,
                save_raw=is_final_round and config.save_final_artifacts,
            )

        if is_final_round:
            loss, acc, confusion_matrix, predictions = pool.servers.map(
                obtain_metrics_with_predictions, config=config, device=logger.device
            )[0]
            if config.save_final_artifacts:
                save_predictions(predictions, config, round_number)
        else:
            loss, acc, confusion_matrix = pool.servers.map(
                obtain_metrics, config=config, device=logger.device
            )[0]

        if logger.writer:
            log_server_metrics(logger, loss, acc, confusion_matrix, round_number)

        print(f"ROUND {round_number}: loss {loss:7}, acc {acc:7}")


def run_server_pool(config: ExperimentConfig, logger: LoggerState) -> None:
    """Initialize and run the federated learning server pool.

    Args:
        config: Experiment configuration.
        logger: Logger state for tensorboard/wandb.
    """
    flex_dataset, test_data, must_have_indices = get_dataset(config.dataset)
    flex_dataset["server"] = test_data

    pool = FlexPool.client_server_pool(
        flex_dataset, build_server_model, dataset=config.dataset, l2_factor=config.l2
    )

    train_pool(pool, config, logger, flex_dataset, test_data, must_have_indices)


def main() -> None:
    """Main entry point."""
    config = parse_args()
    if config.seed is not None:
        seed_everything(config.seed)
    logger = setup_logging(config)

    try:
        run_server_pool(config, logger)
    finally:
        finish_logging(logger)


if __name__ == "__main__":
    main()
