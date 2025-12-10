import argparse
from copy import deepcopy
from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from crp.image import imgify
from flex.data import Dataset, LazyIndexable
from flex.model import FlexModel
from flex.pool import FlexPool, fed_avg, aggregate_weights
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from datasets import get_dataset
from models import get_transforms, get_relevance_layer
from utils.flex_boilerplate import (
    build_server_model,
    copy_server_model_to_clients,
    set_aggregated_weights_to_server,
    get_clients_weights,
    clean_up_models,
    create_transfer_model,
    get_clients_fc_weights,
    set_aggregated_fc_weights_to_server,
)
from utils.prueba_crp import extract_heatmap

assert torch.cuda.is_available(), "CUDA not available"
device = "cuda"

round_number = 0
parser = argparse.ArgumentParser(
    description="Conditional Relevance Propagation for Causal Learning"
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["cifar_10", "imagenet", "celeba", "waterbirds"],
    default="cifar_10",
    help="Dataset to use",
)
parser.add_argument(
    "--clients", type=int, default=100, help="Number of clients per round"
)
parser.add_argument(
    "--lognum", type=int, default=0, help="Number of logs to keep in the tensorboard"
)

parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument(
    "--irm_epochs",
    type=int,
    default=10,
    help="Number of epochs to train the last layer",
)
parser.add_argument(
    "--batchsize",
    type=int,
    default=64,
    help="Batch size to use for training on clients",
)
parser.add_argument(
    "--samples", type=int, default=2, help="Number of samples per class to select"
)

parser.add_argument(
    "--f", type=int, default=1, help="Parameter for the Krum aggregation operator"
)

parser.add_argument(
    "--no_log", action="store_true", help="If activated, no logs will be saved"
)
parser.add_argument("--rounds", type=int, default=75, help="Number of rounds")
parser.add_argument(
    "--irm_rounds",
    type=int,
    default=25,
    help="Number of rounds for computation of last layer",
)
parser.add_argument("--l1", type=float, default=0.0, help="L1 regularization factor")
parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization factor")

args = parser.parse_args()

CLIENTS_PER_ROUND = args.clients
EPOCHS = args.epochs
round_number = 0
AGG = fed_avg


def get_summary_writer_filename(args):
    parts = [
        f"irmsamples{args.samples}",
        "l1" if args.l1 > 0.0 else "",
        "l2" if args.l2 > 0.0 else "",
        f"lognum{args.lognum}" if args.lognum > 0 else "",
        "sgd" if EPOCHS == 1 else "",
    ]
    return f"runs/{args.dataset}/" + "-".join([part for part in parts if part])


writer = SummaryWriter(get_summary_writer_filename(args)) if not args.no_log else None

flex_dataset, test_data, must_have_indices = get_dataset(args.dataset)
client_ids = list(flex_dataset.keys())

print(f"Running options: {args}")

data_transforms = get_transforms(args.dataset)


def select_waterbirds_label(dataset: Dataset):
    y_data = [y[0] for y in dataset.y_data]
    y_data = LazyIndexable(y_data, len(y_data))
    return Dataset(X_data=dataset.X_data, y_data=y_data)


def train(
    client_flex_model: FlexModel, client_data: Dataset, l1_factor=args.l1, epochs=EPOCHS
):
    train_dataset = client_data.to_torchvision_dataset(transform=data_transforms)
    client_dataloader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True
    )
    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    for _ in range(epochs):
        for images, labels in client_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels)
            if l1_factor > 0.0:
                # TODO: fc is hardcoded
                l1_loss = sum(p.abs().sum() for p in model.fc.parameters())
                loss += l1_factor * l1_loss
            loss.backward()
            optimizer.step()


def obtain_metrics(server_flex_model: FlexModel, test_data: Dataset, is_server=True):
    if "waterbirds" in args.dataset and is_server:
        test_data = select_waterbirds_label(test_data)

    model = server_flex_model["model"]
    model.eval()
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    # get test data as a torchvision object
    test_dataset = test_data.to_torchvision_dataset(transform=data_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=False
    )
    losses = []
    confusion_matrix = torch.zeros(2, 2)

    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

            # Update confusion matrix
            for t, p in zip(target.cpu().view(-1), pred.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    confusion_matrix = confusion_matrix.cpu().numpy()
    test_loss = sum(losses) / len(losses)
    test_acc /= total_count

    return test_loss, test_acc, confusion_matrix


def obtain_accuracy(server_flex_model: FlexModel, test_data: Dataset):
    return obtain_metrics(server_flex_model, test_data)[1]


def select_subsample_server_data(_, dataset: Dataset, k=2) -> Dataset:
    """
    :param _: server flex_model
    :param dataset: server dataset
    :param k: how many samples per class to select
    :return: a Dataset object with k samples per class
    """
    print("Creating subsample")
    labels = dataset.y_data
    if args.dataset == "waterbirds":
        labels = [(label[0], label[1]) for label in labels]
    labels_to_indices = {label: [] for label in labels}

    for i, label in enumerate(labels):
        if len(labels_to_indices[label]) < k:
            labels_to_indices[label].append(i)

    indices = [v for v in labels_to_indices.values()]
    indices = [i for sublist in indices for i in sublist]
    new_dataset = dataset[indices]

    if args.dataset == "waterbirds":
        new_dataset = select_waterbirds_label(new_dataset)

    torch_data = new_dataset.to_torchvision_dataset()
    transform = transforms.ToTensor()
    for i in range(len(torch_data)):
        sample, _ = torch_data[i]
        sample = np.copy(sample)
        if writer:
            writer.add_image(f"sample/{i}", transform(sample), 0)

    return new_dataset


def load_client_model(
    original_model: nn.Module,
    collected_weights: List[List[torch.Tensor]],
    client_id: int,
):
    """
    :param original_model: Pytorch module with the neural network where the weights will be loaded
    :param collected_weights: A list of all collected weights, generated by a collect_clients_weights function
    :param client_id: Index of client
    :return: A copy of the original model with the weights of the client_id-th client loaded
    """
    assert client_id < len(collected_weights), (
        f"Client ID out of bounds, {client_id} >= {len(collected_weights)}"
    )
    client_weights = collected_weights[client_id]
    new_model = deepcopy(original_model).to(device)
    with torch.no_grad():
        weight_dict = new_model.state_dict()
        for layer_key, new in zip(weight_dict, client_weights):
            weight_dict[layer_key].copy_(weight_dict[layer_key].to(device) + new)
    return new_model


def create_server_heatmap(
    server_model: FlexModel, _: Dataset, subsample_dataset: Dataset
):
    model = server_model["model"]
    client_crp(deepcopy(model).to(device), "server", subsample_dataset)


def client_crp(client_model: nn.Module, client_id: int, sample_dataset: Dataset):
    data = sample_dataset.to_torchvision_dataset()
    for sample_id in range(len(data)):
        sample, label = data[sample_id]
        heatmap, _ = extract_heatmap(
            client_model,
            layer=get_relevance_layer(args.dataset),
            transforms=data_transforms,
            sample=sample,
            label=label,
        )
        img = imgify(heatmap, cmap="seismic", symmetric=True, grid=(1, 5))
        transform = transforms.PILToTensor()
        if writer:
            writer.add_image(
                f"crp/client_{client_id}/image_{sample_id}",
                transform(img),
                round_number,
            )


@aggregate_weights
def irm_based_aggregation(weights: List[List[torch.Tensor]]):
    number_of_layers = len(weights[0])
    n_clients = len(weights)
    aggregated_weights = []
    for i in range(number_of_layers):
        # layer_weights shape: (n_clients, *param_shape)
        layer_weights = torch.stack([weight[i] for weight in weights])

        # Get the sign of each weight update for each client and parameter
        # signs shape: (n_clients, *param_shape)
        signs = torch.sign(layer_weights)

        # Sum the signs across clients for each parameter coordinate
        # sum_signs shape: (*param_shape)
        sum_signs = signs.sum(dim=0)

        # Create a mask: True if all clients had the same sign (sum is +/- n_clients)
        # same_sign_mask shape: (*param_shape), boolean
        same_sign_mask = torch.abs(sum_signs) == n_clients

        # Calculate the average weight for each parameter coordinate
        # avg_weights shape: (*param_shape)
        avg_weights = layer_weights.mean(dim=0)

        aggregated_layer_weight = avg_weights * same_sign_mask.float()

        aggregated_weights.append(aggregated_layer_weight)

    return aggregated_weights


def run_federated_round(
    pool: FlexPool,
    round_number: int,
    other_clients: FlexPool,
    must_have_clients: FlexPool,
    CLIENTS_PER_ROUND: int,
    copy_server_model_to_clients_func: Callable,
    train_func: Callable,
    obtain_metrics_func: Callable,
    get_clients_weights_func: Callable,
    clean_up_models_func: Callable,
    AGG_func: Callable,
    set_aggregated_weights_to_server_func: Callable,
    create_server_heatmap_func: Callable,
    subsamples: Dataset,
    writer: SummaryWriter = None,
    epochs: int = EPOCHS,
):
    """Runs a single round of federated learning."""
    selected_clients = other_clients.select(CLIENTS_PER_ROUND - len(must_have_clients))
    pool.servers.map(copy_server_model_to_clients_func, selected_clients)
    pool.servers.map(copy_server_model_to_clients_func, must_have_clients)

    selected_clients.map(train_func, epochs=epochs)
    must_have_clients.map(train_func, epochs=epochs)
    if (round_number + 1) % 5 == 0 and writer:
        client_metrics = selected_clients.map(
            obtain_metrics_func, is_server=False
        ) + must_have_clients.map(obtain_metrics_func, is_server=False)
        losses = [loss for loss, _, _ in client_metrics]
        accs = [acc for _, acc, _ in client_metrics]

        if losses:  # Ensure metrics were collected
            avg_loss = sum(losses) / len(losses)
            median_loss = np.median(losses)
            max_loss = max(losses)
            min_loss = min(losses)
            writer.add_scalar("Average Client Loss", avg_loss, round_number)
            writer.add_scalar("Median Client Loss", median_loss, round_number)
            writer.add_scalar("Max Client Loss", max_loss, round_number)
            writer.add_scalar("Min Client Loss", min_loss, round_number)

        if accs:  # Ensure metrics were collected
            avg_acc = sum(accs) / len(accs)
            median_acc = np.median(accs)
            max_acc = max(accs)
            min_acc = min(accs)
            writer.add_scalar("Average Client Accuracy", avg_acc, round_number)
            writer.add_scalar("Median Client Accuracy", median_acc, round_number)
            writer.add_scalar("Max Client Accuracy", max_acc, round_number)
            writer.add_scalar("Min Client Accuracy", min_acc, round_number)

    pool.aggregators.map(get_clients_weights_func, selected_clients)
    pool.aggregators.map(get_clients_weights_func, must_have_clients)
    selected_clients.map(clean_up_models_func)

    pool.aggregators.map(AGG_func)
    pool.aggregators.map(set_aggregated_weights_to_server_func, pool.servers)

    if (round_number + 1) % 5 == 0 and round_number > 0:
        print("Generating heatmaps")
        pool.servers.map(create_server_heatmap_func, subsample_dataset=subsamples)

    loss, acc, confusion_matrix = pool.servers.map(obtain_metrics_func)[0]
    if writer:
        writer.add_scalar("Loss", loss, round_number)
        writer.add_scalar("Accuracy", acc, round_number)
        fig = plt.figure()
        plt.imshow(confusion_matrix)
        # plt.show() # Consider removing plt.show() if running non-interactively
        writer.add_figure("confusion_matrix", fig, round_number)
        plt.close(fig)  # Close the figure to free memory
        print("Clossing images")
    print(f"ROUND {round_number}: loss {loss:7.4f}, acc {acc:7.4f}")  # Added formatting


def train_base(pool: FlexPool, n_rounds=args.rounds, irm_rounds=args.irm_rounds):
    global round_number
    clients = pool.clients
    subsamples: Dataset = pool.servers.map(
        select_subsample_server_data, k=args.samples
    )[0]

    must_have_clients = clients.select(lambda id, _: id in must_have_indices)
    other_clients = clients.select(lambda id, _: id not in must_have_indices)
    permanent_clients = other_clients.select(CLIENTS_PER_ROUND - len(must_have_clients))

    for i in tqdm(range(n_rounds)):
        round_number = i
        run_federated_round(
            pool=pool,
            round_number=round_number,
            other_clients=permanent_clients,
            must_have_clients=must_have_clients,
            CLIENTS_PER_ROUND=CLIENTS_PER_ROUND,
            copy_server_model_to_clients_func=copy_server_model_to_clients,
            train_func=train,
            obtain_metrics_func=obtain_metrics,
            get_clients_weights_func=get_clients_weights,
            clean_up_models_func=clean_up_models,
            AGG_func=AGG,
            set_aggregated_weights_to_server_func=set_aggregated_weights_to_server,
            create_server_heatmap_func=create_server_heatmap,
            subsamples=subsamples,
            writer=writer,
        )

    # Prepare for transfer (freeze model and reset fc)
    pool.servers.map(create_transfer_model)
    for j in range(irm_rounds):
        round_number = n_rounds + j
        print(f"Starting IRM round {j + 1}/{irm_rounds}")
        run_federated_round(
            pool=pool,
            round_number=round_number,
            other_clients=permanent_clients,
            must_have_clients=must_have_clients,
            CLIENTS_PER_ROUND=CLIENTS_PER_ROUND,
            copy_server_model_to_clients_func=copy_server_model_to_clients,
            train_func=train,
            obtain_metrics_func=obtain_metrics,
            get_clients_weights_func=get_clients_fc_weights,
            clean_up_models_func=clean_up_models,
            AGG_func=AGG,
            set_aggregated_weights_to_server_func=set_aggregated_fc_weights_to_server,
            create_server_heatmap_func=create_server_heatmap,
            subsamples=subsamples,
            writer=writer,
            epochs=args.irm_epochs,
        )


def run_server_pool():
    global flex_dataset
    global test_data
    flex_dataset["server"] = test_data
    pool = FlexPool.client_server_pool(
        flex_dataset, build_server_model, dataset=args.dataset, l2_factor=args.l2
    )
    train_base(pool, n_rounds=args.rounds)


def main():
    run_server_pool()


if __name__ == "__main__":
    main()
