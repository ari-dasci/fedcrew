import argparse
import os
from copy import deepcopy
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.image import imgify
from flex.data import Dataset, LazyIndexable
from flex.model import FlexModel
from flex.pool import FlexPool, fed_avg
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset
from models import get_transforms, get_relevance_layer
from utils.flex_boilerplate import build_server_model, copy_server_model_to_clients, set_aggregated_weights_to_server, \
    get_clients_weights, clean_up_models, causal_weighted_average
from utils.prueba_crp import extract_heatmap

assert torch.cuda.is_available(), "CUDA not available"
device = "cuda"

round_number = 0
parser = argparse.ArgumentParser(description="Conditional Relevance Propagation for Causal Learning")
parser.add_argument("--dataset", type=str, choices=["cifar_10", "imagenet", "celeba", "waterbirds"], default="cifar_10",
                    help="Dataset to use", )
parser.add_argument("--clients", type=int, default=100, help="Number of clients per round")
parser.add_argument("--causal", action="store_true", help="Whether use the causal ponderated aggregation")

parser.add_argument("--lognum", type=int, default=0, help="Number of logs to keep in the tensorboard")

parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
parser.add_argument("--batchsize", type=int, default=64, help="Batch size to use for training on clients", )
parser.add_argument("--clipgradients", action="store_true", help="Whether to clip gradients")

parser.add_argument("--samples", type=int, default=2, help="Number of samples per class to select")

parser.add_argument("--f", type=int, default=1, help="Parameter for the Krum aggregation operator")

parser.add_argument("--no_log", action="store_true", help="If activated, no logs will be saved")
parser.add_argument("--rounds", type=int, default=100, help="Number of rounds")
parser.add_argument("--l1", type=float, default=0.0, help="L1 regularization factor")
parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization factor")
parser.add_argument("--alpha", type=float, default=0.5, help="Threshold for counting a sample as correct")
args = parser.parse_args()

CLIENTS_PER_ROUND = args.clients
EPOCHS = args.epochs

AGG = causal_weighted_average if args.causal else fed_avg


def get_summary_writer_filename(args):
    parts = ["causal" if args.causal else "avg", f"samples{args.samples}", "l1" if args.l1 > 0.0 else "", "l2" if args.l2 > 0.0 else "",
             f"lognum{args.lognum}" if args.lognum > 0 else "", "sgd" if EPOCHS == 1 else "", f"alpha{args.alpha}"]
    return f"runs/{args.dataset}/" + ".".join([part for part in parts if part])


writer = SummaryWriter(get_summary_writer_filename(args)) if not args.no_log else None

flex_dataset, test_data, must_have_indices = get_dataset(args.dataset)
client_ids = list(flex_dataset.keys())

print(f"Running options: {args}")

data_transforms = get_transforms(args.dataset)


def select_waterbirds_label(dataset: Dataset):
    label_index = 0
    y_data = [y[0] for y in dataset.y_data]
    y_data = LazyIndexable(y_data, len(y_data))
    return Dataset(X_data=dataset.X_data, y_data=y_data)


def train(client_flex_model: FlexModel, client_data: Dataset, l1_factor=args.l1):
    train_dataset = client_data.to_torchvision_dataset(transform=data_transforms)
    client_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](model.parameters(), **client_flex_model["optimizer_kwargs"])
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    for _ in range(EPOCHS):
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


def obtain_metrics(server_flex_model: FlexModel, test_data: Dataset):
    if "waterbirds" in args.dataset:
        test_data = select_waterbirds_label(test_data)

    model = server_flex_model["model"]
    model.eval()
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    # get test data as a torchvision object
    test_dataset = test_data.to_torchvision_dataset(transform=data_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=False)
    losses = []
    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count

    return test_loss, test_acc


def obtain_accuracy(server_flex_model: FlexModel, test_data: Dataset): return \
    obtain_metrics(server_flex_model, test_data)[1]


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


def load_client_model(original_model: nn.Module, collected_weights: List[List[torch.Tensor]], client_id: int):
    """
    :param original_model: Pytorch module with the neural network where the weights will be loaded
    :param collected_weights: A list of all collected weights, generated by a collect_clients_weights function
    :param client_id: Index of client
    :return: A copy of the original model with the weights of the client_id-th client loaded
    """
    assert client_id < len(collected_weights), f"Client ID out of bounds, {client_id} >= {len(collected_weights)}"
    client_weights = collected_weights[client_id]
    new_model = deepcopy(original_model).to(device)
    with torch.no_grad():
        weight_dict = new_model.state_dict()
        for layer_key, new in zip(weight_dict, client_weights):
            weight_dict[layer_key].copy_(weight_dict[layer_key].to(device) + new)
    return new_model


def create_client_heatmaps(server_model: FlexModel, _: Dataset, subsample_dataset: Dataset):
    print("Extracting heatmaps")
    model = server_model["model"].to(device)
    weights = server_model["weights"]
    for client_id in range(len(weights)):
        client_model = load_client_model(model, weights, client_id)
        client_crp(client_model, client_id, subsample_dataset)


def create_server_heatmap(server_model: FlexModel, _: Dataset, subsample_dataset: Dataset):
    model = server_model["model"]
    client_crp(deepcopy(model).to(device), "server", subsample_dataset)


def client_crp(client_model: nn.Module, client_id: int, sample_dataset: Dataset):
    data = sample_dataset.to_torchvision_dataset()
    for sample_id in range(len(data)):
        sample, label = data[sample_id]
        heatmap, _ = extract_heatmap(client_model, layer=get_relevance_layer(args.dataset), transforms=data_transforms,
                                     sample=sample, label=label)
        img = imgify(heatmap, cmap="seismic", symmetric=True, grid=(1, 5))
        transform = transforms.PILToTensor()
        if writer:
            writer.add_image(f"crp/client_{client_id}/image_{sample_id}", transform(img), round_number)


def group_correct_class_probs(model: nn.Module, sample_dataset: Dataset):
    """
    :param model: Client neural network model
    :param sample_dataset: Datasets with samples of each class. Exactly balanced
    :return: List with tuples (label, correct_class_prob) for each sample in the dataset
    """
    dataset = sample_dataset.to_torchvision_dataset(transform=data_transforms)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.batchsize)
    logits = {}

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            sample_logits = model(inputs)  # shape: (batch_size, num_classes)
            probs = torch.softmax(sample_logits, dim=1)  # shape: (batch_size, num_classes)

            # labels.unsqueeze(1) changes shape from (batch_size,) to (batch_size, 1)
            # gather returns a tensor of shape (batch_size, 1), which we squeeze to (batch_size,)
            correct_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

            for label, prob in zip(labels.cpu().numpy(), correct_class_probs.cpu().numpy()):
                label = int(label)
                if label not in logits:
                    logits[label] = []

                logits[label].append(prob)
    return logits


def get_crp_attribution(model: nn.Module, sample_dataset: Dataset, layer: str):
    dataset = sample_dataset.to_torchvision_dataset(transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=1)

    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    attribution = CondAttribution(model)

    contributions = {}

    for sample, label in dataloader:
        sample = data_transforms(sample.squeeze()).to(device)
        sample = sample.unsqueeze(0)
        sample.requires_grad = True

        conditions = [{"y": [label]}]
        attr = attribution(sample, conditions, composite, record_layer=layer_names)
        rel_c = cc.attribute(attr.relevances[layer], abs_norm=True)
        rel_c = rel_c[0]
        rel_c_min = torch.min(rel_c)
        rel_c_max = torch.max(rel_c)
        rel_c = (rel_c - rel_c_min) / (rel_c_max - rel_c_min)
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label not in contributions:
            contributions[label] = []
        contributions[label].append(rel_c)

    contributions = {label: torch.stack(contributions[label]) for label in contributions}

    return contributions


def compute_features_weights_per_client(server_model: FlexModel, _, subsamples: Dataset, alpha: float = args.alpha):
    print("Running round CRP")
    model = server_model["model"]
    weights = server_model["weights"]

    clients_probs = {}
    clients_relevances = {}

    for client_id in range(len(weights)):
        client_model = load_client_model(model, weights, client_id)
        clients_probs[client_id] = group_correct_class_probs(client_model,
                                                             subsamples)  # Dict with id -> (Dict with label -> [prob])
        clients_relevances[client_id] = get_crp_attribution(client_model, subsamples, get_relevance_layer(
            args.dataset))  # Dict with id -> (Dict with label -> [relevance])

    # We assume that all labels are present in `subsamples`
    labels = sorted(list(clients_probs[0].keys()))
    client_features_weights = {}

    for label in labels:
        # Order matters, we need to keep the order of the clients
        label_probs = torch.tensor([clients_probs[client_id][label] for client_id in range((len(weights)))],
                                   dtype=torch.float32).to(device)
        label_probs = torch.where(label_probs > alpha, label_probs, float("-inf")) # softmax(-inf) = 0
        sample_weight = torch.softmax(label_probs.flatten(), dim=0).view(*label_probs.shape)  # (n_clients, n_samples)
        label_relevances = torch.stack([clients_relevances[client_id][label] for client_id in range(len(weights))]).to(
            device)  # (n_clients, n_samples, n_features)

        weights_features_clients = torch.sum(label_relevances * sample_weight.unsqueeze(-1),
                                             dim=1)  # (n_clients, n_features)
        client_features_weights[label] = weights_features_clients

    return torch.stack([client_features_weights[label] for label in labels], dim=0)  # (n_labels, n_clients, n_features)


def train_base(pool: FlexPool, n_rounds=100):
    clients = pool.clients
    subsamples: Dataset = pool.servers.map(select_subsample_server_data, k=args.samples)[0]

    must_have_clients = clients.select(lambda id, _: id in must_have_indices)
    other_clients = clients.select(lambda id, _: id not in must_have_indices)

    for i in tqdm(range(n_rounds)):
        global round_number
        round_number = i

        selected_clients = other_clients.select(CLIENTS_PER_ROUND - len(must_have_clients))
        pool.servers.map(copy_server_model_to_clients, selected_clients)
        pool.servers.map(copy_server_model_to_clients, must_have_clients)

        selected_clients.map(train)
        must_have_clients.map(train)
        pool.aggregators.map(get_clients_weights, selected_clients)
        pool.aggregators.map(get_clients_weights, must_have_clients)
        selected_clients.map(clean_up_models)

        ponderation_tensor = pool.servers.map(compute_features_weights_per_client, subsamples=subsamples)[
            0] if args.causal else None

        if args.causal:
            pool.aggregators.map(AGG, ponderation_tensor=ponderation_tensor)
        else:
            pool.aggregators.map(AGG)
        pool.aggregators.map(set_aggregated_weights_to_server, pool.servers)

        if (round_number + 1) % 5 == 0 and round_number > 0:
            pool.servers.map(create_server_heatmap, subsample_dataset=subsamples)

        loss, acc = pool.servers.map(obtain_metrics)[0]
        if writer:
            writer.add_scalar("Loss", loss, round_number)
            writer.add_scalar("Accuracy", acc, round_number)
        print(f"ROUND {round_number}: loss {loss:7}, acc {acc:7}")


def run_server_pool():
    global flex_dataset
    global test_data
    flex_dataset["server"] = test_data
    pool = FlexPool.client_server_pool(flex_dataset, build_server_model, dataset=args.dataset, l2_factor=args.l2)
    train_base(pool, n_rounds=args.rounds)


def main():
    run_server_pool()


if __name__ == "__main__":
    main()
