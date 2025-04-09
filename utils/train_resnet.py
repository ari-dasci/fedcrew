import torch
import torch.nn as nn
import torch.optim as optim
from flex.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from .waterbirds import WaterbirdsDataset

import sys
sys.path.insert(0, '..')
from models import get_model, get_transforms


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Entrenando"):
        labels = labels[0]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, val_loader, criterion, device):
    """Evalúa el modelo en el conjunto de validación"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluando"):
            labels = labels[0]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


def main():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_epochs = 100
    lr = 0.001

    # Cargar dataset (reemplazar CustomDataset con tu dataset)
    train_transforms = get_transforms("waterbirds")
    train_dataset = Dataset.from_torchvision_dataset(WaterbirdsDataset()).to_torchvision_dataset(
        transform=train_transforms)
    val_dataset = Dataset.from_torchvision_dataset(WaterbirdsDataset(train=False)).to_torchvision_dataset(
        transform=train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Modelo
    model = get_model("waterbirds")
    model = model.to(device)

    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Entrenamiento
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')


if __name__ == "__main__":
    main()
