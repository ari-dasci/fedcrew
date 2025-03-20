import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from imagenetsubset import load_tiny_imagenet
from tqdm import tqdm

from imagenetsubset import load_tiny_imagenet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = ResNet18_Weights.DEFAULT.transforms()

trainset = load_tiny_imagenet(train=True, transforms=transform)
testset = load_tiny_imagenet(train=False, transforms=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = resnet18()
# for param in model.parameters():
#     param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)
model = torch.compile(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 6
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    if epoch == 3:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        print("Moving to fine-tuning stage")

    for i, (images, labels) in enumerate(tqdm(trainloader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}")
            running_loss = 0.0

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

    # Save model checkpoint (overwrites previous checkpoint)
    torch.save(model.state_dict(), "resnet.pth")

print("Finished Training")
