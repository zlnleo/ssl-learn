import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from vit_cifar100 import ViT

# ============================
# Config
# ============================

BATCH_SIZE = 128

EPOCHS = 100

LR = 1e-3

NUM_CLASSES = 100


CHECKPOINT = "./checkpoints/vit_cifar100_last.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Device:", device)


# ============================
# Dataset
# ============================


mean = [0.5071, 0.4867, 0.4408]


std = [0.2675, 0.2565, 0.2761]


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


train_dataset = datasets.CIFAR100(
    root="../data", train=True, download=True, transform=transform
)


test_dataset = datasets.CIFAR100(
    root="../data", train=False, download=True, transform=transform
)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================
# Load ViT
# ============================


vit = ViT(num_classes=NUM_CLASSES)


checkpoint = torch.load(CHECKPOINT, map_location=device)


vit.load_state_dict(checkpoint["model"])


vit = vit.to(device)


# ============================
# Freeze ViT encoder
# ============================


for param in vit.encoder.parameters():

    param.requires_grad = False


# ViT feature dimension

feature_dim = vit.encoder.out_dim


print("Feature dim:", feature_dim)


# ============================
# Linear classifier
# ============================


classifier = nn.Linear(feature_dim, NUM_CLASSES)


classifier = classifier.to(device)


# ============================
# Optimizer
# ============================


optimizer = optim.AdamW(classifier.parameters(), lr=LR, weight_decay=0)


criterion = nn.CrossEntropyLoss()


# ============================
# Train
# ============================


def train(epoch):

    classifier.train()

    correct = 0

    total = 0

    loop = tqdm(train_loader, desc=f"Train [{epoch+1}/{EPOCHS}]")

    for images, labels in loop:

        images = images.to(device)

        labels = labels.to(device)

        # ====================
        # Extract feature
        # ====================

        with torch.no_grad():

            features = vit.forward_features(images)

        # ====================
        # Linear classifier
        # ====================

        outputs = classifier(features)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        pred = outputs.argmax(dim=1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

        loop.set_postfix(acc=f"{100*correct/total:.2f}%")


# ============================
# Test
# ============================


@torch.no_grad()
def test():

    classifier.eval()

    correct = 0

    total = 0

    loop = tqdm(test_loader, desc="Test")

    for images, labels in loop:

        images = images.to(device)

        labels = labels.to(device)

        features = vit.forward_features(images)

        outputs = classifier(features)

        pred = outputs.argmax(dim=1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

    acc = 100 * correct / total

    print(f"Test Accuracy: {acc:.2f}%")


# ============================
# Main
# ============================


for epoch in range(EPOCHS):

    train(epoch)

    test()
