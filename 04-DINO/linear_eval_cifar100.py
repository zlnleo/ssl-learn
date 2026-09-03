import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.append("../ViT")

from vit_cifar100 import ViTEncoder

# ============================
# Config
# ============================


BATCH_SIZE = 128

EPOCHS = 100

LR = 1e-3


CHECKPOINT = "./checkpoint/dino_vit_last.pth"


NUM_CLASSES = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(device)


# ============================
# Dataset
# ============================


mean = [0.5071, 0.4867, 0.4408]


std = [0.2675, 0.2565, 0.2761]


train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
)


train_dataset = datasets.CIFAR100(
    "../data", train=True, download=True, transform=train_transform
)


test_dataset = datasets.CIFAR100(
    "../data", train=False, download=True, transform=test_transform
)


train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)


test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)


# ============================
# Load DINO Encoder
# ============================


encoder = ViTEncoder(image_size=32, patch_size=4, embed_dim=256, depth=6, heads=8)


checkpoint = torch.load(CHECKPOINT, map_location=device)


state_dict = checkpoint["model"]


encoder_dict = {}


for k, v in state_dict.items():

    if k.startswith("teacher_encoder."):

        new_k = k.replace("teacher_encoder.", "")

        encoder_dict[new_k] = v


encoder.load_state_dict(encoder_dict)


encoder = encoder.to(device)


print("DINO encoder loaded")


# ============================
# Freeze Encoder
# ============================


for p in encoder.parameters():

    p.requires_grad = False


encoder.eval()


# ============================
# Linear Classifier
# ============================


classifier = nn.Linear(256, NUM_CLASSES)


classifier = classifier.to(device)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR)


# ============================
# Train
# ============================


def train(epoch):

    classifier.train()

    total = 0

    correct = 0

    loop = tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}")

    for images, labels in loop:

        images = images.to(device)

        labels = labels.to(device)

        with torch.no_grad():

            feature = encoder(images)

        output = classifier(feature)

        loss = criterion(output, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        pred = output.argmax(dim=1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

        loop.set_postfix(acc=f"{100*correct/total:.2f}%")

    print("Train Acc:", 100 * correct / total)


# ============================
# Test
# ============================


@torch.no_grad()
def test():

    classifier.eval()

    correct = 0

    total = 0

    for images, labels in tqdm(test_loader, desc="Test"):

        images = images.to(device)

        labels = labels.to(device)

        feature = encoder(images)

        output = classifier(feature)

        pred = output.argmax(dim=1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

    acc = 100 * correct / total

    print(f"Test Accuracy:{acc:.2f}%")

    return acc


# ============================
# Main
# ============================


best = 0


for epoch in range(EPOCHS):

    train(epoch)

    acc = test()

    if acc > best:

        best = acc


print("Best Accuracy:", best)
