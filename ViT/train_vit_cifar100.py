import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from tqdm import tqdm

from scheduler import WarmupCosineLR
from vit_cifar100 import ViT

# ============================
# 1. Config
# ============================

BATCH_SIZE = 128
EPOCHS = 200

LR = 5e-4

NUM_CLASSES = 100


CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "vit_cifar100_last.pth")


LOG_DIR = "./runs/vit_cifar100/" + datetime.now().strftime("%m%d_%H%M")


os.makedirs(CHECKPOINT_DIR, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Device:", device)


# ============================
# 2. Dataset
# ============================

mean = [0.5071, 0.4867, 0.4408]


std = [0.2675, 0.2565, 0.2761]


train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
)


train_dataset = datasets.CIFAR100(
    root="../data", train=True, download=True, transform=train_transform
)


test_dataset = datasets.CIFAR100(
    root="../data", train=False, download=True, transform=test_transform
)


train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)


test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
)


# ============================
# 3. Model
# ============================


model = ViT(num_classes=NUM_CLASSES)


model = model.to(device)


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)


scheduler = WarmupCosineLR(optimizer, warmup_epochs=10, max_epochs=EPOCHS)


# AMP

scaler = torch.cuda.amp.GradScaler()


# ============================
# 4. TensorBoard
# ============================

writer = SummaryWriter(LOG_DIR)


# ============================
# 5. Checkpoint
# ============================


start_epoch = 0


if os.path.exists(CHECKPOINT_PATH):

    print("Loading checkpoint...")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler.load_state_dict(checkpoint["scheduler"])

    scaler.load_state_dict(checkpoint["scaler"])

    start_epoch = checkpoint["epoch"] + 1

    print("Resume epoch:", start_epoch)


# ============================
# 6. Train
# ============================


def train_one_epoch(epoch):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    loop = tqdm(train_loader, desc=f"Train Epoch [{epoch+1}/{EPOCHS}]", ncols=120)

    for images, labels in loop:

        images = images.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item() * images.size(0)

        pred = outputs.argmax(dim=1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

    avg_loss = total_loss / total

    acc = 100 * correct / total

    writer.add_scalar("Train/Loss", avg_loss, epoch)

    writer.add_scalar("Train/Accuracy", acc, epoch)

    return avg_loss, acc


# ============================
# 7. Test
# ============================


@torch.no_grad()
def test_one_epoch(epoch):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0

    loop = tqdm(test_loader, desc="Testing", ncols=120)

    for images, labels in loop:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        pred = outputs.argmax(dim=1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / total

    acc = 100 * correct / total

    writer.add_scalar("Test/Loss", avg_loss, epoch)

    writer.add_scalar("Test/Accuracy", acc, epoch)

    return avg_loss, acc


# ============================
# 8. Main Loop
# ============================


for epoch in range(start_epoch, EPOCHS):

    train_loss, train_acc = train_one_epoch(epoch)

    test_loss, test_acc = test_one_epoch(epoch)

    scheduler.step()

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        CHECKPOINT_PATH,
    )

    print(f"""
Epoch {epoch+1}

Train Loss: {train_loss:.4f}
Train Acc : {train_acc:.2f}%

Test Loss : {test_loss:.4f}
Test Acc  : {test_acc:.2f}%

""")


writer.close()
