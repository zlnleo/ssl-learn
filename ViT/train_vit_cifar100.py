import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
from torchvision.transforms import RandAugment
from vit_cifar100 import ViT   # 你之前写的模型文件

# -----------------------------
# 1. 配置
# -----------------------------
BATCH_SIZE = 128
EPOCHS = 100
LR = 3e-4
os.makedirs("./checkpoints", exist_ok=True)
CHECKPOINT_PATH = "./checkpoints/vit_cifar100.pth"
LOG_DIR = "./runs/vit_cifar100"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 2. 数据增强 + DataLoader
# -----------------------------

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    RandAugment(),   # ★ 强增强
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    ),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    ),
])

train_dataset = datasets.CIFAR100(
    root="../data", train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR100(
    root="../data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False)

# -----------------------------
# 3. 模型、优化器、损失函数
# -----------------------------
model = ViT(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# -----------------------------
# 4. TensorBoard
# -----------------------------
writer = SummaryWriter(LOG_DIR)

# -----------------------------
# 5. Checkpoint 断点重跑
# -----------------------------
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")

# -----------------------------
# 6. 训练函数
# -----------------------------
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=120)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(),
                         acc=100. * correct / total)

    avg_loss = total_loss / total
    avg_acc = 100. * correct / total

    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Accuracy", avg_acc, epoch)

    return avg_loss, avg_acc

# -----------------------------
# 7. 测试函数
# -----------------------------
def test_one_epoch(epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", ncols=120):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    avg_acc = 100. * correct / total

    writer.add_scalar("Test/Loss", avg_loss, epoch)
    writer.add_scalar("Test/Accuracy", avg_acc, epoch)

    return avg_loss, avg_acc

# -----------------------------
# 8. 主训练循环
# -----------------------------
for epoch in range(start_epoch, EPOCHS):
    train_loss, train_acc = train_one_epoch(epoch)
    test_loss, test_acc = test_one_epoch(epoch)

    scheduler.step()

    # 保存 checkpoint
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }, CHECKPOINT_PATH)

    print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

writer.close()
