import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from vit_mae import MAE

# ============================================================
# 配置
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "./checkpoint/mae_last.pth"

BATCH_SIZE = 256

EPOCHS = 100

NUM_CLASSES = 100

LR = 1e-3


# ============================================================
# 1. Patch Embedding
# ============================================================

# 我们直接使用 MAE 里面已经训练好的 PatchEmbedding。
#
# 这里不重新定义。
#
# Linear Probe 的输入：
#
# [B, 3, 32, 32]
#
#         ↓
#
# PatchEmbedding
#
#         ↓
#
# [B, 64, 256]


# ============================================================
# 2. MAE Encoder Feature Extractor
# ============================================================


class MAEFeatureExtractor(nn.Module):

    def __init__(self, mae):

        super().__init__()

        # ----------------------------------------------------
        # 使用 MAE 预训练好的 Patch Embedding
        # ----------------------------------------------------

        self.patch_embedding = mae.patch_embedding

        # ----------------------------------------------------
        # 使用 MAE 预训练好的位置编码
        # ----------------------------------------------------

        self.pos_embedding = mae.pos_embedding

        # ----------------------------------------------------
        # 使用 MAE 预训练好的 Encoder
        # ----------------------------------------------------

        self.encoder = mae.encoder

    def forward(self, x):

        # ====================================================
        # Image
        #
        # [B, 3, 32, 32]
        # ====================================================

        x = self.patch_embedding(x)

        # ====================================================
        # Patch tokens
        #
        # [B, 64, 256]
        # ====================================================

        x = self.pos_embedding(x)

        # ====================================================
        # 注意：
        #
        # 这里！！！！！
        #
        # 不进行 random_masking
        #
        # Linear Probe 时我们希望 Encoder
        # 看到完整图片。
        #
        # ====================================================

        # ====================================================
        # MAE Encoder
        #
        # [B, 64, 256]
        #
        # ====================================================

        x = self.encoder(x)

        # ====================================================
        # Global Average Pooling
        #
        # 64个patch
        #
        #      ↓
        #
        # 一个图片feature
        #
        # [B, 64, 256]
        #
        #      ↓
        #
        # [B, 256]
        # ====================================================

        x = x.mean(dim=1)

        return x


# ============================================================
# 3. Linear Probe
# ============================================================


class LinearProbe(nn.Module):

    def __init__(self, mae):

        super().__init__()

        # ----------------------------------------------------
        # MAE Encoder
        # ----------------------------------------------------

        self.feature_extractor = MAEFeatureExtractor(mae)

        # ----------------------------------------------------
        # Linear Classifier
        #
        # 256维 feature
        #
        #      ↓
        #
        # 100个类别
        # ----------------------------------------------------

        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):

        # ====================================================
        # 得到 MAE feature
        #
        # [B,256]
        # ====================================================

        feature = self.feature_extractor(x)

        # ====================================================
        # Linear classifier
        #
        # [B,256]
        #
        #      ↓
        #
        # [B,100]
        # ====================================================

        out = self.classifier(feature)

        return out


# ============================================================
# 4. Freeze MAE Encoder
# ============================================================


def freeze_mae(model):

    # --------------------------------------------------------
    # 冻结整个 MAE feature extractor
    # --------------------------------------------------------

    for param in model.feature_extractor.parameters():

        param.requires_grad = False


# ============================================================
# 5. Train
# ============================================================


def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    # --------------------------------------------------------
    # 非常重要：
    #
    # model.train() 会把 MAE Encoder 也设置成 train mode。
    #
    # 虽然参数被冻结了，
    # 但 DropPath / Dropout 仍然可能工作。
    #
    # Linear Probe 中我们希望 Encoder 是 eval mode。
    #
    # 所以手动设置：
    # --------------------------------------------------------

    model.feature_extractor.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    loop = tqdm(loader, desc="Train")

    for images, labels in loop:

        images = images.to(DEVICE, non_blocking=True)

        labels = labels.to(DEVICE, non_blocking=True)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(images)

        # ----------------------------------------------------
        # Classification loss
        # ----------------------------------------------------

        loss = criterion(outputs, labels)

        # ----------------------------------------------------
        # Backward
        #
        # 只有 classifier 有梯度
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        pred = outputs.argmax(dim=1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

        total_loss += loss.item() * labels.size(0)

        loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / total:.2f}%")

    avg_loss = total_loss / total

    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ============================================================
# 6. Test
# ============================================================


@torch.no_grad()
def evaluate(model, loader, criterion):

    model.eval()

    # Encoder保持eval

    model.feature_extractor.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    loop = tqdm(loader, desc="Test")

    for images, labels in loop:

        images = images.to(DEVICE, non_blocking=True)

        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        pred = outputs.argmax(dim=1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total

    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ============================================================
# 7. Main
# ============================================================


def main():

    print("=" * 60)

    print("MAE Linear Probe")

    print("=" * 60)

    print("Device:", DEVICE)

    # ========================================================
    # 创建 MAE
    # ========================================================

    mae = MAE()

    mae = mae.to(DEVICE)

    # ========================================================
    # 加载 MAE checkpoint
    # ========================================================

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    mae.load_state_dict(checkpoint["model"])

    print("Loaded MAE checkpoint:")

    print("Epoch:", checkpoint["epoch"])

    # ========================================================
    # 创建 Linear Probe
    # ========================================================

    model = LinearProbe(mae)

    model = model.to(DEVICE)

    # ========================================================
    # 冻结 MAE
    # ========================================================

    freeze_mae(model)

    # ========================================================
    # 检查参数
    # ========================================================

    trainable_params = []

    frozen_params = []

    for name, param in model.named_parameters():

        if param.requires_grad:

            trainable_params.append(name)

        else:

            frozen_params.append(name)

    print()

    print("Trainable parameters:")

    for name in trainable_params:

        print("  ", name)

    print()

    print("Frozen parameters:", len(frozen_params))

    # ========================================================
    # Dataset
    # ========================================================

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )

    train_dataset = torchvision.datasets.CIFAR100(
        root="../data", train=True, download=True, transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root="../data", train=False, download=True, transform=test_transform
    )

    # ========================================================
    # DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Optimizer
    #
    # 注意：
    #
    # 这里只把 classifier 放进去。
    #
    # MAE Encoder不会更新。
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=LR, weight_decay=0.0
    )

    # ========================================================
    # Training
    # ========================================================

    best_acc = 0.0

    for epoch in range(EPOCHS):

        print()

        print("=" * 60)

        print(f"Epoch {epoch + 1}/{EPOCHS}")

        print("=" * 60)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print()

        print(f"Train Loss: {train_loss:.4f}")

        print(f"Train Acc : {train_acc:.2f}%")

        print(f"Test Loss : {test_loss:.4f}")

        print(f"Test Acc  : {test_acc:.2f}%")

        # ====================================================
        # 保存最好结果
        # ====================================================

        if test_acc > best_acc:

            best_acc = test_acc

            torch.save(
                {
                    "epoch": epoch + 1,
                    "classifier": model.classifier.state_dict(),
                    "test_acc": test_acc,
                },
                "./checkpoint/mae_linear_probe_best.pth",
            )

    # ========================================================
    # Final
    # ========================================================

    print()

    print("=" * 60)

    print("Linear Probe Finished")

    print("=" * 60)

    print(f"Best Test Accuracy: " f"{best_acc:.2f}%")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()
