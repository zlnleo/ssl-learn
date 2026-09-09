import os
from datetime import datetime

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from vit_mae import MAE

# ============================================================
# Path
# ============================================================


CHECKPOINT_DIR = "./checkpoint"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


SAVE_PATH = "./checkpoint/mae_last.pth"


LOG_DIR = "./runs/mae_cifar100/" + datetime.now().strftime("%m%d_%H%M")


# ============================================================
# Dataset
# ============================================================


class MAEDataset(torch.utils.data.Dataset):

    def __init__(self, root="../data"):

        self.dataset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=False
        )

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __getitem__(self, index):

        img, _ = self.dataset[index]

        img = self.transform(img)

        return img

    def __len__(self):

        return len(self.dataset)


# ============================================================
# Patchify
# ============================================================


def patchify(imgs, patch_size=4):
    """
    image:

    [B,3,32,32]


    output:

    [B,64,48]


    一个patch:

    4*4*3=48

    """

    B, C, H, W = imgs.shape

    h = H // patch_size

    w = W // patch_size

    x = imgs.reshape(B, C, h, patch_size, w, patch_size)

    # B,C,h,p,w,p

    x = x.permute(0, 2, 4, 3, 5, 1)

    # B,h,w,p,p,C

    x = x.reshape(B, h * w, patch_size * patch_size * C)

    return x


# ============================================================
# Train
# ============================================================


def train(model, loader, optimizer, epochs=100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    model = model.to(device)

    writer = SummaryWriter(LOG_DIR)

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0

    # ==============================
    # resume
    # ==============================

    if os.path.exists(SAVE_PATH):

        checkpoint = torch.load(SAVE_PATH, map_location=device)

        model.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = checkpoint["epoch"]

        print("Resume epoch:", start_epoch)

    # ==============================
    # epoch
    # ==============================

    for epoch in range(start_epoch, epochs):

        model.train()

        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, img in enumerate(loop):

            img = img.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                pred, mask = model(img)

                target = patchify(img)

                # MAE loss

                loss = (pred - target) ** 2

                loss = loss.mean(dim=-1)

                # 只计算mask区域loss

                loss = (loss * mask).sum() / mask.sum()

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            total_loss += loss.item()

            loop.set_postfix(loss=f"{loss.item():.4f}")

            writer.add_scalar(
                "train/loss_step", loss.item(), epoch * len(loader) + step
            )

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch} Loss {avg_loss:.4f}")

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # save

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            },
            SAVE_PATH,
        )

    writer.close()


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":

    dataset = MAEDataset()

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
    )

    model = MAE()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    train(model, loader, optimizer, epochs=200)
