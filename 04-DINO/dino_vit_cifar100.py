import os
import sys
from datetime import datetime

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# ============================
# Path
# ============================

sys.path.append("../ViT")

from vit_cifar100 import ViTEncoder

CHECKPOINT_DIR = "./checkpoint"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


LAST_CHECKPOINT = "./checkpoint/dino_vit_last.pth"


LOG_DIR = "./runs/dino_vit_cifar100/" + datetime.now().strftime("%m%d_%H%M")


# ============================
# Dataset
# ============================


class DINODataset(Dataset):

    def __init__(self, root="../data"):

        self.dataset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True
        )

        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.2, 0.5)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __getitem__(self, index):

        img, _ = self.dataset[index]

        global1 = self.global_transform(img)

        global2 = self.global_transform(img)

        locals = [self.local_transform(img) for _ in range(4)]

        return global1, global2, locals

    def __len__(self):

        return len(self.dataset)


# ============================
# Projection Head
# ============================


class ProjectorHead(nn.Module):

    def __init__(self, in_dim=256, hidden_dim=2048, out_dim=256):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):

        return self.net(x)


# ============================
# DINO Model
# ============================


class DinoNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.student_encoder = ViTEncoder()

        self.student_projector = ProjectorHead()

        self.teacher_encoder = ViTEncoder()

        self.teacher_projector = ProjectorHead()

        # teacher初始化
        for ps, pt in zip(
            self.student_encoder.parameters(), self.teacher_encoder.parameters()
        ):

            pt.data.copy_(ps.data)

            pt.requires_grad = False

        for ps, pt in zip(
            self.student_projector.parameters(), self.teacher_projector.parameters()
        ):

            pt.data.copy_(ps.data)

            pt.requires_grad = False

        self.register_buffer("center", torch.zeros(1, 256))

    def forward(self, views):

        # student

        student_input = torch.cat(views, dim=0)

        student_feature = self.student_encoder(student_input)

        student_output = self.student_projector(student_feature)

        student_outputs = list(torch.chunk(student_output, len(views), dim=0))

        # teacher

        with torch.no_grad():

            teacher_input = torch.cat(views[:2], dim=0)

            teacher_feature = self.teacher_encoder(teacher_input)

            teacher_output = self.teacher_projector(teacher_feature)

            teacher_outputs = list(torch.chunk(teacher_output, 2, dim=0))

        return student_outputs, teacher_outputs


# ============================
# Loss
# ============================


def dino_loss(student_output, teacher_output, center, temp_s=0.1, temp_t=0.04):

    with torch.no_grad():

        teacher_prob = nn.functional.softmax((teacher_output - center) / temp_t, dim=-1)

    student_log_prob = nn.functional.log_softmax(student_output / temp_s, dim=-1)

    loss = -(teacher_prob * student_log_prob).sum(dim=-1).mean()

    return loss


# ============================
# EMA
# ============================


def update_teacher_ema(student, teacher, beta=0.996):

    for ps, pt in zip(student.parameters(), teacher.parameters()):

        pt.data.copy_(beta * pt.data + (1 - beta) * ps.data)


def update_center(center, teacher_out, momentum=0.9):

    batch_center = teacher_out.mean(dim=0, keepdim=True)

    center.data.copy_(momentum * center + (1 - momentum) * batch_center)


# ============================
# Train
# ============================


def train(model, loader, optimizer, epochs=100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    model.to(device)

    writer = SummaryWriter(LOG_DIR)

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0

    if os.path.exists(LAST_CHECKPOINT):

        ckpt = torch.load(LAST_CHECKPOINT, map_location=device)

        model.load_state_dict(ckpt["model"])

        optimizer.load_state_dict(ckpt["optimizer"])

        scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"]

        print("Resume:", start_epoch)

    for epoch in range(start_epoch, epochs):

        model.train()

        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, (g1, g2, locals) in enumerate(loop):

            views = [g1.to(device, non_blocking=True), g2.to(device, non_blocking=True)]

            views.extend([x.to(device, non_blocking=True) for x in locals])

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                student_outputs, teacher_outputs = model(views)

                loss = 0

                count = 0

                for i, t in enumerate(teacher_outputs):

                    for j, s in enumerate(student_outputs):

                        if i == j:

                            continue

                        loss += dino_loss(s, t, model.center)

                        count += 1

                loss /= count

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            update_teacher_ema(model.student_encoder, model.teacher_encoder)

            update_teacher_ema(model.student_projector, model.teacher_projector)

            teacher_cat = torch.cat(teacher_outputs, dim=0)

            update_center(model.center, teacher_cat)

            total_loss += loss.item()

            writer.add_scalar(
                "train/loss_step", loss.item(), epoch * len(loader) + step
            )

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)

        writer.add_scalar("train/loss_epoch", avg_loss, epoch)

        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch} Loss {avg_loss:.4f}")

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            },
            LAST_CHECKPOINT,
        )

    writer.close()


# ============================
# Main
# ============================


dataset = DINODataset()


loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    # num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)


model = DinoNet()


optimizer = torch.optim.AdamW(
    [
        {"params": model.student_encoder.parameters()},
        {"params": model.student_projector.parameters()},
    ],
    lr=1e-3,
    weight_decay=0.04,
)


train(model, loader, optimizer, epochs=100)
