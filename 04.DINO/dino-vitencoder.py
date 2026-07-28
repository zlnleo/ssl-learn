import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from tqdm import tqdm

import sys
import os

# 添加 ViT 目录到 Python 搜索路径
vit_path = r"D:\project\self_supervised_learning\ViT"
sys.path.append(vit_path)

# 引入 ViT 的模块
from vit_cifar100 import PatchEmbedding, CLS_Token, PositionEmbedding, TransformEncoder


writer = SummaryWriter("./runs/dino")
checkpoint_dir = "./checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

last_checkpoint_path = "./checkpoint/dino_last.pth"


# ============================
# Dataset
# ============================
class DINODataset(Dataset):
    def __init__(self, root='../data'):
        self.dataset = torchvision.datasets.CIFAR10(root, train=True, download=True)

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(24, scale=(0.5, 0.8)),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        global1 = self.global_transform(img)
        global2 = self.global_transform(img)
        locals = [self.local_transform(img) for _ in range(4)]
        return global1, global2, locals

    def __len__(self):
        return len(self.dataset)


# ============================
# ViT Encoder（替换 ResNet18）
# ============================
class ViTEncoder(nn.Module):
    def __init__(self, image_size=32, patch_size=4,
                 embed_dim=256, depth=6, heads=8):
        super().__init__()

        self.patch = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        num_tokens = self.patch.num_patches + 1

        self.cls = CLS_Token(embed_dim)
        self.pos = PositionEmbedding(num_tokens, embed_dim)
        self.encoder = TransformEncoder(depth, embed_dim, heads)
        self.norm = nn.LayerNorm(embed_dim)

        self.out_dim = embed_dim

    def forward(self, x):
        x = self.patch(x)
        x = self.cls(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0]   # 取 CLS token


# ============================
# Projector Head
# ============================
class ProjectorHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=2048, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================
# DINO Network
# ============================
class DinoNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.student_encoder = ViTEncoder()
        self.student_projector = ProjectorHead(in_dim=256)

        self.teacher_encoder = ViTEncoder()
        self.teacher_projector = ProjectorHead(in_dim=256)

        # 初始化 teacher = student
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data.copy_(ps.data)
            pt.requires_grad = False

        for ps, pt in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            pt.data.copy_(ps.data)
            pt.requires_grad = False

        self.register_buffer("center", torch.zeros(1, 256))

    def forward(self, views):
        # student 处理所有 view
        student_input = torch.cat(views, dim=0)
        student_feature = self.student_encoder(student_input)
        student_output = self.student_projector(student_feature)
        student_outputs = list(torch.chunk(student_output, len(views), dim=0))

        # teacher 只处理 global crop
        with torch.no_grad():
            teacher_input = torch.cat(views[:2], dim=0)
            teacher_feature = self.teacher_encoder(teacher_input)
            teacher_output = self.teacher_projector(teacher_feature)
            teacher_outputs = list(torch.chunk(teacher_output, 2, dim=0))

        return student_outputs, teacher_outputs


# ============================
# DINO Loss & EMA
# ============================
def update_teacher_ema(student, teacher, beta=0.996):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.copy_(beta * pt.data + (1 - beta) * ps.data)


def dino_loss(student_output, teacher_output, center,
              temp_s=0.1, temp_t=0.04):

    with torch.no_grad():
        teacher_prob = nn.functional.softmax(
            (teacher_output - center) / temp_t, dim=-1
        )

    student_log_prob = nn.functional.log_softmax(
        student_output / temp_s, dim=-1
    )

    loss = -(teacher_prob * student_log_prob).sum(dim=-1).mean()
    return loss


def update_center(center, teacher_out, momentum=0.9):
    batch_center = teacher_out.mean(dim=0, keepdim=True)
    center.data.copy_(momentum * center + (1 - momentum) * batch_center)


# ============================
# Checkpoint
# ============================
def save_checkpoint(epoch, net, optimizer):
    checkpoint = {
        "epoch": epoch,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(checkpoint, last_checkpoint_path)

    if (epoch + 1) % 10 == 0:
        path = os.path.join(checkpoint_dir, f"dino_epoch_{epoch+1}.pth")
        torch.save(checkpoint, path)
        print(f"save checkpoint: {path}")


# ============================
# Train
# ============================
def train(net, loader, optimizer, epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    net.to(device)

    start_epoch = 0
    if os.path.exists(last_checkpoint_path):
        ckpt = torch.load(last_checkpoint_path, map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"继续训练：Epoch {start_epoch}")
    else:
        print("没有checkpoint，从头开始训练")

    for epoch in range(start_epoch, epochs):
        total_loss = 0

        for g1, g2, locals in tqdm(loader):
            views = [g1.to(device), g2.to(device)]
            views.extend([v.to(device) for v in locals])

            student_outputs, teacher_outputs = net(views)

            loss = 0
            count = 0
            for i, t_out in enumerate(teacher_outputs):
                for j, s_out in enumerate(student_outputs):
                    if i == j:
                        continue
                    loss += dino_loss(s_out, t_out, net.center)
                    count += 1
            loss /= count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_teacher_ema(net.student_encoder, net.teacher_encoder)
            update_teacher_ema(net.student_projector, net.teacher_projector)

            teacher_cat = torch.cat(teacher_outputs, dim=0)
            update_center(net.center, teacher_cat)

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"epoch {epoch+1}/{epochs} loss {avg_loss:.4f}")
        save_checkpoint(epoch+1, net, optimizer)

    writer.close()


# ============================
# Run
# ============================
dataset = DINODataset()
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = DinoNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.04)

train(model, loader, optimizer, epochs=100)
