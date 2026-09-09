import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms
from torchvision.models import resnet18

import torch.nn.functional as F
from tqdm import tqdm


# =========================
# 1. Data Augmentation
# =========================

augmentations = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),

    transforms.RandomApply(
        [transforms.ColorJitter(0.8,0.8,0.8,0.2)],
        p=0.8
    ),

    transforms.RandomGrayscale(p=0.2),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )
])


# =========================
# 2. Dataset
# =========================

class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = datasets.CIFAR10("./data",train=True,download=False)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img,_ = self.data[index]
        # 同一张图片生成两个增强版本
        xi = augmentations(img)
        xj = augmentations(img)
        return xi,xj



# =========================
# 3. Encoder
# =========================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)

        # CIFAR10:32x32图片，不适合ImageNet的大卷积
        self.model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.model.maxpool = nn.Identity()
        # 去掉分类器
        self.model.fc = nn.Identity()
    def forward(self,x):
        return self.model(x)

# =========================
# 4. Projection Head
# =========================
class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,128)
        )
    def forward(self,x):
        return self.net(x)

# =========================
# 5. SimCLR
# =========================
class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.projector = ProjectionHead()
    def forward(self,x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

# =========================
# 6. NT-Xent Loss
# =========================
class NTXentLoss(nn.Module):
    def __init__(self,temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self,zi,zj):
        batch = zi.shape[0]
        z = torch.cat([zi,zj],dim=0)
        z = F.normalize(z,dim=1)

        similarity = torch.matmul(z,z.T)
        similarity /= self.temperature
        # 删除自己和自己的相似度，对角线上的相似度设置为负的1e9
        mask = torch.eye(2*batch,device=z.device).bool()
        similarity.masked_fill_(mask,torch.finfo(similarity.dtype).min)
        # 正样本位置
        labels = torch.arange(batch,device=z.device)
        labels = torch.cat([labels+batch,labels])
        loss = F.cross_entropy(similarity,labels)
        return loss

# =========================
# 7. Train
# =========================
def train():
    device = ("cuda"if torch.cuda.is_available()else "cpu")
    print("device:",device)

    # TensorBoard
    writer = SummaryWriter("runs/simclr")
    dataset = SimCLRDataset()
    loader = DataLoader(dataset,batch_size=256,shuffle=True,drop_last=True,pin_memory=True)

    model = SimCLR().to(device)
    criterion = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)

    # =========================
    # AMP 混合精度
    # ⭐新增
    # =========================
    scaler = torch.amp.GradScaler("cuda")
    epochs = 100
    start_epoch = 0
    checkpoint_path = "checkpoint_latest.pth"
    # =========================
    # 自动恢复训练
    # ⭐新增
    # =========================
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )
        model.load_state_dict(
            checkpoint["model"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        start_epoch = checkpoint["epoch"] + 1
        print(f"恢复训练，从epoch {start_epoch}开始")
    else:
        print("没有checkpoint，从头训练")


    for epoch in range( start_epoch,epochs):
        model.train()
        total_loss = 0

        for xi,xj in tqdm(loader):
            xi = xi.to(device)
            xj = xj.to(device)
            with torch.amp.autocast("cuda"):
                zi = model(xi)
                zj = model(xj)
                loss = criterion(zi,zj)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        epoch_loss = (total_loss /len(loader))
        print(f"Epoch {epoch+1}, Loss:{epoch_loss:.4f}")
        # TensorBoard
        writer.add_scalar("Loss/train",epoch_loss,epoch)

        # 保存checkpoint

        checkpoint = {
            "epoch":epoch,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "loss":epoch_loss

        }
        torch.save(checkpoint,checkpoint_path)
    # 保存encoder
    torch.save(model.encoder.state_dict(),"simclr_encoder.pth")
    writer.close()



if __name__=="__main__":
    train()