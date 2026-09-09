import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

import os
os.makedirs("./checkpoint",exist_ok=True)

#augmentation
augmentations = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
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

#Dataset
class BYOLDataset(Dataset):
    def __init__(self,root='../data'):
        self.dataset=torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
        )
        self.transform = augmentations
    def __getitem__(self, index):
        img,_=self.dataset[index]
        view1=self.transform(img)
        view2=self.transform(img)
        return view1,view2
    def __len__(self):
        return len(self.dataset)

#Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet18(weights=None)

        self.net.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.net.maxpool=nn.Identity()
        self.net.fc=nn.Identity()
        self.out_dim=512
    def forward(self,x):
        x = self.net(x)
        x=torch.flatten(x,1)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self,x):
        return self.net(x)

#predictor
class Predictor(nn.Module):
    def __init__(self,dim=256,hidden_dim=4096):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self,x):
        return self.net(x)

#BYOL
class BYOL(nn.Module):
    def __init__(self):
        super().__init__()
        self.online_encoder=Encoder()
        self.online_projector=ProjectionHead()
        self.online_predictor=Predictor()
        #target网络
        self.target_encoder=Encoder()
        self.target_projector=ProjectionHead()

        #初始化target
        for param_online,param_target in zip(self.online_encoder.parameters(),self.target_encoder.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
        for param_online, param_target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
    def forward(self,x1,x2):
        o1=self.online_projector(self.online_encoder(x1))
        o2=self.online_projector(self.online_encoder(x2))
        p1=self.online_predictor(o1)
        p2=self.online_predictor(o2)

        #target
        with torch.no_grad():
            t1=self.target_projector(self.target_encoder(x1))
            t2=self.target_projector(self.target_encoder(x2))
        return p1,p2,t1,t2

#EMA更新update
def update_target(online,target,beta=0.996):
    for param_online, param_target in zip(online.parameters(),target.parameters()):
        param_target.data.copy_(beta * param_target.data+(1 - beta) * param_online.data)

def BYOL_loss(p,z):
    p=nn.functional.normalize(p,dim=1)
    z=nn.functional.normalize(z,dim=1)
    loss=2-2*(p*z).sum(dim=1)
    return loss.mean()
def save_checkpoint(model,optimizer,epoch,loss,path):
    torch.save({
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "loss":loss
    },path)
def load_checkpoint(model,optimizer,path):
    checkpoint=torch.load(
        path,
        map_location="cpu"
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    start_epoch=checkpoint["epoch"]
    print(
        f"resume from epoch {start_epoch}"
    )
    return start_epoch

def save_encoder(model,epoch,path):
    torch.save({
        "epoch":epoch,
        "encoder":model.online_encoder.state_dict()
    },path)
#移动优化器到gpu
def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k,v in state.items():
            if torch.is_tensor(v):
                state[k]=v.to(device)
def train(net,loader,optimizer,epochs=100,start_epoch=0):
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    net.to(device)
    optimizer_to(optimizer,device)
    for epoch in range(start_epoch,epochs):
        total_loss=0
        for x1,x2 in tqdm(loader):
            x1=x1.to(device)
            x2=x2.to(device)
            p1,p2,t1,t2=net(
                x1,x2
            )
            loss=(BYOL_loss(p1,t2)+BYOL_loss(p2,t1))/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_target(net.online_encoder,net.target_encoder)
            update_target(net.online_projector,net.target_projector)
            total_loss+=loss.item()
        avg_loss = total_loss / len(loader)
        print(f"epoch:{epoch}", f"loss:{avg_loss:.4f}")
        if (epoch+1) % 10 == 0:
            save_checkpoint(net,optimizer,epoch+1,avg_loss,"./checkpoint/byol_checkpoint.pth")
        if (epoch+1)%20==0:
            save_encoder(net,epoch+1,f"./checkpoint/byol_encoder_{epoch+1}.pth")
dataset=BYOLDataset()
loader=DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)
model=BYOL()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
#断点恢复
start_epoch=0
checkpoint="./checkpoint/byol_checkpoint.pth"
if os.path.exists(checkpoint):
    start_epoch=load_checkpoint(
        model,
        optimizer,
        checkpoint
    )

train(model,loader,optimizer,epochs=100,start_epoch=start_epoch)