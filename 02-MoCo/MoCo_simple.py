import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader

import os
from tqdm import tqdm
save_dir="./checkpoint"

os.makedirs(
    save_dir,
    exist_ok=True
)


best_loss=float("inf")

# 保存指定epoch
save_epochs=[20,50,100,150,200]

#augmentation的transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(32,scale=(0.2,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)],p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )
])

#augmentation
class Augmentation:
    def __init__(self,transform):
        self.transform = transform
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2

#create encoder
def create_encoder():
    encoder=models.resnet18(weights=None)
    # CIFAR10:
    # 原ImageNet:
    # 7x7 stride2 + maxpool
    # 对32x32图片损失太大
    encoder.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    encoder.maxpool = nn.Identity()
    dim=encoder.fc.in_features
    encoder.fc = nn.Identity()
    return encoder, dim

#MoCo
class MoCo(nn.Module):
    def __init__(self, dim=128,K=4096,m=0.999,T=0.07):
        super().__init__()
        self.K=K
        self.m=m
        self.T=T
        #quert encoder
        self.encoder_q,features_dim=create_encoder()
        #key encoder
        self.encoder_k,_=create_encoder()

        self.projection_q=nn.Sequential(
            nn.Linear(features_dim,256),
            nn.ReLU(),
            nn.Linear(256,dim),
        )
        self.projection_k = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )
        #初始化参数
        self.encoder_k.load_state_dict(
            self.encoder_q.state_dict()
        )
        self.projection_k.load_state_dict(
            self.projection_q.state_dict()
        )
        #key不进行参数更新
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projection_k.parameters():
            param.requires_grad=False

        #queue
        self.register_buffer("queue",torch.randn(dim,K))
        self.queue=F.normalize(self.queue,dim=0)
        self.register_buffer("queue_ptr",torch.zeros(1,dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        #更新key encoder
        for q,k in zip(self.encoder_q.parameters(),self.encoder_k.parameters()):
            k.data=(self.m*k.data+(1-self.m)*q.data)
        for q,k in zip(self.projection_q.parameters(),self.projection_k.parameters()):
            k.data=(self.m*k.data+(1-self.m)*q.data)
    #queue 更新
    @torch.no_grad()
    def deque_queue(self,k):
        batch_size=k.shape[0]
        ptr=int(self.queue_ptr)
        self.queue[:, ptr:ptr+batch_size]=k.T
        ptr=(ptr+batch_size)%self.K
        self.queue_ptr[0]=ptr
    def forward(self,im_q,im_k):
        q=self.encoder_q(im_q)
        q=self.projection_q(q)
        q=F.normalize(q,dim=1)

        with torch.no_grad():
            self.momentum_update()
            k=self.encoder_k(im_k)
            k=self.projection_k(k)
            k=F.normalize(k,dim=1)

        l_pos=torch.einsum('nc,nc->n',[q,k]).unsqueeze(-1)
        l_neg=torch.einsum("nc,ck->nk",[q,self.queue.clone().detach()])

        logits=torch.cat([l_pos,l_neg],dim=1)
        logits/=self.T
        labels=torch.zeros(logits.shape[0],dtype=torch.long,device=logits.device)
        loss=F.cross_entropy(logits,labels)
        self.deque_queue(k)
        return loss

#dataset
dataset=torchvision.datasets.CIFAR10(
    root='../data',
    train=True,
    download=True,
    transform=Augmentation(transform)
)
loader=DataLoader(dataset,batch_size=128,shuffle=True,drop_last=True,pin_memory=True)
device ='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model=MoCo().to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=0.03,momentum=0.9,weight_decay=1e-4)
epochs=20


start_epoch=0
resume=False
if resume:

    checkpoint=torch.load(
        "./checkpoint/moco_last.pth"
    )


    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    start_epoch=checkpoint["epoch"]+1

    best_loss = checkpoint["best_loss"]


for epoch in range(start_epoch,epochs):
    model.train()
    total_loss=0
    for (x1, x2), _ in tqdm(loader):
        x1 = x1.to(device)
        x2 = x2.to(device)

        loss = model(x1, x2)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(loader)

    print(f"epoch:{epoch}",f"loss:{avg_loss:.4f}")
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": avg_loss,
    }
    # 每轮保存最新
    torch.save(
        checkpoint,
        "./checkpoint/moco_last.pth"
    )
    # 保存最佳
    if epoch + 1 in save_epochs:
        print(f"Saving encoder epoch {epoch + 1}")
        torch.save(
            {
                "epoch": epoch + 1,
                "encoder":model.encoder_q.state_dict(),
                "loss":avg_loss
            },
            f"./checkpoint/moco_encoder_epoch_{epoch + 1}.pth"
        )
