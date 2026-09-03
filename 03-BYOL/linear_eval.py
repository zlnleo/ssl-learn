import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm

# =====================
# Encoder
# =====================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=resnet18(weights=None)
        # CIFAR10版本ResNet
        self.net.conv1=nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.net.maxpool=nn.Identity()
        # 去掉分类头
        self.net.fc=nn.Identity()
    def forward(self,x):
        x=self.net(x)
        return x

# =====================
# Linear classifier
# =====================

class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(512,10)
    def forward(self,x):
        return self.fc(x)



# =====================
# load BYOL encoder
# =====================

def load_encoder(encoder,path):
    checkpoint=torch.load(path,map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder"])
    print("encoder loaded")

# =====================
# dataset
# =====================
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )
])


train_dataset=datasets.CIFAR10(root="../data",train=True,download=True,transform=transform)
test_dataset=datasets.CIFAR10(root="../data",train=False,download=True,transform=transform)



train_loader=DataLoader(train_dataset,batch_size=256,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=256,shuffle=False)


# =====================
# train linear
# =====================
def train(net,encoder,epochs=50):
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    encoder.to(device)
    net.to(device)
    #冻结encoder
    for param in encoder.parameters():
        param.requires_grad=False
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
    loss_fn=nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()
        total_loss=0
        correct=0
        total=0
        for X,y in tqdm(train_loader):
            X=X.to(device)
            y=y.to(device)
            with torch.no_grad():
                feature=encoder(X)
            pred=net(feature)
            loss=loss_fn(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()

            correct+=(pred.argmax(1)==y).sum().item()
            total+=y.size(0)
        acc=correct/total
        print(f"epoch:{epoch+1}",f"loss:{total_loss/len(train_loader):.4f}",f"train acc:{acc:.4f}")
        test(net,encoder)



# =====================
# test
# =====================

def test(net,encoder):
    device=next(net.parameters()).device
    net.eval()
    encoder.eval()
    correct=0
    total=0
    with torch.no_grad():
        for X,y in test_loader:
            X=X.to(device)
            y=y.to(device)
            feature=encoder(X)
            pred=net(feature)
            correct+=(pred.argmax(1)==y).sum().item()
            total+=y.size(0)
    print(f"test acc:{correct/total:.4f}")

# =====================
# main
# =====================


encoder=Encoder()


load_encoder(encoder,"./checkpoint/byol_encoder_60.pth")
classifier=LinearClassifier()

train(classifier,encoder,epochs=50)