import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from tqdm import tqdm



# ======================
# device
# ======================

device="cuda" if torch.cuda.is_available() else "cpu"

print(device)



# ======================
# encoder
# 必须和训练保持一致
# ======================

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



encoder,feature_dim=create_encoder()



# ======================
# 加载MoCo encoder_q
# ======================

checkpoint=torch.load(
    "./checkpoint/moco_encoder_epoch_20.pth",
    map_location=device
)


encoder.load_state_dict(
    checkpoint["encoder"]
)


encoder=encoder.to(device)



# ======================
# 冻结encoder
# ======================

encoder.eval()

for p in encoder.parameters():

    p.requires_grad=False



# ======================
# Linear classifier
# ======================


classifier=nn.Linear(
    feature_dim,
    10
).to(device)



# ======================
# 数据
# ======================


transform=transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        [0.4914,0.4822,0.4465],
        [0.2023,0.1994,0.2010]
    )

])


train_set=torchvision.datasets.CIFAR10(

    root="../data",

    train=True,

    download=True,

    transform=transform
)


test_set=torchvision.datasets.CIFAR10(

    root="../data",

    train=False,

    download=True,

    transform=transform
)



train_loader=DataLoader(
    train_set,
    batch_size=256,
    shuffle=True
)


test_loader=DataLoader(
    test_set,
    batch_size=256,
    shuffle=False
)



# ======================
# optimizer
# 注意：
# 这里只训练classifier
# ======================

optimizer=torch.optim.Adam(
    classifier.parameters(),
    lr=1e-3
)


criterion=nn.CrossEntropyLoss()



# ======================
# linear evaluation
# ======================


epochs=20


for epoch in range(epochs):


    classifier.train()


    total=0

    correct=0


    for x,y in tqdm(train_loader):


        x=x.to(device)

        y=y.to(device)



        with torch.no_grad():

            feature=encoder(x)



        pred=classifier(feature)


        loss=criterion(
            pred,
            y
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        total+=y.size(0)

        correct+=(pred.argmax(1)==y).sum().item()



    acc=100*correct/total


    print(
        f"epoch:{epoch}",
        f"train acc:{acc:.2f}%"
    )



# ======================
# test
# ======================


classifier.eval()


correct=0

total=0


with torch.no_grad():


    for x,y in test_loader:


        x=x.to(device)

        y=y.to(device)


        feature=encoder(x)


        pred=classifier(feature)


        correct+=(pred.argmax(1)==y).sum().item()

        total+=y.size(0)



print("test accuracy:",100*correct/total)