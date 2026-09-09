import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from simCLR import Encoder

device="cuda"


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )
])


train_data=datasets.CIFAR10(
    "./data",
    train=True,
    transform=transform,
    download=False
)


test_data=datasets.CIFAR10(
    "./data",
    train=False,
    transform=transform,
    download=False
)


train_loader=DataLoader(
    train_data,
    batch_size=256,
    shuffle=True
)


test_loader=DataLoader(
    test_data,
    batch_size=256
)



# 加载encoder

encoder=Encoder()

encoder.load_state_dict(
    torch.load(
        "simclr_encoder.pth"
    )
)


encoder.to(device)


# 冻结

for p in encoder.parameters():
    p.requires_grad=False



classifier=nn.Linear(
    512,
    10
).to(device)



loss_fn=nn.CrossEntropyLoss()


optimizer=torch.optim.Adam(
    classifier.parameters(),
    lr=1e-3
)



for epoch in range(50):

    classifier.train()

    for x,y in train_loader:

        x=x.to(device)
        y=y.to(device)


        with torch.no_grad():

            feature=encoder(x)


        pred=classifier(feature)


        loss=loss_fn(
            pred,
            y
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



    print(
        "epoch",
        epoch
    )



# 测试

correct=0
total=0


classifier.eval()


with torch.no_grad():

    for x,y in test_loader:

        x=x.to(device)
        y=y.to(device)


        feature=encoder(x)

        pred=classifier(feature)


        correct += (
            pred.argmax(1)==y
        ).sum().item()


        total+=y.size(0)



print(
    "accuracy:",
    correct/total
)