import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm


# ============================
# dataset
# ============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5]*3,
        [0.5]*3
    )
])


train_dataset = torchvision.datasets.CIFAR10(
    "../data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    "../data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)



# ============================
# encoder
# 和DINO保持一致
# ============================

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.net=resnet18(weights=None)

        self.net.conv1=nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.net.maxpool=nn.Identity()

        self.net.fc=nn.Identity()


    def forward(self,x):

        return self.net(x)



# ============================
# 加载DINO encoder
# ============================


device="cuda" if torch.cuda.is_available() else "cpu"


encoder=Encoder()


checkpoint=torch.load(
    "./checkpoint/dino_last.pth",
    map_location=device
)


state_dict=checkpoint["model"]


# 只取student encoder
encoder_dict={}

for k,v in state_dict.items():

    if k.startswith(
        "student_encoder"
    ):

        new_key=k.replace(
            "student_encoder.",
            ""
        )

        encoder_dict[new_key]=v



encoder.load_state_dict(
    encoder_dict
)


encoder.to(device)


# 冻结encoder

for p in encoder.parameters():
    p.requires_grad=False


encoder.eval()



# ============================
# linear classifier
# ============================

classifier=nn.Linear(
    512,
    10
)

classifier.to(device)



optimizer=torch.optim.Adam(
    classifier.parameters(),
    lr=1e-3
)


loss_fn=nn.CrossEntropyLoss()



# ============================
# train linear
# ============================

epochs=20


for epoch in range(epochs):

    classifier.train()

    total_loss=0
    correct=0
    total=0


    for x,y in tqdm(train_loader):

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


        total_loss+=loss.item()


        correct += (
            pred.argmax(1)==y
        ).sum().item()

        total+=y.size(0)


    train_acc=correct/total


    print(
        f"epoch {epoch} "
        f"loss {total_loss/len(train_loader):.4f} "
        f"acc {train_acc:.4f}"
    )



# ============================
# test
# ============================


classifier.eval()

correct=0
total=0


with torch.no_grad():

    for x,y in test_loader:

        x=x.to(device)
        y=y.to(device)


        feature=encoder(x)


        pred=classifier(feature)


        correct+=(
            pred.argmax(1)==y
        ).sum().item()


        total+=y.size(0)


print(
    "test acc:",
    correct/total
)