import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import os


writer = SummaryWriter(
    "./runs/dino"
)
checkpoint_dir = "./checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

# 修改1：固定last checkpoint，用于自动断点续跑
last_checkpoint_path = "./checkpoint/dino_last.pth"


#dataset
class DINODataset(Dataset):
    def __init__(self, root='../data'):
        self.dataset=torchvision.datasets.CIFAR10(root, train=True, download=True)
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                32,
                scale=(0.7, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5] * 3,
                [0.5] * 3
            )
        ])
        #
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                24,
                scale=(0.5, 0.8)
            ),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5] * 3,
                [0.5] * 3
            )
        ])
    def __getitem__(self, index):
        img,_=self.dataset[index]
        global1 =self.global_transform(img)
        global2 =self.global_transform(img)
        locals = [
            self.local_transform(img)
            for _ in range(4)
        ]

        return global1,global2,locals
    def __len__(self):
        return len(self.dataset)

#encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=resnet18(weights=None)
        #按照cifar10修改
        self.net.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.net.maxpool=nn.Identity()
        self.net.fc=nn.Identity()
        self.out_dim=512
    def forward(self,x):
        x=self.net(x)
        x=torch.flatten(x,1)
        return x

#projector
class ProjectorHead(nn.Module):
    def __init__(self,in_dim=512,hidden_dim=2048,out_dim=256):
        super().__init__()
        self.net =nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,out_dim),
        )
    def forward(self,x):
        return self.net(x)

class DinoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_encoder=Encoder()
        self.student_projector=ProjectorHead()

        self.teacher_encoder=Encoder()
        self.teacher_projector=ProjectorHead()

        for params_student,params_teacher in zip(self.student_encoder.parameters(),self.teacher_encoder.parameters()):
            params_teacher.data.copy_(params_student.data)
            params_teacher.requires_grad=False
        for params_student, params_teacher in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            params_teacher.data.copy_(params_student.data)
            params_teacher.requires_grad = False

        self.register_buffer("center", torch.zeros(1, 256))

    def forward(self, views):
        # student一次处理所有view
        # =========================
        student_input = torch.cat(
            views,
            dim=0
        )
        student_feature = self.student_encoder(student_input)
        student_output = self.student_projector(student_feature)
        # 拆回不同view
        student_outputs = list(
            torch.chunk(
                student_output,
                len(views),
                dim=0
            )
        )
        # teacher只看global crop
        # 一次forward
        # =========================
        with torch.no_grad():
            teacher_input = torch.cat(views[:2],dim=0)
            teacher_feature = self.teacher_encoder(teacher_input)
            teacher_output = self.teacher_projector(teacher_feature)
            teacher_outputs = list(
                torch.chunk(
                    teacher_output,
                    2,
                    dim=0
                )
            )

        return student_outputs, teacher_outputs

def update_teacher_ema(student,teacher,beta=0.996):
    for param_student, param_teacher in zip(student.parameters(), teacher.parameters()):
        param_teacher.data.copy_(beta*param_teacher.data+(1.-beta)*param_student.data)

def dino_loss(student_output,teacher_output,center,temp_s=0.1,temp_t=0.04):

    # teacher 不参与梯度
    with torch.no_grad():
        teacher_prob = nn.functional.softmax(
            (teacher_output - center) / temp_t,
            dim=-1
        )
    # student 要 log_softmax
    student_log_prob = nn.functional.log_softmax(
        student_output / temp_s,
        dim=-1
    )
    loss = -(teacher_prob * student_log_prob).sum(dim=-1).mean()

    return loss

def update_center(center,teacher_out,momentum=0.9):
    batch_center=teacher_out.mean(dim=0,keepdim=True)
    center.data.copy_(momentum*center+(1-momentum)*batch_center)

# 修改2：统一保存函数
# last_checkpoint每轮覆盖，用于恢复训练
# epoch_checkpoint每10轮保存，用于比较模型

def save_checkpoint(epoch, net, optimizer):

    checkpoint = {
        "epoch": epoch,
        # 这里包含：
        # student_encoder
        # student_projector
        # teacher_encoder(EMA)
        # teacher_projector(EMA)
        # center buffer
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    # 1. 保存last，用于断点续跑
    torch.save(
        checkpoint,
        last_checkpoint_path
    )
    # 2. 每10轮额外保存
    if (epoch+1) % 10 == 0:
        epoch_path = os.path.join(
            checkpoint_dir,
            f"dino_epoch_{epoch+1}.pth"
        )
        torch.save(
            checkpoint,
            epoch_path
        )
        print(
            f"save checkpoint: {epoch_path}"
        )
def train(net,loader,optimizer,epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    net.to(device)
    start_epoch=0
    #断点续跑
    if os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(
            last_checkpoint_path,
            map_location=device
        )
        net.load_state_dict(
            checkpoint["model"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        start_epoch = checkpoint["epoch"] + 1
        print(f"继续训练：Epoch {start_epoch}")
    else:
        print("没有checkpoint，从头开始训练")

    for epoch in range(start_epoch,epochs):
        total_loss=0
        for g1,g2,locals in tqdm(loader):
            views = [g1.to(device), g2.to(device)]
            views.extend(
                [v.to(device) for v in locals]
            )

            student_outputs, teacher_outputs = net(views)
            loss = 0
            count = 0
            for i, teacher_out in enumerate(teacher_outputs):
                for j, student_out in enumerate(student_outputs):
                    # 跳过同一个Global
                    if i == j:
                        continue
                    loss += dino_loss(
                        student_out,
                        teacher_out,
                        net.center
                    )
                    count += 1
            loss /= count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_teacher_ema(net.student_encoder,net.teacher_encoder)
            update_teacher_ema(net.student_projector,net.teacher_projector)

            teacher_cat = torch.cat(teacher_outputs,dim=0)
            update_center(net.center,teacher_cat)
            total_loss+=loss.item()

        avg_loss = total_loss / len(loader)
        writer.add_scalar(
            "train/loss",
            avg_loss,
            epoch)
        writer.add_scalar(
            "train/lr",
            optimizer.param_groups[0]["lr"],
            epoch
        )
        print(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f}")
        save_checkpoint(epoch + 1,net,optimizer)
    writer.close()
dataset = DINODataset()
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = DinoNet()
optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.04
)

train(model, loader, optimizer, epochs=100)