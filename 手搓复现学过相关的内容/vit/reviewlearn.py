import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 【本版新增】TensorBoard 写入器

try:
    from vit import DROPOUT, ViT
except ImportError:
    from vit_solution import DROPOUT, ViT
DATA_DIR = "../../data"

#构建cifar100
def build_cifar100_loader(args):
    from torchvision import datasets, transforms
    CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
    CIFAR100_STD = (0.2673, 0.2564, 0.2762)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    train_ds = datasets.CIFAR100(root=args.data_dir, train=True,
                                 download=True, transform=train_transforms)
    test_ds = datasets.CIFAR100(root=args.data_dir, train=False,
                                download=True, transform=test_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    return train_loader, test_loader,3,32,100

def build_fashionmnist_loader(args):
    """FashionMNIST loader 工厂（同初始版）。"""
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(root=args.data_dir, train=True,
                                     download=True, transform=transform)
    val_ds = datasets.FashionMNIST(root=args.data_dir, train=False,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    return train_loader, val_loader, 1, 28, 10

def train_one_epoch(model, train_loader,criterion, optimizer, scaler,device, epoch,args):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=scaler.is_enabled()):
            scores = model(images)
            loss = criterion(scores, labels)


        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        correct += (scores.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return total_loss/len(train_loader),correct/total

#eval
@torch.no_grad()
def evaluate(model, test_loader,criterion, device):
    model.eval()
    total_loss,correct,total = 0.0,0,0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        scores = model(images)
        loss = criterion(scores, labels)
        total_loss += loss.item()
        correct += (scores.argmax(-1) == labels).sum().item()
        total += labels.numel()
    return total_loss/len(test_loader),correct/total

def parse_args():
    parser = argparse.ArgumentParser(description="复现各种模版拼接")
    parser.add_argument('--epochs',type=int,default=100,help="训练轮次")
    parser.add_argument('--dataset',type=str,default="cifar100",choices=["cifar100","fashionmnist"],help="数据集")
    parser.add_argument("--data-dir",type=str,default=DATA_DIR,help="数据目录")
    parser.add_argument('--batch-size',type=int,default=128,help="一批")
    parser.add_argument("--lr",type=float,default=1e-3,help="学习率")
    parser.add_argument("--weight-decay",type=float,default=0.05,help="AdaW权重衰退")
    parser.add_argument("--grad-clip",type=float,default=1.0,help="梯度裁剪阈值")
    parser.add_argument("--num-workers",type=int,default=2,help="DataLoader进程数")
    parser.add_argument("--seed",type=int,default=42,help="随机种子默认42")
    parser.add_argument("--embed-size",type=int,default=192,help="token维度")
    parser.add_argument("--num-heads",type=int,default=6,help="多头注意力头的数目")
    parser.add_argument("--num-layers",type=int,default=6,help="encoder内部多少层")
    parser.add_argument("--dropout",type=float,default=DROPOUT,help="dropout")
    parser.add_argument("--amp",action="store_true",default=True,help="混合精度（GPU上默认开启）")
    parser.add_argument("--ckpt-dir",type=str,default="./checkpoint",help="checkpoint输出目录")
    parser.add_argument("--resume",action="store_true",help="断点续跑，从ckpt-dir/last.pt恢复完整的训练状态")
    parser.add_argument("--log-dir",type=str,default="./runs",help="实验记录目录（每次运行生成 run_时间戳/config.txt+train.log）")

    return parser.parse_args()


def main():
    args = parse_args()
    #固定种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #实验记录
    run_dir = os.path.join(args.log_dir,time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir,exist_ok=True)
    log_file = open(os.path.join(run_dir,"train.log"),"a",encoding="utf-8")
    with open(os.path.join(run_dir,"config.txt"),"w",encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write("{}={}\n".format(key,value))
    def log(msg):
        print(msg)
        log_file.write(msg+"\n")
        log_file.flush()

    #tensorboard
    writer = SummaryWriter(os.path.join(run_dir,"tfboard"))

    log(f"device:{device}")
    log(f"run_dir:{run_dir}")
    log(f"tensorboard:训练时另开终端执行 `tensorboard --logdir runs` 查看曲线")
    if args.dataset == "cifar100":
        train_loader,test_loader,in_channels,img_size,num_classes=build_cifar100_loader(args)
    else:
        train_loader,test_loader,in_channels,img_size,num_classes=build_fashionmnist_loader(args)
    log(f"dataset: {args.dataset}, classes: {num_classes}, "
        f"train batches: {len(train_loader)}, test batches: {len(test_loader)}")
    #模型
    model =ViT(img_size=img_size,patch_size=4,in_channels=in_channels,num_classes=num_classes,
               embed_size=args.embed_size,num_heads=args.num_heads,num_layers=args.num_layers,
               dropout=args.dropout).to(device)
    log(f"model parameters:{sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    #损失函数
    criterion = nn.CrossEntropyLoss()
    #优化器
    optimizer =torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    #使用混合精度
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cude",enabled=use_amp)

    #配置文件
    config =dict(img_size=img_size,patch_size=4,in_channels=in_channels,num_classes=num_classes,
               embed_size=args.embed_size,num_heads=args.num_heads,num_layers=args.num_layers,dropout=args.dropout)

    #断点续跑
    os.makedirs(args.ckpt_dir,exist_ok=True)
    best_path=os.path.join(args.ckpt_dir,"best.pt")
    last_path=os.path.join(args.ckpt_dir,"last.pt")
    start_epoch,best_acc =1,0.0
    if args.resume:
        if os.path.exists(last_path):
            ckpt = torch.load(last_path,map_location=device,weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"]+1
            best_acc = ckpt["best_acc"]
            log(f"[resume] 已从 {last_path} 恢复：上次跑到 epoch {ckpt['epoch']}，"
                f"best_acc {best_acc:.4f}，本轮从 epoch {start_epoch} 继续")
        else:
            log(f"[resume] 未找到 {last_path}，从头开始训练")
    #主训练循环
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    start = time.time()
    for epoch in range(start_epoch,args.epochs+1):
        train_loss,train_acc=train_one_epoch(model,train_loader,criterion,optimizer,scaler,device,epoch,args)
        test_loss,test_acc =evaluate(model,test_loader,criterion,device)
        scheduler.step()

        #tensorboard写入
        writer.add_scalar("train/loss",train_loss,epoch)
        writer.add_scalar("train/acc",train_acc,epoch)
        writer.add_scalar("test/loss",test_loss,epoch)
        writer.add_scalar("test/acc",test_acc,epoch)

        #模型调优
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state":model.state_dict(),
                "config":config,
                "best_acc":best_acc,
                "epoch":epoch,
            },best_path)
        #保存断点
        torch.save({
            "model_state":model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "scheduler_state":scheduler.state_dict(),
            "scaler_state":scaler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "config": config,
        },last_path)
        log(f"epoch {epoch:>3}/{args.epochs}, "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f} "
            f"(best: {best_acc:.4f})")

    writer.add_hparams(vars(args),{"best_acc":best_acc})
    writer.close()#关闭写入器
    log(f"training finished in {time.time() - start:.2f}s, "f"best test acc: {best_acc:.4f}")
    log(f"checkpoint:{args.ckpt_dir}/ (best.pt=最优模型, last.pt=续跑存档)")
    log(f"查看曲线: tensorboard --logdir {args.log_dir}")
    log_file.close()

if __name__ == '__main__':
    main()