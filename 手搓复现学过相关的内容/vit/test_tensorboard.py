from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/demo")              # ① 建 writer，指定事件文件目录
for i in range(100):
    writer.add_scalar("loss", 0.5 - i * 0.004, i)   # ② 写点：(标签, 值, 横轴坐标)
writer.close()