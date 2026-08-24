# 04 · train.py 完全讲解：从 argparse 到 AMP 的每个知识点

> 配套文件：`train.py`（CIFAR-100 训练脚本，已逐行注释）。
> 本文按"脚本结构 → 逐块精讲 → 你不会用的知识点深挖 → 常见报错 → 动手练习"组织。
> 适合打开 `train.py` 对照着读，一行行吃透。
> 专项延伸：`05_argparse完全讲解.md`（argparse 专题）、`06_scaler与scheduler完全讲解.md`（AMP 与学习率调度专题）、`07_训练骨架速查.md`（四种训练模板 + 断点续跑模板，写新脚本时抄）、`08_wandb完全讲解.md`（实验记录平台）、`09_tensorboard完全讲解.md`（本地看板）。
> 工业流程全貌：`10_工业训练流程全景与清单.md`（15 个零件总地图 + 你缺什么），其子专题为 `11_yaml配置管理完全讲解.md`、`12_DDP分布式训练完全讲解.md`、`13_可复现性工程完全讲解.md`、`14_推理与模型导出完全讲解.md`、`15_Git与Docker工程工具完全讲解.md`。
> 训练脚本版本对照：`16_训练版本对照与使用指南.md`（train.py + v2 TensorBoard + v3 早停 + v4 可复现 + v5 hydra + v6 DDP，每个版本一个功能）。

---

## 一、这个脚本在干什么（总览）

训练一个手写 ViT 在 CIFAR-100（100 类）上做分类。整个脚本是"五段式"结构：

```
配置层(argparse/常量) → 数据层(Dataset/transform/DataLoader) → 模型层(ViT)
     → 训练层(train_one_epoch/evaluate) → 主流程(main)
```

**一个 epoch 的数据流**：

```
train_loader ──> train_one_epoch（前向→loss→反向→裁剪→更新）──> (train_loss, train_acc)
test_loader  ──> evaluate（纯前向，不动参数）                  ──> (test_loss, test_acc)
                        │
                 scheduler.step()   ← 每轮结束更新学习率
                        │
                 test_acc 创新高？ ──是──> torch.save 最优 checkpoint
```

---

## 二、逐块精讲

### 2.1 配置层（文件顶部 + parse_args）

- `DATA_DIR = "../../data"`：相对路径，从 vit/ 目录向上两级到 `self_supervised_learning/data`；
- `CIFAR100_MEAN/STD`：整个数据集的像素均值和标准差，归一化用。这三个数不是随便编的，是**对 5 万张训练图统计出来的**（预计算好的公开值），作用是把像素分布拉到以 0 为中心、方差约 1，训练更稳定。

### 2.2 数据层

- `transforms.Compose([...])`：把多个变换串成流水线，**按列表顺序执行**；
- `RandomCrop(32, padding=4)`：先把 32×32 的图四周各 pad 4 像素（变成 40×40），再随机裁回 32×32——等价于给每张图随机平移了 0~4 像素，制造"位置变化"的增强；
- `RandomHorizontalFlip()`：50% 概率左右翻转；
- `ToTensor()`：把 PIL 图（H,W,C 的 uint8）变成 (C,H,W) 的 float32 张量，值域 [0,1]；
- `Normalize(mean, std)`：逐通道做 `(x-mean)/std`；
- 测试集只有 ToTensor+Normalize，**不做随机增强**——评估要用"干净分布"，否则测出来的是增强能力不是分类能力。

### 2.3 模型层

`model = ViT(...).to(device)`。`.to(device)` 把模型所有参数搬到 GPU；`sum(p.numel() for p in model.parameters())` 统计参数量。

### 2.4 训练层（train_one_epoch / evaluate）

见下文第四节 AMP 深挖；evaluate 的三个区别见第三节装饰器部分。

### 2.5 主流程（main）

装配 + 循环 + 调度 + 保存最优，见下文各知识点。

---

## 三、你不会用的内容，逐个讲透（重点）

### 3.1 argparse：命令行参数

```python
parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
```

四个要素：**参数名 / 类型 / 默认值 / 帮助文字**。

- `type=int`：命令行传进来的是字符串 `"50"`，argparse 自动转成 int 50；
- `default=100`：不传就用 100；
- `choices=["cifar100", "toy"]`：白名单，拼错会直接报错（帮你挡低级错误）；
- `action="store_true"`：**开关型**。命令行里出现 `--use-wandb` 就是 True，不出现就是 False。注意它不能像 `--epochs 50` 那样带值。

**为什么用 argparse 而不是改代码里的常量？** 换超参不改代码 → 不容易手滑改坏训练逻辑；配置和逻辑分离；脚本之间可以互相调用。用法：`python train.py --epochs 50 --lr 3e-4`。返回值 `args` 是个命名空间，用 `args.epochs` 取值。

### 3.2 `@torch.no_grad()`：装饰器

```python
@torch.no_grad()
def evaluate(...): ...
```

`@` 装饰器 = 把函数"包一层"。上面这句**完全等价于**：

```python
def evaluate(...):
    with torch.no_grad():   # 把整个函数体缩进进这个上下文
        ...原来的所有代码...
```

`torch.no_grad()` 的作用：让里面所有运算**不构建计算图、不记录梯度**。验证/推理阶段只需要输出，不需要反向传播，省显存、跑得快。典型用途：evaluate、推理脚本、你的 transformer 里 `generate` 用的也是它（`@torch.no_grad()` 装饰整个方法）。

### 3.3 `model.train()` 和 `model.eval()`：模式开关

这两个方法**本身不训练也不推理**，只切换模型内部"行为模式"：

| 层 | train() 模式 | eval() 模式 |
|---|---|---|
| Dropout | 开启（随机丢神经元） | 关闭（全保留） |
| BatchNorm | 用当前 batch 的统计量 | 用训练期累计的全局统计量 |

**铁律**：训练前 `model.train()`，验证/推理前 `model.eval()`。忘了 eval 的症状：验证结果每次跑都不一样（dropout 在随机丢），或者加了 BN 后验证准确率崩掉。我们的 ViT 只有 dropout，所以症状是前者。

### 3.4 AMP 混合精度（train_one_epoch 的核心）

**一句话**：前向用 fp16 算（快、省显存），但用 GradScaler 防止 fp16 精度不够导致梯度下溢。

**为什么需要它**：fp16 能表示的最小正数约 6e-5。梯度本来就很小，转成 fp16 时比 6e-5 更小的部分直接变成 0——参数就"学不动"了。

**四行关键代码逐一解释**：

```python
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=...):
    logits = model(images)     # ① 前向自动切 fp16：matmul/conv 用 fp16 加速
    loss = criterion(logits, labels)
```

- `autocast` 是一个"区域声明"：只包**前向**。里面 torch 会自动决定哪些算子用 fp16（矩阵乘、卷积），哪些必须留在 fp32（softmax、LayerNorm、loss——它们对精度敏感，torch 内置了白名单）。
- `enabled=...`：CPU 上关掉（CPU 不支持 fp16 加速）。

```python
scaler.scale(loss).backward()   # ② 放大 loss 再反向
```

- `scaler` 内部维护一个放大系数（初始 65536）。`scale(loss)` 把 loss 乘以这个系数再反向，梯度也同比放大，小梯度就被"抬"回 fp16 能表示的范围；
- 系数会在 `scaler.update()` 时动态调整。

```python
scaler.unscale_(optimizer)      # ③ 把梯度还原成真实尺度
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

- **裁剪前必须先 unscale_**：否则你裁的是放大过的梯度，阈值就失效了；
- `clip_grad_norm_`：把整个模型的梯度范数限制在 1.0 以内。作用：防止某一步梯度爆炸把参数甩飞。带 `_` 后缀 = 原地修改。

```python
scaler.step(optimizer)   # ④ 代替 optimizer.step()
scaler.update()          # ⑤ 更新放大系数
```

- `scaler.step()` 会先检查梯度有没有 inf/nan：**有则自动跳过本次更新**（并触发系数减半），没有才真正更新参数——这是 AMP 的自我保护；
- `scaler.update()`：如果上几步一切正常，系数按计划逐步增大（放大倍数越大，能救回越小的梯度）；如果出现过 inf/nan，系数减半。

**禁用 AMP 时**（CPU 或 `--amp` 关闭）：`enabled=False` 让 `autocast` 和 `GradScaler` 全部变成"透明通道"，四行代码退化成标准的 `loss.backward() → clip → optimizer.step()`，行为完全一致——这就是这套写法的优雅之处。

**fp16 vs bf16**：bf16 表示范围和 fp32 一样大（只是精度低），所以 bf16 训练**不需要 GradScaler**（不会溢出/下溢），但需要 Ampere 以上 GPU 支持。工业上大模型训练普遍用 bf16；你的环境用 fp16 + GradScaler 是通用安全的选择。

### 3.5 学习率调度器：CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
...
scheduler.step()   # 每个 epoch 结束调一次
```

- 学习率随训练进行按**余弦曲线**从初始值平滑衰减到 0：前期大步快学，后期小步精调，收敛比固定 lr 好；
- `T_max=epochs`：一个完整余弦周期覆盖全部训练；
- **调用时机**：我们这个调度器是按 epoch 调度的（`scheduler.step()` 在 epoch 循环里），每次 `step()` 读一次当前 lr 并更新。注意有些调度器（如 OneCycleLR）要**每个 batch** 调一次——用之前必须看文档确认。

**进阶：warmup**。训练刚开始时模型是随机的、梯度很大，直接给满 lr 容易把参数冲散。warmup 的做法是前几个 epoch 把 lr **线性爬升**到峰值，再开始余弦衰减——lr 曲线变成"上坡 + 下坡"。完整配方是 warmup+cosine，研究型训练基本标配。

### 3.6 AdamW 与 weight decay

- **weight decay（权重衰减）**：每次更新参数时，额外把参数朝 0 缩一点（`θ ← θ - lr·λ·θ`），是防过拟合的正则手段；
- **Adam 的坑**：标准 Adam 把 weight decay 混进了自适应学习率的分母里，导致衰减效果被"稀释"，对大权重和小权重的衰减力度不一致；
- **AdamW**：把 weight decay 从梯度更新里**拆出来**，直接对参数做纯衰减——这就是"解耦"的含义。ViT/Transformer 系列几乎都用 AdamW + weight_decay=0.05。

### 3.7 固定随机种子

```python
torch.manual_seed(args.seed)              # CPU 随机数
torch.cuda.manual_seed_all(args.seed)     # 所有 GPU 随机数
```

同样的种子 + 同样的代码 + 同样的数据 → 结果可复现。**但注意**：这只保证"重跑完全一样"，不保证"不同机器/不同 cudnn 版本结果一样"——GPU 上还有 cuDNN 的非确定性（`torch.backends.cudnn.deterministic = True` 可强制确定，但会变慢）。科研里记 seed 是底线，跨机器复现要连环境一起记。

### 3.8 torch.save / state_dict：模型怎么存

```python
torch.save({
    "model_state": model.state_dict(),   # 参数名 -> 张量的字典
    "config": {...},                     # 网络结构超参数
    "best_acc": best_acc, "epoch": epoch,
}, "checkpoint/best.pt")
```

- `state_dict()` 只存**参数**，不含网络结构——加载时必须先按 config 重建 `ViT(...)` 再 `load_state_dict`；
- 所以 config（embed_size、num_layers 等）必须一起存，否则拿到权重也不知道怎么搭模型；
- train.py 现在存两份：`checkpoint/best.pt` 只留"测试准确率创新高"的模型（部署用）；`checkpoint/last.pt` 每个 epoch 更新、额外带 optimizer/scheduler/scaler 状态（断点续跑用，见 07 速查文档）。

### 3.9 wandb：实验记录

```python
wandb.init(project="vit-cifar100", config=vars(args))   # 建一个实验
wandb.log({"train_loss": x, "test_acc": y})             # 记录一个点（循环里调）
```

网页端自动画曲线、存超参数、能对比多次实验。`vars(args)` 把 argparse 的参数变成字典上传，这样每个实验的配置都留档。**你的阶段规划铁律**："每个实验进 wandb"——养成习惯后，调参过程全都有据可查。

### 3.10 f-string 格式化

```python
print(f"epoch {epoch:>3}/{args.epochs}, train_loss: {train_loss:.4f}")
```

- `{x:.4f}`：保留 4 位小数；
- `{x:>3}`：右对齐、占 3 格（`1` 会显示成 `  1`），让多行输出列对齐；
- f-string 是 Python 3.6+ 的字符串内嵌变量写法，比 `"{}".format()` 和 `%` 格式化更快更可读。

### 3.11 time.time() 计时

`start = time.time()` 记录当前时间戳（秒），结束时 `time.time() - start` 就是训练耗时。粗粒度计时够用；精确测量单段代码耗时用 `time.perf_counter()`。

### 3.12 `if __name__ == "__main__":` 守卫

- 直接 `python train.py` 运行：`__name__` 是 `"__main__"` → 执行 main()；
- 被 import（如 `from train import ToyVisionDataset`）：`__name__` 是 `"train"` → 不执行 main()，只提供函数和类。

**Windows 上的特殊重要性**：`num_workers>0` 时 DataLoader 会 spawn 子进程，子进程会重新 import 本文件；没有守卫的话每个子进程都会再执行一次 main() → 递归开进程直到死循环。这就是为什么训练脚本几乎都必须有这个守卫。

### 3.13 DataLoader 的两个参数

- `num_workers`：几个子进程并行读图+预处理。0 = 主进程自己干（慢）；开多了浪费内存。Windows 上开 >0 必须配 3.12 的守卫；
- `pin_memory=True`：把数据放进"锁页内存"，GPU 拷贝（`x.to("cuda")`）走更快的 DMA 通道，训练吞吐提升。

---

## 四、常见报错排查表

| 报错/症状 | 原因 | 解法 |
|---|---|---|
| `RuntimeError: DataLoader worker killed` / 死循环 | Windows 下 num_workers>0 无 `__main__` 守卫 | 加守卫（已加）或 num_workers=0 |
| loss 出现 nan | fp16 溢出或 lr 太大 | 看 scaler 是否报 skipped；调小 lr |
| 验证准确率忽高忽低 | 验证时没 `model.eval()` | 加 eval() |
| `expected scalar type Half but found Float` | 手写算子逃出 autocast 的 fp32 白名单 | 把该算子放进 autocast 里或手动 .half() |
| 保存的 checkpoint 加载报 shape 错 | config 和训练时不一致 | 保存 config、加载时按 config 重建 |
| 训练到一半显存爆掉 | batch 太大或梯度没释放 | 调小 batch；检查 zero_grad 时机 |

---

## 五、动手练习（改一处，看一个现象）

1. **关掉 AMP**：`--amp` 默认开，试试 CPU 跑 toy 数据（自动关）和 GPU 关 `--amp` 对比速度与显存——感受 fp16 的价值；
2. **加 warmup**：给 scheduler 加一个 5 epoch 的线性 warmup（提示：`torch.optim.lr_scheduler.LambdaLR`），对比前几个 epoch 的 loss 曲线；
3. **每个 epoch 打学习率**：`print(scheduler.get_last_lr())` 加进循环，肉眼验证余弦曲线；
4. **把 wandb 接上**：`pip install wandb` 后用 `--use-wandb` 跑 toy 任务，看网页曲线长什么样；
5. **写个加载脚本**：加载 `checkpoint/best.pt`（config 重建模型 + load_state_dict），在测试集上复现 best_acc——这是"训练→保存→部署"闭环的最后一环；
6. **断点续跑体验**：跑 toy 任务到一半 Ctrl+C 中断，然后 `python train.py --dataset toy --resume`，观察它从第几轮接着练。

---

## 六、和你的阶段规划对表

这份脚本覆盖了 `D:\project\new\study\02` 里"必做 4 项"的 **训练循环** 和 **混合精度** 两项。对照完本文后，请给脚本的每个知识点写一句话笔记进 `notes/回顾笔记/`（规划要求的验收动作），然后继续攻 **损失函数**（InfoNCE/NT-Xent 手推）和 **优化器**（SGD vs AdamW 内部机制）两项。
