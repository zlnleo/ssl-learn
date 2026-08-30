# 一步一步写 DeiT (CIFAR-100) —— 从零实现教程

> 目标：跟着本文件，从 0 开始一步步写出 `deit_cifar100.py`。
> 每一步都给出「为什么这么做 + 关键代码 + 对应论文位置 + 验证方法」。
> 全程只依赖 `torch` + `torchvision`，不依赖 timm / 预训练权重。
>
> 📌 2026-08-28 更新：本教程对应完整增强参考版 `deit_cifar100.py`（现已升级为含 EMA 的
> 完全体，`--ema` 开关）。当前仓库的学习主线是手写模块化版
> `deitmodel.py / deitloss.py / deitteacher.py / deittrain.py`（已训练出 63.27% 基线）；
> 两版对照阅读，教程里的每个"为什么"在两版里都成立。v2 学习路线见 `DeiT_v2学习路线.md`。

---

## Step 0：准备环境与数据骨架

**做什么**：建立工程骨架——固定随机种子、加载 CIFAR-100、写好 train/test 的 DataLoader。

**为什么**：Transformer 对随机性敏感，不做种子固定，两次实验结果不可比；CIFAR-100 是
100 类的"小 ImageNet"，正好用来验证 DeiT "有限数据也能训 Transformer" 的核心论点。

```python
import torch, torchvision
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)   # CIFAR 数据集的经验统计值
CIFAR100_STD  = (0.2673, 0.2564, 0.2762)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

train_set = CIFAR100(r'D:\project\self_supervised_learning\data', train=True,  download=True, transform=...)
test_set  = CIFAR100(r'D:\project\self_supervised_learning\data', train=False, download=True, transform=...)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=4)
```

✅ 验证：运行后能打印出 `50000 / 10000` 张训练/测试图。

---

## Step 1：Patch Embedding —— 把图像变成"单词序列"

**做什么**：32×32 的图切成 **4×4 的 patch**，每个 patch 用一个卷积投影成 `embed_dim` 维向量。

**为什么**：Transformer 只吃序列。ViT 用 16×16 patch 切 224 的图；CIFAR 只有 32×32，
**patch 必须用 4**，否则 32/16=2 → 只有 4 个 token，信息量太少。patch=4 → 8×8=**64 个 token**，
与 ImageNet 上的 ViT（196 tokens）保持同量级。用 `Conv2d(stride=patch)` 一行搞定切片+投影。

**对应论文**：Sec 2 "image patches as tokens"。

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2          # 64
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B,3,32,32) -> (B,192,8,8) -> (B,64,192)
        return self.proj(x).flatten(2).transpose(1, 2)
```

✅ 验证：`PatchEmbed()(torch.randn(2,3,32,32)).shape == (2, 64, 192)`。

---

## Step 2：多头自注意力

**做什么**：标准 scaled dot-product attention，`qkv` 用**一个** `Linear(dim, dim*3)` 一次性
投影再切 3 份、分 H 个头。

**为什么**：`qkv = self.qkv(x).reshape(B, N, 3, H, C//H).permute(2,0,3,1,4)` 这一行是
timm/DeiT 官方实现的经典写法——一次矩阵乘比三次快，且便于复用官方权重格式。

**对应论文**：Sec 2 "standard Transformer block"（DeiT 的注意力就是 ViT 原版，未改）。

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) \
                             .permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale      # (B,H,N,N)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))
```

✅ 验证：`Attention(192)(torch.randn(2,64,192)).shape == (2,64,192)`。

---

## Step 3：MLP + Encoder Block（+ Stochastic Depth）

**做什么**：两层 MLP（中间 4 倍宽 + GELU）；Block = **pre-LN** 的注意力残差 + MLP 残差；
再加论文里的 **Stochastic Depth（drop path）**。

**为什么**：
- DeiT/ViT 用 **LayerNorm 在前**（pre-norm），比 post-norm 训练更稳；
- **DropPath 是 DeiT 训练配方的核心正则之一**（论文 Sec 4.1，DeiT-B 用 0.1）：每层以概率
  丢弃整条残差分支，等价于随机训练"更浅"的子网络，防过拟合效果显著。

**对应论文**：Sec 2 + Sec 4.1。

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep = 1 - self.drop_prob
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep).div_(keep)  # 除以 keep 保持期望
        return x * mask

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
```

✅ 验证：`Block(192, 3)(torch.randn(2,64,192)).shape == (2,64,192)`。

---

## Step 4：拼出 ViT 主体（class token + 位置编码）

**做什么**：在 64 个 patch token 前面拼一个可学习的 **class token**，加上可学习位置编码，
过 12 层 Block，取出 class token 走分类头。

**为什么**：class token 是 ViT 的"汇集点"，通过注意力把整张图的信息聚合到自己身上；
位置编码让没有卷积的 Transformer 知道 token 的空间顺序。两者都用截断正态 `std=0.02` 初始化。

```python
class ViT(nn.Module):
    def __init__(self, embed_dim=192, depth=12, num_heads=3, num_classes=100):
        super().__init__()
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 + 1, embed_dim))   # 64 patches + 1 cls
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        x = self.patch_embed(x)                                   # (B,64,dim)
        x = torch.cat([self.cls_token.expand(x.shape[0],-1,-1), x], dim=1)  # (B,65,dim)
        x = x + self.pos_embed
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])                                 # 只取 class token
```

✅ 验证：`ViT()(torch.randn(2,3,32,32)).shape == (2, 100)`。此时你已经有"纯 ViT"基线了。

---

## Step 5：DeiT 关键改造 —— 蒸馏 token + 双分类头

**做什么**：在 class token 旁**再加一个 distillation token**，一起过注意力；最后两个 token
各接一个分类头；训练时输出两个 logits，**推理时取平均**。

**为什么**：这是 DeiT 论文 Fig.2 的灵魂。蒸馏 token 的作用是给教师信号留一条"专用通道"：
class token 学真值标签，dist token 学教师知识，二者在注意力中相互交流但不互相干扰；
推理时两个头的输出平均，等价于"真值视角 + 教师视角"的集成。

**对应论文**：Sec 3.2 "Distillation through attention"。

```python
class DistilledViT(nn.Module):
    def __init__(self, embed_dim=192, depth=12, num_heads=3, num_classes=100, distilled=True):
        super().__init__()
        self.distilled = distilled
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))   # ★ 新增
        self.pos_embed  = nn.Parameter(torch.zeros(1, 64 + 2, embed_dim))  # 65 -> 66
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)                  # 头 1: 真值
        self.head_dist = nn.Linear(embed_dim, num_classes) if distilled else None  # 头 2: 教师

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls  = self.cls_token.expand(x.shape[0], -1, -1)
        if self.distilled:
            dist = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, dist, x), dim=1)        # (B, 66, dim) ★ 一起过注意力
        else:
            x = torch.cat((cls, x), dim=1)
        x = self.blocks(x + self.pos_embed)
        x = self.norm(x)
        return (x[:, 0], x[:, 1]) if self.distilled else (x[:, 0], None)

    def forward(self, x):
        x_cls, x_dist = self.forward_features(x)
        if self.training:
            return self.head(x_cls), (self.head_dist(x_dist) if self.distilled else None)
        # 推理: 双头平均 —— 论文 Sec 3.2
        return (self.head(x_cls) + self.head_dist(x_dist)) / 2 if self.distilled else self.head(x_cls)
```

✅ 验证：训练模式下输出 `(B,100)` 和 `(B,100)` 两个张量；`eval()` 模式下输出一个 `(B,100)`。

---

## Step 6：训练配方 —— RandAugment / Mixup / CutMix / Random Erasing

**做什么**：实现论文 Sec 4.2 的增强四件套（RandAugment 手写，Mixup/CutMix 批内混合，
Random Erasing 用 torchvision）。

**为什么**：DeiT 论文最重要的论点就是 **"recipe matters"**——ViT 在 ImageNet 上训不好，
缺的不是架构而是这些增强与正则。RandAugment 参数严格对齐论文：`n=2, m=9, mstd=0.5, inc=1`
（每个 epoch 幅度 +1）；Mixup α=0.8、CutMix α=1.0、RE p=0.25。

**对应论文**：Sec 4.2（论文明确给出这些超参）。

```python
class RandAugmentCIFAR:
    """n 个随机操作, 幅度 m±mstd, 每 epoch 递增 inc (rand-m9-mstd0.5-inc1)"""
    def __init__(self, n=2, m=9, mstd=0.5, inc=1): ...
    def __call__(self, img):                    # img: [C,H,W], 像素 [0,1]
        for op in np.random.choice(self.ops, self.n, replace=False):
            mag = float(np.clip(np.random.uniform(self.m - self.mstd, self.m + self.mstd), 0, 30))
            img = op(img, mag)
        return img.clamp_(0., 1.)

def mixup_data(x, y, alpha):   # x' = lam*x + (1-lam)*x[idx]
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam, idx

def cutmix_data(x, y, alpha):  # 随机矩形区域替换
    ...

# 训练 transform 顺序: 裁剪+翻转 -> RandAugment -> RandomErasing(0.25) -> 归一化
train_transform = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
    RandAugmentCIFAR(n=2, m=9, mstd=0.5, inc=1),
    T.RandomErasing(p=0.25, scale=(0.02,0.33), ratio=(0.3,3.3), value='random'),
    T.Normalize(MEAN, STD),
])
```

✅ 验证：把增强后的图 `T.ToPILImage()` 回来肉眼检查；`mixup_data` 输出的图应该是两张图的叠影。

---

## Step 7：卷积教师 —— 先教出一个"师傅"

**做什么**：训练一个小卷积网络 `TeacherCNN`（VGG 风格 + BN）作为教师。

**为什么**：DeiT 的教师是 RegNetY-16GF（约 84.2%），我们没这个资源；但**蒸馏只需要教师
比学生强**。小 CNN 在 CIFAR-100 上 30 个 epoch 就能到 65~70%，足够当师傅。另外论文强调
**卷积教师比 Transformer 教师更有效**（蒸馏传递的是卷积的归纳偏置），所以我们特意选 CNN。

**对应论文**：Sec 3.2 "The convnet teacher" + 消融 Table。

```python
class TeacherCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 32 -> 16
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 16 -> 8
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 8 -> 4
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))
    def forward(self, x): return self.head(self.features(x))
```

教师训练用最朴素的 SGD + momentum + 余弦调度即可，30 epoch。教师训好后**冻结**（只推理）。

✅ 验证：教师测试精度 ≥ 65%，且保存 checkpoint 供后续复用。

---

## Step 8：蒸馏损失 —— 论文 Eq.(1) 硬蒸馏 / Eq.(2) 软蒸馏

**做什么**：实现两种蒸馏损失。总损失 = `(1-α)·分类头CE + α·蒸馏头损失`，α=0.5。

**为什么**：
- **硬蒸馏（Eq.1）**：教师先 `argmax` 出硬标签 `y_t`，学生的蒸馏头用 CE 去拟合。论文发现
  它与软蒸馏效果相当甚至更好，且**无需调温度**，实现最简单；
- **软蒸馏（Eq.2）**：教师学生都除温度 τ=3.0 后算 KL，损失乘 τ² 保持梯度尺度，能传递
  教师完整的类别相似性（比如"猫和狗容易混"这种信息）。

**对应论文**：Sec 3.2 公式 (1) (2)。

```python
def distillation_loss(student_out, teacher_logits, targets, args):
    logits_cls, logits_dist = student_out
    base = soft_cross_entropy(logits_cls, targets, smoothing=0.1)   # 头1: 真值 + 标签平滑

    if args.distill == 'hard':                     # Eq.(1)
        y_t = teacher_logits.argmax(dim=1)         # 教师硬标签
        dist = F.cross_entropy(logits_dist, y_t)   # 头2: 拟合教师标签
    else:                                          # Eq.(2), tau=3.0
        p_t = F.softmax(teacher_logits / args.tau, dim=1)
        dist = F.kl_div(F.log_softmax(logits_dist / args.tau, dim=1),
                        p_t, reduction='batchmean') * args.tau ** 2
    return (1 - args.alpha) * base + args.alpha * dist
```

> 💡 Mixup/CutMix 怎么和蒸馏共存？教师只在干净图像上前向一次得到 `Z_t`；若本批做了
> 混合（λ），就把教师 logits 用同一个 λ 混合作为目标：`Z_t = λ·Z_t + (1-λ)·Z_t[idx]`。
> 这样不必为混合图再前向一次教师，省一半算力。

✅ 验证：分别跑 `--distill hard/soft/none`，观察打印的 `dist` 项数值量级合理（CE 约 0~5）。

---

## Step 9：训练循环 —— AdamW + 线性缩放 + 余弦 + warmup

**做什么**：按论文 Sec 4.1 组装优化器与学习率调度。

**为什么**：
- **AdamW**：权重衰减解耦进优化器，是 Transformer 训练的标配；
- **权重衰减分组**：只衰减 2D 权重（Linear/Conv），**不衰减 bias 与 LayerNorm**，这是 DeiT
  官方实现细节，对精度有可感知影响；
- **lr = 1e-3 × batch/512 线性缩放** + 前 5 epoch warmup + 余弦衰减；
- **300 epochs**（CIFAR 上 100~300 均可）。

**对应论文**：Sec 4.1 "Optimization"。

```python
decay, no_decay = [], []
for name, p in student.named_parameters():
    (no_decay if p.ndim <= 1 else decay).append(p)      # bias/LN 权重是 1D
optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': 0.05},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=5e-4)

for epoch in range(1, epochs + 1):
    if epoch <= warmup:                       # 线性 warmup
        lr = base_lr * epoch / warmup
    else:                                     # 余弦衰减
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))
    for g in optimizer.param_groups: g['lr'] = lr
    train_one_epoch(...); evaluate(...)       # 每 epoch 打印 loss/test_acc/吞吐
```

每个 epoch 里还要记得 **RandAugment 幅度递增 1**（`rand-m9-inc1`）：
`for t in train_set.transform.transforms: if isinstance(t, RandAugmentCIFAR): t.m += 1`。

✅ 验证：loss 曲线平滑下降，warmup 阶段无发散；`test_acc` 单调爬升。

---

## Step 10：评估、消融与结论验证

**做什么**：推理时用双头平均输出；做三组对照实验。

**为什么**：论文的结论必须靠对照实验验证——同样的种子、同样的增强，只切换蒸馏开关。

```bash
python deit_cifar100.py --model tiny --distill none --epochs 100   # 基线 ViT
python deit_cifar100.py --model tiny --distill hard --epochs 100   # 论文 Eq.1
python deit_cifar100.py --model tiny --distill soft --epochs 100   # 论文 Eq.2
```

**期望观察**（对应论文结论）：
1. `hard` / `soft` 的最终精度 **比 `none` 高 1~3 个点**（ImageNet 上是 +1.6）；
2. `hard` 与 `soft` 非常接近（论文：83.4 vs 83.2，硬蒸馏略好）；
3. 蒸馏的收益主要在**训练后期**显现（学生学不动时教师兜底）；
4. 把教师换成更强的网络（或把学生换 `--model small` + `--drop-path 0.1`），整体精度再上一档。

✅ 验证：三组实验跑完后，把最优 test_acc 填进下表（这也正是 `solution.md` 中的预期表）：

| 配置 | 教师 acc | 学生 top-1 | Δ vs 基线 |
|---|---|---|---|
| DeiT-Ti（none） | — | | — |
| DeiT-Ti + hard | | | |
| DeiT-Ti + soft | | | |

---

## 附：常见坑清单

1. **忘记 `eval()` 时双头取平均** → 蒸馏模型测试精度虚低；
2. **Dropout/DropPath 在教师前向时没关** → 教师 logits 有噪声，蒸馏质量下降（务必 `teacher.eval()` + `no_grad`）；
3. **RandAugment 加在 Normalize 之后** → 像素范围错乱，模型不收敛；
4. **Mixup 混合了图像但没混合标签**（或反过来）→ loss 发散；
5. **权重衰减打到了 LayerNorm/bias 上** → 精度掉 1~2 个点，很隐蔽；
   1. **只测 1 个 epoch 就下结论** → 蒸馏收益要后期才出现，至少跑完 warmup + 50 epoch。 			
