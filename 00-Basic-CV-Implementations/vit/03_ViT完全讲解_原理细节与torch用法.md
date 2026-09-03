# 03 · ViT 完全讲解：原理、细节与 PyTorch 用法

> 这份文档的目标：把 ViT **讲透**——每个模块为什么这么设计、哪些细节最容易错、
> 写 ViT/Transformer 用到的 PyTorch API 逐个讲清楚。
> 适合在写完 `vit.py` 并对照 `vit_solution.py` 复盘后精读。

---

## 0. 一句话定位 ViT

ViT = **把图片切成小方块当作"词"**，把你 Transformer 的 **Encoder** 拿来用（去掉所有掩码），最后加一个分类头。

| | 你的 transformer (NLP) | ViT (视觉) |
|---|---|---|
| 输入 | 词 id 序列 | patch 向量序列 |
| 结构 | Encoder + Decoder | 只有 Encoder |
| 掩码 | 因果 / padding | 无（双向注意力） |
| 位置编码 | 正弦（固定 buffer） | 可学习（Parameter） |
| 激活 | ReLU | GELU |
| 归一化 | post-LN | pre-LN |
| 输出 | 逐位置 logits | 整图一个分类 |

---

## 1. 数据流总览

```
(B, C, H, W) 图片
      │  ① PatchEmbedding：切块 + 展平 + 线性投影
      ▼
(B, N, E)  patch token 序列（N = (H/P)×(W/P)）
      │  ② 拼 [CLS] token → 加位置编码 → dropout
      ▼
(B, N+1, E)
      │  ③ × num_layers：EncoderBlock（pre-LN 自注意力 + MLP）
      ▼
(B, N+1, E)
      │  ④ 最后 LN → 取 [CLS]（或 patch 平均）→ Linear
      ▼
(B, num_classes)
```

---

## 2. 模块深挖

### 2.1 PatchEmbedding —— 为什么必须切块？

注意力是 **O(n²)** 的：224×224 的图有 50176 个像素，当 5 万个 token 直接爆炸；
切成 16×16 的 patch 后只剩 196 个 token，和一句话差不多长。

**两个等价实现**：

| 写法 | 做法 | 特点 |
|---|---|---|
| Conv2d（你用/答案用） | `nn.Conv2d(C, E, kernel_size=P, stride=P)` | 一行代码，"滑窗=切块，卷积核=共享的投影矩阵" |
| unfold + Linear | `F.unfold(img, P, stride=P)` 得到 (B, C·P·P, N) 再转置过 Linear | 更直观暴露"展平→投影"两步，慢一点 |

关键理解点：**每个 patch 都用同一个投影矩阵**——所以一个卷积核就够，这就是 Conv2d 等价的原因。

细节：Conv2d 输出 `(B, E, H/P, W/P)`，要 `flatten(2)` 成 `(B, E, N)` 再 `transpose(1, 2)` 成 `(B, N, E)`。顺序反了就是 `(B, E, N)`，后续全错。

### 2.2 [CLS] token —— 为什么需要它？

注意力输出是"每个位置一个向量"，但分类要"整张图一个结果"。两种聚合方式：

- **`[CLS]`（原论文）**：一个可学习向量拼在序列最前面。经过 N 层双向注意力，它理论上和每个 patch 都交互过，等于"汇总了全图"。最后取 `x[:, 0]`。
- **GAP（全局平均池化）**：`x[:, 1:].mean(dim=1)` 对所有 patch 向量取平均。后续工作证明效果相当，还省一个 token 的计算。

`[CLS]` 为什么能"汇总"？注意它不是魔法——它只是一个普通的、会被注意力更新的向量，学到最后它自然学会去"看"和分类相关的 patch（可以用注意力可视化验证：`[CLS]` 那行的权重分布就是它"在看哪"）。

### 2.3 位置编码 —— 最容易出错的一节

**为什么必须加**：注意力对输入顺序完全无感（置换不变性）。我们的消融实验实测：去掉位置编码，象限任务准确率从 100% 掉到 25%（随机猜）。

**可学习 vs 正弦**：

| | 可学习（ViT 原论文） | 正弦（你的 transformer） |
|---|---|---|
| 实现 | `nn.Parameter(torch.zeros(1, N+1, E))` | 三角函数算好存 `register_buffer` |
| 优点 | 灵活，让模型自己学空间关系 | 零参数；理论上能外推更长的序列 |
| 缺点 | 换分辨率要插值；多占一点参数 | 图像上通常不如可学习 |

**两个必踩的坑**：

1. **+1**：序列里多了 `[CLS]`，位置编码长度必须是 `N+1`，不是 `N`。这是全 ViT 最高频的 shape 错误；
2. **顺序**：先 `cat([cls, x])` 再 `+ pos_embed`。顺序反了 `[CLS]` 就拿不到"第 0 号位置"编码。

**进阶**：训练用 32×32（N=64），推理想喂 64×64（N=256）怎么办？把 (1, 65, E) 的位置编码按 patch 的二维排布 reshape 成 (1, 8, 8, E)，用 `F.interpolate` 插值到 (1, 16, 16, E)，再展平回 (1, 257, E)——这是 ViT 部署时的常见操作。

### 2.4 多头注意力 —— 你写过，但细节值得再抠一遍

- **切头**：`reshape(B, S, H, d_k)` 再 `transpose(1, 2)` → `(B, H, S, d_k)`。为什么不是直接 view？因为要保证"每个头的 d_k 个维度是连续的一段"——reshape 恰好按内存顺序切成 H 段，每段是一个头。
- **transpose 后不 contiguous 也能算**：`transpose` 只是改了 stride，不搬数据；matmul 会自己处理非连续张量。但注意 `view()` 要求连续内存，非连续张量上要改用 `reshape()`（自动拷贝）。
- **缩放 √d_k**：d_k 越大点积方差越大，softmax 会饱和到梯度消失区，除一下把分数拉回正常范围。
- **dropout 位置**：注意力权重上的 dropout（softmax 后）和输出投影后的 dropout 是两处不同的正则，ViT 原论文两者都有。
- **mask**：ViT 完全不需要——没有"未来"要挡（无自回归），没有 PAD 要屏蔽（长度恒等）。你保留 mask 参数是为了代码和 transformer 一致，调用时永远传 None。
- **einsum 写法（扩展阅读）**：`scores = torch.einsum('bhqd,bhkd->bhqk', q, k)` 和 matmul 写法完全等价，einsum 的好处是下标即文档。

### 2.5 MLP 与 GELU

注意力是"线性组合"（加权平均），多层堆起来还是线性；MLP 提供**逐位置的非线性变换和通道间混合**（升维再降维）。这就是"注意力负责空间交互、MLP 负责逐点加工"的分工。

**GELU vs ReLU**：ReLU 在 0 点一刀切；GELU 是平滑 S 形（`x·Φ(x)`），处处可导，实验上对 Transformer 系更友好。换回 ReLU 也能跑，但这是原论文实测的选择。

### 2.6 pre-LN vs post-LN

```
post-LN（你的 transformer）:  x = LN(x + attn(x))
pre-LN（ViT）:               h = attn(LN(x));  x = x + h
```

区别只在 LayerNorm 放残差前还是后。**pre-LN 的梯度可以从残差支路直通回传**，层数越深越稳，所以现代模型几乎都用 pre-LN。你 transformer 层数少用 post-LN 没问题，但要知道差异。

### 2.7 初始化 —— ViT 比 CNN 更"娇气"

原论文：所有权重 `trunc_normal(std=0.02)`，偏置 0，LayerNorm 置 1/0，`cls_token`/`pos_embed` 也用 trunc_normal。

- `trunc_normal`：截断正态——超出 2σ 的采样值丢弃重采，避免极端初始化；
- 为什么敏感：ViT 没有 CNN 的归纳偏置，收敛路径窄；PyTorch 默认的 kaiming 初始化在玩具任务上也能学会（我们的预检验证过 100%），但上真实数据就会显出差距。**建议按论文初始化**，用 `self.apply(self._init_weights)` 递归应用到所有子模块。

---

##  	3. 高频坑清单（按踩中概率排序）

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | `pos_embed` 忘了 **+1** | cat 后相加 shape 报错 | `(1, N+1, E)` |
| 2 | `mlp_ratio` 是 float | `nn.Linear` 报 `invalid combination of arguments` | `int(embed_size * mlp_ratio)` |
| 3 | 缺接口常量/命名对不上 | 验收脚本静默退回参考答案 | 按规格表提供 `DROPOUT` 等 |
| 4 | 忘了整除断言 | 切头/切块静默出怪 shape | `assert embed_size % num_heads == 0`、`assert img_size % patch_size == 0` |
| 5 | `expand` 出来的张量原地写 | 报"view of a tensor"错误 | expand 是共享内存的视图，只读；要写用 `repeat` |
| 6 | 推理忘了 `model.eval()` | 每次输出不同 | eval() 关 dropout |
| 7 | 掩码建在 CPU | masked_fill 报设备不一致（你 transformer 踩过） | 创建时传 `device=` |
| 8 | `view()` 用在非连续张量 | RuntimeError | 换 `reshape()` |
| 9 | `cat` 和 `pos_embed` 顺序反 | [CLS] 无位置信息 | 先 cat 后加 |
| 10 | 测试耦合内部命名 | 换命名就挂 | 测试验证行为不验证命名（已改） |

---

## 4. PyTorch 用法手册（按你写过的代码组织）

### 4.1 参数与注册

| API | 作用 | 场景 | 易错点 |
|---|---|---|---|
| `nn.Parameter(tensor)` | 把张量注册为可学习参数 | cls_token、pos_embed | 必须包 Parameter，否则不进 `parameters()`、不进 optimizer |
| `self.register_buffer(name, t)` | 注册"跟模型走但不学习"的张量 | 正弦位置编码表 | buffer 会进 `state_dict`、跟 `.to(device)` 搬家，但不被梯度更新 |

**Parameter vs buffer 一句话**：要训练就 Parameter（ViT 位置编码），固定常量就 buffer（transformer 正弦编码）。

### 4.2 形状变换

| API | 作用 | 易错点 |
|---|---|---|
| `x.reshape(B,S,H,d)` | 改形状，必要时自动拷贝 | 最省心，首选 |
| `x.view(...)` | 改形状，要求内存连续 | 非连续（如 transpose 后）会报错 |
| `x.flatten(2)` | 从第 2 维开始压平 | flatten(2) 保留前两维 (B,E) |
| `x.transpose(1,2)` / `x.permute(0,2,1,3)` | 换轴（只改 stride 不搬数据） | transpose 一次换两轴；permute 可任意换 |
| `x.squeeze(d)` / `unsqueeze(d)` | 去/加长度为 1 的维 | mask 那两个 1 就是 unsqueeze 出来的 |
| `x.expand(B,-1,-1)` | 按 batch 广播复制（零拷贝视图） | **只读**！写操作要 `x.repeat(...)`（真拷贝） |

### 4.3 索引与切片（你已经在用）

```python
x[:, 0]        # 每个 batch 的第 0 个位置 -> [CLS] 向量    (B, E)
x[:, 1:]       # 去掉第 0 个位置 -> 所有 patch            (B, N, E)
x[:, -1, :]    # 最后一个位置（transformer generate 用过） (B, E)
scores.masked_fill(~mask, float('-inf'))   # bool 掩码填充
x.mean(dim=1)  # 沿序列维平均 -> GAP
```

### 4.4 初始化

| API | 作用 |
|---|---|
| `nn.init.trunc_normal_(t, std=0.02)` | 截断正态（ViT 论文），带 `_` 表示原地操作 |
| `nn.init.kaiming_uniform_(t)` / `nn.init.xavier_uniform_(t)` | CNN/MLP 常用 |
| `nn.init.zeros_ / ones_ / constant_` | 清零/置一/置常数 |
| `self.apply(fn)` | 递归遍历所有子模块调用 fn——批量初始化的标准姿势 |
| `t.zero_() / t.fill_(v)` | 张量原地清零/填充 |

### 4.5 模块与容器

| API | 作用 | 区别 |
|---|---|---|
| `nn.ModuleList([...])` | 注册"一组模块"，自己写循环调用 | 不自动前向；适合层数不定的堆叠（你的 blocks） |
| `nn.Sequential(...)` | 按顺序自动前向 | 适合固定流水线（如 MLP 也可写成 Sequential） |
| `nn.Linear(in, out)` | 全连接 = 权重矩阵+偏置 | 维度必须 **int**（你的 mlp_ratio 坑） |
| `nn.Conv2d(C, E, k, stride)` | 卷积 | ViT 里当"patch 投影"用 |
| `nn.LayerNorm(E)` | 对最后一维归一化 | NLP/Transformer 标配（注意和 BatchNorm 的区别：LN 不分 batch 统计） |
| `nn.Dropout(p)` | 训练时随机置零，推理自动关闭 | 靠 `model.train()/eval()` 切换 |

### 4.6 函数式接口与计算

| API | 作用 | 注意 |
|---|---|---|
| `F.softmax(x, dim=-1)` | softmax | 也可 `x.softmax(-1)`；`nn.Softmax` 是模块版 |
| `torch.matmul(a, b)` / `a @ b` | 矩阵乘，支持 batch 广播 | 等价 |
| `torch.cat([a, b], dim=1)` | 沿已有维拼接 | 两形状除 dim 外一致 |
| `torch.stack([a, b], dim=0)` | 新建一维堆叠 | 和 cat 的区别：stack 多一维 |
| `torch.einsum('bhqd,bhkd->bhqk', q, k)` | 爱因斯坦求和 | 下标即文档，注意力更优雅的写法 |
| `F.interpolate` | 插值缩放 | 位置编码换分辨率用 |
| `x.argmax(dim=-1, keepdim=True)` | 取最大值的下标 | generate 用它选词 |

### 4.7 训练与调试

| API | 作用 |
|---|---|
| `@torch.no_grad()` / `with torch.no_grad():` | 推理不建计算图，省显存提速 |
| `optimizer.zero_grad() → loss.backward() → optimizer.step()` | 标准三步（不清梯度会累加） |
| `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` | 梯度裁剪，ViT 原论文配方 |
| `model.state_dict() / load_state_dict()` | 保存/加载权重 |
| `model.named_parameters()` | 遍历 (名字, 参数)，调试时找"哪个参数没被更新" |
| `t.item()` | 把 0 维张量变 Python 数字（generate 用过） |
| `t.device / t.dtype / t.shape` | 调试三件套，报错先看这三个 |

---

## 5. 架构改进：把 [CLS] + 位置编码抽成独立类（回答你的问题）

**结论：值得拆，你的直觉是对的。** 理由：

1. **单一职责**：现在的 `ViT.__init__` 里，"拼 [CLS]、建位置编码、dropout"散在三处；抽出来之后每个类只回答"我是干什么的"；
2. **顺序被封装**：cls 先拼、位置编码后加——这个顺序是易错点（坑表 #9），封装进一个类后调用方永远不会写错；
3. **forward 变成装配线**：主类四五行代码正好对应数据流图的四个箭头，读代码像读图；
4. **可独立测试**：这个类可以单独构造、单独验证形状，不用每次搭整个 ViT。

参考实现（两个类合成一个最合适，因为它们的顺序是绑定的）：

```python
class ClassTokenPosEmbed(nn.Module):
    """[CLS] 拼接 + 可学习位置编码 + dropout，三步按固定顺序封装。"""

    def __init__(self, num_patches, embed_size, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        # +1：给 [CLS] 留一个位置（坑表 #1 在这里一次性解决）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_size))
        self.pos_drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, N, E)  patch token 序列
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, E)  只读广播
        x = torch.cat([cls, x], dim=1)           # (B, N+1, E) 先拼
        x = x + self.pos_embed                   # 再加位置编码
        return self.pos_drop(x)
```

拆完之后 `ViT` 变成：

```python
def __init__(self, ...):
    self.patch_embed = PatchEmbedding(...)
    self.cls_pos_embed = ClassTokenPosEmbed(num_patches, embed_size, dropout)
    self.blocks = nn.ModuleList([...])
    self.norm = nn.LayerNorm(embed_size)
    self.head = nn.Linear(embed_size, num_classes)

def forward(self, x):
    x = self.patch_embed(x)          # ① 图 -> token
    x = self.cls_pos_embed(x)        # ② cls + 位置
    for block in self.blocks:        # ③ 逐层编码
        x = block(x)
    x = self.norm(x)
    x = x[:, 0]                      # ④ 取 [CLS] 分类
    return self.head(x)
```

**注意拆分的边界**：一个类做一件事是好事，但不要拆碎——比如把 pos_drop 也单独拆成类就没意义了（工业界的 timm 也把 cls/pos 留在主类里，因为它们是主类"组装图"的一部分；你拆出来是教学清晰度优先，两种风格都成立，知道各自的理由就行）。

---

## 6. 调试技巧

1. **报错先看 shape/device/dtype 三件套**：99% 的 ViT 报错是 shape 问题；
2. **加 shape 断言**：写完 forward 在每步 `assert x.shape == (...)`，训练前就能抓住 90% 的错；
3. **梯度自检**：`next(iter(model.parameters())).grad` 是不是 None——是就说明计算图断了；
4. **看注意力矩阵**：把某层返回的 attn 打印/画热力图，检查有没有全行均匀（退化）或死行；
5. **梯度裁剪前后打印 grad norm**：`p.grad.norm()`，异常大说明 lr 太大或初始化有问题；
6. **小数据过拟合测试**：任何新模型先用几十条数据训练到 100% 准确率，验证"模型有学习能力"再上大数据（我们的象限任务就是干这个的）。

---

## 7. 训练配方（原论文三件套 + 常用扩展）

- **AdamW + weight decay=0.05**：ViT 对 weight decay 敏感，AdamW 把 weight decay 和梯度更新解耦；
- **梯度裁剪 1.0**：防止早期大梯度；
- **trunc_normal(0.02) 初始化**：见 2.7；
- **扩展**：余弦学习率衰减（`CosineAnnealingLR`）、warmup（前几千步线性升 lr）、混合精度（`torch.cuda.amp`）、数据增强（ViT 数据饥渴，AutoAugment/RandAugment 收益很大）。

---

## 8. 学习资源

- 论文 *An Image is Worth 16x16 Words*（重点看 Figure 1 与位置编码附录）；
- 论文 *DeiT*（ViT 怎么少用数据训练）；
- [timm](https://github.com/huggingface/pytorch-image-models) 的 `vision_transformer.py`（工业级实现，手写完再读）；
- 本仓库 `transformer/`（你的 NLP 版，随时对照）+ `01_引导_从零手写ViT.md` + `02_验证与修改方向.md`。
