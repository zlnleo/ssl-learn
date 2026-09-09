# Mixup / CutMix 接入训练教程 —— 从"函数写好了"到"训练跑起来"

> 你的 `Mixup()` / `CutMix()` 已经检查过：**结构正确、实测可运行**（randperm 配对、β 采样、
> CutMix 面积修正 λ 都对）。这份文档只讲一件事：**怎么把它们融进 `deittrain_v2.py` 的
> `train_one_epoch`**。按文档自己改完，我再帮你复查。

---

## 0. 先说结论：函数本身的两个小建议（不是 bug）

| 位置 | 建议 | 原因 |
|---|---|---|
| `idx = torch.randperm(x.size(0))` | 改成 `torch.randperm(x.size(0), device=x.device)` | 我实测你当前写法在 torch 2.11 能跑，但把索引张量放到与 x 同一设备是防御性写法（参考版就是这么做的），大 batch 下也更快 |
| `lam = np.random.beta(...)` | 写成 `lam = float(np.random.beta(...))` | 实测当前环境返回 Python float、不会引发 dtype 升级；但显式 `float()` 保证在任何 numpy 版本下 `lam*x` 都保持 float32（防止悄悄变成 float64 拖慢 AMP） |

两个论文参数提醒：
- Mixup 论文用 **α=0.8**（你的默认 ✓）；CutMix 论文用 **α=1.0**（你目前共用 0.8，建议分开）；
- 每批以 **0.5 概率在 Mixup/CutMix 之间二选一**（timm 惯例，参考版同款）。

---

## 1. 全景：一个训练 batch 的新流水线（2026-08-29 最终版）

这是整个融合的核心，**顺序不能乱**：

```
① 取 batch: X, y (搬到 device)
        │
② 抽签: 本批要不要混合? (概率 = mix_switch)
        │  ┌─ 要 ─→ 再抽: Mixup 还是 CutMix?
        │  │          X_mixed ← 混合后的图
        │  │          target  ← lam·one_hot(y_a) + (1-lam)·one_hot(y_b)   ← 软标签!
        │  └─ 不要 → X_mixed = X, target = y (和 v1 完全一样)
        │
③ 教师前向: teacher_logits = teacher(X_mixed)   ← 教师看混合后的图!
        │
④ 学生前向: out = student(X_mixed)              ← 学生看同一张混合图
        │
⑤ 损失: criterion(out, teacher_logits, target, args)
        │
⑥ backward / scaler / clip (v1 的代码原样不动)
```

### 为什么是"教师看混合图"（最终版）——诚实复盘

这个项目里试过两种方案，最终定为**教师看混合图**：

| 方案 | 做法 | 评价 |
|---|---|---|
| A（旧版，我已放弃） | 教师看**干净图** T(X)，再把教师 logits 用 λ 人工混合 | 是**线性近似**：教师是非线性网络，T(λx₁+(1-λ)x₂) ≠ λT(x₁)+(1-λ)T(x₂)。Mixup 下近似尚可，CutMix 下误差明显。省不了算力（教师前向本来就只有一次） |
| B（最终版 ✓） | 教师直接看**混合图** T(X_mixed) | **精确**，代码更简单，与 timm 的训练管线一致。教师前向仍然只有一次，不存在"翻倍" |

> 📌 关于"官方 DeiT 仓库"的一个事实澄清：官方 facebookresearch/deit 的实现其实**既不是 A 也不是 B**——
> 它的 Mixup 作用在 **logits** 上（batch 模式），图像本身没有混合，教师看到的也是干净图。
> 真正"图像级混合 + 教师看混合图"的是 **timm** 的现代训练管线。所以"哪个更接近官方"这个
> 说法本身不成立；最终版对齐的是 timm 路线，也是社区更常用的路线。

### 两个必须能回答的"为什么"

**为什么②标签要变成软标签 (B, C)？**
混合图"0.7 猫 + 0.3 狗"的正确标签本来就该是"0.7 猫、0.3 狗"。`lam·one_hot(y_a)+(1-lam)·one_hot(y_b)`
一行搞定。而你的 `soft_cross_entropy` 里 `targets.ndim == 2` 的分支就是为此准备的——
**v1 埋的接口，现在派上用场，loss 函数一行都不用改**。

**为什么②教师和学生必须看同一张 X_mixed？**
蒸馏的定义就是"学生模仿教师对**同一输入**的回答"。图、标签、教师信号三者在输入空间上
对齐，损失才自洽；如果学生看混合图而教师看干净图，蒸馏头学到的就是"错位的教师答案"。

---

## 2. 需要新加的两个小工具（各一行核心）

```python
def mix_target(y_a, y_b, lam, num_classes):
    """两行: one-hot 后按 lam 凸组合 -> (B, C) 软标签"""
    return lam * F.one_hot(y_a, num_classes).float() + (1 - lam) * F.one_hot(y_b, num_classes).float()
```

（F 即 `torch.nn.functional`，你的 `deitloss.py` 里已有类似的 one-hot 用法可参考。）

args 增加三个开关（自己加进 `get_args`）：

```python
parser.add_argument("--mixup", type=float, default=0.8, help="Mixup 的 alpha, 0 表示关闭")
parser.add_argument("--cutmix", type=float, default=1.0, help="CutMix 的 alpha, 0 表示关闭")
parser.add_argument("--mix-switch", type=float, default=0.5, help="启用混合的批比例")
```

---

## 3. `train_one_epoch` 的最终写法（已由 AI 落到你的 v2 文件）

```python
for X, y in pbar:
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()

    # ---- ② 先混合 (抽签 + 二选一) ----
    X_mixed, target = X, y                            # 兜底: 不混合时与 v1 一致
    if (args.mixup > 0 or args.cutmix > 0) and np.random.rand() < args.mix_switch:
        if args.mixup > 0 and (args.cutmix <= 0 or np.random.rand() < 0.5):
            X_mixed, y_a, y_b, lam, idx = Mixup(X, y, args.mixup)
        else:
            X_mixed, y_a, y_b, lam, idx = CutMix(X, y, args.cutmix)
        target = mix_target(y_a, y_b, lam, 100)       # 软标签 (B, 100)

    # ---- ③ 教师看混合图 ----
    with torch.no_grad():
        teacher_logits = teacher(X_mixed)

    # ---- ④ 学生看同一张混合图 + 损失 ----
    with torch.amp.autocast('cuda', enabled=not args.no_amp):
        student_logits = student(X_mixed)
        total_loss, base_loss, dist_loss = criterion(student_logits, teacher_logits, target, args)
    ...
```

三个要点（对应你之前踩过的坑）：
1. **混合结果必须写进 `X_mixed`，且教师/学生都吃它**——大小写或变量名不一致 = 增强白做；
2. **内层判断要兼容"只开一个"**：`args.cutmix <= 0 or ...` 防止只开 mixup 时硬币落到
   CutMix 分支（α=0 会崩）；
3. **criterion 第三参放 `target`**（软标签），第二参放教师输出，顺序不可乱。

### 关于 train_acc 的口径（重要但别纠结）

混合后 `student_logits[0].argmax(dim=1) == y` 是"混合图上的近似准确率"——它会**明显下降**
（这是好事，见验收清单），但它不再是严格意义的口径。保持这样即可，只是心里清楚：
**train_acc 变成观察正则强度的仪表盘，test_acc 才是裁判。**

### `evaluate` 一行都不要改

测试集永远不混合（部署时模型看到的就是真实图）。你的 `soft_cross_entropy(logits, y, args.smoothing)`
照旧——测试标签永远是硬标签 (B,)。

---

## 4. 验收清单（改完按顺序查）

1. **先单测函数**（秒级）：
   ```bash
   python -c "import torch, numpy as np; import deittrain_v2 as t;
   x=torch.randn(4,3,32,32).cuda(); y=torch.randint(0,100,(4,)).cuda();
   mx,ya,yb,lam,idx=t.Mixup(x,y); assert mx.shape==x.shape and 0<=lam<=1;
   cx,ya,yb,lam,idx=t.CutMix(x,y,alpha=1.0); assert cx.shape==x.shape;
   tgt=t.mix_target(ya,yb,lam,100); assert tgt.shape==(4,100) and torch.allclose(tgt.sum(1),tgt.new_ones(4));
   print('mixup/cutmix 单测通过')"
   ```
2. **跑 2 个 epoch**：初始 loss 仍在 ≈4.6 量级；无 NaN；
3. **看关键现象**：`train_acc` 相比 v1 的 97% **明显下降**（预期 70~85%），`test_acc` 不降反升——
   看到这个，你就亲眼验证了"训练集精度下降是好事"；
4. **完整对照实验**：同 seed 跑 v2（+Mixup）100 epochs，与 baseline 63.27% 比较，把数字填进
   `DeiT_v2学习路线.md` 第 7 节的消融表（预期 +1~3 点）；
5. **蒸馏头也在工作**：打印里的 `base/dist` 两项都要正常下降（如果 dist 不降，检查教师 logits
   有没有混 λ）。

## 5. 常见坑

| 坑 | 说明 |
|---|---|
| 教师前向写了两次（干净图一次 + 混合图一次） | 这才是"翻倍"；最终版教师只看混合图一次，成本与旧版相同 |
| 混了图、忘了把标签也变软 | loss 发散或涨不动——图和标签必须同一个 λ |
| 教师看干净图 + 人工混 logits（旧方案） | 线性近似，CutMix 下误差明显；已被"教师看混合图"取代 |
| 混合结果写进局部变量 `x`，学生却前向 `X` | 混合图根本没进模型，增强白做 |
| criterion 传参顺序写成 `(logits, target, y, args)` | 软标签被当成教师输出、硬标签被当成目标，蒸馏信号错乱 |
| 只开 mixup 时硬币落到 CutMix 分支 | cutmix=0 时内层判断必须兼容（否则以 alpha=0 调 CutMix） |
| 在 `evaluate` 里也混合 | 测试口径被污染，acc 虚低 |
| 忘加 `--mixup 0` 之类的关闭开关 | 没法做消融（关闭能力 = 实验能力） |
| 混合后的 acc 拿来做早停 | 早停只看 `test_acc`，你现在的代码已经是这样 ✓ |

---

## 7. 逐行解释：训练里为什么那样加

你问"训练时候为什么那个样子加"，把接好的代码逐行拆开（顺序就是流水线的顺序）：

```python
X_mixed, target = X, y            # ① 兜底
if (args.mixup > 0 or args.cutmix > 0) and np.random.rand() < args.mix_switch:   # ② 外层抽签
    if args.mixup > 0 and (args.cutmix <= 0 or np.random.rand() < 0.5):          # ③ 内层二选一
        X_mixed, y_a, y_b, lam, idx = Mixup(X, y, args.mixup)   # ④ 混合图, 写进 X_mixed
    else:
        X_mixed, y_a, y_b, lam, idx = CutMix(X, y, args.cutmix)
    target = mix_target(y_a, y_b, lam, 100)                   # ⑤ 软标签 (B,100)
with torch.no_grad():
    teacher_logits = teacher(X_mixed)                          # ⑥ 教师看混合图 (精确)
student_logits = student(X_mixed)                              # ⑦ 学生看同一张图
criterion(student_logits, teacher_logits, target, args)        # ⑧ 第三参是 target 不是 y
```

**① 为什么先 `target = y`？**
抽签没抽中的 batch 要完全走 v1 的老路。而你的 `soft_cross_entropy` 天生能吃两种输入——
`(B,)` 硬标签和 `(B,C)` 软标签走不同分支，所以 `target` 这个变量"硬也行、软也行"，
一行兜底让两种情况的代码共用。

**② 为什么外层是"开关 + 概率"两个条件？**
`args.mixup > 0 or args.cutmix > 0` 是**开关**——没有它就没法做消融（`--mixup 0 --cutmix 0`
一键回到 baseline）。`np.random.rand() < mix_switch` 是**概率**——0.5 意味着只有一半的批
参与混合，另一半保持原样。不全混合的理由：模型始终要见到一部分"真实图"，混合是正则
而不是替代数据。

**③ 为什么内层还要掷一次硬币？**
Mixup 和 CutMix 是**替代关系**，同一个 batch 只能做一种（先 mixup 再 cutmix 会把样本
混合成四不像）。0.5/0.5 二选一是 timm 的惯例。括号里 `args.cutmix <= 0 or ...` 是兼容
"只开 mixup"的消融场景——硬币落到反面时不能去调一个 alpha=0 的 CutMix。

**④ 为什么混合结果必须写进 `X_mixed`（而不是小写 `x`）？**
第 ⑥⑦ 行教师和学生前向吃的都是 `X_mixed`。写进小写 `x`（你最初的写法）意味着**混合图
被扔掉、师生看到的还是干净图**——增强就白做了。"本批要喂给两个网络的那张图"必须是一个
明确的变量，混合就是改它。

**⑤ 为什么标签要 mix_target？**
混合图的真值本来就不是 one-hot："0.7 猫 + 0.3 狗"的正确答案是 (0.7, 0.3, ...)。
`lam*one_hot(y_a) + (1-lam)*one_hot(y_b)` 就是这条答案，直接喂进你 v1 就写好的
`soft_cross_entropy` 2D 分支。

**⑥ 为什么教师看混合图（最终方案）？**
蒸馏的定义是"学生模仿教师对**同一输入**的回答"。教师直接前向 `T(X_mixed)` 是精确答案；
旧的"干净图 T(X) + 人工混 logits"则是线性近似——教师是非线性网络，
`T(λx₁+(1-λ)x₂) ≠ λT(x₁)+(1-λ)T(x₂)`，Mixup 下近似尚可，CutMix 下误差明显。
而且教师前向无论哪种方案都只有一次，旧方案并不省算力，所以最终版选择精确且更简单的
"教师看混合图"。注：官方 DeiT 仓库其实连图像都不混（mixup 作用在 logits 上），
本方案对齐的是 timm 的现代管线。

**⑦ 为什么学生也必须看同一张 X_mixed？**
否则师生看到不同的输入，蒸馏头学到的就是"错位的教师答案"——正则效果和蒸馏效果同时打折。

**⑧ 为什么第三参放 target 而不是 y？**
`criterion` 的签名是 `(student_out, teacher_logits, targets, args)`。混合时如果放 `y`，
分类头就学不到软标签（Mixup 的标签正则完全失效）；而把 `target` 放到第二位（你原来的
写法）会让 `Distillation_loss` 把软标签当成教师输出——蒸馏头拿到一个假教师。
抽中时放 `target`（软），没抽中时 `target == y`（硬），两种情况下这一行都是对的。

> 一句话记忆：**师生看同一张混合图；标签是混合软标签；criterion 顺序 (学生, 教师, 标签)。**

---

## 6. 完成后

自己改完 → 跑第 4 节验收 → 叫我。我会帮你复查 `train_one_epoch` 的接线（重点看 λ 是否
三处一致：图、标签、教师 logits），然后把消融结果一起填进 v2 路线图。
