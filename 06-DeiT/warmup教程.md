# DeiT 训练 warmup 教程

> 目标：看懂 warmup 为什么存在、论文怎么用，然后**你自己加进 `deittrain.py`**，加完我来检查。
> 对应论文：Sec 4.1 "Optimization" —— "we use AdamW ... with a **5-epoch linear warmup** and cosine decay"

---

## 1. warmup 是什么

一句话：**学习率不从峰值开始，而是前几个 epoch 从 0 线性爬到峰值，再进入正常衰减。**

```
lr
 │        ╱‾‾‾‾‾‾‾‾╲
 │       ╱            ╲        ← 余弦衰减
 │      ╱              ╲
 │─────╱                ╲────
 └────┴────────┴────────┴──► epoch
      0   warmup=5    100
      (线性爬坡)   (余弦下降)
```

## 2. 为什么 Transformer 需要它

三个递进的直觉：

1. **初始梯度又大又乱**：随机初始化的模型前几步梯度方向和尺度都很极端，直接上峰值 lr 容易
   把参数踢飞，loss 震荡甚至 NaN；
2. **Adam 的前几步统计被污染**：Adam 的一阶/二阶矩估计 `m, v` 是累积量，头几步的巨大梯度
   会在 `v` 里留下"创伤记忆"，后面很多步都在为它还债；
3. **Transformer 没有 BatchNorm 兜底**：BN 会隐式地稳定每层输入尺度，ViT/DeiT 只有
   LayerNorm（归一化不约束激活尺度），所以对 lr 更敏感，更需要"先小步预热再大步"。

> 注意：你的**教师**（小 CNN + SGD + lr=0.1 的 MultiStepLR）不需要 warmup——这是 SGD 时代
> 的传统配方。warmup 是给 **AdamW + Transformer** 的。别两边混淆。

## 3. 论文的完整学习率配方（warmup 只是前半段）

| 项 | 论文值 | 说明 |
|---|---|---|
| 峰值 lr | `1e-3 × batch_size / 512` | 线性缩放规则：batch 翻倍 lr 翻倍。你 batch=128 → `1e-3×128/512 = 2.5e-4`；你现在用的 5e-4 略高但 CIFAR 上没问题 |
| warmup | **5 epoch 线性爬坡** | 0 → 峰值 |
| 衰减 | 余弦 | 峰值 → 接近 0 |
| 总时长 | 300 epoch (ImageNet) | CIFAR 100 epoch 也适用同一形状 |

## 4. 公式（这是你要背下来的部分）

设 `warmup_epochs = 5`，当前 epoch 记作 `e`（从 1 开始）：

```
warmup 阶段 (e ≤ warmup):   lr = base_lr × e / warmup_epochs
余弦阶段 (e > warmup):      
		lr = base_lr × 0.5 × (1 + cos(π × (e − warmup) / (epochs − warmup)))
```

验证形状：e=1 → lr≈0；e=5 → lr=base_lr（爬坡结束）；e=(epochs+warmup)/2 → lr=base_lr/2；
e=epochs → lr≈0。

## 5. 加到你的 `deittrain.py` 的两种方式

### 方式 A：per-epoch 手写（教学推荐，5 行）

删掉现在第 207 行的 `CosineAnnealingLR` 和第 241 行的 `scheduler.step()`，在 epoch 循环
**开头**自己算：

```python
for epoch in range(start_epoch, args.epochs + 1):
    # ---- 学习率调度: warmup + 余弦 (论文 Sec 4.1) ----
    if epoch <= args.warmup_epochs:                       # 前 5 epoch 线性爬坡
        lr = args.lr * epoch / args.warmup_epochs
    else:                                                  # 之后余弦衰减
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g['lr'] = lr
    ...
```

优点：公式透明、每行都能回答"为什么"；**对断点续跑天然兼容**（lr 只由 epoch 号决定，不需要
恢复 scheduler 状态）。这是学概念时的最佳选择。

### 方式 B：torch 原生组合调度器

```python
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0,
                                          total_iters=args.warmup_epochs),   # 近似从 0 爬坡
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=args.epochs - args.warmup_epochs),
    ],
    milestones=[args.warmup_epochs],
)
# 每 epoch 结束 scheduler.step() —— 你现在的调用位置不用动
```

优点：代码更"库化"；缺点：黑盒一点，且续跑时必须恢复 scheduler 状态（你已经存了
`scheduler_dict` ✓，所以也能续）。两种方式任选，**先 A 后 B** 学习效果最好。

## 6. 自己动手 + 验收标准

1. 在 `get_args()` 加一行：`parser.add_argument("--warmup-epochs", type=int, default=5)`；
2. 按方式 A 或 B 接入；
3. 验收（跑 1 个 epoch 验证不崩，再跑 8 个 epoch 验证曲线）：
   - 在 epoch 日志里**打印当前 lr**（`log(f"... lr {lr:.2e} ...")`）；
   - 观察 lr：1→0.1e-3、2→0.2e-3 … 5→0.5e-3（线性爬坡）、6→0.49e-3 开始余弦下降；
   - 前几个 epoch 的 loss **不出现 NaN、不剧烈震荡**；
4. 加分实验（强烈建议做）：同 seed 分别跑"有 warmup / 无 warmup"各 5 个 epoch，把两者的
   loss 曲线放进 TensorBoard 对比——你会直观看到 warmup 期间两者的差距，比看十篇教程都管用。

## 7. 方式 A 与方式 B 的保存与断点续跑（针对"版本 A 保存不进模型"的补充）

### 7.1 你困惑的答案：方式 A 根本不需要"保存"

你验证了方式 A，但卡在"怎么把它保存到模型里"——**这个问题本身不成立**。原因：

方式 A 的学习率是 epoch 的**纯函数**：

```
lr = f(epoch)      # 只依赖 epoch 号, 不依赖任何历史状态
```

断点续跑时，`start_epoch` 从 checkpoint 恢复，公式直接用当前 epoch 号重算，
自动得到"如果没中断，此刻应该是什么学习率"。**它没有状态，所以没有东西可保存**
（真正需要保存的只有 epoch 号——你已经存在 last.pth 的 `"epoch"` 里了）。

### 7.2 两种方式的本质：无状态 vs 有状态

| | 方式 A（手写公式） | 方式 B（SequentialLR） |
|---|---|---|
| 性质 | **无状态**（stateless）：lr = f(epoch) | **有状态**（stateful）：内部有 `last_epoch` 计数器 |
| 保存进 checkpoint | 什么都不用存（epoch 已有） | 必须存 `scheduler.state_dict()` |
| 恢复 | 自动（公式重算） | 必须先**重建同结构的 scheduler**，再 `load_state_dict` |
| 优点 | 透明、可解释、续跑零成本 | 官方规范，与生态一致（timm 等库内部也这么干） |
| 缺点 | 要自己保证公式与 epoch 号一致 | 黑盒一点；重建结构与保存时不一致会**静默出错** |

### 7.3 你的代码里，方式 B 的保存/恢复链路（已经写对了，逐行对一遍）

`deittrain.py` 现在的三处，缺一不可：

1. **构造**（`main()` 里那段 `SequentialLR(LinearLR + CosineAnnealingLR, milestones=[warmup])`）
   —— 注意这段必须在 `load_state_dict` **之前**执行（先有同样结构的空壳，才能往里灌状态）；
2. **保存**（last.pth 里的 `"scheduler_dict": scheduler.state_dict()` 一行）→ 进入 last.pth；
3. **恢复**（resume 分支里的 `scheduler.load_state_dict(ckpt['scheduler_dict'])`）
   —— 配套的 `torch.load(..., weights_only=False)` 你 resume 分支里已有 ✓。

所以"官方库能不能保存进模型"——**能，而且你已经在做了**。你缺的不是保存代码，
是"意识到它已经保存了"。这一轮把链路看懂，以后任何有状态的组件（optimizer、scaler、
EMA）都是同一个套路：**构造 → state_dict 存入 → load_state_dict 灌回**。

### 7.4 续跑正确性自检（一次到位，以后每次 resume 都做）

resume 成功后第一行日志加上：

```python
log(f"[resume] 恢复后学习率 = {scheduler.get_last_lr()[0]:.2e}")
```

对照 warmup 公式手算"该 epoch 应有的 lr"，对得上就说明 scheduler 恢复正确。
如果对不上（比如恢复成了峰值 lr），说明 state_dict 与构造结构不一致，或者 load 时机错了。

### 7.5 你代码里与 resume 相关的 off-by-one（已修复 ✅）

> ✅ 已修复（2026-08-28 第二轮）：你采用了第一种写法
> `start_epoch = ckpt['epoch']` + `range(start_epoch+1, ...)`，正确。

当时的原始问题是：第 237 行 `start_epoch = ckpt['epoch'] + 1`，第 248 行又
`range(start_epoch+1, ...)` —— **加了两次 1，会跳过一个 epoch**。二选一：

```python
start_epoch = ckpt['epoch']          # 配 range(start_epoch+1, ...)   ← 推荐, 你已采用
# 或
start_epoch = ckpt['epoch'] + 1      # 配 range(start_epoch, ...)
```

推荐第一种：与 scheduler 的 `last_epoch` 语义一致（last.pth 里存的是"已完成"的 epoch 号）。

---

## 8. 常见坑

| 坑 | 说明 |
|---|---|
| warmup 放在余弦之后 | 顺序反了：必须先爬坡再衰减 |
| 用 `epoch` 和 `epoch-1` 混着算 | 差一个 epoch 不影响大局，但要前后一致 |
| batch 改小了却忘了缩 lr | 线性缩放：`base_lr × batch/512`；batch 从 128 减到 64，lr 该减半 |
| 把 warmup 加到教师训练里 | 不需要（SGD + 小 CNN 传统配方），加了无害但没必要 |
| resume 后 warmup 重来 | 方式 A 天然正确（lr 由 epoch 号决定）；方式 B 靠 scheduler_dict 恢复，恢复后接着爬 |

---

学完加进代码后叫我，我再实机检查（重点看：lr 曲线形状、前 5 epoch 稳定性、以及你有没有
顺手把 `deit手写评价.md` 第 3 节的其他必改项一起修了）。
