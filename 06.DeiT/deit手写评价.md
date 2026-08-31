# deit 手写代码评价（deittrain.py 为主）

> 检查时间：2026-08-28
> 检查范围：`deitmodel.py`、`deitloss.py`、`deitteacher.py`、`deittrain.py`
> 验证方式：`ssl_cv` 环境实机运行 `_check_user_files.py`（模型+损失）+ 逐行静态检查（训练脚本）

---

## 0. 总体评价

**结论：结构推导已经过关，工程接口还差一层。** 你已经做到了最难的 80%：从 ViT 自己推出了 DeiT 的
蒸馏 token + 双头结构（`deitmodel.py` 全绿），损失公式也推导正确（`deitloss.py` 全绿，上次的
`tempetature` 拼写坑也改成了 `tau`）。但 `deittrain.py` 是"半看半写"的产物，**恰恰在"半看"的
部分——接口对接处——有 8 处会崩或会静默出错的 bug**。

这非常正常，而且暴露的是真实的学习漏洞：**抄来的代码块和你自己的接口对不上**。把下面的清单
逐条修掉，你就完成了"模块化开发"这门课最贵的一课。

### 各文件状态一览

| 文件 | 状态 | 一句话评价 |
|---|---|---|
| `deitmodel.py` | ✅ 通过（你已修好 ModuleList） | 结构推导满分，参数量 5.400M 对齐 DeiT-Ti |
| `deitloss.py` | ✅ 通过（hard/soft 公式全对） | 唯一小瑕疵：注释里还写着 `tempetature`/`(4,1000)` |
| `deitteacher.py` | ⚠️ 逻辑可用，但被调用方式坑了 | 见第 2 节 |
| `deittrain.py` | ❌ 当前跑不起来（8 处问题） | 骨架和创意都好，接口全是缝 |

---

## 1. 分模块评价

### 1.1 args 参数区【纯手写】

你采用"需要哪个加哪个"的风格，方向对（避免一次堆 20 个参数）。但暴露了这种风格的典型代价：

| 行 | 问题 | 说明 |
|---|---|---|
| 62 vs 216 | `--ckpt` 定义，代码却用 `args.ckpt_dir` | argparse 会把 `--ckpt` 转成 `args.ckpt`；`ckpt_dir` 不存在 → **AttributeError，跑到第 216 行必崩**。改法：参数名改成 `--ckpt-dir`（自动变 `ckpt_dir`） |
| 61 & 73 | `--teacher-epochs` 定义了两次 | argparse 以后者为准，不崩但脏。**加参数前先 grep 一下同名** |

### 1.2 build_loaders【纯手写】✅

数据管道完全正确（crop+flip+归一化，v1 无增强版符合计划）。一个小风格问题：
`return train_loader, test_loader, 32, 3, 100` —— 把 img_size/通道数/类别数**硬编码在返回值里**，
属于"魔法数字"。能用，但建议收敛到文件顶部的配置区：

```python
IMG_SIZE, IN_CHANNELS, NUM_CLASSES = 32, 3, 100   # CIFAR-100 常量, 别散落在函数里
```

### 1.3 evaluate（含你改的 test loss）【纯手写，想法正确】

**你给测试集加 loss 的想法是对的**（第 6 节专门回答"参考代码为什么不算 test loss"）。
但两处接口问题：

| 行 | 问题 |
|---|---|
| 117 | `eval_total_loss` 是**求和**不是平均 → 数值 ≈ 78 个 batch 的和，看着吓人。改：返回前 `/ len(loader)` |
| 240 vs 109 | 调用时传了 5 个参数（多了 `teacher`），但函数只定义了 4 个 → **TypeError**。且测试时本来就不需要教师（推理只用学生，见第 3 节），删掉调用处的 `teacher` 即可 |

另外：测试 loss 的**口径**应该是纯 CE（学生 eval 输出是双头平均后的单张量，没有蒸馏头）。
你现在的 `criterion(logits, y)` 想表达的就是这个，但 `Distillation_loss` 需要 4 个参数，
测试时应该换成 `soft_cross_entropy(logits, y, args.smoothing)`（从 `deitloss` import）。

### 1.4 train_one_epoch【AMP 部分写得很好，loss 接线全错】

先说**写得好的**：AMP 五步曲顺序完全正确——`autocast 前向 → scale(loss).backward() →
unscale_ → step → update`，连 `unscale_` 这个大多数人会漏的一步你都写了。

再说不好的（按崩溃顺序）：

| 行 | 问题 |
|---|---|
| 147-148 | 学生 train 模式返回 **(cls_logits, dist_logits) 元组**，你却 `criterion(logits, y)` 只传 2 个参数；而且**教师前向完全缺失**——整个循环里 `teacher` 形参一次都没用上 → **蒸馏根本没发生** |
| 154 | `logits.argmax` 会对元组崩（`'tuple' object has no attribute 'argmax'`） |
| 156 | **`return` 缩进在 for 循环里面** → 每 epoch 只训练 1 个 batch 就返回 |

正确的接线（对照着看，自己重写一遍）：

```python
def train_one_epoch(student, teacher, loader, criterion, optimizer, scaler, device, args):
    student.train(); teacher.eval()
    one_epoch_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            teacher_logits = teacher(X)                    # ① 教师只推理, 不进反向图
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):  # ② 注意 enabled 接上开关
            out = student(X)                               # ③ (cls_logits, dist_logits)
            total_loss, base_loss, dist_loss = criterion(out, teacher_logits, y, args)  # ④ 四参!
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        one_epoch_loss += total_loss.item()
        correct += (out[0].argmax(dim=1) == y).sum().item()   # ⑤ 元组取 out[0] 算 acc
        total += y.numel()
    return one_epoch_loss / len(loader), correct / total      # ⑥ 循环外, 且要平均
```

注意 ⑥：你在 `deitteacher.py` 里写的 `return total_loss / len(train_loader), correct/total`
**是对的**——同一件事你在教师文件里做对了，学生文件里缩进错了。这就是"半看半写"时最容易
发生的：抄的时候丢了一级缩进。

### 1.5 教师加载与训练【意图正确，调用错位】

| 行 | 问题 |
|---|---|
| 195 | `train_teacher(args.data_dir, args.teacher_epochs, args.batch_size, args.teacher_lr, teacher_path)` —— 但 `deitteacher.train_teacher` 签名是 `(data_dir, epochs, batch_size, lr, device=None, save_path=None)`。**第 5 个位置参数是 device 不是 save_path** → `model.to('路径字符串')` 直接崩；且 `save_path=None` 教师不会保存 |
| 195 | 返回值 `(model, best_acc)` 被丢弃 → 即使不崩，`teacher` 也还是**随机权重**，蒸馏等于从垃圾里学 |

正确写法（关键词参数 + 接住返回值）：

```python
teacher, teacher_acc = train_teacher(
    data_dir=args.data_dir, epochs=args.teacher_epochs,
    batch_size=args.batch_size, lr=args.teacher_lr,
    device=device, save_path=teacher_path)
```

### 1.6 断点续跑【半看半写，框架正确，一个存储 bug】

框架你搭得对：`start_epoch = ckpt['epoch']+1`、加载 optimizer/scaler/scheduler 三件套、
`weights_only=False`（torch 2.6+ 新默认，你注意到了，很好）。

但存储侧有个复制粘贴 bug：

| 行 | 问题 |
|---|---|
| 269 | `"scaler_dict": scheduler.state_dict()` —— 存进去的是 **scheduler 的字典**，不是 scaler 的。续跑时 `scaler.load_state_dict(ckpt['scaler_dict'])` 会拿到错误内容。改：`scaler.state_dict()` |

顺带说明：续跑时数据 shuffle、drop_path 随机性都无法完全复现，"续跑"追求的是**接着训练不浪费**，
不是 bit 级复现，所以你现在这个框架是标准且正确的做法。

### 1.7 模型保存【参考结构，使用正确】

best.pth（只存权重+acc+epoch）与 last.pth（存全套训练状态）的双文件分工是标准做法，你用得对。
已按你的授权**添加了 `"config": vars(args)`**（代码里用 `【AI 添加】` 标注了两处），理由：
`./checkpoint/best.pth` 是跨实验共享的单例，会被不同配置覆盖，不内嵌配置就无法追溯它是
哪组超参训出来的。

### 1.8 训练主循环 / 日志 / TensorBoard / 早停【纯手写，整体好评】

- `run_dir` 时间戳目录 + `config.txt` + `train.log`（带 flush）+ 闭包 `log()`：**写得好**，是
  这次代码里最亮眼的工程部分；
- 早停逻辑（`bad_epoch` 计数 + patience）正确；
- `writer.add_hparams(vars(args), {...})` 用对了；
- 一个小优化：教师训练时的 print 直接打到 stdout，没进你的 `train.log`，以后可以把教师训练
  输出重定向或改成回调。

### 1.9 两个隐蔽的 Python 坑【纯手写区】

| 行 | 问题 | 说明 |
|---|---|---|
| 209 | `use_amp = ~args.no_amp` | `~` 是**按位取反**不是逻辑非：`~True = -2`（truthy）→ `--no-amp` 永远不生效。改 `not args.no_amp` |
| 146 | `torch.amp.autocast('cuda')` 没接 `enabled=use_amp` | 你算了 `use_amp` 却没喂给 autocast → `--no-amp` 关不掉 autocast。改 `autocast('cuda', enabled=use_amp)` |

---

## 2. criterion 专题（你说不太理解的部分）

### 2.1 criterion 是什么

`criterion` 就是一个**"把（模型输出，标签）变成标量损失"的函数**（或可调用对象）。把它当参数
传给 `train_one_epoch`，只是为了让"训练循环"和"具体损失公式"解耦：循环不用知道今天是 CE 还是
蒸馏损失，换个损失不用动循环。

### 2.2 你的代码里它应该是谁

你的 `deitloss.py` 里 `Distillation_loss` 是个**函数**（不是类），签名：

```python
Distillation_loss(student_out, teacher_logits, targets, args)
#                 ① 学生的两个头  ② 教师输出   ③ 标签  ④ 参数(alpha/tau/smoothing/distill)
```

所以正确用法不是 `criterion = Distillation_loss()`（零参构造，必崩），而是**直接用函数本身**：

```python
criterion = Distillation_loss            # 它就是那个"从输出到标量"的函数
...
total, base, dist = criterion(out, teacher_logits, y, args)
```

为什么它比普通 CE 多两个输入？因为**蒸馏损失天然需要三方信息**：学生两个头（真值信号 vs 教师
信号）+ 教师的输出。你参考实现 `deit_cifar100.py` 里的 `distillation_loss` 也是这个形状——
"顺着流程写"没错，只是少抄了调用处的参数。

### 2.3 训练和测试不能用同一个 criterion

| | 训练 | 测试 |
|---|---|---|
| 学生输出 | (cls_logits, dist_logits) 两个 | 双头**平均**后的一个 |
| 教师 | 参与（出 logits） | **不参与**（论文：推理只用学生） |
| 用哪个损失 | `Distillation_loss(4 参)` | `soft_cross_entropy(logits, y, smoothing)` |

这就是为什么参考实现里 `evaluate` 不接蒸馏损失——测试时蒸馏头已经不存在了。

---

## 3. 必改清单（按崩溃先后排序）

1. ✅ `--ckpt` → `--ckpt-dir`（或统一用 `args.ckpt`）——第 216 行 AttributeError；
2. 教师调用改关键词参数 + 接住返回值（第 195 行）——否则蒸馏从随机教师学习；
3. `criterion = Distillation_loss`（第 204 行，零参构造必崩）；
4. `train_one_epoch` 按第 1.4 节的对照重写接线 + `return` 移出循环；
5. `evaluate` 调用处删掉多余的 `teacher` 参数（第 240 行）；测试 loss 换成
   `soft_cross_entropy` 并除以批数；
6. `"scaler_dict": scaler.state_dict()`（第 269 行）；
7. `use_amp = not args.no_amp`（第 209 行）+ autocast 接 `enabled=use_amp`（第 146 行）；
8. 删掉重复的 `--teacher-epochs` 定义（第 61 行那个）。

改完用这个顺序验证：

```bash
python _check_user_files.py                 # 模型+损失自检 (已全绿, 每次改完都跑一遍)
python deittrain.py --epochs 1              # 先 1 个 epoch: 不崩 + loss 在 4.6 附近开始降
python deittrain.py --epochs 100 --resume   # 再验证断点续跑真的能续
```

---

## 4. 你问的：参考代码为什么不算 test loss？

三个原因，从强到弱：

1. **学术惯例**：分类 benchmark 报的是 top-1 精度，论文 Table 1 全是 acc 没有 loss；loss 的
   数值受标签平滑、蒸馏方式影响，**跨配置不可比**，而 acc 是硬通货。
2. **参考实现里训练 loss 口径特殊**：Mixup/CutMix 下训练 loss 是对"混合软标签"算的，和测试
   集的标准 CE 根本不是同一个量，画在一起反而误导。
3. **模型选择用不上**：best checkpoint / early stop 都看 acc，loss 不提供额外决策信息；省代码。

**你的修改是对的口径**：测试 loss = CE(双头平均 logits, y)，推理不用教师，这恰恰和论文的
测试方式一致。保留它有两个用途：看"train loss 降 / test loss 升"的过拟合拐点、同配置内对比
训练稳定性。注意两点：要除以批数做平均；hard 和 soft 两种蒸馏模式的 test loss 数值不能互比
（口径相同但训练动态不同），只同配置内看趋势。

---

## 5. 下一步（学习路径）

1. 修完第 3 节清单 → 跑通 1 epoch → 跑 100 epochs，记录基线（预期 60~68%）；
2. 按 `warmup教程.md` 自己加 warmup（加完我再检查）；
3. 进阶可选：AdamW 分组权重衰减（论文只衰减 2D 权重，约 +1~2 点）、`--data-dir` 改绝对路径；
4. v2：按论文 Sec 4.2 逐个加回 RandAugment → Mixup/CutMix，对齐论文配方。

**一句话总结**：你的 DeiT 已经"懂"了（模型+损失全绿），现在差的是"接对线"（train 的 8 处接口）。
这 8 处修完，这个项目就闭环了。

---

## 6. 第二轮修改后评价（2026-08-28 晚，含 100 轮训练结果）

### 6.1 修复确认（上一轮遗留的 2 个必改 + 你顺手加的新东西）

| 项目 | 状态 |
|---|---|
| resume key 统一为 `student_dict`（last.pth 存/取一致） | ✅ 已修 |
| resume 双加一：`start_epoch = ckpt['epoch']` + `range(start_epoch+1, ...)` | ✅ 已修 |
| `--grad-clip`（默认 1.0） | ✅ 你自己加的，且位置**完全正确**（`unscale_` 之后、`step` 之前——AMP 下梯度裁剪的标准位置） |
| loss 平均升级为"按样本加权"（`*y.numel()/total`） | ✅ 你自己加的，比 `/len(loader)` 更精确（末批不满时也不偏） |
| tqdm 进度条 | ✅ 本轮 AI 添加（带 `【AI 添加】` 标记，含可选依赖回退） |

### 6.2 训练结果分析（CIFAR-100，hard 蒸馏）

| 模型 | 结果 | 对照预期 |
|---|---|---|
| 教师 TeacherCNN（30 epochs） | **69%** | ✅ 超过 65% 过关门槛 |
| 学生 DeiT-Ti（100 epochs，早停于 88） | **best 63.27% @ epoch 79** | ✅ 落在 v1 预期区间 60~68% 内 |

- **轨迹合理性**：30 epoch ≈ 58.8% → 88 epoch 63.3%，是正常的饱和曲线；早停在 epoch 88 触发
  （best 出现在 79，之后 10 轮没涨），早停逻辑第一次实战生效 ✓；
- **关于"学生 63% < 教师 69%"**：完全正常，不需要担心。教师的 69% 是小 CNN + SGD 的成熟配方；
  学生是无增强的 Transformer，本来就更吃力。论文的 **+1.6 是对比"同模型无蒸馏基线"**，
  不是"学生必须超过教师"。所以 v1 目标"蒸馏比不蒸馏好"还差最后一步：**跑一次 `--distill none`
  消融**（`deitmodel` 已支持 `distilled=False`，只需在 `deittrain` 加个 none 分支：学生单头
  + `soft_cross_entropy` 训练）。这是上传 GitHub 前唯一建议补的实验；
- **checkpoint 内嵌 config 实战验证成功**：`best.pth` 里读出 `best_acc=0.6327, epoch=79,
  config.epochs=100` —— 现在拿着任何 checkpoint 都能追溯训练配置，上次加的这一行值了。

### 6.3 剩余小建议（都不阻塞，按优先级）

1. **补 `--distill none` 消融**（见 6.2）——这是 v1 目标闭环的最后一环；
2. best.pth 的 key 叫 `model_state_dict`，last.pth 叫 `student_dict`，顺手统一成后者；
3. `--data-dir` 默认 `../data` 是相对路径，建议改成绝对路径（防换目录启动找不到数据）；
4. `evaluate` 的 `criterion` 形参已无用，可删或注明"留给未来"；
5. （v2 可选）AdamW 分组权重衰减，约 +1~2 点。

### 6.4 结论

**项目闭环达成。** 回顾全过程：结构推导（deitmodel）→ 损失推导（deitloss）→ 教师（deitteacher）
→ 训练循环（AMP/warmup/梯度裁剪/早停/续跑/checkpoint）→ 实测 63.27%。当初的目标"从 ViT
自己推出 DeiT"已经达成，且每一行代码都经过实机验证或冒烟自检。可以上传 GitHub 了。

---

## 7. v2 消融实验结果评价（2026-08-29，Mixup / CutMix）

### 7.1 实验执行评价

**实验纪律满分**：一次只改一个变量（`--mixup 0.8 --cutmix 0` / `--mixup 0 --cutmix 1.0` /
两者都开）、同 seed、同 100 epochs、config 自动记录，runs 目录里三组实验的配置清晰可追溯。
这正是 GPT 说的"实验型深度学习"的正确姿势。

### 7.2 结果表（已按 runs 日志核对，test_acc 列为 best）

| 配置 | train_acc(末轮) | **test_acc (best)** | train_loss | test_loss | 备注 |
|---|---|---|---|---|---|
| Baseline (v1) | 97.3% | 63.27% | 0.73 | 1.41 | 早停于 88 |
| + Mixup | 67.74% | 66.45% | 1.2509 | 2.0529 | 跑满 100 |
| + CutMix | 72.34% | **67.90%** | 1.5192 | 2.0163 | 跑满 100 |
| + Mixup + CutMix | 61.97% | 67.12% | 1.6037 | 2.0166 | 早停于 80 |

> 你表里 CutMix 的 67.59 是"最后一轮"的 0.6759，best 实为 **67.90**，已修正。

### 7.3 三个值得写进大脑的观察

1. **教科书级的正则现象，你自己跑出来了**：train_acc 从 97.3% 一路降到 61.97%，
   test_acc 却从 63.27% 升到 67.90%。这就是"训练集精度下降反而是好事"的实验证据——
   模型不再背答案（97% 的记忆没了），开始学规律（泛化 +4.6 个点）。GPT 预言的
   "看到这个现象你就真正理解了"——你现在亲眼看到了。
2. **test_loss 不降反升（1.41 → 2.0+）不是 bug，是口径问题**：你的 evaluate 用的是
   平滑 CE（smoothing=0.1），而 Mixup/CutMix 训练出的模型输出**更柔和、更不自信**——
   平滑 CE 里"对其它 99 类的平均惩罚"这一项会变大，于是 test_loss 数值反而上升。
   这恰好呼应了之前 `deit问题解答.md` 里的提醒：**test loss 只在同配置内可比，
   跨配置看 test_acc**。你看到的 acc↑/loss↑ 并存，正是这个原理的活例子。
3. **单次实验的排名不要过度解读**：CutMix(67.90) > 组合(67.12) > Mixup(66.45)，
   但这是**单 seed 各跑一次**，±0.5~1 点以内都是噪声。可信的结论只有两条：
   三个增强都显著优于 baseline（+3~4.6 点）；组合并不保证叠加收益。若想排序，
   每个配置至少跑 2~3 个 seed 取均值。

### 7.4 状态（2026-08-31 终版更新）

- ✅ RandAugment 已实现并接线（幅度递增已升级为无状态公式，续跑安全）；
- ✅ Experiment 4（RA m=9）→ **66.59%**、Experiment 6（全配方 inc=1）→ **66.22%**
  （train_acc 崩到 43.86%：三重增强+递增幅度过强 → 欠拟合）；
- ✅ **M 消融五组全部完成**（`run_m_ablation.py` 一键脚本）：m=0/5/9/15/20 →
  63.27 / 65.47 / 66.59 / 67.16 / **68.36%**——单调上升未拐头，**68.36% 为项目最优**，
  超过 CutMix 单用（67.90%）；结论："全配方失败"的元凶是叠加+递增，不是 RA 本身；
- ⏸ 按你的决定**暂停**：重复增强 RA3、EMA、`--distill none` 对照、Teacher quality、
  M=25/30、CutMix+RA 组合、多 seed —— 留到 SSL/DINO 回访（清单在
  `deit问题解答.md` 第七节）。

**一句话：你已经在做实验型深度学习，而且做得对——继续保持"一次一个变量 + 记录 + 核对"的纪律。**
（最新的教训：全配方 66.22% < CutMix 单用 67.90% < RA 单用 68.36%，"论文配方"也要按
数据集调强度——这比抄对论文配方有价值得多。）
