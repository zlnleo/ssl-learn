# 06 · scaler 与 scheduler 完全讲解：AMP 混合精度 + 学习率调度

> 对应代码：`train.py` 里的 `scaler = torch.amp.GradScaler(...)`、
> `scheduler = CosineAnnealingLR(...)`，以及 `train_one_epoch` 里的四行 AMP 代码。
> 这两个东西是"现代训练脚本"和"教学脚本"的分水岭——搞懂它们，你写的训练代码就进工业化门槛了。

---

# 上篇：scaler（GradScaler / AMP 混合精度）

## 一、背景：为什么要混合精度

GPU 上 fp16 的矩阵乘比 fp32 快（现代 GPU 有专门的 Tensor Core，fp16 吞吐是 fp32 的 2~8 倍），显存占用还减半。理想状态：**前向/反向用 fp16 算，参数本身仍存 fp32**——这就是"混合"精度。

但 fp16 有两个硬伤：

| 问题 | 解释 | 后果 |
|---|---|---|
| 表示范围小 | fp16 最小正数约 6e-5，最大约 65504 | 太小的数下溢成 0，太大的数溢出成 inf |
| 有效精度低 | 只有约 3 位十进制有效数字 | 大数加小数直接丢（1 + 0.0001 ≈ 1） |

**GradScaler 解决"下溢"问题**；"溢出"问题由它配套的跳过机制解决；"精度低"由 autocast 的算子白名单解决（见下）。

## 二、GradScaler 的完整机制（一张流程图）

```
        loss（可能很小的数）
              │  scaler.scale(loss)：乘上放大系数 S（初始 65536）
              ▼
        放大后的 loss ──backward()──> 梯度也被放大 S 倍，落回 fp16 可表示范围
              │
              │  scaler.unscale_(optimizer)：梯度 ÷ S，还原真实尺度
              ▼
        clip_grad_norm_：按真实尺度裁剪（所以必须先 unscale！）
              │
              │  scaler.step(optimizer)：检查梯度有没有 inf/nan
              │     ├─ 干净 ──> 正常更新参数
              │     └─ 有 inf/nan ──> 跳过本次更新（损失一步，保住训练）
              ▼
        scaler.update()：调整 S
              ├─ 一切顺利 ──> S 逐步增大（S 越大，能救回越小的梯度）
              └─ 刚跳过一步 ──> S 减半（降低下次溢出的概率）
```

**关键直觉**：S 是一个"自适应音量旋钮"。模型平稳训练时音量调大（保护小梯度）；模型闹腾（溢出）时音量调小（保护训练本身）。

## 三、train.py 里四行代码逐行拆

```python
# ① 前向：autocast 声明"下面这块区域用 fp16 算"
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
    logits = model(images)
    loss = criterion(logits, labels)
```

- autocast 只包**前向**（反向会自动跟随前向的精度）；
- 里面不是所有算子都切 fp16：matmul/conv 用 fp16 加速，而 **softmax、LayerNorm、loss 这些"精度敏感"算子在 torch 内置白名单里，自动保持 fp32**——你不用手动指定任何东西；
- `enabled=` 为 False 时整个块退化成普通 fp32 前向（CPU 上就是这样）。

```python
# ② 放大 loss 再反向
scaler.scale(loss).backward()
# 等价于：loss * S 之后再 backward()，梯度随之放大 S 倍
```

```python
# ③ 还原梯度 → 裁剪
scaler.unscale_(optimizer)                                  # 梯度 ÷ S
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # 再裁
# 顺序不能反：不 unscale 就裁，裁的是"S 倍放大"的假梯度，阈值形同虚设
```

```python
# ④ 更新参数 + 更新系数
scaler.step(optimizer)   # 代替 optimizer.step()：带 inf/nan 检查
scaler.update()          # 调整 S
```

**如果每一步"忘了会怎样"**：

| 省略 | 后果 |
|---|---|
| 忘了 `scale()` 直接 backward | 小梯度下溢成 0，参数学不动（早期尤其明显） |
| 忘了 `unscale_` 就裁剪 | 裁剪阈值失真，等于没裁 |
| 用 `optimizer.step()` 代替 `scaler.step()` | 溢出时不会自动跳过，训练被 inf 炸掉 |
| 忘了 `update()` | S 永远不变，失去自适应能力（还能跑，但不是 AMP 了） |

## 四、创建与关闭

```python
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
# "cuda"：设备类型；enabled=False 时上面四行全部变成"透明通道"，
# 整段代码自动退化成最朴素的 fp32 训练——这是这套 API 的设计精髓。
```

`scaler.is_enabled()`：返回当前是否真正开启 AMP（CPU 上自动 False），autocast 的 enabled 用它保持同步。

## 五、常见问题

- **loss 出现 nan**：先看 `scaler.step()` 是否在报 "skipped"——频繁跳过说明 lr 太大或数据里有异常值；偶尔跳过是正常现象（AMP 的自我保护在工作）；
- **想观察 S 的变化**：`scaler.get_scale()` 打印系数，你会看到它从 65536 慢慢爬、溢出时掉下来；
- **resume 训练**：中断后继续训练要连 scaler 一起存：`torch.save(scaler.state_dict(), ...)` 加载时 `scaler.load_state_dict(...)`，否则系数重置、行为不连续；
- **bf16**：表示范围和 fp32 一样大（只是精度低），所以 **bf16 训练不需要 GradScaler**（autocast 里 dtype 换成 torch.bfloat16 即可），但要求 Ampere 架构以上 GPU。工业界大模型训练的主流是 bf16；fp16+scaler 是通用兼容的经典方案。

## 六、动手实验

1. toy 任务上对比：`--amp`（默认开）vs CPU 跑（自动关），感受速度差；GPU 上显式关：`python train.py --dataset toy --amp`（store_true 默认开，想关改代码或用 `--no-amp` 需自己加）；
2. 在循环里打印 `scaler.get_scale()`，观察系数自适应曲线；
3. 把 lr 调到 1.0（故意制造爆炸），观察 scaler 的跳过机制如何"救回"训练。

---

# 下篇：scheduler（学习率调度器）

## 一、为什么需要调度学习率

学习率是"每步跨多大"。固定 lr 的尴尬：
- 前期 lr 小 → 收敛太慢；
- 后期 lr 大 → 在最优解附近来回震荡，落不进去。

调度器的思想：**前期大步探索，后期小步精调**。学习率随时间按某个曲线衰减，就是 scheduler 的全部工作。

## 二、PyTorch 常用调度器一览

| 调度器 | 曲线 | 适用 |
|---|---|---|
| `StepLR` | 每 N 个 epoch 乘一个衰减系数 | 简单任务 |
| `MultiStepLR` | 在指定 epoch 列表处各衰减一次 | ResNet 训练经典（30/60/90） |
| `CosineAnnealingLR` | 余弦曲线平滑降到最低值 | ViT/Transformer 标配（train.py 用它） |
| `OneCycleLR` | 先升后降（单周期） | 大 batch 训练常用 |
| `LambdaLR` | 任意自定义函数 | 万能积木（warmup 靠它搭） |
| `ReduceLROnPlateau` | 验证指标不涨了才衰减 | 特殊：**不按 epoch 数，按指标** |

**选型原则**：不知道用什么就用 CosineAnnealingLR——现代视觉/语言模型论文里出现频率最高。

## 三、CosineAnnealingLR 详解（train.py 里的这个）

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
...
scheduler.step()   # 每个 epoch 结束调用一次
```

- `T_max`：余弦完成一个完整下降周期所需的 step 次数。我们设为 `epochs`，即 lr 从初始值 1e-3 平滑衰减到 0，覆盖整个训练；
- 每次 `step()`：scheduler 读当前 epoch 数、算好新 lr、写回 optimizer 的参数组；
- 查看当前 lr：`scheduler.get_last_lr()`（调试神器，打印它就能看到余弦曲线）。

**调用时机的铁律**：**先确认这个调度器是按 epoch 还是按 batch 调度，再看 step() 放哪**。
- CosineAnnealingLR 是"epoch 级"：step() 放在 epoch 循环末尾；
- OneCycleLR 是"batch 级"：step() 要放在 batch 循环里，放错位置曲线完全乱掉。

## 四、进阶：warmup + cosine（研究训练标配）

训练最初模型是随机的、梯度很大，直接满 lr 容易把参数冲散。warmup 让 lr 先**线性爬升**几个 epoch 再进入余弦衰减——曲线变成"上坡 + 下坡"。

骨架（先看懂结构，需要时再写）：

```python
def warmup_lr(epoch):
    # epoch < 5 时线性升到 1.0；之后交给余弦调度
    if epoch < args.warmup_epochs:
        return epoch / args.warmup_epochs
    return None   # None 表示"用主调度器的值"

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda e: 1.0 if e >= args.warmup_epochs
                        else e / args.warmup_epochs,
)
# 更完整的做法：SequentialLR 把 warmup 和 CosineAnnealing 两个调度器串起来
```

## 五、常见坑速查

| 症状 | 原因 |
|---|---|
| lr 一直不变 | 忘了调 `scheduler.step()` |
| loss 曲线像锯齿震荡 | lr 衰减太慢/太大，调小初始 lr 或换更快的衰减 |
| 第一个 epoch loss 爆炸 | 没有 warmup，初始 lr 太大 |
| 验证集早停不工作 | ReduceLROnPlateau 需要手动传验证 loss：`scheduler.step(val_loss)` |
| resume 后 lr 行为不对 | 中断恢复时 scheduler 也要存/取 state_dict |
| OneCycleLR 曲线奇怪 | step() 放错了层级（它要每个 batch 调） |

## 六、scaler 和 scheduler 的配合关系（train.py 全景）

```python
optimizer = AdamW(...)                 # 更新参数的主体
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # 控制 optimizer 的 lr
scaler    = GradScaler("cuda", enabled=use_amp)         # 保护 fp16 的梯度

每个 batch：zero_grad -> autocast 前向 -> scale(loss).backward()
           -> unscale_ -> clip -> scaler.step -> scaler.update
每个 epoch：scheduler.step()   # 独立于 batch 循环，管的是"第几个 epoch 用多大 lr"
```

三者各管一段：**optimizer 管参数更新，scheduler 管 lr 曲线，scaler 管 fp16 数值安全**——互不冲突、层层叠加。这也是所有现代训练脚本的标准骨架，你以后看任何开源训练代码（timm、MAE、DINO 官方仓库）都是这个结构。

## 七、动手实验

1. **打印 lr 曲线**：循环里 `print(scheduler.get_last_lr())`，肉眼验证余弦；
2. **对比实验**：toy 任务分别用 `CosineAnnealingLR` 和固定 lr（scheduler 换成 `LambdaLR(lambda e: 1.0)`）跑 30 轮，对比最终准确率；
3. **打印 scale 曲线**：循环里 `print(scaler.get_scale())`，两个曲线一起看，理解"lr 在降、scale 在爬"的互补关系。
