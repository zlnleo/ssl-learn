# DeiT 手写进阶问答（本轮 5 问解答）

> 对应你修改后的 `deittrain.py`（2026-08-28 版）。行号以当前文件为准。

---

## 0. 本轮代码检查速报

### 已修复 ✅（上一轮评价里的必改项，你基本全部修对）

| 项目 | 状态 |
|---|---|
| `--ckpt-dir` 参数 | ✅ 第 61 行 |
| 教师调用：6 个位置参数顺序正确 + 接住返回值 | ✅ 第 196 行（这是上次最严重的语义 bug，修得对） |
| `criterion = Distillation_loss`（不再零参构造） | ✅ 第 205 行 |
| train_one_epoch：教师前向 + 4 参调用 + `return` 移出循环 + 损失平均 | ✅ 第 145-157 行 |
| evaluate：签名 4 参、调用 4 参、test loss 除以批数 | ✅ 第 108/250/119 行 |
| `scaler_dict` 存 scaler 自己的状态 | ✅ 第 283 行 |
| `~` → `not` + autocast 接上 `enabled` | ✅ 第 147/219 行 |
| 重复的 `--teacher-epochs` 删除 | ✅ |
| warmup（官方库版 SequentialLR） | ✅ 第 208-217 行 |

### 已全部修复 ✅（2026-08-28 第二轮确认）

1. ✅ **key 已统一为 `student_dict`**（last.pth 保存/加载一致，resume 不再 KeyError；
   注意 best.pth 里还叫 `model_state_dict`，建议顺手统一）；
2. ✅ **resume 双加一已修**：现为 `start_epoch = ckpt['epoch']` + `range(start_epoch+1, ...)`；
3. ✅ 另外你自己新增了 `--grad-clip`（AMP 下正确放在 `unscale_` 之后）和按样本加权的 loss
   平均（`*y.numel()/total`，比 `/len(loader)` 更精确）；
4. ✅ 实测结果：教师 69%，学生 100 轮 best 63.27%（详见 `deit手写评价.md` 第 6 节）。

### 小建议（不阻塞）

- `evaluate` 的 `criterion` 形参已经没用了（第 108/115 行直接调 `soft_cross_entropy`），
  可以删掉，或者留着但注明"留给将来测试蒸馏损失用"；
- `--data-dir` 默认 `../data` 是相对路径，换目录启动会找不到，建议改绝对路径；
- （可选）AdamW 分组权重衰减，论文只衰减 2D 权重，约 +1~2 点，v2 再上。

---

## Q1：evaluate 为什么用 `soft_cross_entropy`？是因为测试时只用学生吗？

**对，就是"测试时只用学生"。** 拆成两层：

1. **测试是"部署模拟"**：推理阶段学生 eval 模式已经把双头**平均成一个 logits**
   （第 114 行拿到的就是平均后的单张量），蒸馏头不复存在；教师也不参与推理
   （论文：教师只在训练时当老师）。所以蒸馏损失（需要 teacher_logits + 两个头的
   4 参函数）在测试时**无从算起**，只能用普通 CE。
2. **为什么用它而不是 `F.cross_entropy`**：你调用时没传 smoothing（默认 0.0），此时
   它与 `F.cross_entropy` **逐位相等**（冒烟测试证明过：5.2256 == 5.2256）。
   用它的唯一理由是"复用自己写过、验证过的函数，代码统一"。换 `F.cross_entropy`
   结果一模一样。

一句话：**测试 loss 的口径 = 学生独立分类的 CE**，这正是论文测试方式的正确对应。

---

## Q2：train_one_epoch 里为什么 teacher 也要进来？和以前学的 ViT 有什么区别？

**对，就是因为 DeiT 要"借教师"。** 对比着看：

| | ViT（你以前学的） | DeiT（现在） |
|---|---|---|
| 损失 | `CE(学生输出, 真值 y)` | `(1-α)·CE(cls头, y) + α·KD(dist头, 教师输出)` |
| 训练循环需要谁 | 只有学生 | 学生 + 教师（多一次教师前向） |
| 教师前向的待遇 | — | `eval()` + `no_grad()`：只产出答案，**绝不参与更新** |
| 学生输出 | 一个 logits | 两个 logits（元组） |

所以 `teacher` 进函数的原因就一句话：**蒸馏损失的公式里有一项要用教师的输出**
（`y_t = argmax(Z_t)` 或 `softmax(Z_t/τ)`）。你可以把教师理解成一个"只读的外部参考答案"：
学生交两份作业，一份按真值批改（cls 头），一份按教师的答案批改（dist 头），教师自己
不学习。

---

## Q3：criterion 必须写成那个样子吗？和 ViT 直接 `F.cross_entropy` 差在哪？
`criterion = Distillation_loss` 是不是就相当于换个名字接进来？

**先回答最后一个：是的，就是换个名字。** Python 里函数是"一等公民"，
`criterion = Distillation_loss` 之后 `criterion(...)` 和 `Distillation_loss(...)`
调用的是同一个函数对象。这个"换名字"的**目的**是解耦：训练循环只认 `criterion(...)`
这一个调用口，将来换损失（hard→soft、或换别的函数）只改一行赋值，循环一行不动。

**再回答"必须那个样子吗"：不是必须，是"由你的损失函数签名决定"。**

- 你的 `Distillation_loss(student_out, teacher_logits, targets, args)` 需要 4 个输入，
  调用处就必须传 4 个；
- ViT 的 `F.cross_entropy(logits, y)` 只需要 2 个，所以传 2 个；
- 原则：**criterion 长什么样，取决于"这个损失需要哪些信息"**。DeiT 的损失天然需要
  三方信息（学生双头 + 教师 + 标签），所以比 ViT 多两个输入。

**和 ViT 的另一个差异：返回值的个数。**

- `F.cross_entropy` 返回 1 个标量；
- 你的 `Distillation_loss` 返回 3 个 `(total, base, dist)` —— 后两个只用于打印观察
  （看"分类损失和蒸馏损失各是多少"），所以调用处要接 3 个变量。这多出来的返回值
  不是规矩，是**给你自己看的仪表盘**。

**想更"官方"吗？** 可以把它包成 `nn.Module` 类（`__init__` 存 alpha/tau/smoothing，
`forward` 里算），调用就变成 `criterion(out, teacher_logits, y)`（args 藏进对象里）。
timm 官方就这么干。但那是**风格差异，不是对错差异**——你现在的"函数 + args"风格
完全正确可用，学明白差异比换风格重要。

---

## Q4：warmup 用官方库是不是更好？方式 A 保存不进模型怎么办？

**结论先行：你选方式 B（官方库）是对的，而且"保存进模型"这件事你已经做完了**
——只是没意识到。

- 方式 B 是"有状态"调度器，它的 `scheduler.state_dict()` 你已经存进 last.pth
  （第 284 行），resume 时也 load 了（第 236 行）。**链路是通的**。
- 方式 A 是"无状态"调度器：`lr = f(epoch)` 纯公式，**根本不存在需要保存的东西**。
  你"验证了 A 但不知道怎么保存"——因为你在试图保存一个不存在的东西。续跑时
  `start_epoch` 恢复，公式自动算出正确 lr，这就是 A 的优势。

详细对比、B 的保存/恢复正确性自检、以及和 resume 相关的那个 off-by-one，
见 `warmup教程.md` 新增的第 7 节。

---

## Q5：为什么 ViT 那种用 `F.cross_entropy` 不用除以 len(loader)，手写的就要除？

**这是个误会：除不除与"官方/手写"无关，与"你想要的打印口径"有关。**

1. `F.cross_entropy` 和你的 `soft_cross_entropy` **每次调用都返回"这一个 batch 的平均
   损失"**（一个标量）——两者在这一层完全一样；
2. 一个 epoch 有 N 个 batch（N = len(loader)），`total += loss.item()` 累加 N 个标量
   得到的是 **N 个 batch 损失的总和**；
3. 除以 N 得到"平均每个 batch 的损失"，数值落在 ln100 ≈ 4.6 量级，可读、可跨配置
   比较；不除以数值就是 ~N×4.6，**训练完全不受影响**（backward 用的是当前 batch 的
   loss，打印值只是给人看的）；
4. 所以你看到的"ViT 代码没除"只有两种可能：① 它其实除了（很多教程写
   `running_loss / (i+1)` 或 `/len(trainloader)`）；② 它没除——那只是打印口径不同，
   不是 ViT 代码"不需要除"。

**顺带澄清两个容易混的"平均"**：

| 平均维度 | 谁负责 | 位置 |
|---|---|---|
| batch **内**平均（128 个样本→1 个标量） | `reduction='mean'`（默认） | 损失函数内部 |
| batch **间**平均（N 个标量→1 个标量） | `/len(loader)` | 你的统计代码 |

两个维度互不替代。你现在训练和测试都除了，是对的。

---

## 附：criterion 小抄表

| 场景 | 输入 | 用什么 | 返回 |
|---|---|---|---|
| ViT 训练 | (logits, y) | `F.cross_entropy` | 1 个标量 |
| DeiT 训练 | (双头元组, teacher_logits, y, args) | `Distillation_loss` | 3 个 (total, base, dist) |
| DeiT 测试 | (平均后 logits, y) | `soft_cross_entropy`（=纯 CE） | 1 个标量 |

把这张表看明白，你就不需要"记住"criterion 的样子，而是能从损失公式自己推出来。
