# DeiT 学习问答集（Q&A 全集）

> 用法：Ctrl+F 搜关键词。内容来源三部分：① 本项目全程你问过的问题与解答；
> ② 代码注释里的知识点转成的问答；③ 第七节的"待补充"是你回访 SSL/DINO 时再问的占位。
> 每个答案给结论 + 指向详细教程/代码位置，本文件只做"速查"。

---

## 一、架构与模型

**Q：为什么 CIFAR 用 patch=4 而不是论文的 16？**
32×32 用 patch=16 只剩 4 个 token，信息太少；patch=4 → 8×8=64 个 token，与 ImageNet 上
ViT 的 196 个同量级。

**Q：class token 是干嘛的？**
序列最前面的可学习 token，通过注意力聚合全图信息，最后用它的输出走分类头。

**Q：蒸馏 token 和 class token 有什么区别？**
都拼在序列前、全程参与注意力；区别在输出：class 头用真值监督、dist 头用教师监督；
推理时两个头的 logits 取平均（论文 Sec 3.2 的"distillation through attention"）。

**Q：为什么蒸馏 token 要参与全程注意力，而不是最后才接？**
论文的设计就是让它作为注意力流里的常驻成员与其他 token 充分交互——"通过注意力蒸馏"。

**Q：为什么训练时 forward 返回元组、eval 返回平均？**
训练需要两个头分别被监督（真值 vs 教师）；推理只需要一个输出，双头平均是官方约定。
（`deitmodel.py` 的 `forward`，见注释）

**Q：`nn.ModuleList` 为什么不能直接 `self.blocks(x)`？**
它只是"容器"，没有 `forward`，不能被调用。用 `for block in self.blocks: x = block(x)`
或换成 `nn.Sequential`。（本项目实机踩过，报错 `ModuleList is missing forward`）

**Q：pos_embed 长度为什么是 num_patches+2？**
cls + dist + 64 个 patch；无蒸馏模式序列少一个，用 `pos_embed[:, :x.shape[1]]` 切片。

**Q：参数量 5.4M 是怎么来的？**
DeiT-Ti 规格：embed_dim=192、12 层、3 头、mlp 768。`_check_user_files.py` 第一条断言
就是用 5.4M 做配置级体检。

**Q：pre-norm 是什么、为什么用？**
LayerNorm 放在注意力/MLP **之前**（先归一化再计算），训练比 post-norm 稳，是
ViT/DeiT 的标准结构。

**Q：DropPath（Stochastic Depth）是什么？**
以概率随机丢弃**整层残差分支**，等价于随机训练更浅的子网络。论文 DeiT-B 用 0.1；
你的 tiny 上用了 0.1 也完全合法。

---

## 二、损失与蒸馏

**Q：criterion 是什么？必须写成那个样子吗？**
criterion = "把（模型输出，标签）变成标量损失"的可调用物。样子由**你的损失需要哪些
信息**决定：`F.cross_entropy` 要 2 个输入就传 2 个，`Distillation_loss` 要 4 个
（学生双头+教师输出+标签+args）就传 4 个。

**Q：`criterion = Distillation_loss` 是不是就换个名字？**
是。函数是一等公民，赋值后 `criterion(...)` 和原函数是同一个对象；"换名字"的目的是
**解耦训练循环**（换损失只改一行赋值）。

**Q：硬蒸馏和软蒸馏的区别？**
硬蒸馏（Eq.1）：`y_t = argmax(教师输出)`，学生蒸馏头用 CE 拟合这个硬标签，α=0.5、无温度。
软蒸馏（Eq.2）：师生都除 τ=3.0 后算 KL，损失乘 τ²，能传递教师完整的类别相似性。
论文结论：效果几乎一样，硬蒸馏更简单。

**Q：为什么训练和测试不能用同一个 criterion？**
eval 时学生只输出**双头平均后的一个张量**，且推理不用教师——蒸馏损失的两个输入
（蒸馏头、教师 logits）都不存在了。测试 loss 用 `soft_cross_entropy`（= 纯 CE 口径）。

**Q：evaluate 为什么用 soft_cross_entropy？**
测试 = 部署模拟：只用学生。且 smoothing=0 时它与 `F.cross_entropy` **逐位相等**
（冒烟测试实测 5.2256 == 5.2256），用它是为了代码统一。

**Q：soft_cross_entropy 为什么有 2D 分支？**
`targets.ndim == 2` 吃 (B,C) 软标签——这是 v1 就为 Mixup/CutMix 埋好的接口，v2 直接兑现。

**Q：label smoothing 对数值有什么影响？**
平滑 CE = (1-ε)·CE(真值) + ε·CE(均匀分布)。对自信预测，损失地板被抬高（实测标准 CE
0.22 vs 平滑 0.81）；accuracy 不受影响（argmax 不变）。

**Q：为什么参考代码（deit_cifar100.py）不算 test loss？**
① 分类 benchmark 惯例只报 top-1；② 参考版训练 loss 是 Mixup 混合软标签口径，与标准 CE
不可比；③ 选模型只看 acc。你后来加 test loss 是对的，注意两点：除以批数、跨配置别比较。

**Q：为什么加了 Mixup/CutMix 后 test_loss 不降反升？**
你的 test loss 是平滑 CE（smoothing=0.1）。混合训练让模型输出更柔和 → "对其它 99 类的
平均惩罚"项变大 → 数值上升。**这是口径现象不是退化**，跨配置永远看 test_acc。

**Q：蒸馏损失为什么返回 3 个值？**
`(total, base, dist)`：total 用于 backward，base/dist 是给你看的仪表盘（分类损失与蒸馏
损失各自的数值）。

---

## 三、训练循环与调度

**Q：train_one_epoch 里 teacher 为什么要进来？和 ViT 差在哪？**
DeiT 的损失公式里有一项要用教师输出（`y_t = argmax(Z_t)`）。教师 = **只读的外部参考
答案**：`eval() + no_grad()` 只出 logits 绝不更新；学生交两份作业（cls 头对真值、
dist 头对教师）。

**Q：warmup 为什么需要？**
随机初始化时梯度又大又乱，Adam 的前几步矩估计被污染；Transformer 没有 BN 兜底，
对 lr 敏感。论文：前 5 epoch 线性爬坡 + 之后余弦。（完整教程 `warmup教程.md`）

**Q：warmup 手写版（A）怎么"保存进模型"？**
不用保存——A 是无状态调度，`lr = f(epoch)`，续跑时由 epoch 号直接重算。要保存的是
官方库版（B）的 `scheduler.state_dict()`，你的 last.pth 已经存了。

**Q：为什么 loss 统计要除以 len(loader)？**
`F.cross_entropy` 每次返回的是 batch **内**平均（一个标量）；累加 N 个 batch 是总和，
除以 N 得到 batch **间**平均，数值才落在 ln100≈4.6 的可读量级。只影响打印，不影响训练。
（更精确的按样本加权：`*y.numel()/total`，你 v2 已用）

**Q：AMP 的正确顺序？**
`autocast 前向+loss → scaler.scale(loss).backward() → scaler.unscale_(optimizer) →
(梯度裁剪) → scaler.step() → scaler.update()`。你写的全对，连 unscale_ 都没漏。

**Q：梯度裁剪为什么放在 unscale_ 之后？**
AMP 下 backward 出的梯度是"放大过"的，必须先 unscale 还原真实梯度再裁剪，否则阈值
语义错误。

**Q：AdamW 为什么要分组权重衰减？**
只衰减 2D 权重（Linear/Conv），bias/LayerNorm 不衰减——论文细节，约 +1~2 点。
（你的 deittrain 目前是简单版，回访时可加）

**Q：resume 踩过哪些坑？**
① 保存/加载 key 不一致（`student_dict` vs `model_state_dict`）；② `start_epoch` 双加一
跳轮；③ `scaler_dict` 存成 scheduler 的字典。三个都在 `deit手写评价.md` 有复盘。

**Q：为什么 `~args.no_amp` 是错的？**
`~` 是按位取反（`~True = -2`，恒为真值），逻辑非要用 `not`。

**Q：为什么 config 要存进 checkpoint？**
checkpoint 目录是跨实验共享的单例（会被覆盖），内嵌 `vars(args)` 后任何 best.pth 都能
追溯出它的超参——本项目实战验证过（从 best.pth 读出 63.27% @ epoch 79 / epochs=100）。

**Q：M 消融为什么用 `--ra-inc 0` 而不是注释代码？**
开关写进命令行 → config.txt 可追溯、不会"忘恢复"。注释代码是消融纪律的反面教材。

---

## 四、数据增强（Mixup / CutMix / RandAugment）

**Q：Mixup 为什么要把标签变软？**
"0.7 猫 + 0.3 狗"的正确答案本来就是 (0.7, 0.3)。硬标签会让混合样本与目标自相矛盾；
软标签强迫模型输出连续概率，决策边界被拉平滑。（`Mixup_CutMix接入教程.md`）

**Q：CutMix 和 Mixup 的区别？**
Mixup 全局透明度混合（图变"糊"）；CutMix 局部拼贴（保留真实纹理 + 模拟遮挡）。
标签同样按 λ 混合，所以学完 Mixup 再学 CutMix 非常顺。

**Q：教师看混合图，还是看干净图再混 logits？（最终版为什么）**
最终版：**教师看混合图 T(X_mixed)**——精确。旧版"干净图 + logits 混 λ"是线性近似
（教师非线性，CutMix 下误差明显）。澄清：官方 DeiT 仓库其实做的是 logits 级混合
（图像不混）；图像级混合 + 教师看混合图是 timm 路线，本项目最终对齐 timm。

**Q：RandAugment 的 n 和 m 分别控制什么？**
n = 做**几种**增强（数量）；m = 每种增强**多狠**（强度，统一标尺 0~30）。

**Q：为什么所有操作统一用 0~30 标尺？**
旋转是角度、亮度是强度、海报化是位数——物理量不同，需要统一"强度语义"；
每个操作自己把 m 映射到自己的物理量（如 ±30°×(m/30)）。这是 RandAugment 最漂亮的设计。

**Q：mstd 是什么？**
中心强度 m 的抖动范围：实际幅度在 [m-0.5, m+0.5] 随机，防止所有图"一样狠"。

**Q：inc（幅度递增）为什么续跑要用无状态公式？**
论文 `rand-m9-inc1` 要求幅度逐 epoch +1。`m = ra_m + (epoch-1)*ra_inc` 由 epoch 号直接
算，续跑天然一致；`m += inc` 续跑会从 ra_m 重新开始。注意：M 消融时必须 `--ra-inc 0`。

**Q：为什么要用 `__call__`？**
让实例"像函数一样被调用"（`ra(img)`）。torchvision 的 Compose 对每个 transform 都执行
`t(img)`，实现 `__call__` 才有资格进流水线；`nn.Module` 的 `model(x)` 是同一个套路。

**Q：为什么 RandAugment 放在 Normalize 之前？**
对比度/亮度/色彩类操作都假设像素 ∈ [0,1] 的"图像"；Normalize 后变成零均值分布，
再做这些操作数值语义就乱了。

**Q：13 个操作分别教什么不变性？**
AutoContrast/Equalize→光照曝光；Rotate→旋转；ShearX/Y→仿射形变；TranslateX/Y→位置
（Transformer 最缺）；Contrast→对比度；Color→饱和度；Brightness→亮度；Sharpness→细节；
Posterize→忽略精细色彩；Solarize→过曝。（每个操作的完整注释在 `deittrain_v2.py`）

**Q：`2 * random.random() - 1` 是什么？**
[-1,1] 均匀分布 = 随机**方向**。增强必须双向（左旋也右旋、变亮也变暗），只单向等于
给数据加系统性偏差。

**Q：M 消融的结论是什么？**
CIFAR-100 上只开 RA：M=0/5/9/15/20 → 63.27/65.47/66.59/67.16/**68.36%**——单调上升、
到 20 未拐头，**68.36% 是 v2 最优**。与论文 ImageNet 的 m=9 经验相反：小图需要更强变换。

**Q：全配方（Mixup+CutMix+RA+inc）为什么反而更低？**
train_acc 崩到 43.86%：三重增强 + 递增幅度 20 轮后顶到极限 30 → 正则过强 → 欠拟合。
论文配方是给 ImageNet（128 万图/大模型/300 epoch）调的，CIFAR 上要自己找强度。

---

## 五、工程与工具

**Q：tqdm 怎么加进训练循环？**
`pbar = tqdm(loader, desc=..., leave=False)` 包住循环 + 批内 `pbar.set_postfix(loss=...,
acc=...)`；文件头用 try/except 导入做可选依赖回退（没装 tqdm 也能跑）。

**Q：冒烟测试的最小数据是怎么编造的？**
四个依据：① 形状从真实管道倒推（(B,3,32,32)）；② 数学期望（随机 logits 的 CE ≈ ln100
≈ 4.6）；③ 官方实现当裁判（smoothing=0 必须与 F.cross_entropy 逐位相等）；④ 分支覆盖
（hard/soft/无蒸馏/eval 模式全走一遍）。（完整方法论 `冒烟测试学习.md`）

**Q：PyCharm 空闲 CPU 100% 是为什么？**
元凶：**git 仓库建在了父目录**（D:\project\self_supervised_learning），PyCharm 的 VCS
要扫描整棵树（含 340MB 数据）；打开"已跟踪且已修改"的文件（如 deit_cifar100.py）会
触发 diff + 全仓库刷新，而未跟踪文件（v2）不会。解法：仓库迁移到 06.DeiT 并 commit
干净（详见当时的诊断回复）。

**Q：为什么每组实验要用独立 ckpt 目录？**
`./checkpoint/best.pth` 是跨实验单例，后跑的实验会覆盖前面的最优模型。
`run_m_ablation.py` 已自动为每组分配 `checkpoint/m_ablation_m{m}`。

**Q：run_m_ablation.py 怎么用？**
`python run_m_ablation.py`（全 5 组）/ `--only 0,9`（子集）/ `--dry-run`（预览命令）。
自动收集每组 best acc、写 `m_ablation_results.txt`、每组独立 checkpoint。

---

## 六、实验结论速查（全部数据）

| 配置 | test_acc (best) |
|---|---|
| TeacherCNN（教师, 30 ep） | 69.0% |
| v1 Baseline（无增强） | 63.27% |
| + Mixup | 66.45% |
| + CutMix | 67.90% |
| + RandAugment (m=9) | 66.59% |
| + Mixup + CutMix | 67.12% |
| 全配方 (inc=1) | 66.22% |
| RA 单独: m=5 / 15 / 20 | 65.47% / 67.16% / **68.36%** |

**当前最优：RandAugment 单独（n=2, m=20, inc=0）= 68.36%。**

---

## 七、⏸ 待补充（回访 SSL/DINO 时再问）

以下问题当时决定暂停，你回来后可逐一问我，我把答案补进本文件：

1. **重复增强 RA3** 是什么？论文消融里它贡献多大？怎么实现？
2. **EMA** 的完整推导：decay 怎么选？为什么推理用影子权重？怎么和 checkpoint 配合？
3. `--distill none` 消融还没跑——"蒸馏到底有没有用"待回答；
4. Teacher quality：不同档教师（60/66/69/71%）→ 学生曲线；
5. M=25/30 找拐点；CutMix+RA(m=20) 组合；全配方固定 m 重跑；多 seed 平均；
6. SSL/DINO 回访复盘清单（来自 `DeiT_v2学习路线.md` 第 9 节）：
   - SimCLR：为什么大 batch？temperature？projection head？增强为什么是核心？
   - MoCo：为什么 queue？为什么 momentum encoder？
   - BYOL：为什么不需要负样本？为什么 EMA teacher？
   - DINO：teacher/student？centering/sharpening？multi-crop？
   - MAE：为什么 mask？为什么 encoder 只看可见 patch？decoder 为什么可以很轻？
