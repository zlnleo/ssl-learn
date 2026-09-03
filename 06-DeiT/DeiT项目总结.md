# DeiT 项目总结（CIFAR-100 手写复现 · 完结篇）

> 2026-08-31。从 ViT 出发手写推导 DeiT → 跑通 v1 baseline → v2 增强消融 →
> 以 M 消融收官。**本文件是项目的时间胶囊，回访 SSL/DINO 前看这一篇就够。**

---

## 一、一句话总结

不用 timm、不抄官方仓库，从 ViT 自己推出了 DeiT（蒸馏 token + 双头 + 硬/软蒸馏），
v1 baseline **63.27%**，再用 Mixup / CutMix / RandAugment 的逐项消融把最优推到
**68.36%**（RA 单独, m=20），全程"一次一个变量 + 记录 + 核对"。

## 二、最终实验总表（全部实机数据，runs/ 可追溯）

| 实验 | 配置 | test_acc (best) |
|---|---|---|
| 教师 | TeacherCNN, 30 epochs, SGD | 69.0% |
| 1 | v1 Baseline（无增强, 硬蒸馏） | 63.27% |
| 2 | + Mixup | 66.45% |
| 3 | + CutMix | 67.90% |
| 4 | + RandAugment (n=2, m=9, inc=0) | 66.59% |
| 5 | + Mixup + CutMix | 67.12% |
| 6 | 全配方 (Mixup+CutMix+RA, inc=1) | 66.22% |
| M 消融 | RA 单独: m=0 / 5 / 9 / 15 / 20 | 63.27 / 65.47 / 66.59 / 67.16 / **68.36%** |

**当前最优：`deittrain_v2.py --mixup 0 --cutmix 0 --ra-m 20 --ra-inc 0` → 68.36%。**

## 三、时间线（关键节点与踩过的坑）

1. **模型推导**（deitmodel.py）：蒸馏 token 双头结构一次成型；自修 `ModuleList` 不可调用、
   pos_embed 长度两个坑，冒烟自检全绿；
2. **损失推导**（deitloss.py）：硬/软蒸馏公式正确；`tempetature` 拼写坑 → `tau`；
3. **v1 训练**（deittrain.py）：8 处接口 bug 复盘（ckpt 参数名、教师返回值丢弃、criterion
   零参构造、return 缩进、evaluate 签名、scaler_dict 存错、`~` 按位取反、autocast 未接
   开关）→ 修完全部跑通 → **63.27%**（教师 69%，train 97.3%/test 63.2% 的过拟合 gap
   成为 v2 的动机）；
4. **v2 Mixup/CutMix**（deittrain_v2.py）：三处接线 bug（混合图没喂给学生、教师 logits
   漏 `[idx]`、criterion 参数顺序）→ 最终版定为"教师看混合图"（timm 路线，并澄清了
   官方仓库其实是 logits 级混合）；
5. **RandAugment**：13 个操作手写 + 全量逐行注释 + 幅度递增升级为无状态公式（续跑安全）
   + 一键消融脚本 `run_m_ablation.py`（自动收集结果、独立 checkpoint）→ **M=20 拿到
   68.36%，刷新全场最优**；
6. **工程全景**：AMP（含 unscale_+梯度裁剪）、warmup+余弦（官方库版+断点续跑）、tqdm、
   早停、checkpoint 内嵌 config、冒烟测试方法论、实验纪律（config 可追溯/单 seed 意识）。

## 四、学到的核心知识（回访时先过一遍这份清单）

- **结构**：蒸馏 token 参与全程注意力（不是输出端拼接）；训练双头/推理平均；
- **蒸馏**：硬/软公式 Eq.(1)(2)、α=0.5/τ=3.0、教师 = 只读参考答案、测试只用学生；
- **训练**：AdamW 分组 wd、warmup 的无状态 vs 有状态、AMP 五步曲、resume 三坑；
- **增强**：Mixup/CutMix → 软标签 → soft_cross_entropy 2D 分支的完整链路；RA 的
  n/m 分离、统一 0-30 标尺、13 个操作各教一种不变性、`__call__` 约定；
- **实验方法论**：baseline 固化、一次一个变量、M 消融找"强度旋钮"、**"论文全配方
  不敌单用 CutMix"**的教训（配方必须按数据集调）；
- **最漂亮的一条曲线**：M 消融 63.27→68.36 单调上升未拐头——你自己亲手测出来的、
  与论文 ImageNet 经验相反的结论。

## 五、遗留与暂停（回访清单，详见 `deit问题解答.md` 第七节）

⏸ 重复增强 RA3、EMA、`--distill none` 蒸馏对照、Teacher quality、M=25/30 拐点、
CutMix+RA 组合、多 seed 平均 → 全部留到 SSL/DINO 回访时补。

## 六、文档索引

| 文档 | 用途 |
|---|---|
| `DeiT论文讲解.md` | 论文精读（公式/结果表/消融） |
| `deit手写评价.md` | 三轮代码评审 + 全部 bug 复盘 + 结果分析 |
| `deit问题解答.md` | **问答集**（你问过的所有问题 + 注释知识点 + 待补充占位） |
| `DeiT_v2学习路线.md` | v2 学习地图 + 消融实验总表 |
| `Mixup_CutMix接入教程.md` | Mixup/CutMix 融合教程（含"为什么那样写"逐行解释） |
| `RandAugment教程.md` | RA 14 操作详解 + 语法课堂 + M 消融结果 |
| `warmup教程.md` | 学习率 warmup 与断点续跑 |
| `冒烟测试学习.md` | 最小测试数据编造方法论 |
| `README.md` | 仓库门面 |

代码：`deitmodel.py` / `deitloss.py` / `deitteacher.py` / `deittrain.py`(v1) /
`deittrain_v2.py`(v2) / `deit_cifar100.py`(参考版含 EMA) / `run_m_ablation.py`(一键消融)。
