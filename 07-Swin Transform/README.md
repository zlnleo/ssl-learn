# Swin Transformer 从零学习项目（机制优先，非官方复刻）

> 目标：按 **Window Attention → Window Partition/Reverse → Relative Position Bias →
> Shifted Window → Attention Mask → Patch Merging → Swin Block → BasicLayer → 完整 Swin**
> 的顺序逐模块学懂 Swin 的**核心机制**，并用模块化工程实现 Swin-Tiny
> （argparse / checkpoint / resume / TensorBoard / 日志 / 消融框架），
> 配合 **4 个核心消融实验** 验证每个机制的"为什么"。
>
> 环境：conda 环境 `ssl_cv`（Python 3.10，torch 2.11.0+cu128，RTX 4060 Laptop）。
> 所有代码、文档、测试均在本机实测通过。

## 1. 学习路线

**论文层入口**：先读 [`paper/论文讲解.md`](paper/论文讲解.md)（逐节解读），
按 [`paper/复现学习路径.md`](paper/复现学习路径.md) 的 M0-M8 里程碑动手，
对照 [`paper/minimal_swin.py`](paper/minimal_swin.py) 单文件最小复现。

每个模块**先回答"为什么需要"→ 再实现 → 再做 shape/debug 实验**，配套五件套：

| # | 模块 | 核心问题 | 关键结论（本机实测） |
|---|------|---------|---------------------|
| 01 | [Window Attention](01_window_attention/) | 为什么要把注意力限制在窗口内？ | 全局 MSA 注意力矩阵 $hw\times hw$ 平方级爆炸；窗口化后注意力部分计算量降 $hw/M^2$ 倍（56² 图 M=7 时 **64×**，实测总 MACs 2.00G→145M） |
| 02 | [Window Partition/Reverse](02_window_partition/) | 2D 特征图怎么无损切成窗口序列？ | view+permute 的索引映射双射；partition/reverse 逐元素误差 = 0 |
| 03 | [Relative Position Bias](03_relative_position_bias/) | 注意力天然没有位置信息怎么办？ | 窗口内相对位移只有 $(2M-1)^2$ 种，一张可学习偏置表即可（M=7,h=3 → 507 参数） |
| 04 | [Shifted Window](04_shifted_window/) | 窗口间零交流怎么办？ | `torch.roll` 循环移位让窗口错位跨界；两层后可达 token 从 16 → 64 |
| 05 | [Attention Mask](05_attention_mask/) | roll 把不相邻区域卷进同一窗口怎么办？ | 9 宫格区域编号，伪邻居处 attention 加 -100 → softmax 后严格归零 |
| 06 | [Patch Merging](06_patch_merging/) | 为什么要层级降采样？ | 2×2 相邻 patch 通道拼接：分辨率减半 + 通道翻倍（CNN pooling 的 Transformer 版） |
| 07 | [Swin Block](07_swin_block/) | W-MSA/SW-MSA 怎么组成一个块？ | pre-norm 双残差；偶数块 W、奇数块 SW 成对使用；mask 惰性缓存 |
| 08 | [BasicLayer](08_basic_layer/) | stage 怎么组织？ | blocks + PatchMerging；56→28→14→7，通道 96→192→384→768 |
| 09 | [完整 Swin](09_swin_transformer/) | 全部机制怎么总装？ | Swin-Tiny 参数 **28,288,354**（论文 28.3M），224 输入 MACs ≈ **4.49G** |

每个模块文件夹内 6 个文件：`README.md`（学习文档，七段式）、`math.md`（数学推导）、
`<模块>.py`（最小可运行代码）、`shape_tracking.py`（Tensor Shape 逐步跟踪）、
`experiment.py`（shape/debug 实验）、`test_<模块>.py`（unittest 单元测试）。

## 2. 目录结构

```
07.Swin Transform/
├── 01_window_attention/ ... 09_swin_transformer/   # 9 个学习模块（每个 6 件套）
├── paper/                    # 论文层三件套
│   ├── 论文讲解.md            # ICCV 2021 Best Paper 逐节解读 + 复现要点
│   ├── 复现学习路径.md        # M0-M8 里程碑，每个带可执行验收标准
│   └── minimal_swin.py       # 最小复现参考代码（单文件，验收 PASS）
├── swin/                     # 模块化工程包（最终 Swin-Tiny 实现）
│   ├── window.py             # 窗口划分/还原、相对位置索引、注意力掩码
│   ├── attention.py          # WindowAttention（相对偏置 + 掩码广播）
│   ├── block.py              # SwinBlock（W-MSA/SW-MSA + 双残差 MLP）
│   ├── layer.py              # BasicLayer
│   ├── patch.py              # PatchEmbed / PatchMerging
│   ├── model.py              # SwinTransformer + swin_tiny/small/base/large
│   ├── config.py             # Swin-T/S/B/L 标准配置
│   └── utils.py              # Mlp / DropPath
├── experiments/              # 4 个核心消融实验
│   ├── common.py             # MACs 统计 / 显存 / 速度 / CIFAR-10 / 训练循环
│   ├── exp1_global_vs_window.py
│   ├── exp2_window_vs_shifted.py
│   ├── exp3_patch_merging_ablation.py
│   └── exp4_window_size.py
├── train.py                  # 工程训练脚本（argparse/checkpoint/resume/TB/日志/AMP）
├── data.py                   # CIFAR-10 / ImageNet / 自定义文件夹
├── tests/test_swin_package.py# 工程包集成测试
├── run_all_tests.py          # 一键运行全部测试（10 个测试文件）
└── requirements.txt
```

## 3. 快速开始

```bash
# 激活环境（普通终端）；或直接用解释器全路径（沙箱/CI 场景）
conda activate ssl_cv
python --version   # 3.10.x

# 0) 论文层入门：讲解 + 复现路径 + 最小代码（三个验收一跑即知）
python paper/minimal_swin.py            # 参数 28.3M / MACs 4.5G / 过拟合 三关验收

# 1) 全部单元测试（10 个文件，几十秒）
python run_all_tests.py

# 2) 单个模块学习（以模块 01 为例）
python 01_window_attention/shape_tracking.py   # Tensor Shape 逐步跟踪
python 01_window_attention/experiment.py       # MACs/显存/速度 debug 实验
python 01_window_attention/test_window_attention.py

# 3) 4 个核心消融实验（详见 experiments/README.md）
python experiments/exp1_global_vs_window.py    # 全局 vs 窗口注意力
python experiments/exp2_window_vs_shifted.py   # 窗口 vs 移位窗口（最重要）
python experiments/exp3_patch_merging_ablation.py
python experiments/exp4_window_size.py

# 4) Swin-Tiny 工程化训练（默认 CIFAR-100，本地已有数据不下载）
python train.py --dataset cifar100 --img-size 224 --epochs 100 --batch-size 64
python train.py --resume output/train/last.pt        # 断点续训
python train.py --resume output/train/best.pt --eval-only
tensorboard --logdir output/train/runs
```

> 数据集说明：默认使用本地已有的 `D:\project\self_supervised_learning\data`
> （内含解压好的 CIFAR-10 / CIFAR-100 / FashionMNIST）；已有数据时绝不触发联网下载，
> 缺失时才允许下载。可用 `--dataset cifar10|cifar100` 与 `--data-dir` 切换。

> Windows 沙箱/CI 环境提示：`conda run` 可能被命名管道限制拦截，此时直接用
> `D:\env\anaconda\envs\ssl_cv\python.exe <脚本>` 运行，效果一致。

## 4. 核心机制速查

**复杂度（MAC 计数，1 MAC ≈ 2 FLOPs）**

$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C \qquad\qquad \Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$$

注意力部分比值 = $hw/M^2$：56² 图 M=7 时 **64×**（实测总 MACs 比值 13.8×，随分辨率增大趋近 64×）。

**SW-MSA 三步曲**：`pad` → `torch.roll(-s)` → 分窗注意力(加 mask) → 还原 → `torch.roll(+s)` → crop。
掩码 = 9 宫格区域编号差的 {0, -100} 矩阵，与输入内容无关 → 可缓存。

**Patch Merging**：$H\times W\times C \to \frac{H}{2}\times\frac{W}{2}\times 2C$，
2×2 相邻 patch 按位置分组通道拼接（4C）→ LN → Linear(4C→2C)。

## 5. 四个核心消融实验（本机实测，RTX 4060 Laptop / ssl_cv 环境）

| 实验 | 对照 | 静态结论（实测） | 精度结论（实测） |
|------|------|----------------|---------|
| 1 全局 vs 窗口 | 同一注意力块，仅窗口化 | 56² 图：2.00G→145M MACs（13.8×），显存 225→3.5MB，CPU 耗时 73→2.5ms（30×） | CIFAR-100 10epoch：global **32.1%** vs 纯 window **23.7%**——窗口化省算力但窗口间零交流，精度缺口由实验 2 的移位解决 |
| 2 窗口 vs 移位 | W-W vs W-SW 两层 | 两层后可达 token：**16 vs 64**（ASCII 覆盖图直观展示） | 精度训练留给用户跑（框架已冒烟验证，趋势预期 shifted 显著更优） |
| 3 有无 Merging | Swin-Tiny 开关 patch_merging | with: 27.5M 参数/8.98G；without: 1.35M 参数/9.06G——层级用通道换分辨率 | 同上 |
| 4 window_size | 4 / 7 / 14 | 8.98G(7) < 9.37G(4) < 11.38G(14)：注意力 ∝ M²，非整除 pad 也花钱 | 同上 |

> 实测口径：exp1 为完整运行（静态+端到端）；exp2-4 的静态部分完整实测，
> 端到端训练框架与 exp1 共用同一套 `train_model` 且经 train.py 冒烟验证
> （CIFAR-100、64px、2 epoch、CUDA、val_acc 21.7%），按需用 `--epochs 10` 跑全量。

## 6. 与官方实现的差异（有意为之）

| 点 | 官方 | 本项目 | 原因 |
|---|---|---|---|
| mask 构造 | 构造时按 input_resolution 注册 buffer | **惰性缓存**（首次前向按实际 (H,W) 构造并缓存） | 支持任意输入尺寸，机制更透明 |
| 非整除尺寸 | 假设整除（224/7 配置成立） | **所有 block 先 pad 到 window_size 整数倍，最后 crop** | window_size 消融（exp4）必需；整除时行为完全一致 |
| patch_merging | 固定开启 | 可开关（`patch_merging=False`） | 实验 3 消融 |
| 依赖 | timm | 纯 torch（DropPath 自实现） | 环境干净 |

> 注：pad 分支下被 pad 的零 token 会参与窗口内注意力（边界 token 的输出被轻微稀释，
> 随后被 crop 丢弃）。标准 224/7 配置全程整除、不发生 pad，与官方数值完全一致；
> 该取舍在 `swin/block.py` 注释中已说明。

## 7. 实测记录（ssl_cv 环境）

- 单元测试：10 个测试文件、81 个用例全部 PASS（`python run_all_tests.py`）
- Swin-Tiny 参数：28,288,354（论文 28.3M ✓）；224 前向 MACs ≈ 4.49G（论文 4.5G ✓）
- 训练工程冒烟（CUDA）：train 1 epoch → 保存 best/last checkpoint → resume 续训 →
  eval-only 全链路通过；TensorBoard events 与 train.log 正常落盘
  （CIFAR-100 64px 2 epoch val_acc=0.2167，device: cuda）
- 实验 1 完整实测：global 32.1% vs 纯窗口 23.7%（CIFAR-100 10epoch）；
  实验 2/3/4 静态部分实测通过（精度训练按需自行运行）
