# 4 个核心消融实验

> 用 ssl_cv 环境运行：先 `conda activate ssl_cv`（普通终端），
> 或在脚本路径前显式用 `D:\env\anaconda\envs\ssl_cv\python.exe`。
> 所有脚本在项目根目录运行；实验自动优先使用 GPU（`--device cpu` 可强制 CPU）。

## 总览

| 实验 | 对照 | 观测指标 | 对应学习模块 |
|---|---|---|---|
| exp1 | Global Attention vs Window Attention | FLOPs(MACs)/显存/速度/精度 | 01 Window Attention |
| exp2 | Window vs Shifted Window（**最重要**） | 可达 token 集（感受野）+ 精度 | 04/05/07 |
| exp3 | Swin-Tiny 有/无 Patch Merging | 参数量/FLOPs/速度/精度 | 06/08 |
| exp4 | window_size = 4 / 7 / 14 | 参数量/FLOPs/速度/精度 | 01/03/05 |

每个实验分两部分：**静态测量**（无需训练，秒级出结果，公式 + 实测对照）与
**端到端精度**（CIFAR-10，默认 10 个 epoch，`--skip-train` 可跳过）。

## 运行方式

```bash
# 实验 1：全局 vs 窗口注意力（训练用小模型，10 epoch 很快）
python experiments/exp1_global_vs_window.py
python experiments/exp1_global_vs_window.py --skip-train   # 只看 FLOPs/显存/速度

# 实验 2：窗口 vs 移位窗口（先看结构分析输出的 ASCII 覆盖图）
python experiments/exp2_window_vs_shifted.py
python experiments/exp2_window_vs_shifted.py --skip-train

# 实验 3：有无 Patch Merging（224 输入，训练较慢，先 --skip-train 看静态表）
python experiments/exp3_patch_merging_ablation.py --skip-train
python experiments/exp3_patch_merging_ablation.py           # 全量含训练

# 实验 4：window_size 消融
python experiments/exp4_window_size.py --skip-train
python experiments/exp4_window_size.py --window-sizes 4 7 14
```

## 数据集（默认 CIFAR-100，本地已有数据绝不下载）

- 默认 `--dataset cifar100`（可用 `--dataset cifar10` 切换，`--num-classes` 自动随数据集）。
- 数据目录自动探测：`D:\project\self_supervised_learning\data`（本地已有解压好的
  `cifar-10-batches-py` / `cifar-100-python`）优先，其次 `./data`；
  也可用 `--data-dir` 显式指定。已有数据时 `download=False` 直接加载，绝不触发联网下载；
  只有本地缺失时才允许下载。

## 测量口径（common.py）

- **MACs**：打补丁统计 `torch.matmul`/`bmm` + 前向 hook 统计 `Linear`/`Conv2d`；
  1 MAC = 1 次乘加 ≈ 2 FLOPs（与论文 Ω 公式同口径）。
- **速度**：CUDA 用 `torch.cuda.Event`，CPU 用 `perf_counter`；预热 + 多次取均值。
- **显存**：`torch.cuda.max_memory_allocated` 峰值 + 注意力矩阵显存的解析式
  （`B × h × N² × 4` 字节，全局 N=hw、窗口 N=M²）。
- **精度**：CIFAR-10（resize 到指定尺寸，无 ImageNet 数据也可复现趋势）。

## 预期结论（定性）

1. **exp1**：窗口注意力的注意力部分计算量是全局的 `hw/M²` 分之一（56² 图上 M=7 时为 64×），
   总 MACs 比值随分辨率增长趋近该上限；窗口注意力的显存/速度优势随分辨率扩大而扩大。
2. **exp2**：W-W 两层后 token 的可达集合永远等于本窗口；W-SW 两层后可达集合跨越多个窗口，
   感受野显著扩张——这是精度差异的机制根源。
3. **exp3**：Patch Merging 以更少的参数量/FLOPs 获得不差甚至更好的精度；
   无 merging 变体全部 stage 都在 56×56 高分辨率上做注意力，计算量显著更大。
4. **exp4**：注意力计算量 ∝ M²（16:49:196），window 越大越接近全局注意力；
   7 是精度与成本的工程平衡点。

## 运行时提示

- 数据集默认使用本地已有的 `D:\project\self_supervised_learning\data`（不下载）；
  首次在全新机器上运行且本地无数据时才联网下载（约 170MB）。
- exp3/exp4 的端到端训练在 224 输入上跑 Swin-Tiny，单模型 10 epoch 视 GPU 约
  几分钟到几十分钟；只做趋势观察的话 `--epochs 5` 足够。
- exp4 的 window=14 在最后一个 stage（7×7 特征图）触发 pad-到-窗口尺寸 的分支，
  属正常健壮性设计（见 `swin/block.py` 注释）。
