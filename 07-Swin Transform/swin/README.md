# swin —— 模块化工程包

> 定位：把 01–09 九个学习模块沉淀为一个**可复用、可 import 的 Swin Transformer 工程包**，
> 是最终 Swin-Tiny 的实现。它不重复解释机制原理（原理见各学习模块与 `paper/`），
> 只负责把机制组装成清晰的模块层次，并暴露稳定接口给训练脚本、消融实验与测试复用。

## 模块文件说明

| 文件 | 职责 | 对应学习模块 |
|------|------|-------------|
| `window.py` | 窗口机制工具集：`window_partition`（特征图切窗口）、`window_reverse`（逆运算还原）、`build_relative_position_index`（相对位置索引表）、`build_attn_mask`（SW-MSA 注意力掩码） | 02 / 03 / 05 |
| `attention.py` | `WindowAttention`：窗口内多头自注意力，含可学习相对位置偏置表 + SW-MSA 掩码广播（QKV 投影 → 拆多头 → 缩放点积 → +偏置 → +掩码 → softmax → AV → 输出投影） | 01 / 03 / 05 |
| `block.py` | `SwinBlock`：pre-norm 双残差块（`W-MSA/SW-MSA` + MLP），`shift_size=0` 为 W-MSA、`=window//2` 为 SW-MSA；mask 惰性缓存；先 pad 到 window_size 整数倍、最后 crop 回原尺寸 | 07 |
| `layer.py` | `BasicLayer`：一个 stage = `depth` 个 SwinBlock（偶数位 W-MSA、奇数位 SW-MSA）+ 可选 PatchMerging；drop_path 可为标量或逐块列表 | 08 |
| `patch.py` | `PatchEmbed`（卷积切 patch）与 `PatchMerging`（2×2 合并：分辨率减半、通道翻倍） | 06（PatchEmbed 属 09 组装） |
| `model.py` | `SwinTransformer` 完整组装 + `swin_tiny/small/base/large` 工厂函数 + `build_swin`（按名称构建）；支持 `patch_merging=False` 消融开关、`num_classes=0` 特征模式 | 09 |
| `config.py` | `SWIN_CONFIGS`：T/S/B/L 标准配置表（embed_dim / depths / num_heads）；`DEFAULT_TRAIN_CFG`：ImageNet-1K 训练超参数速查 | — |
| `utils.py` | `Mlp`（两层全连接 + GELU，隐藏层 4×）与 `DropPath`（stochastic depth，自实现、纯 torch） | — |
| `__init__.py` | 包入口：汇总导出上述类与函数，`__version__ = "1.0.0"` | — |

> 组装顺序（也是学习顺序）：`window.py → attention.py → block.py → layer.py → patch.py → model.py`，
> 底层工具 `utils.py` 与配置 `config.py` 被上层模块引用。

## 快速使用

> 需在 `07-Swin Transform/` 目录下运行（或将其加入 `sys.path`），使 `import swin` 能定位到本包。

```python
import torch
from swin import swin_tiny, build_swin

# 1) 直接构建 Swin-Tiny（默认 1000 类）
model = swin_tiny(num_classes=10)                 # embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24)
out = model(torch.randn(1, 3, 224, 224))          # (1, 10)

# 2) 特征模式（num_classes=0，输出全局平均池化后的 768 维特征）
backbone = swin_tiny(num_classes=0)
feat = backbone(torch.randn(1, 3, 224, 224))      # (1, 768)

# 3) 按名称构建（tiny / small / base / large）
m = build_swin("small", num_classes=100)

# 4) 工程开关示例
m2 = swin_tiny(num_classes=10, patch_merging=False)  # 禁用 PatchMerging（消融实验 3）
m3 = swin_tiny(num_classes=10, window_size=14)       # 覆盖 window_size（消融实验 4）
```

`SwinTransformer` 主要参数（真实签名）：`img_size=224, patch_size=4, in_chans=3, num_classes=1000,
embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24), window_size=7, mlp_ratio=4.,
qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
patch_norm=True, patch_merging=True`。构造时 `assert len(depths) == len(num_heads)`。

## 与学习模块的关系

| 学习模块 | 沉淀到本包的实现 |
|---------|----------------|
| 01 Window Attention | `attention.py::WindowAttention`（最简注意力 → 完整版） |
| 02 Window Partition/Reverse | `window.py::window_partition / window_reverse` |
| 03 Relative Position Bias | `window.py::build_relative_position_index` + `attention.py` 中偏置表 |
| 04 Shifted Window | `block.py` 中 `torch.roll` 循环移位 |
| 05 Attention Mask | `window.py::build_attn_mask` + `attention.py` 掩码广播 |
| 06 Patch Merging | `patch.py::PatchMerging` |
| 07 Swin Block | `block.py::SwinBlock` |
| 08 BasicLayer | `layer.py::BasicLayer` |
| 09 完整 Swin | `model.py::SwinTransformer` + 各工厂函数 |

## 与官方实现的差异点（代码注释中真实提到）

- **mask 惰性缓存**：官方按固定 `input_resolution` 注册 buffer；本包在首次前向按实际 `(H, W)` 构造并缓存，支持任意输入尺寸。
- **非整除尺寸先 pad 再 crop**：所有 block 先 pad 到 window_size 整数倍，最后 crop 回原尺寸。标准配置（56/28/14/7 与 window 7）整除成立、不发生 pad，与官方行为一致；pad 分支让 window_size 消融（如 window=14 作用于 7×7 特征图）也能运行。
- **`patch_merging` 可开关**：`patch_merging=False` 时禁用全部 PatchMerging，形成"无层级"对照模型（消融实验 3 专用）。
- **纯 torch 依赖**：`DropPath` 自实现，无 timm 依赖。

---

返回上级：[07-Swin Transform 目录结构 / 快速开始](../README.md)
