# 模块 07 · Swin Block（W-MSA / SW-MSA 总装）

> 学习顺序：`06_patch_merging` → `07_swin_block` → `08_basic_layer`
>
> 运行环境：`D:\env\anaconda\envs\ssl_cv\python.exe`（torch 2.11.0+cu128，脚本默认 CPU）

本模块把 Swin Transformer 的核心机制——**窗口多头自注意力（W-MSA）** 与**移位窗口多头
自注意力（SW-MSA）**——连同相对位置偏置、注意力掩码、MLP、DropPath 组装成一个基本块
`SwinBlock`。这是整个 Swin 架构的灵魂所在。

---

## ① 为什么需要（动机）

标准 ViT 做全局自注意力，复杂度 $O(N^2)$，在高分辨率特征（56×56）上不可承受。Swin 的
对策是**窗口化**：

- 只在 $M\times M$（默认 7×7）窗口内做注意力，复杂度降到 $O(HW\cdot M^2)$（对 token 数线性）；
- 但纯窗口化会让 token 永远困在自己的窗口里，跨窗口信息不流动；
- 于是让相邻两层交替使用 **W-MSA** 与 **SW-MSA**（循环移位窗口），既保持每层计算量恒定，
  又让信息跨窗口流动。

一句话：**SwinBlock 用“窗口化 + 循环移位”在效率与全局建模之间取得平衡。**

---

## ② 核心机制讲解（第一性原理）

### 结构图（ASCII）

```
        ┌────────────────────────────────────────────┐
        │  SwinBlock（pre-norm + 双残差）              │
        │                                              │
 x ─────┼──────────────────────────────────────┐       │
        │   ┌─────┐   ┌──────────────┐         │       │
        ├──►│ LN1 │──►│ W/SW-MSA     │──⊕───────┼──► y  │
        │   └─────┘   └──────────────┘  ▲        │  │    │
        │                              │残差1     │  │    │
        │   ┌─────┐   ┌───────────┐   DropPath  │  │    │
        ├──►│ LN2 │──►│ MLP(4C)   │──⊕──────────┘  │    │
        │   └─────┘   └───────────┘  ▲残差2        │    │
        │                           DropPath      │    │
        └────────────────────────────────────────────┘
```

- **pre-norm**：LN 放在注意力/MLP 支路**之前**，残差是恒等捷径，梯度直通，深层训练稳定
  （对照 ViT 的 post-norm：LN 在残差之后，深层易不稳定）。
- **两处残差 + DropPath**：残差 1 跳过注意力、残差 2 跳过 MLP；DropPath 以概率 $p$ 把整条
  支路置零（stochastic depth），除以 $1-p$ 保持期望，起正则作用。

### W-MSA 与 SW-MSA 为什么必须成对交替

- **偶数块 W（shift=0）**：常规 7×7 分窗，窗口内做注意力；
- **奇数块 SW（shift=window_size//2）**：先循环移位再分窗，窗口跨越了上一层窗口的边界，
  让信息跨窗口流动；
- **每层计算量恒定**：移位不改变窗口数量（仍是不重叠网格），所以 W 与 SW 的复杂度完全一致。

若只用 W-MSA（不交替），token 的感受野永远停在 7×7 窗口内，如 `experiment.py` 所证：
W-W 两层后每个 token 仍只看到 16 个邻居（M=4 时），而 W-SW 两层后能覆盖全图。

### 相对位置偏置（窗口内的几何归纳偏置）

窗口内两 token 的相对位置种类有限（$(2M-1)^2$ 种），查一张偏置表加到注意力分数上。相对
偏置天然平移不变、参数共享高效，比绝对位置编码更适配窗口结构。

### SW-MSA 的注意力掩码

循环移位后，部分窗口混入了原图不相邻区域的 token，用 9 宫格编号 + 掩码 $-100$ 屏蔽它们，
保证“不该注意的不注意”。掩码**只与 (H,W,window,shift) 有关、与输入内容无关**，故可惰性缓存。

---

## ③ 逐段代码讲解

文件：`swin_block.py`（自包含全部小部件）

```python
def window_partition(x, window_size):
    x = x.view(B, H//M, M, W//M, M, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, M, M, C)
    return windows
```

先把 $(H,W)$ 切成网格，`permute(0,1,3,2,4,5)` 把窗口网格的行列提到一起，再展平成
`(B*nW, M, M, C)`。`window_reverse` 是其逆运算。

```python
def build_relative_position_index(window_size):
    coords = meshgrid(arange(M), arange(M))   # (2, M, M)
    rel = coords[:, :, None] - coords[:, None, :]   # (2, M^2, M^2) 相对坐标
    rel[:, :, 0] += M - 1; rel[:, :, 1] += M - 1    # 平移到 [0, 2M-2]
    rel[:, :, 0] *= 2*M - 1; return rel.sum(-1)     # 行优先一维索引 (M^2, M^2)
```

把每对 token 的相对坐标量化成 $(2M-1)^2$ 种偏置之一。

```python
def build_attn_mask(H, W, window_size, shift_size, device="cpu"):
    img_mask = zeros(1, H, W, 1)
    h_slices = (slice(0,-M), slice(-M,-s), slice(-s,None))   # 3 段
    ...
    attn_mask = mask_windows[:,None,:] - mask_windows[:,:,None]  # (nW, M^2, M^2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)    # 不同区域 -> -100
```

给 9 块编号后分窗，窗口内编号不同即屏蔽。

```python
class WindowAttention(nn.Module):
    def forward(self, x, mask=None):
        qkv = self.qkv(x).reshape(B_, N, 3, h, d).permute(2,0,3,1,4)  # (3,B_,h,N,d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2,-1)   # (B_, h, N, N)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, -1).permute(2,0,1).unsqueeze(0)  # (1,h,N,N)
        attn = attn + bias
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, h, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, h, N, N)
        attn = attn.softmax(dim=-1); attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1,2).reshape(B_, N, C)
        return self.proj_drop(self.proj(out))
```

`qkv` 一次投影出 Q/K/V；`scale=1/sqrt(d)` 缩放；相对偏置查表后广播加；mask 从 `(nW,N,N)`
广播到 `(B_, h, N, N)`。

```python
class SwinBlock(nn.Module):
    def forward(self, x, H, W):
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        pad_r = (M - W % M) % M; pad_b = (M - H % M) % M
        x = F.pad(x, (0,0, 0,pad_r, 0,pad_b))       # 1) pad 到整数倍
        Hp, Wp = H + pad_b, W + pad_r
        if self.shift_size > 0:
            x = torch.roll(x, (-s,-s), dims=(1,2))   # 2) 循环移位
        x = window_partition(x, M).view(-1, M**2, C) # 3) 分窗
        x = self.attn(x, mask=self._get_mask(Hp, Wp, x.device))
        x = x.view(-1, M, M, C)
        x = window_reverse(x, M, Hp, Wp)             # 4) 还原
        if self.shift_size > 0:
            x = torch.roll(x, (s, s), dims=(1,2))
        x = x[:, :H, :W, :].contiguous().view(B, L, C)
        x = shortcut + self.drop_path(x)             # 残差 1
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 残差 2
        return x
```

要点：

- **pad 在 roll 之前**：先补到整数倍，再在统一坐标系上做循环移位，最后 crop 回原尺寸；
  整除时 pad 为 0，与官方行为完全一致。
- **mask 惰性缓存**：`_get_mask` 用 `(H, W, device)` 作 key，命中则复用，不重复构造。

---

## ④ Tensor Shape 跟踪总表

与 `shape_tracking.py` 输出一致（SW-MSA，$B{=}2, H{=}W{=}8, C{=}96, M{=}4, s{=}2$，整除无 pad）：

| 步骤 | 操作 | 输出形状 |
|---|---|---|
| 0 | 输入 `x` | $(2, 64, 96)$ |
| 1 | `norm1 + view(B,H,W,C)` | $(2, 8, 8, 96)$ |
| 2 | pad（整除时为 0） | $(2, 8, 8, 96)$ |
| 3 | `roll(-s,-s)` | $(2, 8, 8, 96)$ |
| 4 | `window_partition` | $(2\cdot4, 4, 4, 96)$ |
| 5 | `view(-1, M^2, C)` | $(8, 16, 96)$ |
| 6 | `attention mask` | $(4, 16, 16)$ |
| 7 | `window attention` | $(8, 16, 96)$ |
| 8 | `view(-1, M, M, C)` | $(8, 4, 4, 96)$ |
| 9 | `window_reverse` | $(2, 8, 8, 96)$ |
| 10 | `roll(+s,+s)` | $(2, 8, 8, 96)$ |
| 11 | `crop + view(B,L,C)` | $(2, 64, 96)$ |
| 12 | 残差 1 / 残差 2 | $(2, 64, 96)$ |

非整除例（$H{=}W{=}10, M{=}4$）：pad 后 $(2,12,12,96)$，窗口数 $\frac{12}{4}\cdot\frac{12}{4}=9$，
mask 形状 $(9,16,16)$，最终 crop 回 $(2,100,96)$。

---

## ⑤ Debug 实验指南

运行 `shape_tracking.py` 逐步对照上表；运行 `experiment.py` 看感受野对比与 mask 缓存。

常见排查点：

1. 输出形状不对 → 检查 `L == H*W`、`view(B, H, W, C)` 的 H/W 传参是否与序列长度一致；
2. SW-MSA 结果异常 → 检查 `torch.roll` 正反移位符号是否相反（先 $-s$ 后 $+s$）；
3. 非整除时报错 → 检查是否先 pad 再 roll、mask 是否用 pad 后的 `(Hp, Wp)` 构造；
4. mask 每次都重建 → 检查 `_mask_key` 是否包含 device（CPU/CUDA 切换会触发重建）。

---

## ⑥ 单元测试覆盖点

文件：`test_swin_block.py`

| 测试 | 覆盖点 |
|---|---|
| `test_output_shape_preserved` | W/SW 输出 $(B,L,C)$ 不变 |
| `test_shift0_equals_manual_partition_attention_reverse` | shift=0 与手工 partition+attention+reverse 一致 |
| `test_two_blocks_stacked` | W+SW 两 block 堆叠可运行 |
| `test_mask_cache_hit` | 同尺寸第二次前向命中缓存、换尺寸才重建 |
| `test_drop_path_train_eval` | DropPath 训练置零/缩放、eval 恒等 |
| `test_non_divisible_hw` | 10×10、window 4 非整除也能运行且形状不变 |
| `test_partition_reverse_roundtrip` | window_partition/reverse 互逆 |
| `test_relative_position_index` | 相对位置索引形状与取值范围 |

---

## ⑦ 与前后模块关系

- **上游**：`06_patch_merging` 在 stage 之间降采样；`SwinBlock` 在**同一 stage 内**保持
  分辨率与通道不变（每个 stage 内 H、W、C 恒定）。
- **本模块**：提供 `window_partition/window_reverse/build_relative_position_index/
  build_attn_mask/Mlp/DropPath/WindowAttention/SwinBlock`，全部自包含，可直接被
  `08_basic_layer` 复用。
- **下游**：`08_basic_layer` 把 `depth` 个 `SwinBlock`（偶数位 W、奇数位 SW）+ 末尾一个
  `PatchMerging` 组装成 stage，串成 56→28→14→7、96→192→384→768 的金字塔。
