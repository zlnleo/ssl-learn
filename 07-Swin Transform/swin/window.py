"""窗口机制工具集：窗口划分/还原、相对位置索引、SW-MSA 注意力掩码。

对应学习模块：02（partition/reverse）、03（relative position index）、05（attention mask）。
本文件是工程版实现：接口与各学习模块完全一致，供 swin 包内部与消融实验复用。
"""
import torch

__all__ = ["window_partition", "window_reverse",
           "build_relative_position_index", "build_attn_mask"]


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """把 2D 特征图切成窗口：(B, H, W, C) -> (B*nW, window_size, window_size, C)。

    nW = (H//window_size) * (W//window_size)，窗口按行优先编号。
    本质：view 把每个窗口的二维区域暴露成独立维度，permute 调整维度顺序
    使窗口维相邻，contiguous 后即可 view 成 (nW, M, M, C)。
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """window_partition 的逆运算：(B*nW, window_size, window_size, C) -> (B, H, W, C)。"""
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def build_relative_position_index(window_size: int) -> torch.Tensor:
    """构造相对位置索引表，形状 (M^2, M^2)，元素为相对位置偏置表的行号 [0, (2M-1)^2)。

    推导：token a=(i1,j1) 与 b=(i2,j2) 的相对位移 (i1-i2, j1-j2)
    只有 (2M-1)^2 种取值；平移 M-1 归一到 [0, 2M-2]^2；
    再按 (dh*(2M-1) + dw) 展平成偏置表的行号。
    """
    coords = torch.stack(
        torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
    coords = coords.reshape(2, -1)                  # (2, M^2) 展平 token 坐标
    rel = coords[:, :, None] - coords[:, None, :]   # (2, M^2, M^2) 广播相减 = 相对坐标
    rel = rel.permute(1, 2, 0).contiguous()         # (M^2, M^2, 2)
    rel[:, :, 0] += window_size - 1                 # 行位移平移: [-(M-1), M-1] -> [0, 2M-2]
    rel[:, :, 1] += window_size - 1                 # 列位移平移
    rel[:, :, 0] *= 2 * window_size - 1             # 行坐标 × 表宽（混合进制）
    return rel.sum(-1)                              # (M^2, M^2)，long 类型


def build_attn_mask(H: int, W: int, window_size: int, shift_size: int,
                    device: str = "cpu") -> torch.Tensor:
    """构造 SW-MSA 的注意力掩码，形状 (nW, M^2, M^2)：0 = 允许注意力，-100 = 屏蔽。

    H, W：移位窗口注意力实际分窗的特征图尺寸（本项目为 pad 到 window_size 整数倍之后的尺寸）。

    原理：循环移位后的图按 9 宫格切成 3x3 块并编号 0..8；同一窗口内的两个 token
    若来自不同编号块，说明它们是 roll 造成的"伪邻居"（空间上不相邻），必须屏蔽。
    attn 矩阵上这些位置加 -100，softmax 后权重 ≈ exp(-100) ≈ 0。
    掩码只依赖窗口几何结构，与输入内容、batch 均无关，因此可以缓存复用。
    """
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)            # (nW, M, M, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)   # (nW, M^2)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, M^2, M^2) 区域号差
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
