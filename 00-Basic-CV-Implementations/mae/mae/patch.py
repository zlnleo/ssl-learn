"""Stage 1 模块: patchify / unpatchify —— 图像与 patch 序列的双向映射 (无参数)。

行主序约定 (整个项目唯一的 patch 顺序标准):
    第 i 个 patch = patch 网格第 (i // n_w) 行、第 (i % n_w) 列, n_w = W // P
    patch 内部像素顺序 = (channel, row, col), 与 nn.functional.unfold 一致

后续所有模块 (mask 索引 / 位置编码 / 可视化) 都依赖这个约定。
"""
import torch


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, T), 其中 N = (H/P)*(W/P), T = P*P*C"""
    B, C, H, W = x.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, f"图像尺寸 {H}x{W} 必须能被 patch_size={P} 整除"
    n_h, n_w = H // P, W // P
    # (B, C, n_h, P, n_w, P) -> (B, n_h, n_w, C, P, P) -> (B, N, C*P*P)
    x = x.reshape(B, C, n_h, P, n_w, P)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, n_h * n_w, C * P * P)
    return x


def unpatchify(x: torch.Tensor, patch_size: int, h: int, w: int) -> torch.Tensor:
    """patchify 的逆操作: (B, N, T) -> (B, C, H, W)"""
    B, N, T = x.shape
    P = patch_size
    n_h, n_w = h // P, w // P
    C = T // (P * P)
    assert n_h * n_w == N, f"N={N} 与 (H/P)*(W/P)={n_h * n_w} 不一致"
    x = x.reshape(B, n_h, n_w, C, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, n_h, P, n_w, P)
    x = x.reshape(B, C, h, w)
    return x
