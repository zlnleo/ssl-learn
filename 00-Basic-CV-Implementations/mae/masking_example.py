"""random_masking 的最小数值例子: N=4, V=2。

运行:
    python masking_example.py

目的:
    手把手演示 shuffle -> keep -> restore -> gather 还原的全过程。
    每个 token 的第一个元素是"身份证号"(10/20/30/40), 一眼能看出谁去了哪里。
"""
import torch

from mae.masking import random_masking


def main():
    torch.manual_seed(42)  # 固定种子 -> 每次运行输出相同

    # 手工构造: 1 张图 (B=1), 4 个 token (N=4), 每个 token 3 维 (D=3)
    x = torch.tensor([[
        [10.0, 11.0, 12.0],  # x0
        [20.0, 21.0, 22.0],  # x1
        [30.0, 31.0, 32.0],  # x2
        [40.0, 41.0, 42.0],  # x3
    ]])

    mask_ratio = 0.5  # N=4 -> len_keep = int(4 * 0.5) = 2 -> V=2, M=2
    x_vis, mask, ids_restore, ids_keep, ids_shuffle = random_masking(x, mask_ratio)

    B, N, D = x.shape
    V = ids_keep.shape[1]

    print("=" * 64)
    print("手工构造的 4 个 token  x = [x0, x1, x2, x3]")
    print(x)
    print()
    print(f"ids_shuffle (随机排列)  = {ids_shuffle.tolist()}")
    print(f"ids_keep    (前 V={V} 个)  = {ids_keep.tolist()}")
    print(f"ids_restore (逆排列)      = {ids_restore.tolist()}")
    print(f"mask (True=被遮)          = {mask.tolist()}")
    print()
    print("x_vis = gather(x, ids_keep)  (可见 token, 按 ids_keep 顺序):")
    print(x_vis)
    print()

    # ---- 模拟 decoder 的拼回与还原: 用 -1 代表 mask token ----
    placeholder = torch.full((B, N - V, D), -1.0)
    concat = torch.cat([x_vis, placeholder], dim=1)  # [x_visible + mask_token]
    restored = torch.gather(concat, dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

    print("concat = [x_visible | mask_token(-1)] =")
    print(concat)
    print()
    print("gather(concat, ids_restore) 恢复后的序列 =")
    print(restored)
    print()

    # ---- 验证: 恢复后, 可见位置必须是原始 token, 被遮位置必须是 -1 ----
    expected = x.clone()
    expected[mask] = -1.0
    print("期望结果 (原始顺序, 被遮位置填 -1) =")
    print(expected)
    print()

    assert torch.equal(restored, expected), "逆排列还原失败!"
    print("[OK] 验证通过: gather(cat([x_vis, mask_token]), ids_restore) 精确恢复了原始顺序")

    # ---- 额外观察: mask 与 ids_restore 的等价关系 ----
    print()
    print("额外观察: mask == (ids_restore >= V) ?")
    print("  ids_restore >= V :", (ids_restore >= V).tolist())
    print("  mask             :", mask.tolist())
    print("  两者完全一致: 位置 i 被遮 <=> 还原索引 ids_restore[i] 落在 mask token 区段")


if __name__ == "__main__":
    main()
