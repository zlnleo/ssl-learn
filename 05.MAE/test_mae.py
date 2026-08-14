import torch


def random_masking(x, mask_ratio=0.5):

    print("=" * 70)
    print("Step 0：原始 x")
    print("x =", x.squeeze(-1))

    B, N, D = x.shape

    keep_num = int(N * (1 - mask_ratio))

    print("\n" + "=" * 70)
    print("Step 1：基本信息")
    print("B =", B)
    print("N =", N)
    print("D =", D)
    print("keep_num =", keep_num)

    # ============================================================
    # 1. 生成随机数
    # ============================================================

    noise = torch.rand(B, N)

    print("\n" + "=" * 70)
    print("Step 2：noise")
    print("noise =", noise)

    # ============================================================
    # 2. 根据随机数排序
    # ============================================================

    ids_shuffle = torch.argsort(noise, dim=1)

    print("\n" + "=" * 70)
    print("Step 3：ids_shuffle")
    print("ids_shuffle =", ids_shuffle)

    # ============================================================
    # 3. 保留前 keep_num 个 Token
    # ============================================================

    ids_keep = ids_shuffle[:, :keep_num]

    print("\n" + "=" * 70)
    print("Step 4：ids_keep")
    print("ids_keep =", ids_keep)

    # ============================================================
    # 4. 根据 ids_keep 取出 Token
    # ============================================================

    index = ids_keep.unsqueeze(-1).repeat(1, 1, D)

    x_masked = torch.gather(x, dim=1, index=index)

    print("\n" + "=" * 70)
    print("Step 5：x_masked")
    print("x_masked =", x_masked.squeeze(-1))

    # ============================================================
    # 5. 生成 mask
    # ============================================================

    mask = torch.ones(B, N)

    mask[:, :keep_num] = 0

    print("\n" + "=" * 70)
    print("Step 6：打乱顺序下的 mask")
    print("mask =", mask)

    # ============================================================
    # 6. 得到 ids_restore
    # ============================================================

    ids_restore = torch.argsort(ids_shuffle, dim=1)

    print("\n" + "=" * 70)
    print("Step 7：ids_restore")
    print("ids_restore =", ids_restore)

    # ============================================================
    # 7. 恢复 mask 原来的位置
    # ============================================================

    mask = torch.gather(mask, dim=1, index=ids_restore)

    print("\n" + "=" * 70)
    print("Step 8：恢复后的 mask")
    print("mask =", mask)

    return x_masked, mask, ids_restore


# =================================================================
# 测试 random_masking
# =================================================================

torch.manual_seed(0)

x = torch.tensor([[[10], [20], [30], [40], [50], [60], [70], [80]]])

x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.5)


# =================================================================
# 模拟 MAE Decoder
# =================================================================

print("\n\n")
print("#" * 70)
print("#              开始模拟 MAE Decoder")
print("#" * 70)


B, N, D = x.shape
keep_num = x_masked.shape[1]

# ============================================================
# 1. 假设 mask token
# ============================================================

mask_token = torch.tensor([[-1.0]])

mask_tokens = mask_token.repeat(B, N - keep_num, 1)

print("\n" + "=" * 70)
print("Decoder Step 1：Mask Tokens")

print("x_masked:")
print(x_masked.squeeze(-1))

print("\nmask_tokens:")
print(mask_tokens.squeeze(-1))


# ============================================================
# 2. 拼接
# ============================================================

x_decoder = torch.cat([x_masked, mask_tokens], dim=1)

print("\n" + "=" * 70)
print("Decoder Step 2：cat")

print("x_decoder:")
print(x_decoder.squeeze(-1))


# ============================================================
# 3. 使用 ids_restore 恢复原始顺序
# ============================================================

index = ids_restore.unsqueeze(-1).repeat(1, 1, D)

x_restore = torch.gather(x_decoder, dim=1, index=index)

print("\n" + "=" * 70)
print("Decoder Step 3：ids_restore")

print("ids_restore:")
print(ids_restore)

print("\nx_restore:")
print(x_restore.squeeze(-1))


# ============================================================
# 4. 和原始 x 对比
# ============================================================

print("\n" + "=" * 70)
print("最终对比")

print("原始 x:")
print(x.squeeze(-1))

print("\n恢复后的 x:")
print(x_restore.squeeze(-1))

print("\nmask:")
print(mask)
