"""Phase 1: Toy MAE 完整 forward 数据流冒烟测试。

运行:
    python test_mae_flow.py

覆盖数据流 (对照《00_MAE模块化学习路线与接口契约.md》§1.4):
    images -> patchify target -> patch embedding -> random masking -> encoder
    -> decoder embed -> append mask token -> gather(ids_restore) -> decoder
    -> pred -> masked MSE loss

每个中间张量都会打印 shape 与代表值, 打印完做一组硬断言。
"""
import torch

from mae import TOY_CONFIG, ToyMAE, patchify, unpatchify


def section(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def show(name, t, n_values=6):
    """打印张量的 shape / dtype, 低维直接全打印, 高维打印前几个值。"""
    if not torch.is_tensor(t):
        print(f"{name:16s} = {t}")
        return
    print(f"{name:16s} shape={tuple(t.shape)}  dtype={t.dtype}")
    if t.dim() == 0:
        print(f"{'':16s} value = {t.item():.6f}")
    elif t.dim() <= 2:
        print(f"{'':16s} full = {t.detach().cpu()}")
    else:
        flat = t.detach().cpu().reshape(-1)[:n_values]
        print(f"{'':16s} first {n_values} values = {flat.tolist()}")


def main():
    torch.manual_seed(0)  # 固定种子, 保证每次输出可复现
    print("TOY_CONFIG =", TOY_CONFIG)

    # ---------------- 0) images ----------------
    section("0) images: 输入图像 x = torch.randn(2, 3, 32, 32)")
    x = torch.randn(2, 3, 32, 32)
    show("images", x)
    print(f"   value range: [{x.min():.3f}, {x.max():.3f}]")

    # ---------------- 1) patchify target ----------------
    section("1) patchify target: 原始图像 -> patch 序列 (无参数, 行主序)")
    P = TOY_CONFIG["patch_size"]
    target = patchify(x, patch_size=P)
    show("patchify target", target)
    print(f"   patch 0 前 12 个像素 = {target[0, 0, :12].tolist()}")

    # ---------------- 2~7) 一次完整 forward ----------------
    model = ToyMAE(**TOY_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {n_params / 1e6:.2f} M")
    out = model(x)

    section("2) patch embedding: Conv2d(8x8, stride=8) 线性投影 (B,N,D)")
    show("patch_embed", out["patch_embed"])
    show("tokens_full", out["tokens_full"])
    print("   tokens_full = patch_embed + pos_embed (位置编码加在 shuffle 之前, 官方顺序)")

    section("3) random masking: 切出可见 token + 三件索引工具")
    show("x_vis", out["x_vis"])
    show("mask", out["mask"])
    show("ids_keep", out["ids_keep"])
    show("ids_restore", out["ids_restore"])
    show("ids_shuffle", out["ids_shuffle"])
    V = out["ids_keep"].shape[1]
    M = int(out["mask"].sum(dim=1)[0].item())
    print(f"   V(可见) = {V},  M(被遮) = {M}   (N = V + M = 16)")

    section("4) encoder latent: 4xBlock 只作用于 V 个可见 token")
    show("latent", out["latent"])

    section("5) decoder: Linear(D->Dd) -> append mask token -> gather(ids_restore)")
    show("decoder_embed", out["decoder_embed"])
    show("restored", out["restored"])
    print("   restored = gather(cat([decoder_embed, mask_token]), ids_restore) + decoder_pos_embed")

    section("6) pred: Linear(Dd->T) 重建出的每个 patch 的像素")
    show("pred", out["pred"])

    section("7) masked MSE loss: 只惩罚被遮的 12 个位置")
    show("target_norm", out["target_norm"])
    show("target_mean", out["target_mean"], n_values=4)
    show("target_var", out["target_var"], n_values=4)
    show("loss", out["loss"])
    print("   归一化后 target 方差 ~ 1 -> 随机初始化模型初始 loss 理论值 ~ 1.0")

    # ================= 硬断言 =================
    section("ASSERT 1: 形状契约")
    assert x.shape == (2, 3, 32, 32)
    assert target.shape == (2, 16, 192)
    assert out["patch_embed"].shape == (2, 16, 192)
    assert out["x_vis"].shape == (2, 4, 192)
    assert out["mask"].shape == (2, 16) and out["mask"].dtype == torch.bool
    assert out["ids_keep"].shape == (2, 4)
    assert out["ids_restore"].shape == (2, 16)
    assert out["ids_shuffle"].shape == (2, 16)
    assert out["latent"].shape == (2, 4, 192)
    assert out["restored"].shape == (2, 16, 128)
    assert out["pred"].shape == (2, 16, 192)
    assert out["mask"].sum().item() == 2 * 12
    print("[OK] 所有形状符合契约")

    section("ASSERT 2: ids_restore 是 ids_shuffle 的逆排列")
    identity = out["ids_shuffle"].gather(1, out["ids_restore"])
    assert torch.equal(identity, torch.arange(16).expand(2, -1)), \
        "ids_restore 不是 ids_shuffle 的逆排列!"
    print("[OK] ids_shuffle.gather(1, ids_restore) == arange(16)")

    section("ASSERT 3: gather 还原机制 (decoder 的数学基础)")
    m = out["mask"]  # (2, 16) bool
    # 被遮位置: restored 必须恰好等于 mask_token + 该位置的 decoder_pos_embed
    expected_masked = (model.decoder.mask_token + model.decoder.decoder_pos_embed)
    expected_masked = expected_masked.expand(2, 16, 128)[m]
    assert torch.allclose(out["restored"][m], expected_masked, atol=1e-5), \
        "被遮位置的 restored 不等于 mask_token + pos_embed!"
    print("[OK] 被遮位置: restored == mask_token + decoder_pos_embed (共享占位 token 生效)")
    # 可见位置: restored[b,i] == decoder_embed[b, ids_restore[b,i]] + pos_embed[i]
    for b in range(2):
        for i in range(16):
            if not m[b, i]:
                r = out["ids_restore"][b, i].item()  # 可见位置必有 r < V
                expected = out["decoder_embed"][b, r] + model.decoder.decoder_pos_embed[0, i]
                assert torch.allclose(out["restored"][b, i], expected, atol=1e-5)
    print("[OK] 可见位置: restored == decoder_embed[ids_restore] + pos_embed (真实编码归位)")

    section("ASSERT 4: 初始 loss 数值")
    loss_val = out["loss"].item()
    assert 0.3 < loss_val < 2.0, f"初始 loss={loss_val:.3f} 不在理论区间 (0.3, 2.0)!"
    print(f"[OK] 初始 loss = {loss_val:.4f}  (归一化 target 下理论值 ~ 1.0)")

    section("ASSERT 5: 反向传播与梯度")
    out["loss"].backward()
    mt_grad = model.decoder.mask_token.grad
    assert mt_grad is not None and mt_grad.abs().sum() > 0
    print(f"[OK] mask_token.grad 非零, 梯度穿过了 gather 到达被遮位置 (grad.abs().sum = {mt_grad.abs().sum().item():.4f})")
    assert model.encoder.patch_embed.proj.weight.grad is not None
    print("[OK] encoder.patch_embed.proj.weight.grad 非零 (梯度穿过了 mask 选择)")
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print("[OK] 所有梯度有限, 无 NaN")

    section("ASSERT 6: patchify / unpatchify 自检")
    rt = unpatchify(patchify(x, 8), 8, 32, 32)
    assert torch.allclose(rt, x, atol=1e-6), "patchify <-> unpatchify 往返失败!"
    print("[OK] unpatchify(patchify(x)) 往返误差 < 1e-6")
    ramp = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    patched = patchify(ramp, 8)
    # patch 0 必须是左上角 8x8 块, 通道优先展平:
    # 值 = ch*1024 + row*32 + col, 其中 row, col 在 0..7 (图宽 32, 块内每行跳 32)
    expected_p0 = torch.tensor([ch * 1024 + r * 32 + c
                                for ch in range(3) for r in range(8) for c in range(8)]).float()
    assert torch.equal(patched[0, 0], expected_p0), "patch 0 的顺序不对!"
    assert torch.equal(patched[0, 15], ramp[0, :, 24:, 24:].reshape(-1)), "patch 15 的顺序不对!"
    print("[OK] ramp 图顺序检查: patch 0 = 左上角块, patch 15 = 右下角块 (行主序约定正确)")

    print()
    print("=" * 78)
    print("  全部通过: Toy MAE 完整 forward 数据流正确")
    print("=" * 78)


if __name__ == "__main__":
    main()
