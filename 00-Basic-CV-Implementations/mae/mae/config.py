"""Toy MAE 固定配置 (phase 1 专用, 不用 YAML)。"""

TOY_CONFIG = dict(
    image_size=32,  # 输入图像 32x32
    patch_size=8,  # patch 8x8
    in_chans=3,  # RGB
    # → num_patches = (32/8)^2 = 16
    # → V = int(16 * 0.25) = 4 个可见, M = 12 个被遮, T = 8*8*3 = 192
    encoder_embed_dim=192,
    encoder_depth=4,
    encoder_num_heads=3,  # 192 / 3 = 64 每头
    decoder_embed_dim=128,
    decoder_depth=2,
    decoder_num_heads=4,  # 128 / 4 = 32 每头
    mask_ratio=0.75,
)
