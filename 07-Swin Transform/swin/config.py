"""Swin-T/S/B/L 标准配置（Swin v1, ImageNet-1K, img 224, patch 4, window 7）。"""

SWIN_CONFIGS = {
    "tiny":  {"embed_dim": 96,  "depths": (2, 2, 6, 2),  "num_heads": (3, 6, 12, 24)},
    "small": {"embed_dim": 96,  "depths": (2, 2, 18, 2), "num_heads": (3, 6, 12, 24)},
    "base":  {"embed_dim": 128, "depths": (2, 2, 18, 2), "num_heads": (4, 8, 16, 32)},
    "large": {"embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48)},
}

# 训练超参数速查（ImageNet-1K 惯例，工程脚本默认值与之对齐）
DEFAULT_TRAIN_CFG = {
    "epochs": 300,
    "batch_size": 128,
    "lr": 1e-3,          # Swin-T 官方约 1e-3 / batch 1024，小 batch 时需按比例缩小
    "min_lr": 1e-6,
    "weight_decay": 0.05,
    "warmup_epochs": 20,
    "drop_path_rate": 0.1,
}
