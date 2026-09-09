"""checkpoint 保存/加载工具（best.pt 只存模型，last.pt 存完整训练状态用于断点续跑）。
（本文件初版为 0 字节空文件，已补齐，见《修改记录.md》）"""
import os

import torch

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(path, model, epoch, best_acc, config=None,
                    optimizer=None, scheduler=None, scaler=None):
    """保存 checkpoint。

    用法约定（与 vit/reviewlearn.py 一致）：
      best.pt  -> 只传 model（附带 epoch/best_acc/config），仅存模型权重；
      last.pt  -> 把 optimizer/scheduler/scaler 一起传入，保存完整训练状态以便 --resume。
    """
    ckpt = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "config": config,
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu"):
    """从 checkpoint 恢复模型（及可选的优化器/调度器/缩放器状态），返回原始 ckpt dict。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt
