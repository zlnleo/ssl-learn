import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

from vit_mae import MAE


# ============================================================
# 1. patchify
# ============================================================

def patchify(imgs, patch_size=4):
    """
    把图片转换成原始 patch

    输入:
        [B, 3, 32, 32]

    输出:
        [B, 64, 48]

    64:
        8 x 8 = 64 个 patch

    48:
        4 x 4 x 3 = 48 个像素值
    """

    B, C, H, W = imgs.shape

    h = H // patch_size
    w = W // patch_size

    x = imgs.reshape(
        B,
        C,
        h,
        patch_size,
        w,
        patch_size
    )

    x = x.permute(
        0,
        2,
        4,
        3,
        5,
        1
    )

    x = x.reshape(
        B,
        h * w,
        patch_size * patch_size * C
    )

    return x


# ============================================================
# 2. 检测模型
# ============================================================

def check_mae():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("MAE Check")
    print("=" * 60)

    print("Device:", device)


    # ========================================================
    # 创建模型
    # ========================================================

    model = MAE()

    model = model.to(device)


    # ========================================================
    # 加载 checkpoint
    # ========================================================

    checkpoint_path = "./checkpoint/mae_last.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()


    print("Checkpoint loaded successfully.")

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )


    # ========================================================
    # Dataset
    # ========================================================

    transform = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                [0.5] * 3,
                [0.5] * 3
            )
        ]
    )


    dataset = torchvision.datasets.CIFAR100(
        root="../data",
        train=False,
        download=True,
        transform=transform
    )


    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # ========================================================
    # 统计
    # ========================================================

    total_loss = 0.0

    total_masked_loss = 0.0

    total_visible_loss = 0.0

    total_mask_ratio = 0.0

    total_samples = 0


    # ========================================================
    # 开始检测
    # ========================================================

    with torch.no_grad():

        for step, (img, _) in enumerate(loader):

            img = img.to(
                device,
                non_blocking=True
            )


            # ------------------------------------------------
            # MAE forward
            # ------------------------------------------------

            pred, mask = model(img)


            # ------------------------------------------------
            # 检查 shape
            # ------------------------------------------------

            if step == 0:

                print()
                print("-" * 60)
                print("Shape Check")
                print("-" * 60)

                print(
                    "Image:",
                    img.shape
                )

                print(
                    "Prediction:",
                    pred.shape
                )

                print(
                    "Mask:",
                    mask.shape
                )


            # ------------------------------------------------
            # target
            # ------------------------------------------------

            target = patchify(img)


            # ------------------------------------------------
            # MSE
            # ------------------------------------------------

            patch_loss = (
                pred - target
            ) ** 2


            # 每个 patch 一个 loss

            patch_loss = patch_loss.mean(
                dim=-1
            )


            # ------------------------------------------------
            # mask
            # ------------------------------------------------

            mask_bool = mask.bool()


            masked_loss = (
                patch_loss[mask_bool]
            ).mean()


            visible_loss = (
                patch_loss[~mask_bool]
            ).mean()


            # MAE真正训练的loss

            loss = (
                patch_loss * mask
            ).sum() / mask.sum()


            # ------------------------------------------------
            # mask ratio
            # ------------------------------------------------

            mask_ratio = mask.float().mean()


            # ------------------------------------------------
            # 累计
            # ------------------------------------------------

            batch_size = img.shape[0]

            total_loss += (
                loss.item() * batch_size
            )

            total_masked_loss += (
                masked_loss.item() * batch_size
            )

            total_visible_loss += (
                visible_loss.item() * batch_size
            )

            total_mask_ratio += (
                mask_ratio.item() * batch_size
            )

            total_samples += batch_size


            # ------------------------------------------------
            # 只打印第一批
            # ------------------------------------------------

            if step == 0:

                print()
                print("-" * 60)
                print("First Batch Check")
                print("-" * 60)

                print(
                    f"Mask ratio: "
                    f"{mask_ratio.item():.4f}"
                )

                print(
                    f"Masked loss: "
                    f"{masked_loss.item():.6f}"
                )

                print(
                    f"Visible loss: "
                    f"{visible_loss.item():.6f}"
                )

                print(
                    f"MAE loss: "
                    f"{loss.item():.6f}"
                )


    # ========================================================
    # 平均结果
    # ========================================================

    avg_loss = (
        total_loss / total_samples
    )

    avg_masked_loss = (
        total_masked_loss / total_samples
    )

    avg_visible_loss = (
        total_visible_loss / total_samples
    )

    avg_mask_ratio = (
        total_mask_ratio / total_samples
    )


    # ========================================================
    # 最终结果
    # ========================================================

    print()
    print("=" * 60)
    print("Final Result")
    print("=" * 60)

    print(
        f"Test MAE loss       : "
        f"{avg_loss:.6f}"
    )

    print(
        f"Masked patch MSE    : "
        f"{avg_masked_loss:.6f}"
    )

    print(
        f"Visible patch MSE   : "
        f"{avg_visible_loss:.6f}"
    )

    print(
        f"Mask ratio          : "
        f"{avg_mask_ratio:.4f}"
    )

    print("=" * 60)


    # ========================================================
    # 自动判断
    # ========================================================

    print()
    print("Automatic Check")
    print("-" * 60)


    # 1. mask

    if abs(avg_mask_ratio - 0.75) < 0.02:

        print(
            "✓ Mask ratio correct"
        )

    else:

        print(
            "✗ Mask ratio abnormal"
        )


    # 2. shape

    if pred.shape[1:] == torch.Size([64, 48]):

        print(
            "✓ Prediction shape correct"
        )

    else:

        print(
            "✗ Prediction shape abnormal"
        )


    # 3. loss

    if avg_loss < 0.1:

        print(
            "✓ Reconstruction loss looks reasonable"
        )

    else:

        print(
            "! Reconstruction loss is relatively high"
        )


    # 4. masked vs visible

    if avg_masked_loss < avg_visible_loss:

        print(
            "✓ Masked reconstruction is "
            "better than visible reconstruction"
        )

    else:

        print(
            "! Masked reconstruction error is "
            "not lower than visible reconstruction"
        )


    print()
    print("=" * 60)
    print("Check finished.")
    print("=" * 60)


# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    check_mae()