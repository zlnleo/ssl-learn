import os

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms

from vit_mae import MAE

image_DIR = "./saveimg"

os.makedirs(image_DIR, exist_ok=True)
# ============================================================
# patch -> image
# ============================================================


def unpatchify(patches, patch_size=4):
    """
    输入:

    patches:

    [B,64,48]


    输出:

    image:

    [B,3,32,32]

    """

    B, N, D = patches.shape

    C = 3

    h = w = int(N**0.5)

    x = patches.reshape(B, h, w, patch_size, patch_size, C)

    # B,h,w,p,p,C

    x = x.permute(0, 5, 1, 3, 2, 4)

    # B,C,h,p,w,p

    img = x.reshape(B, C, h * patch_size, w * patch_size)

    return img


# ============================================================
# 根据mask生成遮挡图片
# ============================================================


def mask_image(img, mask, patch_size=4):
    """
    img:

    [B,3,32,32]


    mask:

    [B,64]


    返回:

    mask后的图片

    """

    patches = patchify(img, patch_size)

    mask = mask.unsqueeze(-1)

    patches = patches * (1 - mask)

    return unpatchify(patches, patch_size)


# ============================================================
# image -> patch
# ============================================================


def patchify(imgs, patch_size=4):
    """

    image:

    [B,3,32,32]


    output:

    [B,64,48]

    """

    B, C, H, W = imgs.shape

    h = H // patch_size

    w = W // patch_size

    x = imgs.reshape(B, C, h, patch_size, w, patch_size)

    x = x.permute(0, 2, 4, 3, 5, 1)

    x = x.reshape(B, h * w, patch_size * patch_size * C)

    return x


# ============================================================
# 显示图片
# ============================================================


def show_image(img, title):
    """
    输入:

    [3,32,32]

    """

    img = img.permute(1, 2, 0)

    # 反Normalize

    img = img * 0.5 + 0.5

    img = img.clamp(0, 1)

    plt.imshow(img)

    plt.title(title)

    plt.axis("off")


# ============================================================
# main
# ============================================================


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    # -------------------------
    # 加载模型
    # -------------------------

    model = MAE()

    checkpoint = torch.load("./checkpoint/mae_last.pth", map_location=device)

    model.load_state_dict(checkpoint["model"])

    model.to(device)

    model.eval()

    # -------------------------
    # dataset
    # -------------------------

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )

    dataset = torchvision.datasets.CIFAR100(
        root="../data", train=True, download=True, transform=transform
    )

    img, label = dataset[0]

    img = img.unsqueeze(0).to(device)

    # -------------------------
    # MAE forward
    # -------------------------

    with torch.no_grad():

        pred, mask = model(img)

    # pred:

    # [1,64,48]

    reconstruction = unpatchify(pred.cpu())

    masked = mask_image(img.cpu(), mask.cpu())

    original = img.cpu()

    # -------------------------
    # visualization
    # -------------------------

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)

    show_image(original[0], "Original")

    plt.subplot(1, 3, 2)

    show_image(masked[0], "Masked 75%")

    plt.subplot(1, 3, 3)

    show_image(reconstruction[0], "MAE Reconstruction")

    plt.savefig("./saveimg/mae_reconstruction.png", dpi=300, bbox_inches="tight")

    plt.close()
