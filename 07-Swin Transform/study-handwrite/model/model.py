# 本文件相对初版有修复（num_layers 未定义 / BasicLayer 关键字 / forward_features 流程），
# 改动前因后果见同目录《修改记录.md》，跨文件传递机制见《学习笔记_文件夹之间如何传递.md》
import torch
from torch import nn

from model.patch import PatchMerging
from model.layer import BasicLayer


class SwinTransformer(nn.Module):
    """完整 Swin Transformer（09 总装，手写复习版）。

    数据流（以 img=224、patch_size=4、window=7 的标准配置为例）：
      patch embed: 224x224x3 -> 56x56x96（卷积一次完成切 patch + 线性投影）
      stage0: 2 个 block (56x56, C=96)  + PatchMerging -> 28x28,  C=192
      stage1: 2 个 block (28x28, C=192) + PatchMerging -> 14x14,  C=384
      stage2: 6 个 block (14x14, C=384) + PatchMerging -> 7x7,    C=768
      stage3: 2 个 block (7x7,   C=768) （最后一层不接 merging）
      norm -> 全局平均池化 -> 分类头

    注意点（都是踩过的坑）：
      - 每个 stage 的输入通道 = embed_dim * 2**i（PatchMerging 通道翻倍）；
      - num_features = embed_dim * 2**(stage数-1)，只能在最后一层之后 norm 一次；
      - drop_path 沿全部 block 线性增长（stochastic depth），再按 stage 切段分发。
    """
    def __init__(self, img_size=224, patch_size=4, in_channels=3,
                 num_classes=1000, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., patch_merging=True):
        super().__init__()
        assert len(depths) == len(num_heads), "depths 与 num_heads 长度必须一致"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = len(depths)                 # stage 数量
        self.embed_dim = embed_dim
        self.patch_merging = patch_merging
        # 最终输出通道 = embed_dim * 2**(stage 数-1)（每个非末层 merging 通道翻倍）
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # 1) patch embedding：patch_size 卷积（stride=patch_size）同时完成切块与投影
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # 2) stochastic depth：drop_path 随块索引从 0 线性增长到 drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if patch_merging:
                dim, n_head = int(embed_dim * 2 ** i), num_heads[i]
                # 每层末尾接 PatchMerging（通道翻倍、分辨率减半），最后一层不接
                downsample = PatchMerging(dim) if i < self.num_layers - 1 else None
            else:  # 消融：关掉 merging，所有 stage 保持 embed_dim 且分辨率不变
                dim, n_head = embed_dim, num_heads[0]
                downsample = None
            layer = BasicLayer(
                dim=dim, depth=depths[i], window_size=window_size, num_heads=n_head,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features if patch_merging else embed_dim)
        self.head = nn.Linear(self.num_features if patch_merging else embed_dim,
                              num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        # patch embedding 后的 token 网格尺寸：H_img/patch_size × W_img/patch_size
        H, W = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        x = self.patch_embed(x)                 # (B, embed, H, W)
        x = x.flatten(2).transpose(1, 2)        # (B, H*W, embed) 变成 token 序列
        for layer in self.layers:
            x, H, W = layer(x, H, W)            # H, W 随 PatchMerging 减半
        x = self.norm(x)                        # 所有 stage 之后统一 norm 一次
        return x.mean(dim=1)                    # 全局平均池化 -> (B, num_features)

    def forward(self, x):
        return self.head(self.forward_features(x))
