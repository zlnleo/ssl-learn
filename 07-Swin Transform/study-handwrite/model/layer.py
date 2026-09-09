# 已修复：导入从裸导入改为以 study-handwrite 为根的绝对导入（见《修改记录.md》）
import torch
import torch.nn as nn

from model.block import SwinBlock

class BasicLayer(nn.Module):
    def __init__(self,dim,depth,window_size,num_heads,mlp_ratio=4.0,qkv_bias =True,
                 drop=0.0,attn_drop=0.0,drop_path=0.0,
                 downsample=None):
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size

        #块构造
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim,num_heads=num_heads,window_size=window_size,
                      shift_size=0 if i%2==0 else window_size//2,
                      mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,
                      drop=drop,attn_drop=attn_drop,
                      drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path)
            for i in range(depth)
        ])
        self.downsample =downsample
    def forward(self, x,H,W):
        for block in self.blocks:
            x = block(x,H,W)
        if self.downsample is not None:
           x= self.downsample(x,H,W)
           H,W = (H+1)//2,(W+1)//2
        return x,H,W
