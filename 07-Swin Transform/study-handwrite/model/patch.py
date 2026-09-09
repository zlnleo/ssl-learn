import torch.nn as nn
import torch
import torch.nn.functional as F
class PatchMerging(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.norm=nn.LayerNorm(dim*4)
        self.reduction=nn.Linear(dim*4,dim*2,bias=False)
    def forward(self,x,H,W):
        B,L,C=x.shape
        x=x.view(B,H,W,C)
        if H%2==1or W%2==1:
            x =F.pad(x,(0,0,0,W%2,0,H%2))
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C) 左上
        x1 = x[:, 1::2, 0::2, :]  # 左下
        x2 = x[:, 0::2, 1::2, :]  # 右上
        x3 = x[:, 1::2, 1::2, :]  # 右下
        x=torch.cat([x0,x1,x2,x3],dim=-1)
        x =x.view(B,-1,4*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


