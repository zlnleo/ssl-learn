# 已修复：导入从裸导入改为以 study-handwrite 为根的绝对导入（见《修改记录.md》）
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import WindowAttention
from utils.utils import DropPath
from model.window import window_partition, window_merge, build_attn_mask
class SwinBlock(nn.Module):
    def __init__(self,dim,num_heads,window_size,
                 shift_size,mlp_ratio=4.,
                 qkv_bias=True,drop=0.,attn_drop=0.,drop_path=0.):
        super().__init__()
        assert 0 <= shift_size < window_size ,  "shift_size 必须满足 0 <= shift_size < window_size"
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim,window_size,num_heads,qkv_bias,attn_drop,proj_drop=drop)
        self.drop_path =DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim,int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio),dim),
            nn.Dropout(drop)
        )
        #参照写的
        # mask 惰性缓存：mask 只依赖 (H, W, shift, window_size) 与设备，
        # 与输入内容、batch 无关，同一几何配置只需构造一次
        self._mask_cache = None
        self._mask_key = None
    def _get_mask(self,H,W,device):
        if self.shift_size == 0:
            return None
        key = (H,W,str(device))
        if key!=self._mask_key:
            self._mask_cache =build_attn_mask(H,W,self.window_size,self.shift_size,device)
            self._mask_key = key
        return self._mask_cache

    def forward(self, x,H,W):
        B,L,C =x.shape
        #保留最原始的x
        shortcut =x
        #因为是swin Transformer所以对应的应该是，需要窗口的样子
        x = self.norm1(x).view(B,H,W,C)

        #1冗余设计pad
        pad_r = (self.window_size-W%self.window_size)%self.window_size
        pad_b = (self.window_size-H%self.window_size)%self.window_size
        x = F.pad(x,(0,0,0,pad_r,0,pad_b))
        Hp,Wp =H+pad_b,W+pad_r

        #如果是sw-msa
        if self.shift_size > 0:
            x = torch.roll(x,shifts=(-self.shift_size,-self.shift_size),dims=(1,2))

        #分窗口做注意力
        x = window_partition(x,self.window_size)
        x = x.view(-1,self.window_size**2,C)
        x = self.attn(x,mask=self._get_mask(Hp,Wp,x.device))
        x = x.view(-1,self.window_size,self.window_size,C)
        #还原窗口之后再移回去
        x = window_merge(x,self.window_size,Hp,Wp)
        if self.shift_size > 0:
            x=torch.roll(x,shifts=(self.shift_size,self.shift_size),dims=(1,2))
        x = x[:,:H,:W,:].contiguous().view(B,L,C)
        #attn注意力residual
        x =shortcut+ self.drop_path(x)
        #mlp部分
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

