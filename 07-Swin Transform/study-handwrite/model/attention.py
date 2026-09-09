'''
2026-09-07:zlnleo
基于deepseek 给出的学习资料
实现swin Transformer的attention模块
（已修复：删除硬编码 sys.path，改为以 study-handwrite 为根的绝对导入，见《修改记录.md》）
'''
import torch
import torch.nn as nn

# 相对位置索引构造在 utils/position.py（对应 03 模块），以 study-handwrite 为根的绝对导入
from utils.position import build_relative_position_index
class WindowAttention(nn.Module):
    '''
    窗口内多头自注意力
    输入： x.shape=[B_,N,C]. B_表示的是batch*nW
    流程:
        qkv多头->拆多头->缩放点积->相对位置偏移->掩码 -> softmax ->V点积->输出投影
    输出:[B_,N,C]
    '''
    def __init__(self,dim,window_size,num_heads,qkv_bias=True,attn_drop=0.0,proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) 必须能被 num_heads({num_heads}) 整除"
        self.dim=dim
        self.window_size=window_size
        self.num_heads=num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        #可以学习的相对位置偏移
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)**2,num_heads))
        self.register_buffer("relative_position_index",build_relative_position_index(window_size),persistent=False)
        with torch.no_grad():
            nn.init.trunc_normal_(self.relative_position_bias_table,std=0.02)
    def forward(self,x,mask=None):
        '''
        注意力传播还是[Batch,N(sequence),C]但是因为实现的细节不一样,所以里面需要对应的变化
        首先进行了一次qkv变成我们需要的内容
        [Batch,N(sequence),3,self.num_heads,self.head_dim]
        然后转换拆分变成
        [Batch,num_heads,N,head_dim]做注意力机制得到[Batch,num_heads,N,N]
        然后加入相对偏移
        bias的shape[(2*window_size-1)**2,num_heads]，然后使用了一个高级索引变成了
        bias.shape=[M^2,num_heads],然后拆开换位置得到[num_heads,N,N]
        然后压一个batch利用广播结束

        判断是w-msa还是sw-msa
        不是就直接走msa的刘成军就行了,softmax(dim=-1)->丢弃->AV重塑Batch,N(sequence),C
        若是sw-msa传一个mask回来了.
        mask的形状[nw,M^2,M^2]
        '''
        B_,N,C=x.shape
        qkv = self.qkv(x).reshape(B_,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]#shape[B,num_heads,N,head_dim]
        attn = (q@k.transpose(-2,-1))*self.scale

        #相对位置bias
        #"每一对 token 应该去 bias table 哪一行拿数据”的地图,通过bias[index]一次性的把所有bias都取出来
        bias =self.relative_position_bias_table[self.relative_position_index.view(-1)]
        #再转成N，N，num_heads变成这样的形式和attn相加
        bias = bias.view(N,N,-1).permute(2,0,1).unsqueeze(0)
        attn = attn + bias

        #通过mask判断是不是w-msa和sw-msa
        if mask is not None:
            nW=mask.shape[0]
            attn = attn.view(B_//nW,nW,self.num_heads,N,N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1,self.num_heads,N,N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn@v).transpose(1,2).reshape(B_,N,C)
        return self.proj_drop(self.proj(out))



