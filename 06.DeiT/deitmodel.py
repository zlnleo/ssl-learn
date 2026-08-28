import torch.nn as nn
import torch

class PatchEmbedding(nn.Module):
    def __init__(self,img_size=32,patch_size=4,in_channels=3,embed_dim=192):
        super().__init__()
        self.img_size=img_size
        self.patch_size=patch_size
        self.num_patches=(self.img_size//self.patch_size)**2
        self.proj=nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
        

    def forward(self, x):
        #x.shape=[B,C,H,W]
        #proj(x).shape=[B,embed_dim,H/patch_size,W/patch_size]
        #x.flatten(2).transpose(1,2).shape=[B,num_patches,embed_dim]
        x=self.proj(x).flatten(2).transpose(1,2)
        return x

class Attention(nn.Module):
    def __init__(self,dim,num_head=3,qkv_bias=True,attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.num_head=num_head
        self.scale=(dim//num_head)**-0.5#对应除的那个根号下d_k,就是每个头分到的维度开根号
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self,x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        #q,k,v.shape=[B,num_head,N,head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1))*self.scale
        #attn.shape=[B,num_head,N,N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)

class MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim,hidden_dim)
        self.act = nn.GELU()
        self.drop1=nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim,in_dim)
        self.drop2 = nn.Dropout(dropout)
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class DropPath(nn.Module):
    def __init__(self,drop_prob=0.):
        super().__init__()
        self.drop_prob=drop_prob
    def forward(self, x):
        if self.drop_prob ==0.0 or not self.training:
            return x
        keep =1 - self.drop_prob
        mask =x.new_empty(x.shape[0],1,1).bernoulli_(keep).div(keep)
        return x * mask

class Block(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio=4.,qkv_bias=True,dropout=0.0,attn_drop=0.,drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn=Attention(dim,num_heads,qkv_bias,attn_drop,dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim,int(mlp_ratio*dim),dropout=dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path >0. else nn.Identity()
    def forward(self,x):
        x=x+self.drop_path1(self.attn(self.norm1(x)))
        x=x+self.drop_path2(self.mlp(self.norm2(x)))
        return x

class DistilledVit(nn.Module):
    def __init__(self,img_size=32,patch_size=4,in_channels=3,num_classes=100,
                 embed_dim=192,depth=12,num_heads=3,mlp_ratio=4.,
                 qkv_bias=True,dropout=0.,attn_drop=0.,drop_path=0.,
                 distilled=True):
        super().__init__()
        self.distilled=distilled
        self.num_classes=num_classes
        self.patch_embed=PatchEmbedding(img_size=img_size,patch_size=patch_size,in_channels=in_channels,embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim,num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,dropout=dropout,attn_drop=attn_drop,drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        #双分类头
        self.head = nn.Linear(embed_dim,self.num_classes)
        self.head_dist = nn.Linear(embed_dim,self.num_classes) if distilled else None

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self,x):
        x =self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        if self.distilled:
            dist = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, dist,x), dim=1)
        else:
            x =torch.cat((cls,x),dim=1)
        x= x + self.pos_embed[:,:x.shape[1]]
        x = self.pos_drop(x)
        for block in self.blocks:  # 逐个调用
            x = block(x)
        if self.distilled:
            return x[:,0],x[:,1]
        else:
            return x[:,0],None

    def forward(self,x):
        x_cls ,x_dist= self.forward_features(x)
        logits_cls = self.head(x_cls)
        logits_dist = self.head_dist(x_dist) if self.distilled else None
        if self.training:
            return (logits_cls,logits_dist) if self.distilled else logits_cls
        if self.distilled:
            return (logits_cls+logits_dist)/2
        return logits_cls


