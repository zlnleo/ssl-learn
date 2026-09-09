import torch
'''
基于Deepseek给出的代码进行学习，
window.py负责实现窗口的切分和还原
（已修复 build_attn_mask 里切片少写 slice() 的笔误，见《修改记录.md》）
'''
def window_partition(x,window_size ) -> torch.Tensor:
    '''
    x.shape=[B,H,W,C]变换成[nW,windows_size,windows_size,C]
    思想有一点类似低位编址这种内容。把一个H截成两个部分变成[H//window_size,window_size]
    这样在不变前面部分的情况下我们能够对窗口进行修改
    所以X.shape=[B,H,W,C]进来之后切出来[B,H_num_windows,window_size,W_num_windows,windows_size,C]
    换位置我们要对window进行操作,所以要permute换位置得到
        [B,H_num_windows,W_num_windows,window_size(H),,windows_size(W),C]
    因为view的原因所以要使用contiguous()使在内存中是连续的内容,然后view()改变shape即可
    view(B_num_windows,window_size,window_size,C)
    '''
    B,H,W,C=x.shape
    x = x.view(B,H//window_size,window_size,W//window_size,window_size,C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    return windows
def window_merge(windows,window_size,H,W):
    '''
    windows.shape=[B*num_windows,windows_size,windows_size,C]
    args=[windows,window_size,H,W]
    return [B,H,W,C]
    '''
    B=windows.shape[0]//((H//window_size)*(W//window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    return x

def build_attn_mask(H,W,window_size,shift_size,device):
    img_mask =torch.zeros((1,H,W,1),device=device)
    #H,W切分成三个区域（注意第三个必须是 slice(...)，不能写裸元组 (-shift_size, None)）
    h_slices =(slice(0,-window_size),slice(-window_size,-shift_size),slice(-shift_size,None))
    w_slices =(slice(0,-window_size),slice(-window_size,-shift_size),slice(-shift_size,None))

    #mask区域
    cnt =0
    for h in h_slices:
        for w in w_slices:
            img_mask[:,h,w,:]=cnt
            cnt +=1
    mask_window = window_partition(img_mask,window_size)        #形状是[B=1*nW,M,M,C=1]
    mask_window =mask_window.view(-1,window_size*window_size)   #变成[nW,M^2]
    attn_mask =mask_window.unsqueeze(1)-mask_window.unsqueeze(2)#差值[nW,M^2,M^2]
    attn_mask = attn_mask.masked_fill(attn_mask!=0,float(-100.0))#两种不同的填充方式
    attn_mask = attn_mask.masked_fill(attn_mask==0 , float(0.0))
    return attn_mask#[nW,M^2,M^2]