'''
构造相对位置索引表和mask
相对位置索引表最主要使用的一个方法是先二维[M,M]按行优先转换成一维的index[]数组
'''
import torch


def build_relative_position_index(window_size):
    '''
    torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij") 会返回两个 3×3 的矩阵：
    [[0, 0, 0],
     [1, 1, 1],
     [2, 2, 2]]
     [[0, 1, 2],
     [0, 1, 2],
     [0, 1, 2]]

    M=window_size
    然后reshape之后变成[2,M^2],通过广播机制变成[2,M^2,M^2]
    把第三维变成2,换位置之后就是[:,:0]表示行,[:,:,1]表示列
    0,1平移到[0,2M-1],都是正数
    对行乘列数变成,在对最后一维sum(-1),得到rel
    rel[i][j]：窗口展平后token i相对于token j的相对位置哈希索引，固定窗口大小仅需构建一次

    [0,1,2
    3,4,5
    6,7,8]
    使用哈希公式：hash = dh * (2*M‑1) + dw
    行偏移乘上总偏移类别数，与列偏移相加，得到唯一一维索引
    rel[0][8],8这个位置坐标是[2,2],所以dw=0-2+2(这个是平移需要加的),dh=也是0
    最终得到dw*(2m-1)+dh也就是0这个唯一表示的距离
    返回rel.shape=[M²,M²]，后续可以用这个rel去索引相对位置bias参数表

    相关的内容补充，Transformer注意力机制中，相对位置偏差的公式定义就是
    Bias(i,j)=f(pos_i-pos_j)
    所以swin Transformer里面的代码里面硬性规定了i-j的实现
    用了这个方法之后我们可以在预训练的时候加载官方训练权重
    '''
    coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size),indexing="ij"))
    coords = coords.reshape(2,-1)
    rel = coords[:,:,None]-coords[:,None,:]
    rel = rel.permute(1,2,0).contiguous()
    rel[:,:,0]+=window_size-1
    rel[:,:,1]+=window_size-1
    rel[:,:,0]*= (2 * window_size - 1 )
    return rel.sum(-1)
