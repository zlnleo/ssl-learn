import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                       smoothing: float = 0.0) -> torch.Tensor:
    """带(可选)标签平滑的交叉熵。

    logits : (B, C)  模型输出
    targets: (B,) 类别索引 long, 或 (B, C) 软标签 float (v2 上 Mixup 时才需要软标签)
    smoothing = 0 时退化为普通 CE; 论文训练用了 0.1。
    实现提示:
      1) targets 是 1D 时, 把 one-hot 换成 (1-smoothing)*one_hot + smoothing/C;
         是 2D 时, t = (1-smoothing)*targets + smoothing/C
      2) loss = -(t * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    """
    if targets.ndim==1:
        with torch.no_grad():
            #生成形如logits（B,C）的tensor,里面填充smoothing/num_classes
            t=torch.full_like(logits, smoothing/logits.size(-1))
            #然后scatter再对应的dim位置，填充成平滑后的内容
            t.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing + smoothing / logits.size(-1))
    else:
        t = (1-smoothing)*targets + smoothing/logits.size(-1)

    return -(t*F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()



def Distillation_loss(student_out, teacher_logits, targets, args):
    logits_cls,logits_dist = student_out
    base_loss=soft_cross_entropy(logits_cls, targets, args.smoothing)

    if args.distill=='hard':
        with torch.no_grad():
            # 取教师预测概率最大的类别作为硬标签
            y_t =teacher_logits.argmax(dim=1)
        # 用学生的蒸馏头去拟合教师的预测结果
        dist_loss=F.cross_entropy(logits_dist,y_t)
    elif args.distill=='soft':
        with torch.no_grad():
            # 教师概率分布：Shape (batch_size, num_clasees)，除args.tempetature得到更柔和的概率分布
            p_t=F.softmax(teacher_logits/args.tau,dim=1)
        #学生log概率分布
        p_s_log = F.log_softmax(logits_dist/args.tau,dim=1)
        # Student 模仿 Teacher 的概率分布
        dist_loss=F.kl_div(p_s_log,p_t,reduction='batchmean')*(args.tau**2)
    else:
        raise ValueError(args.distill)
    # 3. 两个监督信号加权
    total_loss=(1-args.alpha)*base_loss + args.alpha*dist_loss
    return total_loss,base_loss,dist_loss
