# 12 · DDP 分布式训练完全讲解：多卡并行的标准方案

> 状态：**阶段二内容**（你的规划里"分布式入门（DDP）"在缓做清单，入学前不用动手）。
> 本文先建立概念 + 给你一套"改造模板"，等进组拿到多卡机器再实战。
> 你的机器单卡也能用 CPU 模拟练手（gloo 后端），见第六节。

---

## 一、为什么需要 DDP

一张卡显存放不下大 batch / 大模型时，把训练拆到多张卡上并行。
PyTorch 官方方案叫 **DDP（Distributed Data Parallel，数据并行）**：

```
        每张卡：同一份模型 + 不同的数据切片
        前向：各算各的
        反向：各算各的梯度
        然后：所有卡的梯度 all-reduce 求平均（每张卡拿到相同的平均梯度）
        更新：各卡用相同梯度更新 -> 参数永远一致
```

**关键理解**：DDP 不是"一张卡算一半模型"（那是模型并行），而是"每张卡都有完整模型，各自吃不同数据"——所以显存需求没变，变的是**吞吐**（n 张卡 ≈ n 倍速度，理论上）。

---

## 二、必懂的概念（面试必问）

| 概念 | 含义 | 类比 |
|---|---|---|
| world_size | 总共几个进程（= 几卡） | 几个人开会 |
| rank | 本进程编号 0 ~ world_size-1 | 第几号参会者 |
| local_rank | 本进程在**这台机器**上的 GPU 编号 | 用机器上第几块卡 |
| init_process_group | 初始化"通信群组"（backend=nccl/gloo） | 拉个群 |
| DistributedSampler | 把数据按 rank 均分、互不重复 | 分活儿 |
| all_reduce | 所有进程把梯度求和/平均，人人拿到同结果 | 对答案 |

**nccl vs gloo**：GPU 间通信用 nccl（快），CPU/单机模拟用 gloo。

---

## 三、把 train.py 改造成 DDP（六处改动模板）

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ① 初始化（main 开头，任何模型/数据操作之前）
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])          # torchrun 自动注入
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# ② 只有 rank 0 打印和写日志（否则 n 个进程打 n 份）
def log(msg):
    if dist.get_rank() == 0:
        print(msg)

# ③ 数据：训练集用 DistributedSampler（每个 rank 分到不重叠的 1/n 数据）
train_sampler = DistributedSampler(train_ds)
train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=train_sampler, shuffle=False)  # 有 sampler 就不能 shuffle
# 测试集一般只在 rank 0 跑，或也分片后汇总

# ④ 模型包一层 DDP（.to(device) 之后再包）
model = ViT(...).to(device)
model = DDP(model, device_ids=[local_rank])

# ⑤ 每个 epoch 开始前调 set_epoch：让每个 epoch 的数据划分随机化
train_sampler.set_epoch(epoch)

# ⑥ 保存时只 rank 0 存（否则 n 个进程写同一文件打架）
if dist.get_rank() == 0:
    torch.save(...)

# 结束
dist.destroy_process_group()
```

**启动方式（不用 python，用 torchrun）**：

```bash
torchrun --nproc_per_node=4 train.py --batch-size 32
# 自动开 4 个进程，各占一张卡；LOCAL_RANK 等环境变量自动注入
```

---

## 四、必须知道的四个"换算"

1. **总 batch = 每卡 batch × 卡数**：`--batch-size 32` + 4 卡 = 实际 batch 128。改卡数后 lr 要按 batch 比例调（线性缩放法则）；
2. **lr 缩放**：总 batch 翻倍，lr 大致翻倍（配合 warmup）；
3. **随机种子**：每个 rank 的种子 = 基础种子 + rank，否则各卡增强出相同数据；
4. **num_workers**：每卡的 DataLoader 各配各的，总进程数 = 卡数 × (num_workers+1)，别把机器进程数撑爆。

---

## 五、常见坑速查

| 症状 | 原因 |
|---|---|
| 挂在 init_process_group 不动 | 用 python 直接跑的，没走 torchrun |
| 每个 epoch 数据都一样 | 忘了 `sampler.set_epoch(epoch)` |
| 显存没省下来 | 误解了 DDP——它是吞吐并行不是显存并行 |
| 日志打印了 n 份 | 没做 rank 0 判断 |
| checkpoint 损坏 | 多个进程同时写同一文件 |
| Windows 下 nccl 报错 | 换 gloo 后端或上 Linux 服务器（实际多卡训练都在 Linux） |

---

## 六、单机练手方案（现在就能做）

```bash
# CPU 上模拟 2 个进程（gloo 后端），体验 rank/world_size 的机制
torchrun --nproc_per_node=2 train.py --dataset toy --epochs 1
```

把第六节的六处改动做进一个**副本文件**（别动原 train.py），用上面命令跑通，观察：
- 两个进程各打印自己的 rank；
- 只有 rank 0 出日志和 checkpoint。

跑通这一次，你就理解了 DDP 的全部核心概念——真正的多卡只是把 backend 换成 nccl、卡数改大而已。

---

## 七、和你的规划对表

勾掉 `00-现状盘点` 的"分布式训练（DDP）——缺口"（概念层面）。实战留给进组拿到多卡机器后，配合导师的课题做。
