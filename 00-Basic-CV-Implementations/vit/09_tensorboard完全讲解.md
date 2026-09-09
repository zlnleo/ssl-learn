# 09 · TensorBoard 完全讲解：本地可视化看板

> 对应关系：和 `08_wandb完全讲解.md` 是同一类工具，wandb 是"云端看板"，
> TensorBoard 是"本地看板"——不需要注册账号、不需要联网，写完就能看。
> 两者的关系：wandb 适合远程/长期记录，TensorBoard 适合本地即时调试，都学。

---

## 一、TensorBoard 是什么

TensorFlow 出品的可视化工具，PyTorch 内置了写入接口（`torch.utils.tensorboard`）。
训练时往"事件文件"里写指标，然后一条命令起本地网页看板：

```bash
tensorboard --logdir runs
```

| | TensorBoard | wandb |
|---|---|---|
| 联网 | 不需要 | 需要（或 offline 模式） |
| 账号 | 不需要 | 需要注册 |
| 曲线/图像/直方图/计算图 | ✅ 全支持 | ✅ 全支持 |
| 多实验对比 | 本地目录组织 | 网页勾选更顺手 |
| 远程服务器上看 | 要端口转发 | 登录即看 |

---

## 二、安装与三步跑通

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/demo")              # ① 建 writer，指定事件文件目录
for i in range(100):
    writer.add_scalar("loss", 0.5 - i * 0.004, i)   # ② 写点：(标签, 值, 横轴坐标)
writer.close()                                   # ③ 关闭
```

```bash
tensorboard --logdir runs      # 启动看板
# 浏览器打开 http://localhost:6006
```

---

## 三、常用写入接口（全部在 SummaryWriter 上）

| 接口 | 作用 | 用法 |
|---|---|---|
| `add_scalar(tag, value, step)` | 一条曲线（loss/acc/lr） | 最常用 |
| `add_scalars(main, dict, step)` | 一张图多条曲线（train/test 对比） | `add_scalars("loss", {"train": tl, "test": vl}, epoch)` |
| `add_image(tag, img, step)` | 看数据/增强效果 | img 是 (C,H,W) 张量或 PIL |
| `add_images(tag, imgs, step)` | 一批图拼成网格 | 数据检查 |
| `add_histogram(tag, values, step)` | 权重/梯度分布 | 调试梯度消失/爆炸 |
| `add_hparams(hparam_dict, metric_dict)` | 超参数对比视图 | 训练结束后调一次 |
| `add_graph(model, input)` | 可视化计算图 | 需要一份样例输入 |

---

## 四、接进 train.py（和 runs/ 目录融合）

你的 `train.py` 已经给每次运行建了 `runs/run_时间戳/`（config.txt + train.log）。
TensorBoard 的事件文件也写进同一次 run 目录，本地日志就"三件套"齐全了：

```python
from torch.utils.tensorboard import SummaryWriter

# main() 里，建 run_dir 之后：
writer = SummaryWriter(os.path.join(run_dir, "tfboard"))   # ① 子目录，互不干扰

# 训练循环里，每个 epoch 末尾（和 print 指标同一处）：
writer.add_scalar("train/loss", train_loss, epoch)
writer.add_scalar("train/acc", train_acc, epoch)
writer.add_scalar("test/loss", test_loss, epoch)
writer.add_scalar("test/acc", test_acc, epoch)
writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)   # lr 曲线也记

# 训练结束：
writer.add_hparams(vars(args), {"best_acc": best_acc})   # 超参数对比视图
writer.close()
```

`tag` 里带 `/`（如 `train/loss`）会自动分组，看板左侧变成 train/test 两个分组，清爽。

**启动看板**（看所有历史 run）：

```bash
tensorboard --logdir runs
```

每个 run 是一次独立的曲线集合，可以在看板里勾选任意几次 run 叠加对比——效果和 wandb 的对比视图一样，只是全部本地完成。

---

## 五、常见坑速查

| 症状 | 原因 | 解法 |
|---|---|---|
| 打开 6006 端口是空的 | logdir 指错了目录 | 指到事件文件的**父目录**（runs，不是 run_xxx） |
| 曲线乱成一团 | 每次运行都写进同一个目录 | 用时间戳子目录隔离（train.py 已做） |
| 训练变慢 | 每个 batch 都 add_scalar | 攒到 epoch 级记录 |
| 中文路径报错 | TensorBoard 对非 ASCII 路径兼容差 | logdir 路径保持英文 |
| 服务器上看不了 | 6006 端口没转发 | ssh -L 6006:localhost:6006 user@server |

---

## 六、练习

1. 按第四节把 writer 接进 `train.py`，toy 任务跑 10 轮，打开看板看 5 条曲线；
2. 跑两次不同 lr，在同一个 `--logdir runs` 下勾选两次 run 对比 loss 曲线；
3. 用 `add_images` 把 CIFAR-100 训练集的前 16 张图（含增强效果）打上看板，直观理解 RandomCrop/Flip 在干什么；
4. 用 `add_hparams` 跑 3 组不同 lr，打开超参数对比视图，按 test_acc 排序。

---

## 七、和你的规划对表

TensorBoard + wandb 一起勾掉 `00-现状盘点` 的"实验管理——缺口"。二者选型口诀：
**本地快速调试用 TensorBoard，长期记录/团队协作用 wandb**；两个都接上也完全不冲突（各写各的）。
