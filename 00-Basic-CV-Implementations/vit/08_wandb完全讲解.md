# 08 · wandb 完全讲解：实验记录平台从安装到接进 train.py

> 对应内容：`train.py` 里埋好的 `--use-wandb` 钩子（目前只做了 init，本文教你补全整个用法）。
> wandb = "weights & biases"，目前最流行的机器学习实验记录平台（网页看板 + 云端留档）。
> 你的阶段规划铁律第一条："**每个实验进 wandb**，每轮实验记 config/seed/结论"。

---

## 一、wandb 是什么，为什么用它

**一句话**：训练时把指标（loss/acc/lr）、超参数、图表实时上传到网页看板，随时回看、对比任意两次实验。

| | 本地 print / runs 日志（train.py 已有） | wandb |
|---|---|---|
| 看曲线 | 自己画或肉眼看数字 | 自动实时曲线 |
| 超参数留档 | config.txt 手动打开 | 网页表格 + 自动 diff |
| 多次实验对比 | 翻文件 | 网页勾选叠加 |
| 换电脑/换服务器 | 文件要拷 | 登录账号就有 |
| 离线环境 | ✅ 永远可用 | 支持 offline 模式（见第六节） |

**结论**：runs/ 本地日志是"底线备份"，wandb 是"体验升级"。两者并行不冲突。

---

## 二、安装、登录、跑通第一个实验

```bash
pip install wandb
wandb login          # 输入 API Key
```

1. 打开 https://wandb.ai/authorize 注册并复制 API Key（免费）；
2. `wandb login` 粘贴 Key，之后就不用再登录了（Key 存在本地 `~/.netrc`）；
3. 快速验证：

```python
import wandb
wandb.init(project="hello")          # 创建一个名为 hello 的项目里的 run
wandb.log({"loss": 0.5, "acc": 0.8}) # 记一个点
wandb.finish()                       # 结束并同步
```

跑完打开 wandb.ai 网页，就能看到 `hello` 项目里多了一次 run 和一条曲线。

**国内网络慢的应对**：wandb 同步走外网，慢/不稳时用 `wandb offline`（第六节）。

---

## 三、三个核心概念：run / project / config / log

```python
import wandb

wandb.init(
    project="vit-cifar100",      # 项目名：一类实验放一个项目（相当于文件夹）
    name="run_xxx",              # 本次 run 的名字（不写会自动起名）
    config=vars(args),           # 超参数：自动上传成网页表格
)
wandb.log({"test_acc": 0.6}, step=epoch)   # 记一个点，step 是横轴坐标
wandb.finish()
```

| 概念 | 含义 | 类比 |
|---|---|---|
| project | 一组相关实验 | 文件夹 |
| run | 一次完整训练 | 文件夹里的一个文件 |
| config | 本次实验的超参数 | 实验记录卡 |
| log(key=value, step=n) | 记录一个指标点 | 往坐标纸上打点 |

要点：
- `config=vars(args)`：把 argparse 的 17 个参数一次性上传——这就是"命令即实验记录"的落地；
- `log` 里的 key 随便起名，相同 key 自动画进同一张图；
- `step` 不传也行（自动递增），但**传 epoch 数**能让多个 run 的曲线按 epoch 对齐比较。

---

## 四、常用功能清单（按使用频率排序）

### 4.1 记录指标（最高频，train.py 就用它）

```python
wandb.log({"train_loss": tl, "train_acc": ta,
           "test_loss": vl, "test_acc": va,
           "lr": scheduler.get_last_lr()[0],   # 把学习率曲线也记上！
           "amp_scale": scaler.get_scale()},   # 以及 AMP 放大系数
          step=epoch)
```

`lr` 和 `amp_scale` 强烈建议一起记——06 文档里让你"打印曲线"的实验，用 wandb 就变成了网页上三条并排的曲线。

### 4.2 记录图片（看数据增强效果、看注意力可视化）

```python
wandb.log({"aug_samples": [wandb.Image(img) for img in imgs[:8]]}, step=epoch)
```

### 4.3 监控模型（梯度/参数直方图）

```python
wandb.watch(model, log="gradients", log_freq=100)   # init 之后调用一次
```

网页端能看到每层参数的梯度分布——调试"梯度消失/爆炸"时非常好用。

### 4.4 记录表格（混淆矩阵 / 预测样例）

```python
table = wandb.Table(columns=["image", "pred", "true"])
table.add_data(wandb.Image(img), pred_label, true_label)
wandb.log({"predictions": table}, step=epoch)
```

### 4.5 Artifact：版本化地存数据集/模型（进阶）

```python
artifact = wandb.Artifact("vit-cifar100-best", type="model")
artifact.add_file("checkpoint/best.pt")
wandb.log_artifact(artifact)          # 网页端可以直接下载这个 checkpoint
```

### 4.6 Sweep：超参数自动搜索（阶段二再学）

```python
sweep_config = {"method": "bayes", "metric": {"name": "test_acc", "goal": "maximize"},
                "parameters": {"lr": {"min": 1e-4, "max": 1e-2},
                               "weight_decay": {"values": [0.01, 0.05, 0.1]}}}
sweep_id = wandb.sweep(sweep_config, project="vit-cifar100")
wandb.agent(sweep_id, function=train)   # train 函数里照常 wandb.init/log
```

---

## 五、把 wandb 完整接进 train.py（动手改一次）

现在 `train.py` 里只有 init 没有 log。补全三步（对应 `use_wandb` 三处）：

**① init 处（约第 355 行）**——给 run 起个和本地日志一致的名字：

```python
if args.use_wandb:
    try:
        import wandb
        wandb.init(project="vit-cifar100", config=vars(args),
                   name=os.path.basename(run_dir))   # 和 runs/run_xxx 一一对应
    except ImportError:
        log("[提示] 未安装 wandb（pip install wandb），本次不记录实验")
```

**② 训练循环里（epoch 指标打印处）**——屏幕打印的同时上传：

```python
        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss, "train_acc": train_acc,
                "test_loss": test_loss, "test_acc": test_acc,
                "lr": scheduler.get_last_lr()[0],
                "amp_scale": scaler.get_scale(),
            }, step=epoch)
```

**③ 训练结束（log_file.close() 之前）**——优雅收尾：

```python
    if args.use_wandb:
        wandb.finish()   # 不 finish 的话进程结束时也会自动同步，但显式调用更稳
```

改完后跑 `python train.py --dataset toy --epochs 5 --use-wandb`，打开网页就能看到 6 条曲线。**注意**：不要每个 batch 都 log（会拖慢训练），攒到每个 epoch log 一次是标准频率。

---

## 六、离线模式（你的网络环境可能用得上）

同步走外网，网差时两种方案：

```bash
wandb offline     # 本次运行不联网：数据先存本地 wandb/ 目录
# ... 训练完，网络好了以后 ...
wandb sync wandb/offline-run-xxx    # 手动把离线数据推到云端
```

- `wandb offline` 后照常 init/log，一切本地进行；
- 恢复在线：`wandb online`；
- 本地缓存目录 `wandb/` **别手滑删掉**，删了离线数据就没了。

（上一轮下载 CIFAR-100 只有 ~100KB/s 的网络，先用 offline 训练、晚上再 sync 是合理工作流。）

---

## 七、常见坑速查

| 症状 | 原因 | 解法 |
|---|---|---|
| `wandb: ERROR Error communicating with wandb` | 没登录或没网 | `wandb login` 或 `wandb offline` |
| 曲线只有最后一个点 | 每个 epoch 都覆盖了同一个 step | log 时传 `step=epoch` |
| 训练变慢明显 | 每个 batch 都 log | 攒到 epoch 级再 log |
| 网页上找不到这次 run | 项目名写错 / 没 finish | 检查 project 名；结束调 finish() |
| 多个 run 曲线对不齐 | step 没统一 | 统一用 epoch 作 step |
| 报编码错误 | project/name 用了中文 | wandb 的 run 名保持英文/数字 |

---

## 八、练习（按顺序做）

1. 跑通第二节的 hello 实验，在网页上找到自己的曲线；
2. 按第五节把 `train.py` 补全三处，用 toy 任务跑 5 轮，网页确认 6 条曲线 + 超参数表格；
3. 跑两次不同 lr（`--lr 1e-3` 和 `--lr 3e-3`），在网页上勾选两个 run 叠加对比 loss 曲线——**这一步做完你就体验到 wandb 的核心价值了**；
4. 试一次 `wandb offline` + `wandb sync`，掌握慢网工作流；
5. 进阶：给 toy 任务加 4.4 的混淆矩阵表格。

---

## 九、和你的规划对表

这一步勾掉 `00-现状盘点` 里"实验管理（wandb/tensorboard）——缺口"这一项。做完练习 1-3，你的训练闭环就完整了：

```
argparse 配参数 → train.py 训练 → runs/ 本地日志 + wandb 云端看板 → checkpoint 存档 → --resume 续跑
```

至此阶段一的"手写回顾收尾 + 工程工具"骨架基本齐了，剩下的就是损失函数/优化器两个必做项和导师邮件 + LeetCode。
