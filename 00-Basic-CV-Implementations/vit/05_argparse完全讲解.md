# 05 · parse_args 完全讲解：argparse 从入门到够用

> 对应代码：`train.py` 里的 `parse_args()` 函数。
> 目标：搞懂"为什么这么写、每个参数怎么用、怎么给自己以后的脚本写参数"。

---

## 一、为什么需要 argparse

**核心思想：配置与逻辑分离——改超参数不碰代码。**

| 做法 | 问题 |
|---|---|
| 改代码里的常量再重跑 | 每次改都要动代码；改回来容易漏；多人协作/多组实验一团乱 |
| argparse 命令行传参 | 同一份代码跑无数组配置；命令即实验记录；脚本间可互相调用 |

你的阶段规划铁律"每个实验记 config/seed"——argparse 就是 config 的载体：
跑实验的命令行本身就是一条完整的实验记录（配 wandb 的 `config=vars(args)` 自动留档）。

---

## 二、最小可运行示例

```python
import argparse

parser = argparse.ArgumentParser(description="训练脚本")
parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
args = parser.parse_args()
print(args.epochs)   # python demo.py            -> 100
                     # python demo.py --epochs 5 -> 5
```

三行要素：**建 parser → 加参数 → 解析**。`parse_args()` 读 `sys.argv`（命令行内容），返回一个 Namespace 对象，参数值用 `args.参数名` 取。

---

## 三、add_argument 的完整解剖

`add_argument("--名字", 类型, 默认值, 行为, 帮助)`

### 3.1 参数名的两种形态

| 写法 | 含义 | 例 |
|---|---|---|
| `"--epochs"` | 可选参数：`--epochs 5` 传值，不传用 default | `python train.py --epochs 5` |
| `"data_dir"` | 位置参数：按顺序传，无 default | 脚本里少见，`cp src dst` 就是这种 |

深度学习脚本 99% 用可选参数（带 `--`），因为参数多、还要有默认值。

### 3.2 type：自动类型转换

命令行传进来的永远是**字符串**，`type=` 负责转换：

```python
parser.add_argument("--epochs", type=int)     # "50"  -> int 50
parser.add_argument("--lr", type=float)       # "1e-3" -> float 0.001（科学计数法直接认）
parser.add_argument("--ckpt-dir", type=str)  # 默认就是 str，可省略
```

坑：`type=int` 传 `--epochs 50.0` 会报错（"50.0" 不是合法 int 字符串）——类型校验是好事，帮你挡低级错误。

### 3.3 default：缺省值

不传该参数时用的值。`default=100`、`default=1e-3`、`default="./checkpoint"`。

### 3.4 choices：白名单

```python
parser.add_argument("--dataset", choices=["cifar100", "toy", "fashionmnist"])
```

传 `--dataset cifar10` 会直接报错退出并列出合法选项——拼写保护。这是防止"训练跑了两小时才发现数据集名拼错了"的廉价保险。

### 3.5 action：参数的行为模式（重点理解）

| action | 行为 | 适用 |
|---|---|---|
| 默认（store） | 存下传进来的值 | `--epochs 5` |
| `"store_true"` | 出现即 True，不出现即 False，**后面不跟值** | `--use-wandb` |
| `"store_false"` | 出现即 False，不出现即 True | `--no-amp` 这类"否定开关" |
| `"count"` | 出现次数计数 | `-v -v -v` 详细级别 |

**最容易踩的坑**：`store_true` 的参数不能这样传——

```bash
python train.py --use-wandb True   # ❌ 报错！"True" 会被当成位置参数
python train.py --use-wandb        # ✅ 对，出现就代表 True
```

train.py 里两个开关型参数：
- `--amp` 配 `default=True`：默认开启混合精度（GPU 上）；
- `--use-wandb` 无 default：默认关闭，加 flag 才记录。

### 3.6 help：帮助文字

`python train.py --help` 会自动生成使用说明，每个参数一行。写清楚"这个参数改了什么"，未来自己（和三个月后的自己）都会感谢你。

---

## 四、train.py 的 parse_args 逐组解读

| 组 | 参数 | 关键点 |
|---|---|---|
| 数据集 | `--dataset` | choices 白名单防拼写错 |
| | `--data-dir` | default=DATA_DIR，指向本地 CIFAR-100 |
| 训练 | `--epochs/--batch-size/--lr` | 最常改的三个 |
| | `--weight-decay` | AdamW 专用，0.05 是 ViT 配方 |
| | `--grad-clip` | 梯度裁剪阈值 1.0 |
| | `--num-workers` | DataLoader 进程数，Windows 注意 |
| | `--seed` | 复现性，42 |
| 模型 | `--embed-size/--num-heads/--num-layers/--dropout` | 改模型大小和正则 |
| 工程 | `--amp` | store_true + default=True = 默认开 |
| | `--ckpt-dir` | checkpoint 输出目录（best.pt/last.pt） |
| | `--resume` | store_true 开关：断点续跑 |
| | `--log-dir` | 实验记录目录（runs/run_时间戳/） |
| | `--use-wandb` | store_true 开关 |

**返回值**：`args` 是 Namespace，`args.epochs`、`args.weight_decay` 这样取。整个 main() 里 `args.xxx` 出现几十次，全是"配置注入"。

---

## 五、实战练习

**练习 1（必做）**：把 toy 任务改成命令行可控，跑三组对比：

```bash
python train.py --dataset toy --epochs 30
python train.py --dataset toy --epochs 30 --lr 3e-3        # lr 调大十倍
python train.py --dataset toy --epochs 30 --amp             # 默认已开，这条对比意义在 CPU 上
```

**练习 2**：在 PyCharm 里配置参数（不用每次敲命令行）：
Run → Edit Configurations → 选中脚本 → Parameters 框里填 `--dataset toy --epochs 10` → Run。

**练习 3（进阶）**：给 train.py 加一个自己的参数，比如 `--warmup-epochs`，在 main() 里打印它验证取值正确——练完你就真正会写 argparse 了。

---

## 六、常见坑速查

| 症状 | 原因 |
|---|---|
| `error: unrecognized arguments: True` | 给 store_true 参数传了值 |
| `error: argument --epochs: invalid int value: '50.0'` | type 和传值不匹配 |
| 参数没生效，一直用默认值 | 命令行里参数名拼错了（有 choices 的会报错，没 choices 的**静默忽略**——这是加 choices 的另一个理由） |
| help 信息里 `%` 显示异常 | help 字符串里的 `%` 要写 `%%` |

---

## 七、什么时候升级（了解即可）

- 参数多到几十个、还有嵌套结构 → **yaml 配置**（OmegaConf / hydra）：配置写成文件、支持继承和覆盖；
- 要"一组实验网格搜索" → **hydra 的 multirun**：一行命令自动跑完所有超参组合；
- 阶段二做科研实验时再学 yaml 就来得及，现在 argparse 完全够用。

**一句话总结**：parse_args 是训练脚本的"操作面板"——面板按钮（参数）设计得好，后面调参、复现、协作都顺。
