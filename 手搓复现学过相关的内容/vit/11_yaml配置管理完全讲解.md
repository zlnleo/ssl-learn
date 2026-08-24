# 11 · yaml 配置管理完全讲解：从 argparse 到 hydra

> 前情：`05_argparse完全讲解.md` 讲了"参数少"时的方案。
> 本文讲参数多到管不过来时的升级方案：yaml 配置文件 + hydra。
> **什么时候升级**：参数超过 20 个、出现嵌套结构、要跑网格搜索、组内要共享配置——你的阶段二科研实验大概率会遇到。

---

## 一、为什么 argparse 会不够用

| argparse 的痛点 | yaml/hydra 的解法 |
|---|---|
| 参数多了命令行又臭又长 | 配置写成文件，一目了然 |
| 超参和代码混在一起 | 配置与代码彻底分离 |
| 网格搜索要写 shell 循环 | hydra multirun 一行搞定 |
| 实验配置难分享/难复现 | config 文件本身就是实验记录 |

**注意**：argparse 没白学——hydra 底层就是 argparse 的扩展，而且很多公司项目仍用 argparse，两者都要会。

---

## 二、yaml 基础（10 分钟学会）

yaml 是"人类可读的配置文件格式"，规则极简：

```yaml
# configs/vit_cifar100.yaml
dataset: cifar100
epochs: 100
lr: 1e-3              # 科学计数法直接写
batch_size: 128
weight_decay: 0.05
use_wandb: false      # 布尔：true/false

model:                # 嵌套：用缩进表示层级（和 Python 一样，但不能用 Tab！）
  embed_size: 192
  num_heads: 6
  num_layers: 6
  dropout: 0.1

augmentations:        # 列表：用 - 开头
  - random_crop
  - horizontal_flip
```

**读出来**：

```python
import yaml

with open("configs/vit_cifar100.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)          # 注意是 safe_load，不要用 load()（有安全风险）

print(cfg["model"]["embed_size"])    # 192 —— 嵌套 dict 直接取
print(cfg["lr"])                     # 0.001 —— 自动转成 float
```

**三个必记的坑**：缩进只能用空格；`1e-3` 不加引号才是数字（加引号变字符串）；`None`/`null` 表示空值。

---

## 三、OmegaConf：让 yaml 更好用（推荐）

```bash
pip install omegaconf
```

```python
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/vit_cifar100.yaml")
print(cfg.model.embed_size)          # 点号访问，比 dict["key"] 顺眼
cfg.epochs = 200                      # 可修改
print(OmegaConf.to_yaml(cfg))        # 打印整份配置（调试用）
```

优势：点号访问、类型校验、和 hydra 无缝集成。

---

## 四、hydra：配置管理的事实标准（重点）

```bash
pip install hydra-core
```

**改造 train.py 只需要两步**：

```python
# train.py 顶部
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="vit_cifar100")
def main(cfg: DictConfig):
    print(cfg.epochs)        # 参数直接从 cfg 取，不再用 args.xxx
    # ...训练代码里把 args.xxx 全部换成 cfg.xxx...

if __name__ == "__main__":
    main()
```

**三个杀手级用法**：

```bash
# 1. 跑默认配置
python train.py

# 2. 命令行覆盖任意配置项（不用改文件！）
python train.py lr=3e-4 batch_size=256 model.num_layers=4

# 3. 网格搜索：multirun 一次跑完所有组合，每个组合自动存到独立目录
python train.py -m lr=1e-3,3e-3 weight_decay=0.01,0.05
# 生成 4 次运行，结果各存 outputs/2026-08-20/xx-xx-xx/ 下（含完整配置快照）
```

**hydra 自动给你的**：每次运行自动建独立输出目录、自动存配置快照、`-m` 一键网格搜索——这就是"配置管理 + 实验记录"的工业级组合。

---

## 五、什么时候用哪套（决策表）

| 场景 | 方案 |
|---|---|
| 学习项目、参数 < 20 个 | argparse（你现在的状态） |
| 组内项目、参数多、要分享配置 | yaml + OmegaConf |
| 科研实验、要网格搜索 | hydra（强烈建议阶段二就切） |

**迁移路线**：先把手头 train.py 的 17 个参数写进一份 yaml，跑通 → 再装 hydra 改造 main → 最后体验一次 `-m` 网格搜索。三步做完，这条技能线就通了。

---

## 六、练习

1. 把 train.py 的全部参数抄成 `configs/vit_cifar100.yaml`，用 `yaml.safe_load` 读出来打印；
2. 用 OmegaConf 加载并改一个值再打印；
3. 装 hydra，把 main 改成 `@hydra.main` 版本，跑 `python train.py lr=3e-3` 验证覆盖生效；
4. toy 任务上跑一次 `-m lr=1e-3,3e-3`，去 outputs/ 目录看两次运行的配置快照。
