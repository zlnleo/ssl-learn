# RandAugment 学习手册（v2 完整版）

> 覆盖：核心机制（N vs M）、magnitude 详解、`__call__` 为什么、Python 语法小课堂、
> 13 个操作详解、分步实现、**M 消融实验**、以及 Gemini 提到的两个"全自动"方案
> （torchvision / timm 内置 RandAugment）。
> 学习顺序（GPT 的建议，我完全同意）：**先手写跑通 → 做 M 消融 → 和 Mixup/CutMix 对比
> → 最后再看内置实现做交叉验证**。不要倒过来。

---

## 0. 学习路线（先读这段）

```
RandAugment
   ├── N: 做几种增强 (数量)
   ├── M: 增强多狠 (强度, 统一标尺 0~30)
   ├── magnitude → 每个操作映射成自己的物理量
   ├── 13 个操作 (像素/色彩/几何 三大类)
   │        ↓
   │   自己实现 + 实验 M 消融
   │        ↓
   │   和 Mixup / CutMix 对比
   │        ↓
   └── 最后看 torchvision / timm 内置实现 (此时不再觉得是黑盒)
```

---

## 1. 核心机制：最核心的代码只有这几行

```python
class RandAugment:
    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m
        self.ops = [...]          # 操作池

    def __call__(self, img):
        for op in np.random.choice(self.ops, self.n, replace=False):
            mag = ...             # 由 m 生成实际幅度
            img = op(img, mag)
        return img
```

它表达的是：

```
一张图片 → 随机选 N 个增强 → 每个增强用幅度 M → 依次执行 → 增强后的图
```

**两个参数一定要分清**：
- **N 控制"做多少种"**——每张图随机抽几个不同的操作（论文 n=2）；
- **M 控制"增强有多狠"**——所有操作共享的强度标尺（论文 m=9）。

## 2. magnitude 详解（最值得理解的部分）

### 2.1 为什么实际幅度不是固定的 9？

```python
mag = float(np.clip(np.random.uniform(self.m - self.mstd, self.m + self.mstd), 0, 30))
```

假设 m=9、mstd=0.5：实际幅度是 **[8.5, 9.5] 之间的随机数**（第 1 张图 8.73、第 2 张 9.21…）。
所以 **m 是中心强度，mstd 是强度的随机扰动**——防止所有图都"一样狠"，保持增强多样性。
（GPT 建议第一版直接用 `n=2, m=9`，mstd/inc 当论文配方的扩展项，理解即可。）

### 2.2 为什么所有操作统一用 0~30？（RandAugment 最漂亮的设计）

不同操作的物理量完全不同：Rotate 是角度、Translate 是像素、Brightness 是强度、
Posterize 是 bit 数、Solarize 是阈值。如果各自设参数（Rotate=17°、Brightness=0.63…），
"强度"就没有统一意义了。RandAugment 的做法：

```
              M ∈ [0,30]  (统一强度标尺)
                │
     ┌──────────┴──────────┐
     ↓                     ↓
  Rotate                Contrast
     ↓                     ↓
  角度映射             系数映射
  ±30°×(M/30)        1±0.9×(M/30)
```

**每个操作自己把 M 映射到自己的物理量**——M=30 时所有操作都到极限，M=9 时都只用极限的
30%。这个"统一标尺 + 各自映射"的思想一定要记住。

## 3. 专题：为什么要用 `__call__`？

```python
ra = RandAugmentCIFAR(n=2, m=9)
out = ra(img)      # ← 实例"像函数一样"被调用, 全靠 __call__
```

- `__call__` 是 Python 的**魔术方法**：实现了它，实例对象就可以用 `ra(img)` 的语法被调用，
  内部实际执行的是 `ra.__call__(img)`；
- **为什么增强类需要它**：torchvision 的 `T.Compose([...])` 对流水线里的每个 transform
  都是执行 `t(img)`——它不关心 t 是函数还是类实例，只要求"可调用"。实现了 `__call__`，
  你的类就自动获得塞进 Compose 的资格；
- **同一个套路你其实早就见过**：`nn.Module` 内部就是 `__call__` 包了一层 `forward`，
  所以 `model(x)` 也能这么写。`model.train()`、hook 机制都挂在 `__call__` 上；
- 一句话：**`__call__` = 让对象变成函数**。凡是"会被当函数用的对象"（transform、
  criterion、scheduler），Python 生态都用这个约定。

## 4. 语法小课堂（代码里每个语法点）

| 语法 | 含义 | 在本代码里 |
|---|---|---|
| `np.random.choice(list, n, replace=False)` | 无放回随机抽 n 个 | 抽 n 个不同操作；`replace=False` 防止 Rotate+Rotate |
| `np.random.uniform(a, b)` | [a,b) 均匀分布 | 在 [m-mstd, m+mstd] 里取随机幅度 |
| `np.clip(x, 0, 30)` | 把 x 夹进 [0,30] | 幅度抖动后防越界 |
| `float(...)` | 转成 Python float | 防与 float32 张量运算时 dtype 升级成 float64 |
| `2 * random.random() - 1` | [-1,1] 均匀分布 = **随机方向** | 旋转左/右、亮度增/减；只单向就是系统性偏差 |
| `torch.where(cond, a, b)` | 逐元素三元选择 | Solarize：高于阈值取反 |
| 形参名 `_` | 约定：**这个参数我不使用** | autocontrast/equalize 不需要 mag，但签名要统一 |
| `@staticmethod` | 方法不依赖实例（无 self） | 无幅度的两个操作；调用方式仍是 `self.autocontrast` |
| `self.ops = [self.rotate, ...]` | 存**方法本身**（绑定方法=可调用对象），不是调用结果 | 后面统一 `op(img, mag)` 调用 |
| `dim=(1,2), keepdim=True` | 沿 H,W 两维求值，保留维度 | 求每通道 min/max/mean 后还能广播 |
| `unsqueeze(0)/squeeze(0)` | 加/删一个 batch 维 | avg_pool2d 要求输入有 batch 维 |
| `v >> k << k` | 位运算：低位清零 | Posterize 量化（低 k 位归零） |
| `clamp_(0,1)` | **原地**夹回（带下划线=in-place） | 最终防像素越界 |
| `TF.affine(img, angle, translate, scale, shear, fill)` | 通用仿射变换 | shear/translate 操作；**shear 必传**（版本坑） |

## 5. 13 个操作详解（三大类）

| 类别 | 操作 | mag 控制什么 | 教模型什么 |
|---|---|---|---|
| 像素/直方图 | AutoContrast | **不使用** | 光照不变性 |
| 像素/直方图 | Equalize | **不使用** | 曝光不变性 |
| 像素/直方图 | Posterize | bit 数 (8→4) | 忽略精细色彩 |
| 像素/直方图 | Solarize | 反转阈值 (1.0→0.5) | 过曝不变性 |
| 色彩/视觉 | Contrast | 对比度系数 (0.1~1.9) | 对比度不变性 |
| 色彩/视觉 | Color | 饱和度系数 | 色彩强度不变性 |
| 色彩/视觉 | Brightness | 亮度偏移 | 亮度不变性 |
| 色彩/视觉 | Sharpness | 锐化/模糊系数 | 细节不变性 |
| 几何 | Rotate | 角度 (±30°) | 旋转不变性 |
| 几何 | ShearX / ShearY | 错切量 (±0.3) | 仿射形变不变性 |
| 几何 | TranslateX / TranslateY | 位移 (±45% 边长) | 位置不变性（Transformer 最缺） |

> 每个操作"做什么/公式推导/完整代码"见 `deittrain_v2.py` 里带【AI 注释】的
> `RandAugmentCIFAR` 类——那份代码每个操作都有逐行中文注释，本手册不再重复贴代码，
> 两份配合着看：**手册讲思想，代码注释讲细节**。

## 6. 分步实现（第一版：只用 n=2, m=9）

1. **类**：照 `deittrain_v2.py` 的注释版实现（你已完成 ✓，20 次随机增强单测已通过）；
2. **接入 transform**（ToTensor 之后、Normalize 之前——因为操作都假设像素 ∈ [0,1]）：
   ```python
   train_transforms = T.Compose([
       T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
       T.ToTensor(),
       RandAugmentCIFAR(n=2, m=9),       # ★ 这里
       T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
   ])
   ```
3. **（可选）args 开关**：`--ra-n 2 --ra-m 9`，`0` 表示关闭（消融能力）；
4. **（可选，论文 inc=1）幅度递增**：主循环里 `for t in train_set.transform.transforms: if isinstance(t, RandAugmentCIFAR): t.m += 1`——第一版可先不接；
5. **单测 + 肉眼检查**（教程前版第 4 节有命令；`ra_preview.png` 一定看一眼）。

## 7. M 消融实验（GPT 强烈建议，比背代码有价值得多）

固定 n=2、其它一切不变，只改 m：

| 实验 | 配置 | test_acc (best) |
|---|---|---|
| M=0 | `RandAugmentCIFAR(n=2, m=0)` ≈ 无增强 | ? |
| M=5 | 轻度 | ? |
| M=9 | 论文默认 | ? |
| M=15 | 中度 | ? |
| M=20 | 重度 | ? |

观察：**增强强度从弱到强，test_acc 如何变化**——大概率是一条"先升后平/后降"的曲线：
太弱等于没增强（过拟合），太强把语义都破坏了（欠拟合）。这个实验让你真正理解
"m 是正则强度旋钮"，比单纯调库有价值得多。把结果画进 TensorBoard（你已有
`writer.add_scalar('test/acc', ...)`），横轴 m、纵轴 best acc。

## 8. 两个"全自动"方案（Gemini 提的，放最后用）

等你自己的版本跑通、M 消融做完，再用内置实现做一次**交叉验证**（数值细节不同是正常的，
本教学版的每个映射系数是教学近似，与官方库不逐位一致——GPT 也提醒过这点）。

### 8.1 torchvision 内置（你的环境已有，0.26 支持）

```python
import torchvision.transforms as T
train_transforms = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.RandAugment(num_ops=2, magnitude=9),      # ← 一行替代手写类
    T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])
```

- `num_ops=2` 就是你的 n；`magnitude=9` 就是你的 m（内部同样映射到 0~30 标尺，
  `num_magnitude_bins=31` 控制分档）；
- 位置同样必须放在 Normalize 之前（它内部按 [0,1] 图像语义操作）；
- 注意它没有"幅度逐 epoch 递增"，也没有 mstd 抖动——**先做自己的版本，你才能看懂
  它少了什么**。

### 8.2 timm 的 `rand_augment_transform`（完全对齐论文字符串）

```bash
pip install timm        # 你的环境还没装; 装不上就跳过, torchvision 版已够用
```

```python
from timm.data.auto_augment import rand_augment_transform
ra = rand_augment_transform('rand-m9-mstd0.5-inc1', hparams={'translate_const': 117})
# 返回一个可直接塞进 Compose 的 callable (支持 PIL 图)
```

- 配置字符串 `'rand-m9-mstd0.5-inc1'` **就是论文里的写法**——m=9、mstd=0.5、inc=1 一个
  字符串全表达，这正是你手写版 `__init__(n=2, m=9, mstd=0.5, inc=1)` 的由来；
- 到这里你应该能一眼认出它的内部结构：operation pool → sampling → magnitude mapping
  → image transform——和你的类是同一张设计图。

### 8.3 手写 vs 内置（什么时候用哪个）

| | 手写版（你的） | torchvision / timm |
|---|---|---|
| 学习价值 | **极高**（每个操作、每个映射都懂） | 低（黑盒一行） |
| 论文对齐 | 教学近似 | timm 逐位对齐论文字符串 |
| 实验灵活性 | 可改任意操作/映射/加 inc | 有限 |
| 建议 | 学习期 + M 消融用 | 最终交叉验证用 |

## 9. 常见坑（速查）

| 坑 | 说明 |
|---|---|
| 放在 Normalize 之后 | 色彩/亮度操作数值语义全乱 |
| 忘 `clamp_(0,1)` | 多步叠加越界 |
| 忘 `np.clip(mag, 0, 30)` | mstd 抖动越界 |
| `TF.affine` 不传 shear | 部分 torchvision 版本报错（本仓库踩过） |
| 操作只做单向 | 缺 `2*random()-1` = 系统性偏差 |
| `np.random.choice` 忘 `replace=False` | 同操作重复抽中 |
| 直接照抄官方数值细节 | 教学版是近似，交叉验证时用内置版对齐 |

---

**下一步行动**：类已注释好且验证通过 → 按第 6 节接入 `build_loaders`（加 `--ra-n/--ra-m`
开关）→ 2 epoch 冒烟 → 跑 Experiment 4（n=2, m=9）填消融表 → 跑第 7 节的 M 消融 →
最后用第 8 节的内置版交叉验证。每步完成叫我检查。
