# tests —— 工程包集成测试

> 定位：对 `swin/` 工程包做**集成与数值校验**（与 09 模块的单元测试互补，重点在包接口与工程开关）。
> 当前目录仅含 `test_swin_package.py` 一个测试文件。

## 测试内容说明

`test_swin_package.py` 基于 `unittest`，先把上级目录加入 `sys.path` 后 `from swin import ...`，
共 4 个测试类、15 个用例：

| 测试类 | 覆盖内容 |
|--------|---------|
| `TestSwinTiny` | Swin-Tiny 前向输出 shape（224 输入 → `(2, 1000)`）、参数量区间（27.5M~29.5M）、特征模式（`num_classes=0` 输出 `(1, 768)`）、各 stage 分辨率/通道序列、eval 确定性、梯度回传 |
| `TestAblationSwitches` | 工程开关：`patch_merging=False`（分辨率全程不变）、`window_size=4/14` 覆盖、非正方形输入（128×256） |
| `TestFactoryAndConfigs` | `build_swin` 四个名称与 `SWIN_CONFIGS` 的一致性、未知名称抛 `ValueError`、`depths`/`num_heads` 长度不匹配抛 `AssertionError` |
| `TestWindowUtils` | `window_partition`/`window_reverse` 往返无损、相对位置索引取值范围、注意力掩码取值（0 / -100）与对角线为 0 |

## 如何运行

在 `07-Swin Transform/` 目录下（conda 环境 `ssl_cv`，Python 3.10 + torch）：

```bash
# 方式一：一键运行全部测试（模块 01-09 + 本工程包，共 10 个测试文件）
python run_all_tests.py

# 方式二：单独运行本集成测试
python tests/test_swin_package.py
```

> 说明：`run_all_tests.py` 用 `runpy` 在进程内逐一执行各测试文件并汇总 PASS/FAIL，无需 pytest；
> 本测试文件自带 `sys.path` 引导（把上级目录插入路径），因此也可单独运行。

---

返回上级：[07-Swin Transform 目录结构 / 快速开始](../README.md)
