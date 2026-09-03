"""消融实验包：4 个核心实验 + 公共测量工具。

实验与学习模块的对应关系：
  实验 1（Global vs Window）   -> 模块 01 的 mechanism 部分 + exp1 端到端
  实验 2（Window vs Shifted）  -> 模块 07 的 mechanism 部分 + exp2 端到端
  实验 3（有无 Patch Merging） -> 模块 06/08 + exp3 端到端
  实验 4（window_size 消融）   -> exp4（贯穿 01/03/05 的机制）
"""
