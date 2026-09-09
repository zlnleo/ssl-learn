# vit.py 问题与修改建议

本次没有修改原来的 `vit.py`，修正版单独放在 `vit_.py`。

## 主要问题

1. `ViT` 类名没有对上测试脚本

   `test_vit.py` 和 `train_vit.py` 使用的是 `from vit import ViT`，但手写文件里定义的是 `class Vit`。这会导致导入失败，然后脚本退回去使用 `vit_solution.py`，也就是说测试结果可能不是你的实现。

   修改建议：主类命名为 `ViT`，如果想兼容旧写法，可以再加 `Vit = ViT`。

2. Attention 里有拼写错误

   原代码第 14 行是 `k.traspose(-2, -1)`，正确写法应为 `k.transpose(-2, -1)`。这个会在第一次前向传播时直接报错。

3. Attention 缩放因子写法不够稳

   原代码使用 `torch.sqrt(torch.tensor(q.size(-1)))`。如果模型在 GPU 上跑，这个临时 tensor 默认在 CPU，可能造成设备不一致；而且这里本来只是一个 Python 数值。

   修改建议：使用 `math.sqrt(q.size(-1))`。

4. `PatchEmbedding` 没有接收 `embed_size`

   在 `ViT.__init__` 里创建 patch embedding 时，原代码没有传入 `embed_size`：

   ```python
   self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels)
   ```

   这样 `PatchEmbedding` 会一直使用默认的 `embed_size=128`。当测试脚本构造 `embed_size=64` 的小模型时，patch token 是 128 维，但 `cls_token` 和 `pos_embed` 是 64 维，拼接或相加时会形状不匹配。

   修改建议：创建时传入 `embed_size=embed_size`。

5. MLP 隐藏层维度是浮点数

   原代码里 `embed_size * mlp_ratio` 的结果是 `512.0` 这种浮点数，但 `nn.Linear` 的维度需要整数。

   修改建议：写成 `int(embed_size * mlp_ratio)`。

6. 最后输出的是每个 token 的分类结果

   原代码最后直接：

   ```python
   x = self.norm(x)
   x = self.head(x)
   return x
   ```

   这会输出 `(batch, num_patches + 1, num_classes)`。图像分类任务需要整张图一个结果，测试脚本期望 `(batch, num_classes)`。

   修改建议：先取 `[CLS]` token，再送入分类头：

   ```python
   x = self.norm(x)
   x = x[:, 0]
   return self.head(x)
   ```

7. 缺少 `DROPOUT` 常量

   `train_vit.py` 会导入 `DROPOUT`。如果只导入手写文件，它需要存在这个常量。

8. 属性名和测试脚本不一致

   测试脚本检查的是 `model.patch_embed.proj.weight.grad`。手写版本用的是 `patch_embedding.patch_embedding`，逻辑没错，但验收脚本对不上。

   修改建议：主模型里用 `self.patch_embed`，PatchEmbedding 里用 `self.proj`。

## 可以添加的改进

1. 给 `embed_size % num_heads == 0` 加断言，避免切多头时静悄悄出错。

2. 给 `img_size % patch_size == 0` 加断言，避免图片不能整齐切 patch。

3. 对 `cls_token`、`pos_embed`、`Linear`、`Conv2d` 使用 `trunc_normal_(std=0.02)` 初始化，更接近 ViT 常见写法，训练会更稳。

4. MLP 的第二个线性层后也可以加 dropout，这是 ViT 里常见配置。

5. 可以支持 `pool="cls"` 和 `pool="mean"` 两种分类聚合方式，便于对比 `[CLS]` token 和平均池化的效果。

## 修正版文件

`vit_.py` 里已经包含以上修正，并额外保留了几个兼容点：

- 主类是 `ViT`
- 兼容别名 `Vit = ViT`
- 主模型属性是 `patch_embed`
- 兼容访问 `model.patch_embedding`
- PatchEmbedding 主投影层是 `proj`
- 兼容访问 `patch_embedding.patch_embedding`
