# 15 · 工程工具补全：Git 分支与 Docker 用法

> 对应你规划 `00-现状盘点` 的两个缺口："Git 分支/rebase/PR 流程"和"Dockerfile 编写"。
> 这两项是**大厂实习日常要求**，阶段二用 1-2 周集中补。本文是"够用版"速成指南。

---

# 上篇：Git 分支与 PR 流程

## 一、为什么要有分支

主分支（main）永远保持"能跑的稳定版本"；任何新功能/实验都在**分支**上做，做完合并回去。好处：坏了大不了扔掉分支，主线不受影响；多人协作互不干扰。

```bash
git checkout -b feature/lr-schedule   # ① 开一个分支并切过去
# ...改代码...
git add train.py                      # ② 把改动加入暂存区
git commit -m "feat: add cosine lr schedule"   # ③ 提交（写好信息，别写"改了点东西"）
git push origin feature/lr-schedule   # ④ 推到远程
```

## 二、PR（Pull Request）流程：大厂协作标准姿势

```
fork 官方仓库 → 在自己仓库开分支改代码 → 发起 PR → 同事 review 提意见
→ 修改后继续 push → 通过后 merge 进主仓库
```

**commit message 规范**（实习第一天就会遇到）：`feat:` 新功能、`fix:` 修 bug、`docs:` 文档、`refactor:` 重构——用英文前缀，这是通用惯例。

## 三、必会的几个命令

```bash
git status                        # 看当前状态（改动/未提交）
git diff                          # 看具体改了什么
git log --oneline                 # 看提交历史
git checkout main && git merge feature/xxx   # 合并分支
git pull                          # 拉最新代码（先 pull 再 push！）
git reset --hard HEAD~1           # 撤销上一次提交（慎用）
```

**冲突怎么处理**：两边改了同一行 → git 会在文件里标出 `<<<<<<<` 和 `>>>>>>>`，手动选择保留哪边，删掉标记，再 add + commit。第一次遇到会慌，处理过一次就熟了。

## 四、.gitignore：什么不该进仓库

```gitignore
# 你的项目必备
__pycache__/
*.pyc
.idea/
checkpoint/
runs/
data/
*.pt            # 权重文件不进 git（太大）
```

**权重文件怎么办**：用 wandb Artifact / 网盘 / git-lfs 存，不要直接 commit 进仓库（几十 MB 的 .pt 会让仓库膨胀）。

---

# 下篇：Docker

## 一、为什么需要 Docker

"在我电脑上能跑，在服务器上跑不了"——Docker 把**代码 + 依赖 + 系统库**打包成镜像，任何机器拉下来环境完全一致。实验室服务器、公司训练集群都用它交付环境。

## 二、Dockerfile 基本语法（够用版）

```dockerfile
# 基础镜像：带 CUDA 的 PyTorch 官方镜像（版本号去 hub.docker.com 查对应组合）
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace                      # 容器内工作目录

COPY requirements.txt .                 # 先拷依赖文件（分层缓存：依赖没变就不用重装）
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY vit/ vit/                          # 再拷代码
CMD ["python", "vit/train.py"]          # 容器启动时执行的命令
```

## 三、构建与运行

```bash
docker build -t vit-cifar100 .                    # 构建镜像（打个标签）
docker run --gpus all -v D:\project\...\data:/workspace/data vit-cifar100 \
    python vit/train.py --epochs 5                # 跑起来
# --gpus all：把 GPU 透传给容器；-v 主机目录:容器目录：挂载数据盘（数据不进镜像！）
```

**三条铁律**：① 数据用 `-v` 挂载，绝不打进镜像；② 镜像要小，用 runtime 版基础镜像而非 devel 版；③ Dockerfile 写进仓库，同事才能复现你的环境。

---

## 四、练习（阶段二再做，现在可先建文件）

1. 给 `vit/` 项目写 `.gitignore`（现在就能做，5 分钟）；
2. 把项目推到 GitHub，体验一次"开分支 → 改 README → PR → merge"（现在就能做）；
3. 写 `requirements.txt` 和 `Dockerfile`，本地 `docker build` 跑通一次 toy 训练（需要装 Docker Desktop，阶段二再折腾）。

---

## 五、和你的规划对表

这两项勾掉 `00-现状盘点` 的"Git 分支/PR"和"Dockerfile"缺口。加上 08/09 的 wandb/tensorboard，工程工具三件套（Git、Docker、实验管理）就齐了——正是大厂实习简历"工程能力"栏的标准配置。
