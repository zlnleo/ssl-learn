# -*- coding: utf-8 -*-
"""
run_m_ablation.py —— RandAugment M 消融一键脚本

依次运行 5 组实验 (M = 0 / 5 / 9 / 15 / 20), 每组:
    - 关闭 Mixup/CutMix (只考察 RandAugment 单独的作用)
    - --ra-inc 0 (固定强度, 不做逐轮递增)
    - 独立 checkpoint 目录 (互相不覆盖 best.pth)
    - 跑完自动从 runs/ 最新日志抓 best test acc
最后打印汇总表并写入 m_ablation_results.txt。

用法 (先 conda activate ssl_cv, 在本项目目录下):
    python run_m_ablation.py                 # 全部 5 组, 预计 3.5~5 小时
    python run_m_ablation.py --only 0,9      # 只跑指定 M 值
    python run_m_ablation.py --dry-run       # 只打印将执行的命令, 不真的跑

说明:
    - M=0 这一组会自动加 --ra-n 0: 因为 AutoContrast/Equalize 不使用幅度,
      若 m=0 仍会改变图像, 所以用 ra-n 0 得到"真正的无增强"左端点;
    - 教师已缓存在 runs/teacher/, 不会重复训练; 若不存在则第一组自动训练;
    - 结果表可直接填进 RandAugment教程.md 第 7 节。
"""
import argparse
import os
import re
import subprocess
import time

# 切到脚本所在目录, 保证相对路径 (runs/, deittrain_v2.py) 无论从哪启动都对
os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE = ["python", "deittrain_v2.py",
        "--mixup", "0", "--cutmix", "0", "--ra-inc", "0"]


def latest_run_dir(after: float):
    """返回 after 时间之后新建的最新的 runs/run_* 目录 (None 表示没找到)。"""
    best, best_t = None, 0.0
    for d in os.listdir("runs"):
        p = os.path.join("runs", d)
        if not os.path.isdir(p) or not d.startswith("run_"):
            continue
        mt = os.path.getmtime(p)
        if mt > after and mt > best_t:
            best, best_t = p, mt
    return best


def parse_best_acc(run_dir: str) -> float:
    """从 train.log 里抓 'best test acc: 0.xxxx'。"""
    with open(os.path.join(run_dir, "train.log"), encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"best test acc:\s*([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return float("nan")


def main():
    ap = argparse.ArgumentParser(description="RandAugment M 消融一键脚本")
    ap.add_argument("--only", default="", help="逗号分隔的 M 值子集, 如 0,9")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = ap.parse_args()

    m_list = [0, 5, 9, 15, 20]
    if args.only:
        m_list = [int(x) for x in args.only.split(",") if x.strip()]

    print(f"共 {len(m_list)} 组: M = {m_list}  (每组约 45~60 分钟, 全部约 3.5~5 小时)")
    results = []

    for i, m in enumerate(m_list, 1):
        cmd = BASE + ["--ra-m", str(m)]
        if m == 0:
            cmd += ["--ra-n", "0"]                       # M=0 用 ra-n 0 得到真正的无增强
        cmd += ["--ckpt-dir", f"./checkpoint/m_ablation_m{m}"]   # 每组独立 checkpoint, 互不覆盖

        print("\n" + "=" * 70)
        print(f"[{i}/{len(m_list)}] M={m}  命令: {' '.join(cmd)}")
        if args.dry_run:
            continue

        t0 = time.time()
        rc = subprocess.call(cmd)                        # 输出直接透传到终端, 不捕获
        if rc != 0:
            print(f"[M={m}] 运行失败 (退出码 {rc}), 跳过结果收集")
            continue

        run_dir = latest_run_dir(t0)
        acc = parse_best_acc(run_dir) if run_dir else float("nan")
        results.append((m, acc))
        print(f"[M={m}] 完成, 用时 {(time.time() - t0) / 60:.1f} 分钟, "
              f"best test acc = {acc:.4f}  (log: {run_dir})")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("M 消融汇总表:")
    print(f"{'M':>4}  {'best test acc':>14}")
    for m, acc in results:
        print(f"{m:>4}  {acc:>14.4f}")
    if results:
        with open("m_ablation_results.txt", "w", encoding="utf-8") as f:
            f.write("M\tbest_test_acc\n")
            for m, acc in results:
                f.write(f"{m}\t{acc:.4f}\n")
        print("\n已写入 m_ablation_results.txt, 可把数字填进 RandAugment教程.md 第 7 节")


if __name__ == "__main__":
    main()
