"""一键运行全部单元测试（模块 01-09 + 工程包）。

在项目根目录运行（ssl_cv 环境）：
  python run_all_tests.py

说明：测试文件彼此独立、自包含（各模块测试自带 sys.path 引导），
本脚本用 runpy 在进程内逐一执行并收集结果，无需 pytest。
"""
import os
import runpy
import sys
import time

# Windows 控制台默认 GBK，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))

TEST_FILES = [
    os.path.join("01_window_attention", "test_window_attention.py"),
    os.path.join("02_window_partition", "test_window_partition.py"),
    os.path.join("03_relative_position_bias", "test_relative_position_bias.py"),
    os.path.join("04_shifted_window", "test_shifted_window.py"),
    os.path.join("05_attention_mask", "test_attention_mask.py"),
    os.path.join("06_patch_merging", "test_patch_merging.py"),
    os.path.join("07_swin_block", "test_swin_block.py"),
    os.path.join("08_basic_layer", "test_basic_layer.py"),
    os.path.join("09_swin_transformer", "test_swin_transformer.py"),
    os.path.join("tests", "test_swin_package.py"),
]


def run_one(path: str):
    old_argv = sys.argv
    sys.argv = [path]
    # 模块测试用 `from <模块> import ...`，需要其所在目录在 sys.path（runpy 不会自动加）
    module_dir = os.path.dirname(path)
    added = module_dir not in sys.path
    if added:
        sys.path.insert(0, module_dir)
    try:
        runpy.run_path(path, run_name="__main__")
        return 0, ""
    except SystemExit as e:  # unittest.main() 结束时抛出 SystemExit
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        return code, ""
    except Exception as e:  # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"
    finally:
        if added:
            sys.path.remove(module_dir)


def main():
    print(f"Python: {sys.version.split()[0]} | torch: {__import__('torch').__version__}")
    print(f"共 {len(TEST_FILES)} 个测试文件\n")
    failed = []
    t0 = time.time()
    for rel in TEST_FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"[缺失] {rel}")
            failed.append(rel)
            continue
        code, err = run_one(path)
        status = "PASS" if code == 0 else "FAIL"
        print(f"[{status}] {rel}")
        if code != 0:
            failed.append(rel)
            if err:
                print(f"       {err}")
    print(f"\n{'=' * 60}")
    print(f"通过 {len(TEST_FILES) - len(failed)}/{len(TEST_FILES)}，用时 {time.time() - t0:.1f}s")
    if failed:
        print("失败：")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print("全部通过 ✓")


if __name__ == "__main__":
    main()
