"""像素级基线对比：把 ui-verify/current/ 下的截图与 ui-verify/baseline/ 同名文件逐像素对比。

用法:
  # 1) 建立/更新基线
  .venv/bin/python ui-verify/diff.py --save-baseline
  # 2) 对比（current/ 下放本次截图，文件名与基线一致）
  .venv/bin/python ui-verify/diff.py

阈值: 默认容忍 8/255 的亮度差（JPEG/抗锯齿微差）；>2% 像素超阈值判 FAIL。
输出: 每文件 max/mean 差、超阈值像素率，并在 ui-verify/diff-<name>.png 生成放大 16 倍的差异热力图。
退出码: 0=PASS, 1=FAIL, 2=缺文件
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(ROOT, "baseline")
CURRENT = os.path.join(ROOT, "current")
THRESHOLD = 8  # 亮度差阈值 /255
FAIL_RATIO = 0.02  # 超阈值像素占比 > 2% 判 FAIL


def load_luma(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    return (arr[:, :, 0] * 299 + arr[:, :, 1] * 587 + arr[:, :, 2] * 114) // 1000


def compare(name: str) -> bool:
    a = load_luma(os.path.join(BASELINE, name))
    b = load_luma(os.path.join(CURRENT, name))
    if a.shape != b.shape:
        print(f"{name}: SIZE MISMATCH {a.shape} vs {b.shape}")
        return False
    d = np.abs(a - b)
    exceed = (d > THRESHOLD).mean()
    ok = exceed <= FAIL_RATIO
    print(
        f"{name}: {'PASS' if ok else 'FAIL'} "
        f"max_diff={d.max()} mean_diff={d.mean():.2f} exceed={exceed:.4%} "
        f"shape={a.shape}"
    )
    if not ok:
        # 差异热力图：差异像素红色标出，其余黑色；放大 16 倍便于肉眼定位
        heat = np.zeros(a.shape, dtype=np.uint8)
        heat[d > THRESHOLD] = 255
        img = Image.fromarray(heat).resize((a.shape[1] // 16, a.shape[0] // 16), Image.Resampling.NEAREST)
        img.save(os.path.join(ROOT, f"diff-{name}"))
        print(f"  heatmap -> diff-{name}")
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-baseline", action="store_true", help="把 current/ 复制为 baseline/")
    args = ap.parse_args()

    if args.save_baseline:
        try:
            os.makedirs(BASELINE, exist_ok=True)
            files = sorted(f for f in os.listdir(CURRENT) if f.endswith(".png"))
        except OSError as e:
            print(f"cannot read/save directories: {e}")
            sys.exit(2)
        if not files:
            print(f"nothing to save: {CURRENT} is empty")
            sys.exit(2)
        for f in files:
            Image.open(os.path.join(CURRENT, f)).save(os.path.join(BASELINE, f))
        print(f"baseline updated: {len(files)} files")
        return

    try:
        base_files = sorted(f for f in os.listdir(BASELINE) if f.endswith(".png"))
        cur_files = sorted(f for f in os.listdir(CURRENT) if f.endswith(".png"))
    except OSError as e:
        print(f"cannot read directories: {e}")
        sys.exit(2)
    missing = set(base_files) - set(cur_files)
    if missing:
        print("MISSING in current/: " + ", ".join(sorted(missing)))
        sys.exit(2)

    results = [compare(f) for f in base_files]
    if all(results):
        print(f"ALL PASS: {len(results)} screenshots pixel-identical within threshold")
    else:
        print("FAIL: see heatmaps")
        sys.exit(1)


if __name__ == "__main__":
    main()
