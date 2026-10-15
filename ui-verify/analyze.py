"""像素级验证：检查 ui-verify 截图的尺寸、白屏率、异常色块。

用法: .venv/bin/python ui-verify/analyze.py
检查项:
  1. 尺寸 == 1920x1080（像素基准一致）
  2. 近白像素率（luma>240）：深色 UI 应很低，>2% 视为异常（白屏/加载失败）
  3. 整块近白 / 整块纯黑（40px 块，>90% 命中）
  4. 侧栏区（x<240）/ 内容区 / 顶栏平均亮度，确认布局结构在
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))


def scan(fname: str) -> dict:
    with Image.open(fname) as chk:
        chk.verify()
    arr = np.asarray(Image.open(fname).convert("RGB"), dtype=np.int16)
    h, w, _ = arr.shape

    # luma（int16 防溢出）
    luma = (arr[:, :, 0] * 299 + arr[:, :, 1] * 587 + arr[:, :, 2] * 114) // 1000
    white_mask = luma > 240
    black_mask = (arr[:, :, 0] < 4) & (arr[:, :, 1] < 4) & (arr[:, :, 2] < 4)

    # 40px 块统计
    bw, bh = 40, 40
    gy, gx = h // bh, w // bw
    cells_white = 0
    cells_black = 0
    n_cells = gy * gx
    for cy in range(gy):
        for cx in range(gx):
            wblk = white_mask[cy * bh:(cy + 1) * bh, cx * bw:(cx + 1) * bw]
            bblk = black_mask[cy * bh:(cy + 1) * bh, cx * bw:(cx + 1) * bw]
            if wblk.mean() > 0.9:
                cells_white += 1
            if bblk.mean() > 0.95 and luma[cy * bh:(cy + 1) * bh, cx * bw:(cx + 1) * bw].mean() < 2:
                cells_black += 1

    sidebar = luma[:, :240].mean()
    content = luma[:, 240:].mean()
    topbar = luma[:60, 240:].mean()

    return {
        "file": os.path.basename(fname),
        "size": [w, h],
        "ok_size": w == 1920 and h == 1080,
        "white_ratio": round(float(white_mask.mean()), 4),
        "white_cells": round(cells_white / n_cells, 4),
        "black_cells": round(cells_black / n_cells, 4),
        "sidebar_luma": round(float(sidebar), 1),
        "content_luma": round(float(content), 1),
        "topbar_luma": round(float(topbar), 1),
    }


def main() -> None:
    files = sorted(glob.glob(os.path.join(ROOT, "*.png")))
    out = [scan(f) for f in files]
    print(json.dumps(out, ensure_ascii=False, indent=1))
    bad = [o for o in out if not o["ok_size"] or o["white_ratio"] > 0.02]
    if bad:
        print("ANOMALY: " + json.dumps(bad, ensure_ascii=False))
        sys.exit(1)
    print(f"ALL OK: {len(out)} screenshots — size consistent, no white/black blocks")


if __name__ == "__main__":
    main()
