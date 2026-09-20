#!/usr/bin/env python3
"""Generate official resolution table from ComfyUI's ResolutionSelector.

Algorithm source: ComfyUI comfy_extras/nodes_resolution.py (ResolutionSelector),
H3 official workflows use multiple=32. Native canvas: short edge 768 / long edge 1344.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "resolutions_official.json"

ASPECTS = [
    ("16:9", "16:9 横屏", (16, 9)),
    ("3:2", "3:2 照片横", (3, 2)),
    ("4:3", "4:3 标准", (4, 3)),
    ("1:1", "1:1 方形", (1, 1)),
    ("3:4", "3:4 竖版", (3, 4)),
    ("2:3", "2:3 照片竖", (2, 3)),
    ("9:16", "9:16 竖屏", (9, 16)),
    ("21:9", "21:9 宽幅", (21, 9)),
]

MPS = [0.1, 0.4, 0.5, 0.7, 1.0, 2.0]

# 0.1MP 在社区实测中出现闪帧，标注提示（官方节点允许，但不推荐）
NOTES = {0.1: "易闪帧"}


def main():
    out = {
        "meta": {
            "updated": "2026-09-18",
            "source": "ComfyUI comfy_extras/nodes_resolution.py ResolutionSelector (v0.34); H3 official workflows use multiple=32",
            "native_canvas": "短边 768px / 长边上限 1344px",
            "mp_range": [0.4, 2.0],
            "formula": "scale = sqrt(mp*1024*1024/(w_ratio*h_ratio)); width = round(w_ratio*scale/multiple)*multiple",
        },
        "aspects": [],
    }
    for aid, label, (rw, rh) in ASPECTS:
        sizes = []
        for mp in MPS:
            total = mp * 1024 * 1024
            scale = math.sqrt(total / (rw * rh))
            w = round(rw * scale / 32) * 32
            h = round(rh * scale / 32) * 32
            native = min(w, h) <= 768 and max(w, h) <= 1344
            entry = {"mp": mp, "width": w, "height": h, "native": native}
            if mp in NOTES:
                entry["note"] = NOTES[mp]
            sizes.append(entry)
        out["aspects"].append({"id": aid, "label": label, "ratio": [rw, rh], "sizes": sizes})

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", OUT)
    print("16:9 ladder:")
    for s in out["aspects"][0]["sizes"]:
        mark = "原生" if s["native"] else "超原生"
        print(f"  {s['mp']}MP -> {s['width']}x{s['height']} ({mark})")


if __name__ == "__main__":
    main()
