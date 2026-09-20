#!/usr/bin/env python3
"""Fit H3 runtime constants from measured benchmarks -> data/calibration.json

Usage:
    python tools/calibrate.py

Pure stdlib (runs on the bundled python_embeded interpreter).
Formula (verified against comfy/ldm/minimax/model.py @ ComfyUI 0.34.0):
    S            = ceil(W/32)*ceil(H/32)*((frames-1)//4+1) + 40*round(frames/24) + 400
    FLOPs/step   = 38.5e9*S + 1.434e6*S^2        (20B active params + 50x attention quad term)
    compute_s/it = FLOPs/step / (k_bw * vram_bw_gbps * 1e12) * modifiers
"""
import json
import math
import statistics
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

A_LINEAR = 38.5e9
B_QUAD = 1.434e6
TEXT_TOKENS = 400.0
CONF_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}


def h3_tokens(shape):
    w, h, f = shape["width"], shape["height"], shape["frames"]
    lat_t = (f - 1) // 4 + 1
    return math.ceil(w / 32) * math.ceil(h / 32) * lat_t + 40 * round(f / 24) + TEXT_TOKENS


def flops_per_step(shape):
    s = h3_tokens(shape)
    return A_LINEAR * s + B_QUAD * s * s


def weighted_median(pairs):
    """pairs: [(value, weight)] -> (median, n_effective)"""
    if not pairs:
        return None, 0
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v, len(pairs)
    return pairs[-1][0], len(pairs)


def main():
    gpus = {g["id"]: g for g in json.loads((DATA / "gpus.json").read_text(encoding="utf-8"))["gpus"]}
    bms = json.loads((DATA / "benchmarks.json").read_text(encoding="utf-8"))["benchmarks"]

    anchors = []
    for b in bms:
        if b.get("exclude_from_fit"):
            continue
        m = b.get("metrics", {})
        sit = m.get("s_per_it")
        if not sit and m.get("sample_s") and b.get("config", {}).get("steps"):
            sit = m["sample_s"] / b["config"]["steps"]
        if not sit:
            continue
        fl = flops_per_step(b["shape"])
        eff_tops = fl / sit / 1e12
        bw = gpus.get(b["gpu_id"], {}).get("mem_bw_gbps")
        ratio = (eff_tops / bw) if bw else None
        anchors.append({
            "id": b["id"], "gpu_id": b["gpu_id"], "s_per_it": round(sit, 2),
            "tokens": round(h3_tokens(b["shape"])), "flops_step_tflops": round(fl / 1e12, 1),
            "eff_tops": round(eff_tops, 1),
            "bw_gbps": bw, "ratio": round(ratio, 4) if ratio else None,
            "attn": b["config"].get("attn"), "steps": b["config"].get("steps"),
            "lora": b["config"].get("lora"), "confidence": b.get("confidence", "medium"),
        })

    # Split by attention class: sage-class (sage / kitchen / sage-kj) vs plain.
    # Mixing them poisons the fit (plain attention reaches ~0.73x of sage effective TOPS).
    sage_class = {"sage", "ck-attention", "sage-kj"}
    groups = {"sage": [], "plain": []}
    for a in anchors:
        if not a["ratio"]:
            continue
        cls = "sage" if a["attn"] in sage_class else "plain"
        groups[cls].append((a["ratio"], CONF_WEIGHT.get(a["confidence"], 1.0)))
    k_sage, n_sage = weighted_median(groups["sage"])
    k_plain, n_plain = weighted_median(groups["plain"])
    k_bw, n_eff = k_sage, n_sage  # site default: sage-class (recommended setup)

    # per-GPU medians (split by attention class)
    per_gpu = {}
    for a in anchors:
        g = per_gpu.setdefault(a["gpu_id"], {"eff_tops": [], "ratio": [], "ratio_sage": [], "ratio_plain": [], "anchors": []})
        g["eff_tops"].append(a["eff_tops"])
        if a["ratio"]:
            g["ratio"].append(a["ratio"])
            cls_list = g["ratio_sage"] if a["attn"] in sage_class else g["ratio_plain"]
            cls_list.append(a["ratio"])
        g["anchors"].append(a["id"])
    per_gpu_clean = {}
    for gid, g in per_gpu.items():
        per_gpu_clean[gid] = {
            "anchors": g["anchors"],
            "eff_tops_median": round(statistics.median(g["eff_tops"]), 1),
            "ratio_median": round(statistics.median(g["ratio"]), 4) if g["ratio"] else None,
            "ratio_sage_median": round(statistics.median(g["ratio_sage"]), 4) if g["ratio_sage"] else None,
            "ratio_plain_median": round(statistics.median(g["ratio_plain"]), 4) if g["ratio_plain"] else None,
        }

    # sage modifier from the 3090 paired anchors
    by_id = {a["id"]: a for a in anchors}
    sage_pair = None
    nosage = by_id.get("pepikir-3090-864x480x124-base20-nosage")
    sage = by_id.get("pepikir-3090-864x480x124-base20-sage")
    if nosage and sage:
        sage_pair = round(nosage["eff_tops"] / sage["eff_tops"], 3)

    # residuals against the attention-matched k (for the report)
    residuals = []
    for a in anchors:
        if not a["bw_gbps"]:
            continue
        k_used = k_sage if a["attn"] in sage_class else k_plain
        pred = a["bw_gbps"] * k_used * 1e12
        residuals.append({
            "id": a["id"], "attn_class": "sage" if a["attn"] in sage_class else "plain",
            "measured_eff_tops": a["eff_tops"],
            "predicted_eff_tops": round(pred / 1e12, 1),
            "err_pct": round((pred / (a["eff_tops"] * 1e12) - 1) * 100, 1),
        })

    calibration = {
        "generated": date.today().isoformat(),
        "constants": {
            "A_linear": A_LINEAR, "B_quad": B_QUAD, "text_tokens": TEXT_TOKENS,
            "latent_temporal_divisor": 4, "patch_px": 32, "audio_tokens_per_s": 40,
        },
        "k_bw_tops_per_gbps": round(k_bw, 4),
        "k_bw_tops_per_gbps_sage": round(k_sage, 4),
        "k_bw_tops_per_gbps_plain": round(k_plain, 4),
        "k_bw_n_anchors": n_eff,
        "k_bw_n_anchors_sage": n_sage,
        "k_bw_n_anchors_plain": n_plain,
        "modifiers": {
            "attn_plain_vs_sage": round(k_plain / k_sage, 3) if k_sage and k_plain else None,
            "sage_measured_pair": sage_pair,
            "dtype_bf16": 0.55,
            "gguf": 0.8,
            "lora_offload_patch_penalty": 1.08,
            "measured_overhead_step_s": 0.4
        },
        "offload_model": {
            "pcie_eff_ratio": 0.76,
            "ddr4_eff_ratio": 0.70,
            "ddr5_eff_ratio": 0.74,
            "ssd_eff_ratio": 0.75,
            "note": "非驻留权重每步流式传输：step_time >= bytes_not_resident / min(pcie_bw, src_tier_bw)"
        },
        "arch_floors_tops": {
            "120": 235,
            "89": 140,
            "note": "Sage/Kitchen 类注意力在现代架构上的绝对效率下限（TOPS，int8 基准）。Blackwell 三条独立实测（5080/5070Ti/5090）均落在 235-270 TOPS，与显存带宽不成比例；Ampere 无下限（带宽规律成立）。sm_89 为保守估计，证据较弱。"
        },
        "load_model": {
            "te_encode_floor_s": 8.0,
            "weight_read_overhead_mult": 1.5,
            "vae_decode_s_per_mp_frame": 0.25,
            "sources": ["local-log", "pepikir-speedup", "hf-discussion-6"]
        },
        "per_gpu": per_gpu_clean,
        "anchors_used": anchors,
        "residuals": residuals,
        "notes": [
            "k_bw 按注意力后端分组：sage/kitchen 类用 k_bw_tops_per_gbps_sage，plain 用 k_bw_tops_per_gbps_plain。两组比值≈0.73，与 3090 配对实测一致。",
            "网站应对单个 GPU 优先用 per_gpu.eff_tops_median（仅 sage 类锚点），无实测时回退到 k*显存带宽。",
            "二次注意力项在大 token 数（>3 万）不可忽略；15s 高分辨率时长偏差最大来源。",
            "benchmark 间的 per-step 效率差异可达 ±25%（LoRA patch/卸载/调度），UI 必须显示区间而非单值。",
            "modifiers.dtype_bf16 / gguf 无直接实测锚点，为编辑估计值，后续用投稿数据替换。",
            "低显存+量化+LoRA 场景存在每步重打补丁的额外开销（aptech 文档明确），当前以区间形式呈现，待专项实测后加入公式。"
        ]
    }

    out = DATA / "calibration.json"
    out.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"k_bw(sage) = {k_sage:.4f}  k_bw(plain) = {k_plain:.4f}  "
          f"anchors sage/plain = {n_sage}/{n_plain}  sage->plain factor = {sage_pair}")
    print("\nresiduals (attention-matched k):")
    for r in sorted(residuals, key=lambda x: abs(x["err_pct"]), reverse=True):
        print(f"  {r['err_pct']:+7.1f}%  [{r['attn_class']:5}] {r['id']}  "
              f"(measured {r['measured_eff_tops']} vs pred {r['predicted_eff_tops']} TOPS)")

    # sanity matrix for the reference rig (RTX 5090, sage-class, 4-step lightx2v)
    bw5090 = gpus["rtx-5090-32g"]["mem_bw_gbps"]
    print(f"\nRTX 5090 sanity estimates (int8 + sage, k={k_sage:.4f}, 4 steps):")
    for w, h, dur in [(736, 416, 5), (832, 480, 5), (960, 544, 5), (1152, 640, 5), (1344, 768, 5), (1344, 768, 10)]:
        shape = {"width": w, "height": h, "frames": 24 * dur + 1, "fps": 24}
        sit = flops_per_step(shape) / (bw5090 * k_sage * 1e12)
        print(f"  {w}x{h} {dur}s -> {sit:.1f} s/it, x4 = {sit*4/60:.1f} min (sampling only)")


if __name__ == "__main__":
    main()
