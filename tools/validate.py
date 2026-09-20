#!/usr/bin/env python3
"""Cross-file integrity checks for the data repo. Exit 1 on errors."""
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
CTX_KEYS = {"windows", "safetensors_gt_20gb", "pagefile_used", "non_blackwell",
            "sm_120", "seq_gt_160k_tokens", "diffusers_format_lora"}


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


errors, warnings = [], []

gpu_ids = {g["id"] for g in load("gpus.json")["gpus"]}
models = load("models.json")
model_ids = {f["id"] for f in models["files"]} | {l["id"] for l in models["lora_files"]}
sets = load("model_sets.json")["sets"]
acc = load("accelerations.json")["accelerations"]
acc_ids = {a["id"] for a in acc}
rules = load("conflicts.json")["rules"]
rule_ids = {r["id"] for r in rules}
bms = load("benchmarks.json")["benchmarks"]
cal = load("calibration.json")

for b in bms:
    if b.get("gpu_id") and b["gpu_id"] not in gpu_ids:
        errors.append(f"benchmark {b['id']}: unknown gpu_id {b['gpu_id']}")
    for key in ("dit", "te", "lora"):
        v = b.get("config", {}).get(key)
        if v and v not in model_ids:
            warnings.append(f"benchmark {b['id']}: config.{key}={v} not in models.json")

for s in sets:
    for f in s.get("files", []) + s.get("alt_files", []):
        if f not in model_ids:
            errors.append(f"set {s['id']}: unknown file {f}")
    a = s.get("acceleration_default")
    if a and a not in acc_ids:
        errors.append(f"set {s['id']}: unknown acceleration {a}")

for a in acc:
    for c in a.get("conflicts", []):
        if c not in rule_ids:
            errors.append(f"acceleration {a['id']}: unknown conflict {c}")

for r in rules:
    for w in r["when"]:
        if w in acc_ids or w in model_ids or w in CTX_KEYS:
            continue
        warnings.append(f"rule {r['id']}: unrecognized when-ref {w}")

if cal["k_bw_tops_per_gbps_sage"] <= 0 or cal["k_bw_tops_per_gbps_plain"] <= 0:
    errors.append("calibration k values must be positive")
if not (cal["k_bw_tops_per_gbps_plain"] < cal["k_bw_tops_per_gbps_sage"]):
    warnings.append("calibration: plain k should be lower than sage k")

print(f"gpus={len(gpu_ids)} models={len(model_ids)} sets={len(sets)} "
      f"accel={len(acc)} rules={len(rules)} benchmarks={len(bms)}")
for e in errors:
    print("ERROR:", e)
for w in warnings:
    print("WARN :", w)
print("PASS" if not errors else "FAIL")
sys.exit(1 if errors else 0)
