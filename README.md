# WhichH3 Data — MiniMax H3 硬件适配数据集

**在线计算器（本数据集的服务端展示）：<https://whichh3.com>**

> 选好显卡和内存，30 秒算出：该下哪套模型组合、一条视频要生成多久、显存/内存/SSD 压力。
> 支持中英双语、电脑/手机，免费无需注册。

本仓库是 whichh3.com 计算器的**原始数据与校验工具**，全部公开，欢迎引用、提 issue 纠正数据、提 PR 补充实测锚点。

---

## 数据文件一览（`data/`）

| 文件 | 内容 |
| --- | --- |
| `gpus.json` | **49 张显卡**：显存、带宽、CUDA 核心、架构、PCIe、备注（含 6 张魔改卡：2080 Ti 22G / 3070 16G / 3080 20G / 4080 32G / 4080 Super 32G / 4090 48G） |
| `models.json` | **48 个模型文件**：主模型 / 文本编码器 / VAE 的体积、量化、依赖、备注 |
| `model_sets.json` | 6 套推荐组合（入门 / 均衡 / 质量 / Blackwell / 旗舰 / 研究） |
| `accelerations.json` | 加速方案：步数蒸馏、注意力后端（Sage / Kitchen / SLA / VSA）、缓存，含加速倍率与质量代价 |
| `conflicts.json` | 组合冲突与风险提示（含依据来源） |
| `benchmarks.json` | 30+ 条社区实测锚点（来源、配置、s/it），用于校准估算模型 |
| `calibration.json` | 回归系数与每卡修正（估算是「带宽 × 回归系数 + 架构修正」，不是峰值算力） |
| `resolutions_official.json` | ComfyUI ResolutionSelector 官方尺寸表（0.1–2.0MP × 8 种宽高比） |
| `download_library.json` | 精选下载库（仅收录 ComfyUI 可直接加载的版本） |
| `memory.json` | DDR / PCIe / SSD 有效带宽与页文件模型 |
| `sources.json` | 全部数据来源（官方文档、GitHub issue、社区实测） |

## 工具（`tools/`）

```bash
python tools/validate.py        # 数据完整性校验（ID 引用、体积/带宽范围等）
python tools/calibrate.py       # 用 benchmarks.json 重新拟合校准系数
python tools/gen_resolutions.py # 重新生成官方尺寸表
```

## 估算方法

见 [`docs/data-notes.md`](docs/data-notes.md)：时间 = 固定开销 + 加载 + 编码 + 采样（线性项 + 二次注意力项）+ VAE，
核心是「显存带宽 × 实测回归系数」，对 Ada/Blackwell 附加绝对效率下限，对无 Tensor Core 的旧卡做保守折算。
当前 19 条锚点回归平均误差 **13.9%**。

## 引用与许可

- 代码（`tools/`）：MIT
- 数据（`data/`）：CC BY 4.0 —— 引用请注明 **WhichH3（https://whichh3.com）**

## 纠错与补充

- 数据有误 / 有你的实测想加进来：开 **Issue** 附上配置、分辨率、时长与 s/it（或总时长）
- 采纳后会进入锚点回归并更新线上计算器

---

# English

**Online calculator: <https://whichh3.com>** (bilingual EN/中文, free, client-side)

This repo is the raw dataset and validation tooling behind the WhichH3 calculator:
49 GPUs (including 6 modded cards), 48 model files, 6 recommended combos, acceleration & conflict tables,
30+ community benchmark anchors, and the calibrated estimation model (mean error 13.9% across 19 anchors).

- Data: CC BY 4.0 — please cite WhichH3 (https://whichh3.com)
- Tools: MIT
- Found an error or have a measured anchor to add? Open an issue with your config, resolution, duration and s/it.
