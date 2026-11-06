#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Чарт результатов v4 (LLaDA2.2-mini, 2×T4, int8): chart_v4_gsm8k.png.

2×2 панели:
  (0,0) Точность P0-пары с Wilson 95% CI (усами)
  (0,1) Парный NFE на задачу (n=49)
  (1,0) Парный churn на задачу (n=49)
  (1,1) Усечение gen_length=160: доля завершённых/усечённых + точность в каждой группе
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from math import comb

FONT_DIR = "/usr/share/fonts"
fm.fontManager.addfont(f"{FONT_DIR}/truetype/dejavu/DejaVuSans.ttf")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WORK_DIR = "/home/z/my-project/scripts/report_assets"
V4_PATH = "/home/z/my-project/wet_ink/results/results_v4.json"
TOK_CAP = 160

C_NOEDIT = "#7a9bb8"   # контроль (T2T OFF)
C_STOCK = "#1a4a7a"    # stock / wet-ink ON
C_TRUNC = "#c9d8e8"    # усечённые генерации


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


with open(V4_PATH) as f:
    V4 = json.load(f)

g = V4["gsm8k"]
ARMS = [("noedit_b32", "noedit (T2T OFF)", C_NOEDIT),
        ("stock_b32", "stock (T2T ON)", C_STOCK)]

fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.8), constrained_layout=True)

# ---- (0,0) accuracy + Wilson CI ----
ax = axes[0][0]
labels, accs, errs, cols = [], [], [], []
for k, lab, col in ARMS:
    det = [d for d in g[k]["details"] if "error" not in d]
    n = len(det)
    kk = sum(bool(d["ok"]) for d in det)
    p, lo, hi = wilson(kk, n)
    labels.append(f"{lab}\n(n={n})")
    accs.append(100 * p)
    errs.append([100 * (p - lo), 100 * (hi - p)])
    cols.append(col)
x = range(len(labels))
bars = ax.bar(list(x), accs, color=cols, edgecolor="white", linewidth=0.6,
              zorder=3, width=0.55,
              yerr=[[e[0] for e in errs], [e[1] for e in errs]],
              error_kw=dict(ecolor="#142840", capsize=5, lw=1.1, zorder=4))
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=8.4)
ax.set_ylabel("Точность GSM8K, %", fontsize=9.5)
ax.set_ylim(0, 48)
ax.set_title("Точность с Wilson 95% CI", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6, zorder=0)
for b, v in zip(bars, accs):
    ax.text(b.get_x() + b.get_width() / 2, v + 3.4, f"{v:.1f}",
            ha="center", fontsize=8.6, color="#142840", fontweight="bold")

# ---- (0,1) paired NFE ----
ax = axes[0][1]
d0 = {d["i"]: d for d in g["noedit_b32"]["details"] if "error" not in d}
d1 = {d["i"]: d for d in g["stock_b32"]["details"] if "error" not in d}
common = sorted(set(d0) & set(d1))
nfe0 = sum(d0[i]["nfe"] for i in common) / len(common)
nfe1 = sum(d1[i]["nfe"] for i in common) / len(common)
bars = ax.bar([0, 1], [nfe0, nfe1], color=[C_NOEDIT, C_STOCK],
              edgecolor="white", linewidth=0.6, zorder=3, width=0.55)
ax.set_xticks([0, 1])
ax.set_xticklabels(["noedit", "stock"], fontsize=8.6)
ax.set_ylabel("Ср. NFE на задачу", fontsize=9.5)
ax.set_ylim(0, 60)
ax.set_title(f"NFE на задачу (парно, n={len(common)})", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6, zorder=0)
for b, v in zip(bars, [nfe0, nfe1]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
            ha="center", fontsize=8.6, color="#142840", fontweight="bold")
ax.annotate("×0,84", xy=(1, nfe1), xytext=(0.38, 55),
            fontsize=10, color="#1a4a7a", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a4a7a", lw=1.1))

# ---- (1,0) paired churn ----
ax = axes[1][0]
ch0 = sum(d0[i].get("churn", 0) for i in common) / len(common)
ch1 = sum(d1[i].get("churn", 0) for i in common) / len(common)
bars = ax.bar([0, 1], [ch0, ch1], color=[C_NOEDIT, C_STOCK],
              edgecolor="white", linewidth=0.6, zorder=3, width=0.55)
ax.set_xticks([0, 1])
ax.set_xticklabels(["noedit", "stock"], fontsize=8.6)
ax.set_ylabel("Ср. churn на задачу", fontsize=9.5)
ax.set_ylim(0, 9.5)
ax.set_title(f"Churn: перезаписи открытых токенов (n={len(common)})", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6, zorder=0)
for b, v in zip(bars, [ch0, ch1]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}",
            ha="center", fontsize=8.6, color="#142840", fontweight="bold")
ax.annotate("×5,5", xy=(1, ch1), xytext=(0.5, 8.6),
            fontsize=10, color="#1a4a7a", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a4a7a", lw=1.1))

# ---- (1,1) truncation ----
ax = axes[1][1]
fin_share, tru_share, fin_acc, tru_acc = [], [], [], []
for k, lab, col in ARMS:
    det = [d for d in g[k]["details"] if "error" not in d]
    fin = [d for d in det if d.get("n_tokens", 0) < TOK_CAP]
    tru = [d for d in det if d.get("n_tokens", 0) >= TOK_CAP]
    n = len(det)
    fin_share.append(100 * len(fin) / n)
    tru_share.append(100 * len(tru) / n)
    fin_acc.append(100 * sum(bool(d["ok"]) for d in fin) / max(len(fin), 1))
    tru_acc.append(100 * sum(bool(d["ok"]) for d in tru) / max(len(tru), 1))
x = [0, 1]
ax.bar(x, fin_share, color=[C_NOEDIT, C_STOCK], edgecolor="white",
       linewidth=0.6, zorder=3, width=0.55, label="завершено (EOS)")
ax.bar(x, tru_share, bottom=fin_share, color=C_TRUNC, edgecolor="white",
       linewidth=0.6, zorder=3, label="усечено лимитом 160")
ax.set_xticks(x)
ax.set_xticklabels(["noedit", "stock"], fontsize=8.6)
ax.set_ylabel("Доля генераций, %", fontsize=9.5)
ax.set_ylim(0, 112)
ax.set_title("Лимит 160 токенов: доля и точность", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6, zorder=0)
for i in x:
    ax.text(i, fin_share[i] / 2, f"{fin_acc[i]:.0f}%\nверны",
            ha="center", va="center", fontsize=8.2, color="white",
            fontweight="bold", zorder=5)
    ax.text(i, fin_share[i] + tru_share[i] / 2 - 2,
            f"{tru_share[i]:.0f}%\n{tru_acc[i]:.0f}% верны",
            ha="center", va="center", fontsize=8.2, color="#5a7a96",
            fontweight="bold", zorder=5)
ax.legend(fontsize=7.8, loc="upper center", ncol=2, frameon=False,
          bbox_to_anchor=(0.5, 1.02))

p1 = os.path.join(WORK_DIR, "chart_v4_gsm8k.png")
fig.savefig(p1, dpi=200)
plt.close(fig)
print("wrote", p1)
