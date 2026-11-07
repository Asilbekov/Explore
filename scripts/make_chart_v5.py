#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Чарт результатов v5 (wet-ink декодер, LLaDA2.2-mini): chart_v5_gsm8k.png.

2×2 панели:
  (0,0) Точность: базлайны v4b + руки v5 с Wilson 95% CI (горизонтальные бары)
  (0,1) Парный NFE на общих задачах: stock_b32 vs wetink_w16 (×4,88)
  (1,0) Парные расхождения McNemar (исправлено/сломано) по 4 сравнениям
  (1,1) Сегменты задач 0–21 и 22–44: одинаковая точность всех рук по сегментам
        (показывает, что разница общих accuracy — состав задач, не эффект правки)
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FONT_DIR = "/usr/share/fonts"
fm.fontManager.addfont(f"{FONT_DIR}/truetype/dejavu/DejaVuSans.ttf")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WORK_DIR = "/home/z/my-project/scripts/report_assets"
V4_PATH = "/home/z/my-project/wet_ink/results/results_v4.json"
V5_PATH = "/home/z/my-project/wet_ink/results/results_v5.json"

C_V4B = "#7a9bb8"      # базлайны v4b (штатный generate)
C_WET = "#1a4a7a"      # wet-ink декодер v5 (главная рука)
C_CTRL = "#4a7aa8"     # контроль v5 (фронтир без правки)
C_ABL = "#9ab8d0"      # абляция W=8
C_FIX = "#2d7ab3"      # исправлено (win)
C_BRK = "#b8545a"      # сломано (loss)


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
with open(V5_PATH) as f:
    V5 = json.load(f)

g4, g5 = V4["gsm8k"], V5["gsm8k"]


def det_of(src, name):
    return [d for d in src[name]["details"] if "error" not in d]


# ---- парные сравнения по индексу задач ----
def pair(base, arm):
    m0 = {d["i"]: d for d in det_of(g4, base)}
    m1 = {d["i"]: d for d in det_of(g5, arm)}
    com = sorted(set(m0) & set(m1))
    b = sum(1 for i in com if not m0[i]["ok"] and m1[i]["ok"])
    c = sum(1 for i in com if m0[i]["ok"] and not m1[i]["ok"])
    nfe0 = sum(m0[i]["nfe"] for i in com) / len(com)
    nfe1 = sum(m1[i]["nfe"] for i in com) / len(com)
    return dict(n=len(com), b=b, c=c, nfe0=nfe0, nfe1=nfe1)


P_stock_w16 = pair("stock_b32", "wetink_w16")
P_noed_w16 = pair("noedit_b32", "wetink_w16")
# внутренний контроль v5: фронтир без правки vs главная рука (оба из v5)
_mn = {d["i"]: d for d in det_of(g5, "wetink_noedit_w16")}
_mw = {d["i"]: d for d in det_of(g5, "wetink_w16")}
_com = sorted(set(_mn) & set(_mw))
P_ctrl_w16 = dict(
    n=len(_com),
    b=sum(1 for i in _com if not _mn[i]["ok"] and _mw[i]["ok"]),
    c=sum(1 for i in _com if _mn[i]["ok"] and not _mw[i]["ok"]),
)
P_stock_ctrl = pair("stock_b32", "wetink_noedit_w16")

fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.8), constrained_layout=True)

# ---- (0,0) accuracy + Wilson CI ----
ax = axes[0][0]
ARMS = [
    (g4, "stock_b32", "stock_b32 · v4b (n=80)", C_V4B),
    (g4, "noedit_b32", "noedit_b32 · v4b (n=80)", C_V4B),
    (g5, "wetink_w16", "wetink_w16 · v5 (n=45)", C_WET),
    (g5, "wetink_noedit_w16", "noedit_w16 · v5 (n=22)", C_CTRL),
    (g5, "wetink_w8", "wetink_w8 · v5 (n=10)", C_ABL),
]
labels, accs, errs, cols = [], [], [], []
for src, k, lab, col in ARMS:
    det = det_of(src, k)
    n = len(det)
    kk = sum(bool(d["ok"]) for d in det)
    p, lo, hi = wilson(kk, n)
    labels.append(lab)
    accs.append(100 * p)
    errs.append([100 * (p - lo), 100 * (hi - p)])
    cols.append(col)
y = list(range(len(labels)))
ax.barh(y, accs, color=cols, edgecolor="white", linewidth=0.6, zorder=3,
        height=0.62, xerr=[[e[0] for e in errs], [e[1] for e in errs]],
        error_kw=dict(ecolor="#142840", capsize=3, lw=1.0, zorder=4))
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=7.4)
ax.invert_yaxis()
ax.set_xlabel("Точность GSM8K, %", fontsize=9)
ax.set_xlim(0, 55)
ax.set_title("Точность: базлайны v4b и декодер v5 (Wilson 95% CI)", fontsize=9.6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.6, zorder=0)
for b, v in zip(ax.patches, accs):
    ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.1f}",
            va="center", fontsize=7.8, color="#142840", fontweight="bold")

# ---- (0,1) paired NFE: stock vs wetink_w16 ----
ax = axes[0][1]
nfe0, nfe1 = P_stock_w16["nfe0"], P_stock_w16["nfe1"]
bars = ax.bar([0, 1], [nfe0, nfe1], color=[C_V4B, C_WET],
              edgecolor="white", linewidth=0.6, zorder=3, width=0.55)
ax.set_xticks([0, 1])
ax.set_xticklabels(["stock_b32\n(параллельный)", "wetink_w16\n(фронтир 1 токен)"], fontsize=8.0)
ax.set_ylabel("Ср. NFE на задачу", fontsize=9.5)
ax.set_ylim(0, 250)
ax.set_title(f"Парный NFE, n={P_stock_w16['n']} общих задач", fontsize=9.6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6, zorder=0)
for b, v in zip(bars, [nfe0, nfe1]):
    ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.1f}",
            ha="center", fontsize=8.6, color="#142840", fontweight="bold")
ax.annotate(f"×{nfe1 / nfe0:.2f}", xy=(1, nfe1), xytext=(0.30, 228),
            fontsize=10.5, color="#1a4a7a", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a4a7a", lw=1.1))

# ---- (1,0) discordant pairs ----
ax = axes[1][0]
COMPS = [
    (f"w16 vs stock_b32 (n={P_stock_w16['n']})", P_stock_w16["b"], P_stock_w16["c"]),
    (f"w16 vs noedit_b32 (n={P_noed_w16['n']})", P_noed_w16["b"], P_noed_w16["c"]),
    (f"w16 vs контроль noedit_w16 (n={P_ctrl_w16['n']})", P_ctrl_w16["b"], P_ctrl_w16["c"]),
    (f"контроль vs stock_b32 (n={P_stock_ctrl['n']})", P_stock_ctrl["b"], P_stock_ctrl["c"]),
]
yy = list(range(len(COMPS)))[::-1]
for yv, (lab, b, c) in zip(yy, COMPS):
    ax.barh(yv, b, color=C_FIX, edgecolor="white", linewidth=0.6, zorder=3, height=0.58)
    ax.barh(yv, c, left=b, color=C_BRK, edgecolor="white", linewidth=0.6, zorder=3, height=0.58)
    tot = b + c
    ax.text(tot + 0.12, yv, f"{b}/{c}  (p=1,0)", va="center", fontsize=7.6, color="#142840")
ax.set_yticks(yy)
ax.set_yticklabels([c[0] for c in COMPS], fontsize=7.2)
ax.set_xlabel("Парные исходы: исправлено / сломано", fontsize=9)
ax.set_xlim(0, 5.6)
ax.set_title("Расхождения McNemar: все p = 1,0", fontsize=9.6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.6, zorder=0)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=C_FIX, label="исправлено"),
                   Patch(color=C_BRK, label="сломано")],
          fontsize=7.6, loc="lower right", frameon=False)

# ---- (1,1) task-mix segments ----
ax = axes[1][1]
SEGS = [("задачи 0–21", 0, 21), ("задачи 22–44", 22, 44)]
ROWS = [
    ("stock_b32", g4, "stock_b32 · v4b", C_V4B),
    ("wetink_w16", g5, "wetink_w16 · v5", C_WET),
    ("wetink_noedit_w16", g5, "noedit_w16 · v5", C_CTRL),
]
import numpy as np  # noqa: E402
x = np.arange(len(SEGS))
w = 0.26
for j, (k, src, lab, col) in enumerate(ROWS):
    det = det_of(src, k)
    vals = []
    for _, lo, hi in SEGS:
        seg = [d for d in det if lo <= d["i"] <= hi]
        vals.append(100 * sum(bool(d["ok"]) for d in seg) / max(len(seg), 1))
    bars = ax.bar(x + (j - 1) * w, vals, width=w, color=col,
                  edgecolor="white", linewidth=0.6, zorder=3, label=lab)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.0f}",
                ha="center", fontsize=7.8, color="#142840", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in SEGS], fontsize=8.4)
ax.set_ylabel("Точность, %", fontsize=9.5)
ax.set_ylim(0, 46)
ax.set_title("Разница общих accuracy = состав задач,\nне эффект правки", fontsize=9.6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.6, zorder=0)
ax.legend(fontsize=7.0, loc="upper left", frameon=False)

p = os.path.join(WORK_DIR, "chart_v5_gsm8k.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print("wrote", p)
