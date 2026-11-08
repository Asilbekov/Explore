#!/usr/bin/env python3
"""Method snapshot figure for the wet-ink paper (fig_method.png).

Publication-quality static version of the decoding banner: dry / wet / frontier /
masked zones with labels. Large fonts, no title (caption lives in LaTeX).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_DRY = "#1a4a7a"
C_WET = "#cfe6f5"
C_MASK = "#f0f0f0"
C_FLASH = "#d16a6a"
DARK = "#142840"

TOKENS = ["Janet", "'s", "ducks", "lay", "16", "eggs", "a", "day", ".",
          "She", "eats", "3", ",", "bakes", "with", "4", ",",
          "and", "sells", "the", "rest", "for", "$2", "each", ".",
          "How", "much", "does", "she", "make", "every", "day", "?"]
N = len(TOKENS)
W = 16
F = 24          # frontier position in the snapshot
REWRITE = (6, "per")   # a token rewritten by the T2T gate inside the window

fig, ax = plt.subplots(figsize=(10.0, 2.35), dpi=220, constrained_layout=True)
state = list(TOKENS)
state[REWRITE[0]] = REWRITE[1]

ax.set_xlim(-1.2, N + 0.2)
ax.set_ylim(-1.75, 1.9)
ax.axis("off")

for i in range(N):
    if i < F:
        in_win = (F - W) <= i
        if i == REWRITE[0]:
            fc, tc, lw, ec = C_FLASH, "white", 1.8, "white"
        elif in_win:
            fc, tc, lw, ec = C_WET, DARK, 1.1, "#4a7aa8"
        else:
            fc, tc, lw, ec = C_DRY, "white", 0.9, "white"
    else:
        fc, tc, lw, ec = C_MASK, "#b8b8b8", 0.9, "#dcdcdc"
    ax.add_patch(mpatches.FancyBboxPatch(
        (i - 0.43, -0.52), 0.86, 1.04,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
        linestyle="--" if (i < F and (F - W) <= i) else "-"))
    txt = state[i] if i < F else "\u00b7\u00b7\u00b7"
    ax.text(i, 0.0, txt, ha="center", va="center", fontsize=8.2,
            color=tc, zorder=4, fontweight="bold" if i < F else "normal")

# frontier marker
ax.text(F, 0.62, "\u25bc", ha="center", va="bottom", fontsize=13,
        color=DARK, zorder=5)
ax.text(F, 0.98, "frontier: opens ONE token per step", ha="center",
        va="bottom", fontsize=9.5, color=DARK, zorder=5, fontweight="bold")

# wet window bracket
lo, hi = F - W, F
ax.annotate("", xy=(hi - 0.5, -0.75), xytext=(lo - 0.5, -0.75),
            arrowprops=dict(arrowstyle="<->", color="#4a7aa8", lw=1.5))
ax.text((lo + hi - 1) / 2, -1.06,
        "wet window $W$=16: trained T2T gate may rewrite ($p_0>\\tau_e$ and $x_i \\neq x_0$)",
        ha="center", fontsize=9.5, color="#2a5e8f")

# dry bracket
ax.annotate("", xy=(lo - 0.5, -0.75), xytext=(-0.5, -0.75),
            arrowprops=dict(arrowstyle="<->", color=C_DRY, lw=1.5))
ax.text((lo - 1.5) / 2, -1.06, "dry: frozen forever", ha="center",
        fontsize=9.5, color=C_DRY)

# rewrite callout
ax.annotate("T2T rewrite\n(wet ink)", xy=(REWRITE[0], 0.55),
            xytext=(REWRITE[0] - 1.5, 1.55), fontsize=9, color="#a34a4a",
            ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#a34a4a", lw=1.2))

# masked label
ax.text(N - 0.1, 1.55, "still masked", fontsize=9, color="#8a8a8a",
        ha="right", va="center")

# hard dry edge marker
ax.annotate("hard dry edge at distance $W$", xy=(lo - 0.5, -0.30),
            xytext=(lo - 1.0, 0.55), fontsize=9, color=C_DRY, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_DRY, lw=1.2))

out = "/home/z/my-project/scripts/paper/fig_method.png"
fig.savefig(out, dpi=220)
plt.close(fig)
print("wrote", out)
