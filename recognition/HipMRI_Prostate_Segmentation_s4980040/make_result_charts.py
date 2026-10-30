"""
make_result_charts.py - Draws the summary charts used in the README from the
measured results (results/*.json values, copied here so the script runs anywhere).
Run: python make_result_charts.py   -> saves PNGs into images/
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C2D, C3D = "#2a78d6", "#eb6834"          # 2D U-Net = blue, 3D Improved U-Net = orange
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
OUT = "images"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def grouped(ax, labels, a, b, ylim, fmt="{:.2f}", ylabel=""):
    x = np.arange(len(labels)); w = 0.36
    for off, vals, col, name in ((-w / 2, a, C2D, "2D U-Net"), (w / 2, b, C3D, "3D Improved U-Net")):
        bars = ax.bar(x + off, vals, w - 0.03, color=col, label=name)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.01, fmt.format(v),
                    ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, labels); ax.set_ylim(*ylim); ax.set_ylabel(ylabel); style(ax)


# ---- 1. Dice comparison: per class, and prostate by region ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [5, 3]})
grouped(a1, ["Body", "Bone", "Bladder", "Rectum", "Prostate"],
        [0.984, 0.933, 0.966, 0.851, 0.863], [0.977, 0.896, 0.945, 0.818, 0.859], (0.6, 1.0), ylabel="Test Dice")
a1.set_title("Test Dice per class (full resolution)", loc="left", fontsize=11)
grouped(a2, ["Boundary\nslices", "Middle\nslices", "Whole\nprostate"],
        [0.707, 0.898, 0.863], [0.778, 0.903, 0.859], (0.6, 1.0))
a2.set_title("Prostate Dice by region", loc="left", fontsize=11)
a2.annotate("+0.07", xy=(0.18, 0.825), ha="center", fontsize=10, color=INK, weight="bold")
fig.legend(*a1.get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2)
fig.text(0.01, 0.01, "Y-axis starts at 0.6. Test set: 9 scans from 3 patients.", fontsize=8, color=INK2)
plt.tight_layout(rect=(0, 0.03, 1, 0.93)); plt.savefig(f"{OUT}/dice_comparison.png", dpi=150); plt.close()

# ---- 2. Per-scan paired comparison (dumbbell) ----
scans = ["040 w0", "040 w1", "040 w2", "040 w3", "040 w4", "040 w5", "040 w7", "041 w0", "042 w0"]
d2 = [0.8746, 0.8786, 0.8918, 0.8663, 0.8398, 0.8837, 0.8644, 0.8582, 0.8404]
d3 = [0.8767, 0.8537, 0.8882, 0.8377, 0.8044, 0.8832, 0.8806, 0.9048, 0.8329]
fig, ax = plt.subplots(figsize=(7.5, 4.4))
y = np.arange(len(scans))[::-1]
for yi, a, b in zip(y, d2, d3):
    ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1)
ax.scatter(d2, y, s=55, color=C2D, label="2D U-Net", zorder=3, edgecolor=BG, linewidth=1.5)
ax.scatter(d3, y, s=55, color=C3D, label="3D Improved U-Net", zorder=3, edgecolor=BG, linewidth=1.5)
ax.set_yticks(y, scans); ax.set_xlabel("Prostate Dice")
ax.set_xlim(0.79, 0.92); ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_title("Prostate Dice per test scan (same 9 scans for both models)", loc="left", fontsize=11)
ax.legend(frameon=False, loc="upper left")
fig.text(0.01, 0.01, "Means: 2D 0.866 ± 0.017, 3D 0.863 ± 0.030. 3D is higher on 3 of 9 scans.", fontsize=8, color=INK2)
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(f"{OUT}/per_scan_dice.png", dpi=150); plt.close()

# ---- 3. Resource comparison (small multiples, one measure per panel, no dual axes) ----
panels = [("Parameters (millions)", [31.0, 9.5], "{:.1f}"),
          ("Peak VRAM, training (GB)", [1.95, 1.99], "{:.2f}"),
          ("Peak VRAM, inference (GB)", [3.61, 0.52], "{:.2f}"),
          ("Time per epoch, A100 (s)", [34, 9], "{:.0f}"),
          ("Inference per scan, A100 (ms)", [95, 23], "{:.0f}")]
fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
for ax, (title, vals, fmt) in zip(axes, panels):
    bars = ax.bar([0, 1], vals, 0.6, color=[C2D, C3D])
    top = max(vals) * 1.22
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + top * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks([0, 1], ["2D", "3D"]); ax.set_ylim(0, top); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.tick_params(length=0)
    ax.set_title(title, fontsize=9.5, loc="left")
fig.suptitle("Resource profile (A100-PCIE-40GB; blue = 2D U-Net, orange = 3D Improved U-Net)", x=0.01, ha="left", fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(f"{OUT}/resource_comparison.png", dpi=150); plt.close()
print("Saved dice_comparison.png, per_scan_dice.png, resource_comparison.png")