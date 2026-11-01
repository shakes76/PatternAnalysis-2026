import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

IN_CH = 1
NUM_CLASSES = 6
NETWORK_WIDTH = 24

DILATIONS = [
	(1, 1, 1),
	(2, 2, 2),
	(2, 4, 4),
	(2, 8, 8),
	(1, 1, 1),
]

MARGIN = 0.4
GAP = 0.55
IO_W, IO_H = 1.7, 1.8
BLOCK_W, BLOCK_H = 2.4, 3.0
HEAD_W, HEAD_H = 2.2, 2.2
CY = 3.0  # vertical centre line

total_w = (
	2 * MARGIN
	+ IO_W + HEAD_W + IO_W
	+ len(DILATIONS) * BLOCK_W
	+ (len(DILATIONS) + 2) * GAP  # gaps: in->b1, between blocks (n-1), last->head, head->out
	+ 0
)
total_h = 6.0

def draw_block(ax, x, w, h, title, details):
	"""Draw a block whose left edge is x, centred vertically on CY."""
	y = CY - h / 2
	ax.add_patch(
		FancyBboxPatch(
			(x, y), w, h,
			boxstyle="round,pad=0,rounding_size=0.12",
			linewidth=2, facecolor="white", edgecolor="black",
		)
	)
	ax.text(x + w / 2, y + h * 0.84, title,
			ha="center", va="center", fontsize=12, fontweight="bold")
	ax.text(x + w / 2, y + h * 0.40, details,
			ha="center", va="center", fontsize=9, linespacing=1.3)


def draw_arrow(ax, x1, x2):
	ax.add_patch(
		FancyArrowPatch(
			(x1, CY), (x2, CY),
			arrowstyle="-|>", mutation_scale=18, linewidth=2,
			color="black", shrinkA=0, shrinkB=0,
		)
	)

# Create figure
fig, ax = plt.subplots(figsize=(total_w * 0.8, total_h))
ax.set_xlim(0, total_w)
ax.set_ylim(0, total_h)
ax.axis("off")

x = MARGIN

# Input
draw_block(ax, x, IO_W, IO_H, "INPUT", f"{IN_CH} channel\nD x H x W")
x_end = x + IO_W
x = x_end + GAP
draw_arrow(ax, x_end, x)

# Dilated blocks
in_ch = IN_CH
for i, d in enumerate(DILATIONS):
	details = (
		f"Conv3D {in_ch} → {NETWORK_WIDTH}\n"
		f"kernel 3 x 3 x 3\n"
		f"dilation = {d}\n"
		f"padding = {d}\n\n"
		f"InstanceNorm3D\n"
		f"LeakyReLU"
	)
	draw_block(ax, x, BLOCK_W, BLOCK_H, f"DilatedBlock {i + 1}", details)
	in_ch = NETWORK_WIDTH

	x_end = x + BLOCK_W
	x = x_end + GAP
	draw_arrow(ax, x_end, x)

# Head
draw_block(ax, x, HEAD_W, HEAD_H, "HEAD",
		   f"1 x 1 x 1 Conv3D\n\n{NETWORK_WIDTH} → {NUM_CLASSES} channels")
x_end = x + HEAD_W
x = x_end + GAP
draw_arrow(ax, x_end, x)

# Output
draw_block(ax, x, IO_W, IO_H, "OUTPUT", f"{NUM_CLASSES} classes\nD x H x W")

# Titles
ax.text(total_w / 2, 5.5, "CAN3D Architecture",
		ha="center", va="center", fontsize=20, fontweight="bold")
ax.text(total_w / 2, 0.6,
		"Each DilatedBlock: Conv3D → InstanceNorm3D → LeakyReLU",
		ha="center", va="center", fontsize=11)

plt.savefig("model_architecture.png", dpi=200, bbox_inches="tight",
			pad_inches=0.2, facecolor="white")