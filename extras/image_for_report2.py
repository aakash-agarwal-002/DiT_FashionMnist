import os
from PIL import Image
import matplotlib.pyplot as plt

folder = "/Users/aakashagarwal/Downloads/vaani/Adaptive_layer_norm_zero_conditioning_ddim/samples/test"

epochs = [5, 10, 15, 20, 25]
temps  = [100, 500, 1000]

intervals = [1, 2, 5, 10, 20]   # edit or reorder as desired


def load_png_interval(epoch, temp, interval):
    return Image.open(os.path.join(folder, f"e_{epoch}_T{temp}_I{interval}.png"))


def add_axis_close(fig, axes):
    fig.subplots_adjust(hspace=0)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.yaxis.set_label_coords(-0.03, 0.5)


# -------------------------------------------------------------
# new figure: interval 20, T = 1000, epochs 5 to 25
# -------------------------------------------------------------
interval_fixed = 20
T_fixed = 1000

fig, axes = plt.subplots(len(epochs), 1, figsize=(20, 8))
if len(epochs) == 1:
    axes = [axes]

for i, e in enumerate(epochs):
    im = load_png_interval(e, T_fixed, interval_fixed)
    axes[i].imshow(im, aspect="auto")
    axes[i].set_ylabel(str(e), rotation=0, ha="right", va="center",
                       labelpad=-10)

add_axis_close(fig, axes)
fig.supylabel("Epochs")
fig.tight_layout(rect=(0.05, 0, 1, 1))

out = os.path.join(folder, "summary_I20_T1000.png")
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
plt.close(fig)


# -------------------------------------------------------------
# new figure: epoch 25, T = 1000, intervals stacked vertically
# -------------------------------------------------------------
epoch_fixed = 20

fig, axes = plt.subplots(len(intervals), 1, figsize=(20, 8))
if len(intervals) == 1:
    axes = [axes]

for i, I in enumerate(intervals):
    im = load_png_interval(epoch_fixed, T_fixed, I)
    axes[i].imshow(im, aspect="auto")
    axes[i].set_ylabel(str(I), rotation=0, ha="right", va="center",
                       labelpad=-10)

add_axis_close(fig, axes)
fig.supylabel("Intervals")
fig.tight_layout(rect=(0.05, 0, 1, 1))

out = os.path.join(folder, "summary_epoch25_T1000_intervals.png")
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
plt.close(fig)
