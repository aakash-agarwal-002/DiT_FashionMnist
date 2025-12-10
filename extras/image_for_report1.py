import os
from PIL import Image
import matplotlib.pyplot as plt

folder = "/Users/aakashagarwal/Downloads/vaani/Extra_Input_Tokens_Conditioning/samples/test"

epochs = [5, 10, 15, 20, 25]
temps  = [100, 500, 1000]

def load_png(epoch, temp):
    return Image.open(os.path.join(folder, f"e_{epoch}_T{temp}.png"))

def add_axis_close(fig, axes):
    fig.subplots_adjust(hspace=0)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # move the y label closer
        ax.yaxis.set_label_coords(-0.03, 0.5)   # <-------------------

# -------------------------------------------------------------
# one figure per T
# -------------------------------------------------------------
for T in temps:
    fig, axes = plt.subplots(len(epochs), 1, figsize=(20, 8))

    if len(epochs) == 1:
        axes = [axes]

    for i, e in enumerate(epochs):
        im = load_png(e, T)
        axes[i].imshow(im, aspect="auto")
        axes[i].set_ylabel(str(e), rotation=0, ha="right", va="center",
                           labelpad=-10)         # <-------------------

    add_axis_close(fig, axes)
    fig.supylabel("Epochs")

    fig.tight_layout(rect=(0.05, 0, 1, 1))       # shrink left margin

    out = os.path.join(folder, f"summary_T{T}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# -------------------------------------------------------------
# one figure (epoch25) stacking T vertically
# -------------------------------------------------------------
fig, axes = plt.subplots(len(temps), 1, figsize=(20, 8))

if len(temps) == 1:
    axes = [axes]

for i, T in enumerate(temps):
    im = load_png(25, T)
    axes[i].imshow(im, aspect="auto")
    axes[i].set_ylabel(str(T), rotation=0, ha="right", va="center",
                       labelpad=-10)

add_axis_close(fig, axes)
fig.supylabel("Timesteps")

fig.tight_layout(rect=(0.05, 0, 1, 1))

out = os.path.join(folder, "summary_epoch25.png")
fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
plt.close(fig)
