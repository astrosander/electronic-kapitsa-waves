import numpy as np
import matplotlib.pyplot as plt
import glob, os
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, MultipleLocator
from itertools import cycle

# --- Helpers ---
def estimate_k_star(x, y):
    """Estimate k* from FFT peak of the final profile"""
    x = np.asarray(x); y = np.asarray(y)
    y = y - np.mean(y)  # detrend
    L = x[-1] - x[0]
    Y = np.fft.rfft(y)
    ks = 2*np.pi*np.fft.rfftfreq(len(x), d=(x[1]-x[0]))
    peak = np.argmax(np.abs(Y[1:])) + 1  # exclude k=0
    return ks[peak] if peak < len(ks) else ks[1]

def tint(color, amount=0.25):
    """Tint a color toward white to create a lighter version (less than before to keep visibility)"""
    r, g, b = mcolors.to_rgb(color)
    return (1 - amount) * r + amount, (1 - amount) * g + amount, (1 - amount) * b + amount

# --- Publication rcParams ---
mpl.rcParams.update({
    "text.usetex": False,            # Use MathText for portability
    "font.family": "serif",          # Serif font like LaTeX
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",        # Computer Modern math font
    "mathtext.rm": "serif",          # Math text in serif
    "axes.unicode_minus": False,     # Proper minus sign
    "font.size": 18,                 # base
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 18,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "lines.linewidth": 1.7,
})

# --- Files ---
npz_files = sorted(glob.glob("npz/complete_m*.npz"))

# --- Figure geometry ---
nrows = len(npz_files)
fig, axes = plt.subplots(nrows, 1, sharex=True, figsize=(10.5, 2.2 * nrows))
if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]

# Larger plot area with more space for panels
fig.subplots_adjust(left=0.08, right=0.95, top=0.964, bottom=0.10, hspace=0.05)

# --- Styles: 1 strong color + 1 muted neutral ---
# Final selected pattern: strong accent color (universal attractor)
final_color = "#009E73"  # Teal/bluish green - robust, selected, universal (colorblind-safe)

# Initial conditions: muted neutral gray (clearly secondary)
seed_color = "#999999"  # Medium gray - transient, unimportant

# Axes and labels: plain black or very dark gray
axis_color = "black"  # Standard black for axes/ticks/labels
grid_color = "#E0E0E0"  # Very light gray for gridlines

# Line weights and styles for visual hierarchy
seed_lw, final_lw = 1.4, 2.4  # Final is thicker to emphasize selection
seed_alpha, final_alpha = 1.0, 1.0  # Both fully opaque
seed_linestyle = "--"  # Dashed for initial conditions (grayscale-safe)
final_linestyle = "-"  # Solid for final pattern

# Legend labels (clear semantics)
label_init = "initial condition"
label_final = "long-time state"

# Mode labels for panel text
mode_labels = {
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: "",
}

# Optional: fix common xlim or derive from data
xlim = (0, 10)

# Panel letters
letters = cycle(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"])

for idx, (ax, fn) in enumerate(zip(axes, npz_files)):
    d = np.load(fn, allow_pickle=True)
    if not all(k in d.files for k in ("n", "x", "t")):
        continue

    n, x, t = d["n"], d["x"], d["t"]
    mode_num = int(fn.split('_m')[1].split('_')[0])

    # Two snapshots: t=0 and t=final with clear visual hierarchy
    # Plot initial seed: muted gray, thin, dashed (clearly secondary)
    ax.plot(x, n[:, 0],
            color=seed_color, 
            linewidth=seed_lw, 
            alpha=seed_alpha,
            linestyle=seed_linestyle,
            solid_capstyle="round",
            label=label_init if idx == 0 else None, 
            zorder=2)
    
    # Plot final profile: strong teal, thick, solid (strongly emphasized - universal attractor)
    ax.plot(x, n[:, -1],
            color=final_color, 
            linewidth=final_lw, 
            alpha=final_alpha,
            linestyle=final_linestyle,
            solid_capstyle="round",
            label=label_final if idx == 0 else None, 
            zorder=4)
    
    # Very light grid or none (minimal visual interference)
    ax.grid(axis="y", color=grid_color, alpha=0.2, linewidth=0.5)

    # Inward ticks; modest number of ticks
    ax.tick_params(which="both", direction="in", length=3, colors=axis_color)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    
    # Set axis spine colors to black
    for spine in ax.spines.values():
        spine.set_color(axis_color)

    # Panel letter + mode formula in single box with black text
    letter = next(letters)
    label = mode_labels.get(mode_num, fr"Mode {mode_num}")
    combined_text = f"{letter} {label}"
    ax.text(0.01, 0.92, combined_text, transform=ax.transAxes,
            ha="left", va="top", color=axis_color, zorder=5,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6, edgecolor='none'), fontsize=24)

    # Optionally harmonize y-lims across panels (comment out if not desired)
    # if idx == 0:
    #     ypad = 0.0001 * (np.nanmax(n) - np.nanmin(n))
    #     ylims = (np.nanmin(n) - ypad, np.nanmax(n) + ypad)
    # ax.set_ylim(*ylims)

    ax.set_xlim(*xlim)

# Shared labels in black
axes[-1].set_xlabel(r"Distance $x$", fontsize=22, color=axis_color)
fig.text(0.02, 0.5, "Modulation amplitude", rotation=90, va="center", fontsize=22, color=axis_color)

# LaTeX-style legend at top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, borderpad=0.2, handlelength=2.5)

# fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, 
#            borderpad=0.1, handlelength=2.0, columnspacing=1.5, handletextpad=0.5)

# Save vector + raster
plt.savefig("figure3_panels.pdf", bbox_inches="tight", metadata={"Creator": "Matplotlib"})
plt.savefig("figure3_panels.svg", bbox_inches="tight")
plt.savefig("figure3_panels.png", dpi=450, bbox_inches="tight")
plt.close()
