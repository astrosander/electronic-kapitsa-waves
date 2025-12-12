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
    "font.size": 24,                 # Base font size (increased for publication)
    "axes.labelsize": 24,            # Axis labels (increased for visibility)
    "axes.titlesize": 24,
    "xtick.labelsize": 22,           # Tick labels (larger for better visibility)
    "ytick.labelsize": 22,           # Tick labels (larger for better visibility)
    "legend.fontsize": 24,           # Legend (increased)
    "axes.linewidth": 1.2,           # Thicker axes for visibility
    "xtick.major.width": 1.2,        # Thicker major ticks
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,        # Minor ticks
    "ytick.minor.width": 0.8,
    "xtick.major.size": 5,           # Longer major ticks
    "ytick.major.size": 5,
    "xtick.minor.size": 3,           # Minor tick size
    "ytick.minor.size": 3,
    "lines.linewidth": 1.7,
})

# --- Files ---
npz_files = sorted(glob.glob("npz/complete_m*.npz"))

# --- Figure geometry ---
nrows = len(npz_files)
# Publication-ready figure size (slightly wider for better readability)
fig, axes = plt.subplots(nrows, 1, sharex=True, figsize=(11, 2.3 * nrows))
if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]

# Publication-ready spacing: more room for labels and ticks
fig.subplots_adjust(left=0.10, right=0.95, top=0.98, bottom=0.12, hspace=0.08)

# --- Styles: Modern orange and blue ---
# Final selected pattern: optimistic orange (universal attractor)
final_color = "#FF9121"  # Bright, cheerful, optimistic orange (between FF8C42 and FF9500) - robust, selected, universal

# Initial conditions: modern blue (clearly secondary)
seed_color = "#3B82F6"  # Modern vibrant blue - transient, initial state

# Axes and labels: plain black or very dark gray
axis_color = "black"  # Standard black for axes/ticks/labels
grid_color = "#E0E0E0"  # Very light gray for gridlines

# Line weights and styles for visual hierarchy (publication-ready)
seed_lw, final_lw = 2.6, 2.6  # Final is thicker to emphasize selection (increased for print)
seed_alpha, final_alpha = 1.0, 1.0  # Both fully opaque
seed_linestyle = "-"  # Solid line for initial conditions
final_linestyle = "-"  # Solid line for final pattern

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
    # Plot initial seed: modern blue, thin, solid (clearly secondary)
    ax.plot(x, n[:, 0],
            color=seed_color, 
            linewidth=seed_lw, 
            alpha=seed_alpha,
            linestyle=seed_linestyle,
            solid_capstyle="round",
            zorder=2)
    
    # Plot final profile: modern orange, thick, solid (strongly emphasized - universal attractor)
    ax.plot(x, n[:, -1],
            color=final_color, 
            linewidth=final_lw, 
            alpha=final_alpha,
            linestyle=final_linestyle,
            solid_capstyle="round",
            zorder=4)
    
    # Add time labels in each panel (combined in one box with two lines)
    # Position in upper right corner
    x_pos = 0.87  # Near right edge in axes coordinates
    y_pos = 0.90  # Upper position in axes coordinates
    
    # Combined label with newline - two lines in one box
    combined_text = "t=0 s\nt=100 s"
    
    # Create background box first (with transparent text for sizing)
    ax.text(x_pos, y_pos, combined_text, fontsize=24, color='white', alpha=0.01,
            transform=ax.transAxes, ha="left", va="top", zorder=5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='none'))
    
    # Overlay colored text for each line, precisely aligned
    # Top line: t=0 s in blue
    ax.text(x_pos, y_pos, "t=0 s", fontsize=24, color=seed_color,
            transform=ax.transAxes, ha="left", va="top", zorder=6)
    
    # Bottom line: t=100 s in orange (positioned one line down)
    # Increased spacing to prevent overlap - for fontsize 20, need ~0.045-0.05
    line_spacing = 0.2
    ax.text(x_pos, y_pos - line_spacing, "t=100 s", fontsize=24, color=final_color,
            transform=ax.transAxes, ha="left", va="top", zorder=6)
    
    # Very light grid or none (minimal visual interference)
    ax.grid(axis="y", color=grid_color, alpha=0.2, linewidth=0.5)

    # Publication-ready ticks: larger, more visible
    ax.tick_params(which="major", direction="in", length=5, width=1.2, 
                   labelsize=22, colors=axis_color, pad=4)
    ax.tick_params(which="minor", direction="in", length=3, width=0.8, 
                   colors=axis_color)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    
    # Set axis spine colors and width to black (thicker for visibility)
    for spine in ax.spines.values():
        spine.set_color(axis_color)
        spine.set_linewidth(1.2)

    # Panel letter + mode formula in single box with black text - publication-ready
    letter = next(letters)
    label = mode_labels.get(mode_num, fr"Mode {mode_num}")
    combined_text = f"{letter} {label}"
    ax.text(0.01, 0.92, combined_text, transform=ax.transAxes,
            ha="left", va="top", color=axis_color, zorder=5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'), 
            fontsize=24, fontweight='normal')
    
    # Optionally harmonize y-lims across panels (comment out if not desired)
    # if idx == 0:
    #     ypad = 0.0001 * (np.nanmax(n) - np.nanmin(n))
    #     ylims = (np.nanmin(n) - ypad, np.nanmax(n) + ypad)
    # ax.set_ylim(*ylims)
    ax.set_ylim(0.11, 0.31)
    ax.set_xlim(*xlim)

# Shared labels in black - publication-ready sizes
axes[-1].set_xlabel(r"Distance $x$", fontsize=24, color=axis_color, 
                    fontweight='normal', labelpad=12)
fig.text(0.01, 0.55, "Modulation amplitude", rotation=90, va="center", 
         fontsize=24, color=axis_color, fontweight='normal')

# fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, 
#            borderpad=0.1, handlelength=2.0, columnspacing=1.5, handletextpad=0.5)

# Save vector + raster - publication quality
plt.savefig("figure3_panels.pdf", bbox_inches="tight", dpi=300, 
            metadata={"Creator": "Matplotlib"}, facecolor='white', edgecolor='none')
plt.savefig("figure3_panels.svg", bbox_inches="tight", facecolor='white', edgecolor='none')
plt.savefig("figure3_panels.png", dpi=600, bbox_inches="tight", 
            facecolor='white', edgecolor='none')  # High DPI for publication
plt.close()
