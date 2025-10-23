import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import matplotlib as mpl


# Publication-ready settings
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "serif",        # serif font for publication
    "font.size": 12,               # smaller font for publication
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "lines.linewidth": 1.2,
})

# Get all NPZ files in the npz directory
npz_files = glob.glob("npz/complete_m*.npz")
npz_files.sort()  # Sort to ensure consistent order

print(f"Found {len(npz_files)} NPZ files:")
for f in npz_files:
    print(f"  {f}")

# Create a combined plot for all modes
fig, axes = plt.subplots(
    len(npz_files), 1, sharex=True,
    figsize=(8, 2.2 * len(npz_files))
)

# Adjust subplot parameters to provide space for ylabel
fig.subplots_adjust(left=0.15, bottom=0.1, top=0.95, right=0.95, hspace=0.1)
if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]

# Mode labels for legend positioning
mode_labels = {
    1: r"$\cos(3x) + \cos(5x)$",
    2: r"$\cos(5x) + \cos(8x)$", 
    3: r"$\cos(8x) + \cos(15x)$",
    4: r"$\cos(7x) + \cos(13x)$",
    5: r"$\cos(21x) + \cos(34x)$",
    6: r"$\cos(34x) + \cos(55x)$"
}

for panel_idx, (ax, fn) in enumerate(zip(axes, npz_files)):
    print(f"\nProcessing {fn}...")
    
    # Load the .npz file
    d = np.load(fn, allow_pickle=True)
    
    if "n" in d.files and "x" in d.files and "t" in d.files:
        n = d["n"]
        x = d["x"]
        t = d["t"]
        
        # Extract mode number from filename
        mode_num = int(fn.split('_m')[1].split('_')[0])
        
        # Create snapshots at different time points with publication colors
        colors = ['#1f77b4', '#d62728']  # Professional blue and red
        for i, frac in enumerate([0.0, 1.0]):
            j = int(frac*(len(t)-1))
            if i == 0:  # First line (t=0.0) - add mode label to legend
                ax.plot(x, n[:, j], linewidth=1.5, color=colors[i], 
                       label=f"$t={t[j]:.1f}$", alpha=0.9)
            else:  # Second line (t=final) - just time label
                ax.plot(x, n[:, j], linewidth=1.5, color=colors[i], 
                       label=f"$t={t[j]:.1f}$", alpha=0.9)

        # Show legend only on the first panel to avoid repetition
        if panel_idx == 0:
            ax.legend(loc="upper right", frameon=True, fancybox=False, shadow=False, fontsize=14)
        
        # Add mode label as text annotation on each panel with blue color to match t=0 line
        if mode_num in mode_labels:
            ax.text(0.02, 0.95, mode_labels[mode_num], transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', color='#1f77b4',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax.text(0.02, 0.95, f"Mode {mode_num}", transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', color='#1f77b4',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
        ax.grid(True, alpha=0.2, linewidth=0.3)
        ax.tick_params(which='both', width=0.5, length=2, labelsize=9)
        ax.set_xlim(0, 10)
    else:
        print(f"  Required fields (n, x, t) not found in {fn}")

# Set labels for the figure
axes[-1].set_xlabel("Distance $x$", fontsize=16)
# Add single ylabel for all panels with less padding
fig.text(0.06, 0.5, "Modulation amplitude", fontsize=16, 
         rotation=90, verticalalignment='center')

# Improve overall appearance
# plt.tight_layout()  # Commented out since we're using subplots_adjust
plt.savefig("snapshots_panels_m1to6.png", dpi=300, bbox_inches='tight')
plt.savefig("snapshots_panels_m1to6.svg", dpi=300, bbox_inches='tight')
# plt.show()
plt.close()
