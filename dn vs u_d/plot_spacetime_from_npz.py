import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set matplotlib parameters to match main.py
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 14,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
    "axes.labelsize": 18,           # axis label text
    "xtick.labelsize": 16,          # x-tick labels
    "ytick.labelsize": 16,          # y-tick labels
})

# Load data
data_file = "out_ds_open_unstable_M07/data_m07_ud0p7000_ds_open_unstable_M07.npz"
data = np.load(data_file)

# Extract data
t = data['t']
n_t = data['n_t']
p_t = data['p_t']
L = float(data['L'])
Nx = int(data['Nx'])

# Get t_final from meta if available, otherwise use t.max()
try:
    meta = data['meta'].item()
    t_final = meta.get('t_final', t.max())
except:
    t_final = t.max()

# Create spatial grid
x = np.linspace(0.0, L, Nx, endpoint=False)
extent = [x.min(), x.max(), t.min(), t.max()]

# Create panel figure with 2 rows, 2 columns
fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.0))
ax1, ax2 = axes[0, 0], axes[0, 1]  # First row: spacetime plots
ax3, ax4 = axes[1, 0], axes[1, 1]  # Second row: snapshots

# Plot n(x,t) in first panel (top left)
im1 = ax1.imshow(n_t.T, origin="lower", aspect="auto",
                 extent=extent, cmap="inferno")
ax1.set_xlabel("$x$")
ax1.set_ylabel("$t$")
ax1.set_title(f"$n(x,t)$  [lab]")
cbar1 = plt.colorbar(im1, ax=ax1, label="$n$", fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=14)

# Plot p(x,t) in second panel (top right)
im2 = ax2.imshow(p_t.T, origin="lower", aspect="auto",
                 extent=extent, cmap="inferno")
ax2.set_xlabel("$x$")
ax2.set_ylabel("$t$")
ax2.set_title(f"$p(x,t)$  [lab]")
cbar2 = plt.colorbar(im2, ax=ax2, label="$p$", fraction=0.046, pad=0.04)
cbar2.ax.tick_params(labelsize=14)

# Plot n snapshots (bottom left)
percentages = [0, 20, 40, 60, 80, 100]
colors = plt.cm.tab10(np.linspace(0, 1, len(percentages)))
for i, pct in enumerate(percentages):
    idx = int((pct / 100) * (len(t) - 1))
    idx = min(idx, len(t) - 1)
    ax3.plot(x, n_t[:, idx], label=f"$t={t[idx]:.1f}/{t_final:.1f}$", 
            color=colors[i], linewidth=2)
ax3.legend()
ax3.set_xlabel("$x$")
ax3.set_ylabel("$n$")
ax3.set_title(f"Density snapshots")
ax3.set_xlim(0, L)

# Plot p snapshots (bottom right)
for i, pct in enumerate(percentages):
    idx = int((pct / 100) * (len(t) - 1))
    idx = min(idx, len(t) - 1)
    ax4.plot(x, p_t[:, idx], label=f"$t={t[idx]:.1f}/{t_final:.1f}$", 
            color=colors[i], linewidth=2)
ax4.legend()
ax4.set_xlabel("$x$")
ax4.set_ylabel("$p$")
ax4.set_title(f"Momentum snapshots")
ax4.set_xlim(0, L)

plt.tight_layout()

# Save figure
output_file = "spacetime_panel_from_data.png"
plt.savefig(output_file, dpi=300)
print(f"Saved spacetime panel plot → {os.path.abspath(output_file)}")
plt.show()
plt.close()

