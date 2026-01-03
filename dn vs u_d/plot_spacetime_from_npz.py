import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider
from matplotlib.widgets import Slider

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


# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams["legend.frameon"] = False
# # Publication-ready font sizes
# plt.rcParams['font.size'] = 26
# plt.rcParams['axes.labelsize'] = 26
# plt.rcParams['axes.titlesize'] = 26
# plt.rcParams['xtick.labelsize'] = 26
# plt.rcParams['ytick.labelsize'] = 26
# plt.rcParams['legend.fontsize'] = 26
# plt.rcParams['figure.titlesize'] = 26


# Load data
data_file = r"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p02\out_drift_ud0p5000\data_m07_ud0p5000_ud0.5.npz"
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
ax1.axvline(2.5, color='white', linestyle='--', linewidth=2, alpha=1.0)
ax1.axvline(7.5, color='white', linestyle='--', linewidth=2, alpha=1.0)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$t$")
ax1.set_title(f"$n(x,t)$  [lab]")
cbar1 = plt.colorbar(im1, ax=ax1, label="$n$", fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=26)

# Plot p(x,t) in second panel (top right)
im2 = ax2.imshow(p_t.T, origin="lower", aspect="auto",
                 extent=extent, cmap="inferno")
ax2.axvline(2.5, color='white', linestyle='--', linewidth=2, alpha=1.0)
ax2.axvline(7.5, color='white', linestyle='--', linewidth=2, alpha=1.0)
ax2.set_xlabel("$x$")
ax2.set_ylabel("$t$")
ax2.set_title(f"$p(x,t)$  [lab]")
cbar2 = plt.colorbar(im2, ax=ax2, label="$p$", fraction=0.046, pad=0.04)
cbar2.ax.tick_params(labelsize=26)

# Plot n snapshots (bottom left)
percentages = [100]
colors = plt.cm.tab10(np.linspace(0, 1, len(percentages)))
lines_n = []
for i, pct in enumerate(percentages):
    idx = int((pct / 100) * (len(t) - 1))
    idx = min(idx, len(t) - 1)
    line_n, = ax3.plot(x, n_t[:, idx], label=f"$t={t[idx]:.0f}/{t_final:.0f}$", 
            color=colors[i], linewidth=2)
    lines_n.append(line_n)
ax3.axvline(2.5, color='black', linestyle='--', linewidth=2, alpha=0.9)
ax3.axvline(7.5, color='black', linestyle='--', linewidth=2, alpha=0.9)
ax3.legend()
ax3.set_xlabel("$x$")
ax3.set_ylabel("$n$")
ax3.set_title(f"Density snapshots")
ax3.set_xlim(0, L)

# Plot p snapshots (bottom right)
lines_p = []
for i, pct in enumerate(percentages):
    idx = int((pct / 100) * (len(t) - 1))
    idx = min(idx, len(t) - 1)
    line_p, = ax4.plot(x, p_t[:, idx], label=f"$t={t[idx]:.0f}/{t_final:.0f}$", 
            color=colors[i], linewidth=2)
    lines_p.append(line_p)
ax4.axvline(2.5, color='black', linestyle='--', linewidth=2, alpha=0.9)
ax4.axvline(7.5, color='black', linestyle='--', linewidth=2, alpha=0.9)
ax4.legend()
ax4.set_xlabel("$x$")
ax4.set_ylabel("$p$")
ax4.set_title(f"Momentum snapshots")
ax4.set_xlim(0, L)

plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3)

# Save figure
output_file = "spacetime_panel_from_data.png"
plt.savefig(output_file, dpi=300)
print(f"Saved spacetime panel plot → {os.path.abspath(output_file)}")

# Create separate window for slider
fig_slider = plt.figure(figsize=(8, 0.5))
fig_slider.patch.set_facecolor('white')
fig_slider.canvas.manager.set_window_title('')
try:
    toolbar = fig_slider.canvas.manager.toolbar
    if hasattr(toolbar, 'hide'):
        toolbar.hide()
    elif hasattr(toolbar, 'pack_forget'):
        toolbar.pack_forget()
except:
    pass
ax_slider = plt.axes([0.07, 0.0, 1, 1])
ax_slider.set_facecolor('white')
ax_slider.axis('off')
slider = Slider(ax_slider, '%', 0, 100, valinit=100, valstep=0.1)

def update_snapshots(val):
    pct = slider.val
    idx = int((pct / 100) * (len(t) - 1))
    idx = min(idx, len(t) - 1)
    
    # Update n snapshot
    lines_n[0].set_ydata(n_t[:, idx])
    lines_n[0].set_label(f"$t={t[idx]:.1f}/{t_final:.1f}$")
    ax3.legend()
    
    # Update p snapshot
    lines_p[0].set_ydata(p_t[:, idx])
    lines_p[0].set_label(f"$t={t[idx]:.1f}/{t_final:.1f}$")
    ax4.legend()
    
    fig.canvas.draw_idle()

slider.on_changed(update_snapshots)

plt.show()
plt.close()

