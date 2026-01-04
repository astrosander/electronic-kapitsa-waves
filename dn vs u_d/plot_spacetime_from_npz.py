import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, Button

# Set matplotlib parameters to match main.py
# mpl.rcParams.update({
#     "text.usetex": False,          # use MathText (portable)
#     "font.family": "STIXGeneral",  # match math fonts
#     "font.size": 14,
#     "mathtext.fontset": "stix",
#     "axes.unicode_minus": False,   # proper minus sign
#     "axes.labelsize": 18,           # axis label text
#     "xtick.labelsize": 16,          # x-tick labels
#     "ytick.labelsize": 16,          # y-tick labels
# })


# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams["legend.frameon"] = False
# # Publication-ready font sizes
# plt.rcParams['font.size'] = 28
# plt.rcParams['axes.labelsize'] = 28
# plt.rcParams['axes.titlesize'] = 28
# plt.rcParams['xtick.labelsize'] = 28
# plt.rcParams['ytick.labelsize'] = 28
# plt.rcParams['legend.fontsize'] = 28
# plt.rcParams['figure.titlesize'] = 28


# Load data
data_file = r"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\last\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\last\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\last\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=2.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p00\out_drift_ud0p0000\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=2.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p00\out_drift_ud0p0000\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p00\out_drift_ud0p0000\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p00\out_drift_ud0p0000\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p00\out_drift_ud0p0000\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p00\out_drift_ud0p0000\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p00\out_drift_ud0p0000\data_m07_ud0p0000_ud0.npz""D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p00\out_drift_ud0p0000\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p00\out_drift_ud0p0000\data_m07_ud0p0000_ud0.npz"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p01\out_drift_ud0p5000\data_m07_ud0p5000_ud0.5.npz"

# data_file = r"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.14_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)_Dn=0p00_Dp=0p01\out_drift_ud0p5000\data_m07_ud0p5000_ud0.5.npz"
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
fig, axes = plt.subplots(2, 2, figsize=(19.2, 19.2/11.69*8.27))
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
cbar1.ax.tick_params(labelsize=28)

# Plot p(x,t) in second panel (top right)
im2 = ax2.imshow(p_t.T, origin="lower", aspect="auto",
                 extent=extent, cmap="inferno")
ax2.axvline(2.5, color='white', linestyle='--', linewidth=2, alpha=1.0)
ax2.axvline(7.5, color='white', linestyle='--', linewidth=2, alpha=1.0)
ax2.set_xlabel("$x$")
ax2.set_ylabel("$t$")
ax2.set_title(f"$p(x,t)$  [lab]")
cbar2 = plt.colorbar(im2, ax=ax2, label="$p$", fraction=0.046, pad=0.04)
cbar2.ax.tick_params(labelsize=28)

# Plot n snapshots (bottom left)
percentages = [0.2, 100]
colors = ["blue", "red"]#plt.cm.tab10(np.linspace(0, 1, len(percentages)))
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

# Calculate step size for one index
nt = len(t)
pct_per_index = 100.0 / (nt - 1) if nt > 1 else 100.0

# Create slider with adjusted step
ax_slider = plt.axes([0.15, 0.0, 0.7, 1])
ax_slider.set_facecolor('white')
ax_slider.axis('off')
slider = Slider(ax_slider, '%', 0, 100, valinit=100, valstep=pct_per_index)

# Create arrow buttons
ax_prev = plt.axes([0.02, 0.0, 0.1, 1])
ax_prev.set_facecolor('white')
ax_prev.axis('off')
btn_prev = Button(ax_prev, '◀', color='white', hovercolor='lightgray')

ax_next = plt.axes([0.87, 0.0, 0.1, 1])
ax_next.set_facecolor('white')
ax_next.axis('off')
btn_next = Button(ax_next, '▶', color='white', hovercolor='lightgray')

def update_snapshots(val):
    pct = slider.val
    idx = int((pct / 100) * (nt - 1))
    idx = min(max(idx, 0), nt - 1)
    
    # Update n snapshot
    lines_n[0].set_ydata(n_t[:, idx])
    lines_n[0].set_label(f"$t={t[idx]:.1f}/{t_final:.1f}$")
    ax3.legend()
    
    # Update p snapshot
    lines_p[0].set_ydata(p_t[:, idx])
    lines_p[0].set_label(f"$t={t[idx]:.1f}/{t_final:.1f}$")
    ax4.legend()
    
    fig.canvas.draw_idle()

def step_prev(event):
    current_idx = int((slider.val / 100) * (nt - 1))
    current_idx = min(max(current_idx, 0), nt - 1)
    new_idx = max(0, current_idx - 1)
    new_pct = (new_idx / (nt - 1)) * 100.0 if nt > 1 else 0.0
    slider.set_val(new_pct)

def step_next(event):
    current_idx = int((slider.val / 100) * (nt - 1))
    current_idx = min(max(current_idx, 0), nt - 1)
    new_idx = min(nt - 1, current_idx + 1)
    new_pct = (new_idx / (nt - 1)) * 100.0 if nt > 1 else 100.0
    slider.set_val(new_pct)

slider.on_changed(update_snapshots)
btn_prev.on_clicked(step_prev)
btn_next.on_clicked(step_next)

plt.show()
plt.close()

