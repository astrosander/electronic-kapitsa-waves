import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

# --- Publication-ready appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "serif",        # serif font for publication
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 11,               # base font size
    "axes.labelsize": 12,          # axis labels slightly larger
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "mathtext.fontset": "stix",    # STIX math fonts
    "axes.unicode_minus": False,   # proper minus sign
    "axes.linewidth": 0.8,         # thinner axes
    "xtick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.major.size": 4,
    "ytick.minor.size": 2,
    "xtick.direction": "in",       # ticks inside
    "ytick.direction": "in",
    "xtick.top": True,             # ticks on all sides
    "ytick.right": True,
    "lines.linewidth": 1.8,        # thicker lines
    "legend.frameon": True,
    "legend.fancybox": False,      # rectangular legend
    "legend.framealpha": 0.95,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "figure.dpi": 300,             # high resolution
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# --- constants (SI) ---
hbar = 1.054571817e-34   # J·s
m_e  = 9.1093837e-31     # kg
pi   = np.pi
cm2_to_m2 = 1e4

# --- model / material params (BLG) ---
g = 4
m = 0.04 * m_e

# --- densities (entered in cm^-2, converted to m^-2) ---
n_min_cm2, n_max_cm2 = 0.5e10, 10.0e10
n_vals_cm2 = np.linspace(n_min_cm2, n_max_cm2, 800)
n_vals = n_vals_cm2 * cm2_to_m2

# --- w (entered in cm^-2, converted to m^-2) ---
w_cm2 = 3.0e10
w = w_cm2 * cm2_to_m2

# --- coupling U ≈ 2/nu (TF screened, k≈0 constant) ---
nu = g * m / (2 * pi * hbar**2)   # 1/(J·m^2)
U_factor = 1.0
U  = U_factor * (2.0 / nu)        # J·m^2

# --- instability boundary: u_c(n) = w * sqrt(U / (m * n)) ---
u_c = w * np.sqrt(U / (m * n_vals))

# --- Fermi velocity: v_F = (hbar/m) * sqrt(4*pi*n/g) ---
kF  = np.sqrt(4 * pi * n_vals / g)
v_F = (hbar * kF) / m

# --- crossover where u_c = v_F ---
idx_star = np.argmin(np.abs(u_c - v_F))
n_star_cm2 = n_vals_cm2[idx_star]
u_star = 0.5 * (u_c[idx_star] + v_F[idx_star])
n_star_display = n_star_cm2 / 1e10  # for annotation formatting

# --- plot ---
# Journal standard: single column (3.5 inch) or double column (7 inch) width
fig, ax = plt.subplots(figsize=(6.0, 4.5))  # Standard journal width

# Beautiful, optimistic, yet professional color scheme (colorblind-friendly)
# Inspired by Tol Bright and scientific color palettes
stable_color = '#00A693'      # vibrant emerald/teal (optimistic, growth)
unstable_color = '#FF6B8A'    # soft coral pink (gentle warning)
reachable_color = '#FFA726'   # warm amber/gold (opportunity, accessible)
vF_color = '#5C88DA'          # vibrant sky blue (clear, professional)

# curves (journal standard line widths)
ax.plot(u_c, n_vals_cm2, 'k-',  lw=2.0,
        label=r'$u_{\mathrm{c}}(n)$', zorder=4)
ax.plot(v_F, n_vals_cm2, color=vF_color, linestyle='--', lw=2.0,
        label=r'$v_{\mathrm{F}}(n)$', zorder=4)

# shading (enhanced to showcase beautiful colors while maintaining clarity)
u_max_display = 1.5e5  # Fixed upper limit for display
ax.fill_betweenx(n_vals_cm2, 0, u_c, color=stable_color, alpha=0.16,
                 label=r'$\mathrm{Stable}$', zorder=1)
ax.fill_betweenx(n_vals_cm2, u_c, u_max_display,
                 color=unstable_color, alpha=0.14, 
                 label=r'$\mathrm{Unstable}$', zorder=1)
u_right = np.minimum(v_F, u_max_display)
ax.fill_betweenx(n_vals_cm2, u_c, u_right, where=(v_F > u_c),
                 color=reachable_color, alpha=0.20,
                zorder=1)

# crossover marker & annotation (journal style: minimal, clear)
ax.plot([u_star], [n_star_cm2], 'ko', ms=6, zorder=10, 
        markeredgewidth=1.2, markeredgecolor='white')
ax.annotate(
    (r'$u_{\mathrm{c}} = v_{\mathrm{F}}$' + '\n' + 
     r'$n = {:.1f} \times 10^{{10}}\ \mathrm{{cm}}^{{-2}}$'.format(n_star_display)),
    xy=(u_star, n_star_cm2), xytext=(0.65, 0.52), textcoords='axes fraction',
    arrowprops=dict(arrowstyle='->', lw=1.0, color='black',
                    connectionstyle='arc3,rad=0.1', shrinkA=3, shrinkB=3), 
    fontsize=9, va='center', ha='left', zorder=10,
    bbox=dict(boxstyle='round,pad=0.35', facecolor='white', 
              edgecolor='black', alpha=0.9, linewidth=0.6)
)

# axes, labels, legend
ax.set_xlim(0, 1.5e5)  # Fixed range: 0 to 1.5 × 10^5 m/s
ax.set_ylim(n_min_cm2, n_max_cm2)

# X-axis formatter: scale by 10^5 (journal style: numbers only, multiplier in label)
def x_formatter(x, pos):
    value = x / 1e5
    # Show integer if close to integer, otherwise 1 decimal
    if abs(value - round(value)) < 0.01:
        return f'${int(round(value))}$'
    else:
        return f'${value:.1f}$'

ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
ax.set_xlabel(r'$u\ (10^5\ \mathrm{m/s})$', fontsize=13, labelpad=6)
ax.set_ylabel(r'$n\ \mathrm{(cm^{-2})}$', fontsize=13, labelpad=8)

# Grid (minimal for journal clarity)
ax.grid(True, alpha=0.2, linewidth=0.5, linestyle='-', which='major')
ax.set_axisbelow(True)

# Legend (journal standard: clean, minimal)
ax.legend(loc='upper left', fontsize=9, framealpha=0.98,
          fancybox=False, shadow=False, edgecolor='black', 
          facecolor='white', frameon=True, borderpad=0.6,
          columnspacing=0.8, handlelength=2.0, handletextpad=0.5)

# Ensure all spines are visible
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout(pad=1.0)
plt.savefig('blg_phase_diagram.svg', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('blg_phase_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('blg_phase_diagram.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
# plt.show()

# quick readout
n_spec_cm2 = 2.0e10
n_spec = n_spec_cm2 * cm2_to_m2
u_spec = w * np.sqrt(U / (m * n_spec))
vF_spec = (hbar/m) * np.sqrt(4*pi*n_spec/g)
print(f"w = {w_cm2:.2e} cm^-2 | u_c(2e10 cm^-2) = {u_spec:.3e} m/s | v_F(2e10 cm^-2) = {vF_spec:.3e} m/s")
