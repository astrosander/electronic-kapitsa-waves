import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator

img_scale = 1.3
mpl.rcParams.update({
    "figure.figsize": (3.4*img_scale, 2.6*img_scale),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.edgecolor": "#2C3E50",
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "mathtext.fontset": "stixsans",
})

mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})


hbar = 1.054_571_817e-34
kB   = 1.380_649e-23
eV   = 1.602_176_634e-19

g = 4
vF = 1.0e6
gamma1_eV = 0.39
gamma1 = gamma1_eV * eV

mstar = gamma1 / (2.0 * vF**2)

a = 1.0
C = 1.0*10

n_min = 1e9
n_max = 6e12
Npts  = 700

n_cm2 = np.linspace(n_min, n_max, Npts)
n_m2  = n_cm2 * 1e4

eps_F = np.pi * hbar**2 * n_m2 / (g * mstar)
mu = eps_F

eta = 2.0 * eps_F / gamma1

def gamma_ee(T, mu):
    x = kB * T
    return (x / hbar) * (x / (x + a * mu))

def S_col(T, mu):
    x = kB * T
    mu_safe = np.maximum(mu, 1e-40)
    return np.minimum(1.0, C * (x / mu_safe)**2)

def Gamma_intra_corrected(T, mu, eta):
    return gamma_ee(T, mu) * (eta**2) * S_col(T, mu)

def Gamma_intra_uncorrected(T, mu, eta):
    return gamma_ee(T, mu) * (eta**2)

def Gamma_inter(T, mu):
    x = kB * T
    return (x / hbar) * np.exp(-mu / np.maximum(x, 1e-40))

temperatures = np.array([20, 30, 40, 50, 70, 90, 110, 140, 170, 200, 250, 300])

INCLUDE_INTERBAND = True
USE_COLORBAR = True

cmap = plt.get_cmap("plasma")
norm = mpl.colors.Normalize(vmin=temperatures.min(), vmax=temperatures.max())

UNCORRECTED_ALPHA = 0.5
UNCORRECTED_LINEWIDTH = 1.0

def lighten_color(color, factor=0.6):
    if isinstance(color, str):
        color_rgb = np.array(mpl.colors.to_rgb(color))
    else:
        color_array = np.array(color)
        if len(color_array) == 4:
            color_rgb = color_array[:3]
        else:
            color_rgb = color_array
    
    white = np.array([1.0, 1.0, 1.0])
    lightened = color_rgb + (white - color_rgb) * (1 - factor)
    return tuple(lightened)

fig, ax = plt.subplots(figsize=(3.4*img_scale, 2.6*img_scale))
ax.set_facecolor("#FAFAFA")

y_min = np.inf
y_max = -np.inf

for T in temperatures:
    color = cmap(norm(T))

    Gintra_c = Gamma_intra_corrected(T, mu, eta)
    Gintra_u = Gamma_intra_uncorrected(T, mu, eta)

    if INCLUDE_INTERBAND:
        Gtot_c = Gintra_c + Gamma_inter(T, mu)
        Gtot_u = Gintra_u + Gamma_inter(T, mu)
    else:
        Gtot_c = Gintra_c
        Gtot_u = Gintra_u

    y_min = min(y_min, np.nanmin(Gtot_c), np.nanmin(Gtot_u))
    y_max = max(y_max, np.nanmax(Gtot_c), np.nanmax(Gtot_u))

    ax.loglog(n_cm2, Gtot_c, color=color, linewidth=1.4, alpha=0.9)

    light_color = lighten_color(color, factor=0.5)
    ax.loglog(n_cm2, Gtot_u, color=light_color, 
                ls="--", alpha=UNCORRECTED_ALPHA, linewidth=UNCORRECTED_LINEWIDTH)

ax.set_xlim(n_min, n_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel(r"density $n$ (cm$^{-2}$)", color="#2C3E50")
ax.set_ylabel(r"current relaxation rate $\Gamma_J$ (s$^{-1}$)", color="#2C3E50")

ax.minorticks_on()

# Set log scale formatters for both axes
ax.xaxis.set_major_formatter(LogFormatter())
ax.yaxis.set_major_formatter(LogFormatter())
ax.xaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.4)))
ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.4)))

ax.tick_params(colors="#2C3E50")

if USE_COLORBAR:
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("$T$ (K)", color="#2C3E50")
    cbar.ax.tick_params(colors="#2C3E50")
    cbar.outline.set_edgecolor("#2C3E50")
    cbar.outline.set_linewidth(0.8)
else:
    for T in temperatures[::2]:
        ax.plot([], [], color=cmap(norm(T)), label=f"{T} K")
    ax.legend(frameon=False, loc="best", ncols=2)

fig.tight_layout(pad=0.4)

fig.savefig("GammaJ_vs_n_BLG_PRB.pdf", dpi=300)
fig.savefig("GammaJ_vs_n_BLG_PRB.png", dpi=300)

plt.show()
