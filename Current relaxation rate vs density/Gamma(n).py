import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator, FuncFormatter

img_scale = 1.3
mpl.rcParams.update({
    "figure.figsize": (3.4*img_scale, 2.6*img_scale),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "axes.edgecolor": "#2C3E50",
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "mathtext.fontset": "stixsans",
})

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


hbar = 1.054_571_817e-34
kB   = 1.380_649e-23
eV   = 1.602_176_634e-19

g = 4
vF = 1.0e6
gamma1_eV = 0.39
gamma1 = gamma1_eV * eV

mstar = gamma1 / (2.0 * vF**2)

a = 1
C = 1.0*10

n_min = 1e9
n_max = 1e14
Npts  = 700000

n_cm2 = np.linspace(n_min, n_max, Npts)
n_m2  = n_cm2 * 1e4

eps_F = np.pi * hbar**2 * n_m2 / (g * mstar)
mu = eps_F

def gamma_ee(T, mu):
    x = kB * T
    mu_shifted = mu + x / a
    return (x / hbar) * (x / (x + a * mu_shifted))

def S_col(T, mu):
    x = kB * T
    mu_shifted = mu + x / a
    mu_safe = np.maximum(mu_shifted, 1e-40)
    return C * (x / mu_safe)**2

def Gamma_intra_corrected(T, mu, eps_F):
    x = kB * T
    eps_F_shifted = eps_F + x / a
    eta = 2.0 * eps_F_shifted / gamma1
    return gamma_ee(T, mu) * (eta**2) * S_col(T, mu)

def Gamma_intra_uncorrected(T, mu, eps_F):
    x = kB * T
    eps_F_shifted = eps_F + x / a
    eta = 2.0 * eps_F_shifted / gamma1
    return gamma_ee(T, mu) * (eta**2)

def Gamma_inter(T, mu):
    x = kB * T
    mu_shifted = mu + x / a
    return (x / hbar) * np.exp(-mu_shifted / np.maximum(x, 1e-40))

# temperatures = np.array([20, 30, 40, 50, 70, 90, 110, 140, 170, 200, 250, 300])
temperatures = np.array([20, 30, 40, 50, 70, 90, 110, 140, 170, 200, 250, 300])
temperatures = temperatures/10
temperatures = np.array([2, 4, 8, 16, 32])
INCLUDE_INTERBAND = False

from matplotlib.colors import LogNorm
cmap = plt.get_cmap("plasma")
norm = LogNorm(vmin=temperatures.min(), vmax=temperatures.max())

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

fig, ax = plt.subplots(figsize=(2.6*img_scale, 3.4*img_scale))
ax.set_facecolor("#FAFAFA")

y_min = np.inf
y_max = -np.inf

for T in temperatures:
    norm_val = np.clip(norm(T), 0, 1)
    color_val = 0.1 + norm_val * 0.6
    color = cmap(color_val)

    Gintra_c = Gamma_intra_corrected(T, mu, eps_F)
    Gintra_u = Gamma_intra_uncorrected(T, mu, eps_F)

    if INCLUDE_INTERBAND:
        Gtot_c = Gintra_c + Gamma_inter(T, mu)
        Gtot_u = Gintra_u + Gamma_inter(T, mu)
    else:
        Gtot_c = Gintra_c
        Gtot_u = Gintra_u

    y_min = min(y_min, np.nanmin(Gtot_c))
    y_max = max(y_max, np.nanmax(Gtot_c))

    ax.loglog(n_cm2, Gtot_c, color=color, linewidth=1.8, alpha=0.95)
    
    x_label = n_max*0.4
    y_label = Gtot_c[-1]*1.5
    ax.text(x_label, y_label, f'{T} K', color=color, fontsize=12, 
            ha='right', va='center', alpha=0.95, weight='medium')

    light_color = lighten_color(color, factor=0.5)
    # ax.loglog(n_cm2, Gtot_u, color=light_color, 
    #             ls="--", alpha=UNCORRECTED_ALPHA, linewidth=UNCORRECTED_LINEWIDTH)

x_min = n_min
x_max = n_max

ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)
ax.set_ylim(100, 1e11)

ax.set_xlabel(r"density $n$ (cm$^{-2}$)", color="#2C3E50", fontsize=10)
ax.set_ylabel(r"current relaxation rate $\Gamma_J$ (s$^{-1}$)", color="#2C3E50", fontsize=10)

ax.minorticks_on()

def format_power_of_10(x, pos):
    if x <= 0:
        return ''
    exp = int(np.log10(x))
    return r'$10^{' + str(exp) + r'}$'

ax.xaxis.set_major_formatter(FuncFormatter(format_power_of_10))
ax.yaxis.set_major_formatter(FuncFormatter(format_power_of_10))
ax.xaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.4)))
ax.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.4)))

ax.tick_params(colors="#2C3E50")

fig.tight_layout(pad=0.5)

fig.savefig("GammaJ_vs_n_BLG_PRB.pdf", dpi=600, bbox_inches='tight', pad_inches=0.05)
fig.savefig("GammaJ_vs_n_BLG_PRB.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)
fig.savefig("GammaJ_vs_n_BLG_PRB.png", dpi=600, bbox_inches='tight', pad_inches=0.05)

plt.show()
