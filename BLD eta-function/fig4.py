import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator, FuncFormatter
from matplotlib.colors import LogNorm

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

hbar = 1.054571817e-34
kB   = 1.380649e-23
eV   = 1.602176634e-19

vF     = 1.0e6
gamma1 = 0.39 * eV
g      = 4

mstar = gamma1 / (2 * vF**2)
U     = 2 * np.pi * hbar**2 / (g * mstar)

a_interp = 1.0

def kF_from_n(n_m2: np.ndarray) -> np.ndarray:
    return np.sqrt(4 * np.pi * n_m2 / g)

def eps_low_band(k: np.ndarray) -> np.ndarray:
    return 0.5 * (np.sqrt(gamma1**2 + 4 * (hbar * vF * k)**2) - gamma1)

def v_group(k: np.ndarray) -> np.ndarray:
    return (2 * hbar * vF**2 * k) / np.sqrt(gamma1**2 + 4 * (hbar * vF * k)**2)

def eta_of_n(n_m2: np.ndarray) -> np.ndarray:
    kF = kF_from_n(n_m2)
    v  = v_group(kF)
    v_gal = (hbar * kF) / mstar
    chi = v - v_gal
    return np.abs(chi) / np.maximum(np.abs(v), 1e-30)

def gamma_ee(T: float, mu: np.ndarray, a: float = 1.0) -> np.ndarray:
    x = kB * T
    return (x / hbar) * (x / (x + a * mu))

def gamma_total(n_m2: np.ndarray, T: float, a: float = 1.0) -> np.ndarray:
    mu = eps_low_band(kF_from_n(n_m2))
    eta = eta_of_n(n_m2)
    return gamma_ee(T, mu, a=a) * eta**2

def u0_of_n(n_m2: np.ndarray) -> np.ndarray:
    return np.sqrt(U * n_m2 / mstar)

temps = [15, 30, 50, 100, 200]

n_cm2 = np.linspace(1e10, 1e12, 300)
n_m2  = n_cm2 * 1e4

u0 = u0_of_n(n_m2)

uc_dict = {}
lam_um_dict = {}

for T in temps:
    gam = gamma_total(n_m2, T, a=a_interp)
    R = np.gradient(np.log(gam), np.log(n_m2))
    uc = u0 / np.maximum(np.abs(R), 1e-30)
    lam = 4 * np.pi * u0 / np.maximum(gam, 1e-30)

    uc_dict[T] = uc
    lam_um_dict[T] = lam * 1e6

cmap = plt.get_cmap("plasma")
norm = LogNorm(vmin=min(temps), vmax=max(temps))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4*img_scale, 5.2*img_scale), sharex=True)
ax1.set_facecolor("#FAFAFA")
ax2.set_facecolor("#FAFAFA")

dsdsa = 0.95
dasda = 1.00

for T in temps:
    norm_val = np.clip(norm(T), 0, 1)
    color_val = 0.1 + norm_val * 0.6
    color = cmap(color_val)
    ax1.plot(n_cm2, uc_dict[T], color=color, linewidth=1.8, alpha=0.95, zorder=1)
    x_label = n_cm2[-1] * dsdsa#0.95
    dsdsa -= 0.08
    y_label = uc_dict[T][-1]*dasda
    dasda-= 0.06
    # ax1.text(x_label, y_label, f'{T} K', color=color, fontsize=9, 
    #          ha='right', va='center', weight='medium', zorder=10)

ax1.set_ylabel(r"Critical velocity $u_c$ (m/s)", color="#2C3E50", fontsize=10)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1.minorticks_on()
ax1.tick_params(colors="#2C3E50")
dadas=1.0
for T in temps:
    norm_val = np.clip(norm(T), 0, 1)
    color_val = 0.1 + norm_val * 0.6
    color = cmap(color_val)
    ax2.plot(n_cm2, lam_um_dict[T], color=color, linewidth=1.8, alpha=0.95, zorder=1)
    x_label = n_cm2[-1] * 0.95
    dadas+=0.05
    y_label = lam_um_dict[T][-1] * 1.5*dadas
    ax2.text(x_label, y_label, f'{T} K', color=color, fontsize=9, 
             ha='right', va='center', weight='medium', zorder=10)

ax2.set_xlabel(r"density $n$ (cm$^{-2}$)", color="#2C3E50", fontsize=10)
ax2.set_ylabel(r"Wavelength $\lambda$ ($\mu$m)", color="#2C3E50", fontsize=10)
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.minorticks_on()
ax2.tick_params(colors="#2C3E50")

def format_power_of_10(x, pos):
    if x <= 0:
        return ''
    exp = int(np.log10(x))
    return r'$10^{' + str(exp) + r'}$'

ax2.xaxis.set_major_formatter(FuncFormatter(format_power_of_10))
ax2.yaxis.set_major_formatter(FuncFormatter(format_power_of_10))
ax2.xaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.4)))
ax2.yaxis.set_minor_formatter(LogFormatter(minor_thresholds=(2, 0.4)))

fig.tight_layout(pad=0.5)

# Save in high DPI PNG and SVG formats
fig.savefig('fig4.png', dpi=600, format='png', bbox_inches='tight', pad_inches=0.05)
fig.savefig('fig4.svg', dpi=600, format='svg', bbox_inches='tight', pad_inches=0.05)

plt.show()
