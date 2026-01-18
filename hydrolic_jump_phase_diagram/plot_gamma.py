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


def gamma_fn(n):
    """Finite-contrast CNP peak (logistic/tanh step) function."""
    nabs = np.abs(n)
    gamma_lo = 2.5e-7
    gamma_hi = 7.5e-7
    n_c = 21.95
    Delta = 0.05
    return gamma_lo + (gamma_hi - gamma_lo) / (1.0 + np.exp((nabs - n_c) / Delta))


def dgamma_dn(n):
    """Derivative of gamma_fn with respect to n."""
    nabs = np.abs(n)
    gamma_lo = 2.5e-7
    gamma_hi = 7.5e-7
    n_c = 21.95
    Delta = 0.05

    z = (nabs - n_c) / Delta
    ez = np.exp(z)
    dgamma_dn_abs = -(gamma_hi - gamma_lo) * ez / (Delta * (1.0 + ez)**2)

    # Include sign from |n| if n is negative
    return dgamma_dn_abs * np.sign(n)


# Density range from phase diagram parameters
nmin = 20
nmax = 23
Npts = 10000

n_vals = np.linspace(nmin, nmax, Npts)
gamma_vals = gamma_fn(n_vals)
dgamma_dn_vals = dgamma_dn(n_vals)

# Also compute log-slope for reference
log_slope = np.abs(dgamma_dn_vals / gamma_vals)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.6*img_scale, 3.4*img_scale), sharex=True)
ax1.set_facecolor("#FAFAFA")
ax2.set_facecolor("#FAFAFA")

# Plot gamma(n)
ax1.plot(n_vals, gamma_vals, color="#2C3E50", linewidth=1.8, alpha=0.95)
ax1.set_ylabel(r"$\gamma(n)$ (s$^{-1}$)", color="#2C3E50", fontsize=10)
ax1.set_ylim(gamma_vals.min() * 0.95, gamma_vals.max() * 1.05)
ax1.minorticks_on()
ax1.tick_params(colors="#2C3E50")
ax1.grid(True, alpha=0.2, linewidth=0.7)

# Plot log-slope |d(ln gamma)/dn|
ax2.plot(n_vals, log_slope, color="#E74C3C", linewidth=1.8, alpha=0.95)
ax2.set_xlabel(r"density $n_0$", color="#2C3E50", fontsize=10)
ax2.set_ylabel(r"$\left|\frac{d\ln\gamma}{dn}\right|$", color="#2C3E50", fontsize=10)
ax2.minorticks_on()
ax2.tick_params(colors="#2C3E50")
ax2.grid(True, alpha=0.2, linewidth=0.7)

# Add vertical line at n_c
n_c = 21.95
ax1.axvline(n_c, color="#3498DB", linestyle="--", linewidth=1.2, alpha=0.6, label=f"$n_c = {n_c}$")
ax2.axvline(n_c, color="#3498DB", linestyle="--", linewidth=1.2, alpha=0.6)
ax1.legend(loc='best', fontsize=8, framealpha=0.9)

# Add horizontal line at 4π for reference (Condition 1 threshold)
ax2.axhline(4 * np.pi, color="#9B59B6", linestyle=":", linewidth=1.2, alpha=0.6, label=r"$4\pi$")
ax2.legend(loc='best', fontsize=8, framealpha=0.9)

ax1.set_xlim(nmin, nmax)

# Add title with equation
fig.suptitle(r"$\gamma(n) = \gamma_{\rm lo} + \frac{\gamma_{\rm hi} - \gamma_{\rm lo}}{1 + \exp\left(\frac{|n| - n_c}{\Delta}\right)}$", 
             fontsize=12, y=0.98)

fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.96])

fig.savefig("gamma_vs_n.pdf", dpi=600, bbox_inches='tight', pad_inches=0.05)
fig.savefig("gamma_vs_n.svg", dpi=600, bbox_inches='tight', pad_inches=0.05)
fig.savefig("gamma_vs_n.png", dpi=600, bbox_inches='tight', pad_inches=0.05)

plt.show()

