import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------
# Physics / model parameters
# ----------------------------
U0     = 0.9
n      = 0.94
w      = 0.28
gamma0 = 16.5
m      = 1.3

u_c = 2.0                 # fixed drift velocity for the plot
eta0 = 0.0                # compare eta=0
eta1 = 0.1                # vs eta>0  (here Dp=Dn=eta)

# ----------------------------
# Dispersion: Eq. (18)-(19)
# ----------------------------
def growth_rates(k, u, *, n, w, gamma0, U0, m, Dn, Dp=None):
    """
    Returns Im(omega_plus), Im(omega_minus).
    We follow your convention e^{ikx - i omega t}.
    """
    if Dp is None:
        Dp = Dn

    gamma   = gamma0 * np.exp(-n / w)    # gamma(n)
    gamma_n = -gamma / w                # dgamma/dn for n>0 in this model

    p  = n * u
    Pn = U0 * n - p**2 / (m * n**2)      # Pi_n
    Pp = 2 * p / (m * n)                # Pi_p
    Lam = (gamma_n - gamma / n) * p     # Lambda = (dgamma/dn - gamma/n) p

    G_tilde = gamma + (Dp - Dn) * k**2
    Delta = (G_tilde + 1j * k * Pp)**2 + 4j * k * Lam / m - 4 * (k**2) * Pn / m

    # Choose sqrt branch consistently (avoid random sign flips)
    sqrtD = np.sqrt(Delta)
    sqrtD = np.where(np.real(sqrtD) < 0, -sqrtD, sqrtD)

    omega_plus  = (-1j * G_tilde + k * Pp + 1j * sqrtD) / 2 - 1j * Dn * k**2
    omega_minus = (-1j * G_tilde + k * Pp - 1j * sqrtD) / 2 - 1j * Dn * k**2

    return np.imag(omega_plus), np.imag(omega_minus)

# ----------------------------
# Choose plotting k-range from eta>0 peak (finite k*)
# ----------------------------
k_probe = np.linspace(-10, 10, 12001)
z1p_probe, _ = growth_rates(
    k_probe, u_c, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=eta1, Dp=eta1
)
k_star = abs(k_probe[np.argmax(z1p_probe)])
kmax = max(4.0 * k_star, 4.0)
kmax=6
kmin = -kmax


N = 6000
k = np.linspace(kmin, kmax, N)

# Compute curves
z0p, z0m = growth_rates(k, u_c, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=eta0, Dp=eta0)
z1p, z1m = growth_rates(k, u_c, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=eta1, Dp=eta1)

# ----------------------------
# PRL-friendly styling
# ----------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "savefig.transparent": True,
})

# Square figure for 1:1 aspect ratio
fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)

c_plus  = "tab:blue"     # ω+
c_minus = "tab:orange"   # ω-

# Plot all curves on the same axes
# Mode color distinguishes ω+ (blue) and ω- (orange)
# Line style distinguishes η=0 (solid) and η=0.1 (dashed)
ax.plot(k, z0p, color=c_plus,  ls="-", linewidth=2.0)
ax.plot(k, z1p, color=c_plus,  ls="--", linewidth=2.0)
ax.plot(k, z0m, color=c_minus, ls="-", linewidth=2.0)
ax.plot(k, z1m, color=c_minus, ls="--", linewidth=2.0)

# Shade unstable region for eta>0 (helps intuitive reading)
ax.fill_between(k, 0, z1p, where=(z1p > 0), color=c_plus, alpha=0.12, linewidth=0)

# Mark ±k* for eta>0 (wavelength selection)
idx = np.argmax(z1p)
kpk = abs(float(k[idx]))
zpk = float(z1p[idx])

ax.axvline(+kpk, color="0.5", ls=":", lw=1.0, zorder=0)
ax.axvline(-kpk, color="0.5", ls=":", lw=1.0, zorder=0)
ax.plot([+kpk, -kpk], [zpk, zpk], "o", ms=4.5, color=c_plus, clip_on=False)
ax.annotate(
    r"$\pm k^{\ast}$", xy=(kpk, zpk), xytext=(0.60, 0.86),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", lw=1.2, color="0.35"),
    ha="left", va="center", fontsize=11
)

# Cosmetics
ax.axhline(0, color="0.2", lw=1.2)
ax.grid(True, ls=":", lw=0.8, color="0.85", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

# Improve tick appearance
ax.tick_params(axis='both', which='major', labelsize=13, width=1.0, length=5)
ax.tick_params(axis='both', which='minor', width=0.8, length=3)

# Set y-limits to accommodate both ω+ and ω-
all_data = np.concatenate([z0p, z1p, z0m, z1m])
ymin, ymax = float(np.min(all_data)), float(np.max(all_data))
# Ensure zero is visible
ymin = min(ymin, 0.0)
ymax = max(ymax, 0.0)
rng = max(1e-12, ymax - ymin)
ax.set_ylim(ymin - 0.06 * rng, ymax + 0.08 * rng)

ax.set_xlim(kmin, kmax)
ax.set_ylim(-6, 2)#ymax)

# Labels with larger font sizes
ax.set_xlabel(r"Wavenumber $k$", fontsize=16, fontweight='medium')
ax.set_ylabel(r"$\mathrm{Im}\,\omega_{\pm}(k)$", fontsize=16, fontweight='medium')

# Two-part legend: mode colors + diffusion line styles (larger fonts)
mode_handles = [
    Line2D([0], [0], color=c_plus,  lw=2.2, label=r"$\omega_{+}$"),
    Line2D([0], [0], color=c_minus, lw=2.2, label=r"$\omega_{-}$"),
]
diffusion_handles = [
    Line2D([0], [0], color="0.2", lw=2.2, ls="-",  label=rf"$\eta={eta0:g}$"),
    Line2D([0], [0], color="0.2", lw=2.2, ls="--", label=rf"$\eta={eta1:g}$"),
]

leg1 = ax.legend(handles=mode_handles, loc="upper left", frameon=False,
                 handlelength=2.8, borderaxespad=0.3, fontsize=12)
ax.add_artist(leg1)
ax.legend(handles=diffusion_handles, loc="upper right", frameon=False,
          handlelength=2.8, borderaxespad=0.3, fontsize=12)

# Save (vector PDF + high-res PNG for publication)
fig.savefig("linear_instability_increment.svg", bbox_inches="tight", dpi=300)
fig.savefig("linear_instability_increment.png", dpi=600, bbox_inches="tight")
# plt.show()