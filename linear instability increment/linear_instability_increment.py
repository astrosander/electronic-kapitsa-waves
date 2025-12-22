import numpy as np
import matplotlib.pyplot as plt

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
# Choose plotting k-range for log-log plot (small k, positive only)
# ----------------------------
# Focus on small positive k values for log-log analysis
kmin_log = 1e-3
kmax_log = 1.0
N_log = 5000
k_log = np.logspace(np.log10(kmin_log), np.log10(kmax_log), N_log)

# Compute curves for log-log plot
z0p_log, z0m_log = growth_rates(k_log, u_c, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=eta0, Dp=eta0)
z1p_log, z1m_log = growth_rates(k_log, u_c, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=eta1, Dp=eta1)

# Use absolute value of growth rates for log-log (focus on magnitude)
z0p_abs = np.abs(z0p_log)
z1p_abs = np.abs(z1p_log)
z0m_abs = np.abs(z0m_log)
z1m_abs = np.abs(z1m_log)

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

# Plot all curves on log-log axes
# Mode color distinguishes ω+ (blue) and ω- (orange)
# Line style distinguishes η=0 (solid) and η=0.1 (dashed)
ax.loglog(k_log, z0p_abs, color=c_plus,  ls="-", linewidth=2.0, label=r"$\omega_{+}$, $\eta=0$")
ax.loglog(k_log, z1p_abs, color=c_plus,  ls="--", linewidth=2.0, label=r"$\omega_{+}$, $\eta=0.1$")
ax.loglog(k_log, z0m_abs, color=c_minus, ls="-", linewidth=2.0, label=r"$\omega_{-}$, $\eta=0$")
ax.loglog(k_log, z1m_abs, color=c_minus, ls="--", linewidth=2.0, label=r"$\omega_{-}$, $\eta=0.1$")

# Add reference lines for k and k^2 scaling
# Find appropriate scaling factors to match the data visually
k_ref = k_log
# Use a point in the middle of the range for scaling
mid_idx = len(k_ref) // 3
if z1p_abs[mid_idx] > 0:
    # Reference for |k| scaling (slope = 1 on log-log)
    scale_k1 = z1p_abs[mid_idx] / k_ref[mid_idx]
    k1_ref = k_ref * scale_k1
    # Reference for k^2 scaling (slope = 2 on log-log)
    scale_k2 = z1p_abs[mid_idx] / (k_ref[mid_idx]**2)
    k2_ref = k_ref**2 * scale_k2
else:
    # Fallback if data is zero
    k1_ref = k_ref * 1e-2
    k2_ref = k_ref**2 * 1e-4

ax.loglog(k_ref, k1_ref, color="gray", ls=":", linewidth=1.5, alpha=0.7, label=r"$\propto |k|$")
ax.loglog(k_ref, k2_ref, color="gray", ls="-.", linewidth=1.5, alpha=0.7, label=r"$\propto k^2$")

# Cosmetics
ax.grid(True, ls=":", lw=0.8, color="0.85", alpha=0.6, which="both")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

# Improve tick appearance
ax.tick_params(axis='both', which='major', labelsize=13, width=1.0, length=5)
ax.tick_params(axis='both', which='minor', width=0.8, length=3)

# Set limits
ax.set_xlim(kmin_log, kmax_log)

# Labels with larger font sizes
ax.set_xlabel(r"Wavenumber $k$", fontsize=16, fontweight='medium')
ax.set_ylabel(r"$|\mathrm{Im}\,\omega_{\pm}(k)|$", fontsize=16, fontweight='medium')

# Legend
ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=11, 
          handlelength=2.8, borderaxespad=0.3)

# Save (vector SVG + high-res PNG for publication)
fig.savefig("linear_instability_increment_loglog.svg", bbox_inches="tight", dpi=300)
fig.savefig("linear_instability_increment_loglog.png", dpi=600, bbox_inches="tight")
# plt.show()