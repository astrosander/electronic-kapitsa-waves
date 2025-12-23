import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

U0     = 1.0
n      = 1
w      = 0.5
gamma0 = 20
m      = 0.5

u_c1 = 0.55
u_c2 = 0.2
eta0 = 0.0
eta1 = 0.1

def growth_rates(k, u, *, n, w, gamma0, U0, m, Dn, Dp=None):
    if Dp is None:
        Dp = Dn

    gamma   = gamma0 * np.exp(-n / w)
    gamma_n = -gamma / w

    p  = n * u
    Pn = U0 * n - p**2 / (m * n**2)
    Pp = 2 * p / (m * n)
    Lam = (gamma_n - gamma / n) * p

    G_tilde = gamma + (Dp - Dn) * k**2
    Delta = (G_tilde + 1j * k * Pp)**2 + 4j * k * Lam / m - 4 * (k**2) * Pn / m

    sqrtD = np.sqrt(Delta)
    sqrtD = np.where(np.real(sqrtD) < 0, -sqrtD, sqrtD)

    omega_plus  = (-1j * G_tilde + k * Pp + 1j * sqrtD) / 2 - 1j * Dn * k**2
    omega_minus = (-1j * G_tilde + k * Pp - 1j * sqrtD) / 2 - 1j * Dn * k**2

    return np.imag(omega_plus), np.imag(omega_minus)

def omega_plus_k2_asymptotics(k, u, *, n, w, gamma0, U0, m, Dn, Dp=None):
    if Dp is None:
        Dp = Dn
    
    k1 = 1e-3
    k2 = 2e-3
    
    z1, _ = growth_rates(k1, u, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=Dn, Dp=Dp)
    z2, _ = growth_rates(k2, u, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=Dn, Dp=Dp)
    
    A1 = z1 / (k1**2)
    A2 = z2 / (k2**2)
    A = (A1 + A2) / 2
    
    return A * k**2

def omega_plus_large_k_asymptote(u, *, n, w, gamma0, U0, m, Dn, Dp=None):
    if Dp is None:
        Dp = Dn
    
    k_large = 1e3
    z_asymp, _ = growth_rates(k_large, u, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=Dn, Dp=Dp)
    
    return z_asymp
k_probe = np.linspace(-10, 10, 12001)
z1p_probe, _ = growth_rates(
    k_probe, u_c1, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=eta1, Dp=eta1
)
k_star = abs(k_probe[np.argmax(z1p_probe)])
kmax = max(4.0 * k_star, 4.0)
kmax=4
kmin = -kmax

N = 600
k = np.linspace(kmin, kmax, N)

z0p_u1, _ = growth_rates(k, u_c1, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=eta0, Dp=eta0)
z1p_u1, _ = growth_rates(k, u_c1, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=eta1, Dp=eta1)
z0p_u2, _ = growth_rates(k, u_c2, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=eta0, Dp=eta0)
z1p_u2, _ = growth_rates(k, u_c2, n=n, w=w, gamma0=gamma0, U0=U0, m=m, Dn=eta1, Dp=eta1)

k_small_for_fit = 1e-3
z_small_u1, _ = growth_rates(k_small_for_fit, u_c1, n=n, w=w, gamma0=gamma0, 
                              U0=U0, m=m, Dn=eta0, Dp=eta0)
A_k2_u1 = z_small_u1 / (k_small_for_fit**2)
z0p_asymp_u1 = A_k2_u1 * k**2

z_small_u2, _ = growth_rates(k_small_for_fit, u_c2, n=n, w=w, gamma0=gamma0, 
                              U0=U0, m=m, Dn=eta0, Dp=eta0)
A_k2_u2 = z_small_u2 / (k_small_for_fit**2)
z0p_asymp_u2 = A_k2_u2 * k**2

omega_plus_infty_u1 = omega_plus_large_k_asymptote(u_c1, n=n, w=w, gamma0=gamma0, 
                                                    U0=U0, m=m, Dn=eta0, Dp=eta0)
omega_plus_infty_u2 = omega_plus_large_k_asymptote(u_c2, n=n, w=w, gamma0=gamma0, 
                                                    U0=U0, m=m, Dn=eta0, Dp=eta0)
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

fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)

c_u2 = "tab:blue"
c_u1 = "tab:orange"

ax.plot(k, z0p_u1, color=c_u1, ls="--", linewidth=2.0)
ax.plot(k, z1p_u1, color=c_u1, ls="-", linewidth=2.0)
ax.plot(k, z0p_u2, color=c_u2, ls="--", linewidth=2.0)
ax.plot(k, z1p_u2, color=c_u2, ls="-", linewidth=2.0)

ax.plot(k, z0p_asymp_u1, color="red", linewidth=1.0, alpha=1, 
        zorder=1, label='_nolegend_')
# ax.plot(k, z0p_asymp_u2, color="black", linewidth=1.0, alpha=0.7, 
#         zorder=1, label='_nolegend_')

ax.axhline(omega_plus_infty_u1, color="red", ls="-", linewidth=1.0, alpha=1,
           zorder=1, label='_nolegend_')
# ax.axhline(omega_plus_infty_u2, color="black", ls="-", linewidth=1.0, alpha=0.7,
#            zorder=1, label='_nolegend_')

ax.fill_between(k, 0, z1p_u1, where=(z1p_u1 > 0), color=c_u1, alpha=0.12, linewidth=0)
ax.fill_between(k, 0, z1p_u2, where=(z1p_u2 > 0), color=c_u2, alpha=0.12, linewidth=0)

idx = np.argmax(z1p_u1)
kpk = abs(float(k[idx]))
zpk = float(z1p_u1[idx])

ax.axvline(+kpk, color="0.5", ls=":", lw=1.0, zorder=0)
ax.axvline(0, color="0.5", ls="-.", lw=1.0, zorder=0)
ax.axvline(-kpk, color="0.5", ls=":", lw=1.0, zorder=0)
ax.plot([+kpk, -kpk], [zpk, zpk], "o", ms=4.5, color=c_u1, clip_on=False)
ax.annotate(
    r"$\pm k^{\ast}$", xy=(kpk, zpk), xytext=(0.60, 0.86),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", lw=1.2, color="0.35"),
    ha="left", va="center", fontsize=11
)

ax.axhline(0, color="0.2", lw=1.2)
ax.grid(True, ls=":", lw=0.8, color="0.85", alpha=0.6)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.spines["top"].set_linewidth(1.2)
ax.spines["right"].set_linewidth(1.2)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

ax.tick_params(axis='both', which='major', labelsize=13, width=1.0, length=5)
ax.tick_params(axis='both', which='minor', width=0.8, length=3)

all_data = np.concatenate([z0p_u1, z1p_u1, z0p_u2, z1p_u2])
ymin, ymax = float(np.min(all_data)), float(np.max(all_data))
ymin = min(ymin, 0.0)
ymax = max(ymax, 0.0)
rng = max(1e-12, ymax - ymin)
ax.set_ylim(ymin - 0.06 * rng, ymax + 0.08 * rng)

ax.set_xlim(kmin, kmax)
ax.set_ylim(-0.65, 0.85)

ax.set_xlabel(r"Wavenumber $k$", fontsize=16, fontweight='medium')
ax.set_ylabel(r"$\mathrm{Im}\,\omega_{+}(k)$", fontsize=16, fontweight='medium')

u_handles = [
    Line2D([0], [0], color=c_u1, lw=2.2, label=rf"$u_c={u_c1:g}$"),
    Line2D([0], [0], color=c_u2, lw=2.2, label="$u_c=0.20$"),#$u_c={u_c2:g}$"),
]
diffusion_handles = [
    Line2D([0], [0], color="0.2", lw=2.2, ls="-",  label=rf"$\eta={eta0:g}$"),
    Line2D([0], [0], color="0.2", lw=2.2, ls="--", label=rf"$\eta={eta1:g}$"),
]

leg1 = ax.legend(handles=u_handles, loc="upper left", frameon=False,
                 handlelength=2.8, borderaxespad=0.3, fontsize=12)
ax.add_artist(leg1)
ax.legend(handles=diffusion_handles, loc="upper right", frameon=False,
          handlelength=2.8, borderaxespad=0.3, fontsize=12)

fig.savefig("linear_instability_increment.svg", bbox_inches="tight", dpi=300)
fig.savefig("linear_instability_increment.png", dpi=600, bbox_inches="tight")
plt.show()