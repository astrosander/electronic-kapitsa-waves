import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Pars:
    m: float = 1.0
    e: float = 1.0
    n0: float = 1.0
    U: float = 0.06
    Gamma0: float = 0.08
    w: float = 1.0
    u0: float = 0.40
    Dn: float = 0.02
    Dp: float = 0.02
    include_poisson: bool = True
    eps: float = 20.0

def gamma_at_n0(par: Pars):
    G0 = par.Gamma0 * np.exp(-par.n0/par.w)
    Gp = -G0/par.w
    return G0, Gp

def steady_eE(par: Pars):
    G0, _ = gamma_at_n0(par)
    return G0 * par.m * par.u0

def B_of_k(k, par: Pars, drive_factor: float):
    G0, Gp = gamma_at_n0(par)
    p0 = par.m * par.n0 * par.u0
    eE = drive_factor * steady_eE(par)

    term_const = Gp * p0 - eE
    term_ik    = 1j * k * (par.U * par.n0 - par.m * par.u0**2)
    term_poiss = 0.0
    if par.include_poisson:
        invk = np.where(k != 0.0, 1.0/k, 0.0)
        term_poiss = 1j * (par.n0 * par.e / par.eps) * invk
    return term_const + term_ik + term_poiss

def sigma_branches(k, par: Pars, drive_factor: float):
    k = np.asarray(k, dtype=float)
    G0, _ = gamma_at_n0(par)

    C1 = (G0 + (par.Dn + par.Dp) * k**2) + 1j * (2.0 * par.u0 * k)
    C0 = par.Dn * k**2 * (G0 + par.Dp * k**2 + 1j * 2.0 * par.u0 * k) - 1j * k * B_of_k(k, par, drive_factor)
    disc = C1**2 - 4.0 * C0
    sqrt_disc = np.sqrt(disc)
    s_plus  = -0.5 * (C1 - sqrt_disc)
    s_minus = -0.5 * (C1 + sqrt_disc)
    return np.vstack([s_plus, s_minus])

def plot_family(ax, k, par, drive_factors, label_prefix=""):
    colors = plt.cm.tab10.colors
    for j, df in enumerate(drive_factors):
        sig = sigma_branches(k, par, df)
        upper = np.max(np.real(sig), axis=0)
        lower = np.min(np.real(sig), axis=0)

        col = colors[j % 10]
        ax.plot(k, upper, lw=2.2, color=col, label=f"{label_prefix}η={df:.2f} (upper)")
        ax.plot(k, lower, lw=1.8, ls="--", color=col, alpha=0.9, label=f"{label_prefix}η={df:.2f} (lower)")

def make_figure(title, par, kmax=8.0, Nk=1201, drive_factors=(0.85, 0.95, 1.05, 1.15), save_as=None):
    k = np.linspace(-kmax, kmax, Nk)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    plot_family(ax, k, par, drive_factors)

    cs = np.sqrt(par.U * par.n0 / par.m)
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlim(-kmax, kmax)
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Instability increment  Re σ(k)")
    ax.set_title(f"{title}\n"
                 f"u0={par.u0:g},  c_s={cs:.3f},  "
                 f"Dp={par.Dp:g}, Dn={par.Dn:g},  Poisson={'on' if par.include_poisson else 'off'}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=False, fontsize=9)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=180)
    plt.show()

if __name__ == "__main__":
    base = Pars(
        m=1.0, e=1.0, n0=1.0,
        U=0.06, Gamma0=0.08, w=1.0,
        u0=0.40,
        Dn=0.02, Dp=0.02,
        include_poisson=True, eps=20.0
    )

    dfs = (0.85, 0.95, 1.05, 1.15)

    make_figure("Increment vs k — Equal diffusion (Dp = Dn)", base, drive_factors=dfs,
                save_as="increment_equal_DpDn.png")

    hiDp = Pars(**vars(base)); hiDp.Dp = 0.30; hiDp.Dn = 0.01
    make_figure("Increment vs k — Momentum-dominated diffusion (Dp ≫ Dn)", hiDp, drive_factors=dfs,
                save_as="increment_hiDp.png")

    hiDn = Pars(**vars(base)); hiDn.Dp = 0.01; hiDn.Dn = 0.30
    make_figure("Increment vs k — Density-dominated diffusion (Dp ≪ Dn)", hiDn, drive_factors=dfs,
                save_as="increment_hiDn.png")
