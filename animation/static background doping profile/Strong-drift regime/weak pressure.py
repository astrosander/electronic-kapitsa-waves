import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp
import os

@dataclass
class Par:
    m: float = 1.0
    e: float = 1.0
    n0: float = 1.0
    p0: float = 0.0
    U: float = 1.0
    Gamma0: float = 0.05
    w: float = 1.0
    epsilon: float = 15.0
    E_ext: float = 0.80
    n1: float = 0.05
    lam: float = 0.005/4
    x0: float = 0.55
    L: float = 100.0
    Nx: int = 256
    t_final: float = 150.0
    n_save: int = 220
    rtol: float = 1e-7
    atol: float = 1e-9
    n_floor: float = 1e-12
    dealias: bool = True
    save_frames: bool = True
    base_dir: str = "frames_U_%s"
    cmap: str = "inferno"

par = Par()

U_list = [0.80, 0.20, 0.05, 0.00]

x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j*k
inv_k2 = np.zeros_like(k, dtype=np.complex128)
nz = (k != 0); inv_k2[nz] = 1.0/(k[nz]**2)

def Dx(f): 
    return (ifft(ik * fft(f))).real

def dealias_23(f):
    if not par.dealias:
        return f
    F = fft(f)
    N = len(F)
    kcut = int(np.floor(N/3))
    F[(N//2 - kcut + 1):(N//2 + kcut)] = 0.0
    return ifft(F).real

def Gamma_of(n): return par.Gamma0 * np.exp(-n/par.w)
def Pi0_of(n):   return 0.5 * par.U * n**2

def periodic_delta(x, x0, L):
    return (x - x0 + 0.5*L) % L - 0.5*L

def nbar_profile():
    xc = par.x0 * par.L
    d = periodic_delta(x, xc, par.L)
    return par.n0 + par.n1 * np.exp(-0.5 * par.lam * d**2)

def phi_from_n_with_bg(n, nbar):
    rhs_hat = fft((par.e/par.epsilon) * (n - nbar))
    phi_hat = inv_k2 * rhs_hat
    phi_hat[0] = 0.0
    return ifft(phi_hat).real

def initial_fields():
    n0 = np.full_like(x, par.n0)
    p0 = np.full_like(x, par.p0)
    nbar = nbar_profile()
    return n0, p0, nbar

def rhs_pde(t, y):
    N = par.Nx
    n = y[:N]; p = y[N:]
    n_eff = np.maximum(n, par.n_floor)
    v = p/(par.m*n_eff)

    nv = dealias_23(n_eff * v)
    dn_dt = -Dx(nv)

    Pi = Pi0_of(n_eff) + (p**2)/(par.m*n_eff)
    Pi = dealias_23(Pi)

    nbar = nbar_profile()
    phi  = phi_from_n_with_bg(n, nbar)
    Ex   = par.E_ext - Dx(phi)

    dp_dt = -Gamma_of(n_eff)*p - Dx(Pi) + par.e*n_eff*Ex
    return np.concatenate([dn_dt, dp_dt])

def run_once():
    n0, p0, nbar = initial_fields()
    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)
    sol = solve_ivp(rhs_pde, (0.0, par.t_final), y0, t_eval=t_eval,
                    method="BDF", rtol=par.rtol, atol=par.atol)
    N = par.Nx
    n_t = sol.y[:N, :]
    p_t = sol.y[N:, :]
    return sol.t, n_t, p_t, nbar

def spacetime_dn(t, n_t, nbar, Uval):
    dn = n_t - nbar[:, None]
    extent = [x.min(), x.max(), t.min(), t.max()]
    plt.figure(figsize=(10,4.6))
    plt.imshow(dn.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    x0 = par.x0 * par.L
    plt.plot([x0, x0], [t.min(), t.max()], 'w--', lw=1.2, alpha=0.8, label="defect")
    plt.colorbar(label=r"$\delta n$")
    plt.xlabel("x"); plt.ylabel("t")
    plt.title(rf"$\delta n(x,t)=n-\bar n$  |  U={Uval:.2f},  E={par.E_ext:.2f}")
    plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(f"spacetime_dn_U_{Uval:.2f}.png", dpi=160); plt.close()

def save_animation_frames(t, n_t, p_t, nbar, Uval):
    Gbar = Gamma_of(nbar)
    p_naive = par.e * nbar * par.E_ext / np.maximum(Gbar, 1e-16)

    outdir = par.base_dir % f"{Uval:.2f}".replace('.', 'p')
    os.makedirs(outdir, exist_ok=True)

    for j in range(len(t)):
        n = n_t[:, j]
        p = p_t[:, j]
        n_eff = np.maximum(n, par.n_floor)
        Pi = Pi0_of(n_eff) + (p**2)/(par.m*n_eff)
        phi = phi_from_n_with_bg(n, nbar)
        Ex = par.E_ext - Dx(phi)
        R = Gamma_of(n_eff)*p + Dx(Pi) - par.e*n_eff*Ex

        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axs[0].plot(x, n - nbar, lw=2, label=r"$\delta n$")
        axs[0].axhline(0, ls='--', lw=1)
        axs[0].set_ylabel(r"$\delta n$")
        axs[0].set_title(f"U={Uval:.2f}, E={par.E_ext:.2f}, t={t[j]:.2f}")

        axs[1].plot(x, p, lw=2, label="p(x,t)")
        axs[1].plot(x, p_naive, lw=1.4, ls='--', label=r"$p_{\rm naive}=e\,\bar n E/\Gamma(\bar n)$")
        axs[1].set_ylabel("p")
        axs[1].legend(loc="best", fontsize=9)

        axs[2].plot(x, R, lw=1.8)
        axs[2].set_ylabel("residual")
        axs[2].set_xlabel("x")

        plt.tight_layout()
        fname = f"{outdir}/frame_{j:04d}.png"
        plt.savefig(fname, dpi=140); plt.close()

for Uval in U_list:
    par.U = float(Uval)
    t, n_t, p_t, nbar = run_once()
    spacetime_dn(t, n_t, nbar, Uval)
    if par.save_frames:
        save_animation_frames(t, n_t, p_t, nbar, Uval)

G0 = par.Gamma0 * np.exp(-par.n0/par.w)
p_ss_uniform_est = (par.e * par.n0 * par.E_ext) / G0
print(f"[info] With E={par.E_ext:.2f}, Γ(n0)={G0:.3f} → nominal uniform p_ss≈{p_ss_uniform_est:.3f}")
print("[info] Created per-U folders like frames_U_0p05/, each with animation frames.")
