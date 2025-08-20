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
    Gamma0: float = 0.08
    w: float = 1.0
    epsilon: float = 15.0
    E_ext: float = 0.10
    n1: float = 0.02
    lam: float = 0.005 / 4
    x0: float = 0.55
    L: float = 100.0
    Nx: int = 256
    t_final: float = 160.0
    n_save: int = 220
    rtol: float = 1e-7  
    atol: float = 1e-9
    n_floor: float = 1e-12
    save_frames: bool = True
    frames_dir: str = "frames_nbar"
    cmap: str = "inferno"

par = Par()

x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j*k
inv_k2 = np.zeros_like(k, dtype=np.complex128)
nz = (k != 0); inv_k2[nz] = 1.0/(k[nz]**2)

def Dx(f): return (ifft(ik * fft(f))).real

def Gamma(n): return par.Gamma0 * np.exp(-n/par.w)
def Pi0(n):   return 0.5 * par.U * n**2

def periodic_delta(x, x0, L):
    return (x - x0 + 0.5*L) % L - 0.5*L

def nbar_profile():
    x0 = par.x0 * par.L
    d = periodic_delta(x, x0, par.L)
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

    dn_dt = -Dx(n_eff * v)

    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)

    nbar = nbar_profile()
    phi  = phi_from_n_with_bg(n, nbar)
    Ex   = par.E_ext - Dx(phi)

    dp_dt = -Gamma(n_eff)*p - Dx(Pi) + par.e*n_eff*Ex

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
    u_t = p_t/(par.m*np.maximum(n_t, par.n_floor))
    return sol.t, n_t, p_t, u_t, nbar

def spacetime_dn(t, n_t, nbar, title=r"$\delta n(x,t)=n-\bar n$"):
    dn = n_t #- nbar[:, None]
    extent = [x.min(), x.max(), t.min(), t.max()]
    plt.figure(figsize=(10,4.6))
    plt.imshow(dn.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    x0 = par.x0 * par.L
    plt.plot([x0, x0], [t.min(), t.max()], 'w--', lw=1.3, alpha=0.8, label="defect center")
    plt.colorbar(label=r"$\delta n$")
    plt.xlabel("x"); plt.ylabel("t"); plt.title(title)
    plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig("spacetime_dn.png", dpi=160)

def save_animation_frames(t, n_t, p_t, nbar):
    os.makedirs(par.frames_dir, exist_ok=True)
    Gbar = Gamma(nbar)
    p_naive = par.e * nbar * par.E_ext / np.maximum(Gbar, 1e-16)

    for j in range(len(t)):
        n = n_t[:, j]
        p = p_t[:, j]
        n_eff = np.maximum(n, par.n_floor)
        Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)
        nphi = phi_from_n_with_bg(n, nbar)
        Ex = par.E_ext - Dx(nphi)
        R = Gamma(n_eff)*p + Dx(Pi) - par.e*n_eff*Ex

        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        axs[0].plot(x, n - nbar, lw=2, label=r"$\delta n$")
        axs[0].axhline(0, ls='--', lw=1)
        axs[0].set_ylabel(r"$\delta n$")
        axs[0].set_title(f"t = {t[j]:.2f}")

        axs[1].plot(x, p, lw=2, label="p(x,t)")
        axs[1].plot(x, p_naive, lw=1.4, ls='--', label=r"$p_{\rm naive}=e\,\bar n\,E/\Gamma(\bar n)$")
        axs[1].set_ylabel("p")
        axs[1].legend(loc="best", fontsize=9)

        axs[2].plot(x, R, lw=1.8)
        axs[2].set_ylabel("force-balance residual")
        axs[2].set_xlabel("x")

        plt.tight_layout()
        fname = f"{par.frames_dir}/frame_{j:04d}.png"
        plt.savefig(fname, dpi=140); plt.close()

def norms(a):
    return np.sqrt(np.mean(a*a)), np.max(np.abs(a))

t, n_t, p_t, u_t, nbar = run_once()

spacetime_dn(t, n_t, nbar)

if par.save_frames:
    save_animation_frames(t, n_t, p_t, nbar)

dn0 = n_t[:, 0] - nbar
dnT = n_t[:, -1] - nbar
pG  = Gamma(nbar)
p_naive = par.e * nbar * par.E_ext / np.maximum(pG, 1e-16)

l2_0, li_0 = norms(dn0)
l2_T, li_T = norms(dnT)
l2_p, li_p = norms(p_t[:, -1] - p_naive)

print(f"||n - nbar||_L2:  t=0 -> {l2_0:.3e},   t_end -> {l2_T:.3e}")
print(f"||n - nbar||_Linf: t=0 -> {li_0:.3e},  t_end -> {li_T:.3e}")
print(f"Final p vs naive (ignoring gradients)  L2={l2_p:.3e}, Linf={li_p:.3e}")
print('Note: discrepancy from "naive" comes from pressure & electrostatic gradient terms.')
