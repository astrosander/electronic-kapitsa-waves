import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp

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
    E_ext: float = 0.10
    L: float = 100.0
    Nx: int = 128
    t_final: float = 120.0
    n_save: int = 200
    rtol: float = 1e-7
    atol: float = 1e-9
    n_floor: float = 1e-12
    save_frames: bool = True
    frames_dir: str = "frames_uniformE"
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

def phi_from_n(n):
    rhs_hat = fft( (par.e/par.epsilon) * n )
    phi_hat = inv_k2 * rhs_hat
    phi_hat[0] = 0.0
    return ifft(phi_hat).real

def initial_fields():
    n0 = np.full_like(x, par.n0)
    p0 = np.full_like(x, par.p0)
    return n0, p0

def rhs_pde(t, y):
    N = par.Nx
    n = y[:N]; p = y[N:]
    n_eff = np.maximum(n, par.n_floor)
    v = p/(par.m*n_eff)

    dn_dt = -Dx(n_eff * v)

    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)

    phi = phi_from_n(n)
    Ex_int = -Dx(phi)
    Ex = par.E_ext + Ex_int

    dp_dt = -Gamma(n_eff)*p - Dx(Pi) + par.e*n_eff*Ex

    return np.concatenate([dn_dt, dp_dt])

def run_once():
    n0, p0 = initial_fields()
    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)
    sol = solve_ivp(rhs_pde, (0.0, par.t_final), y0, t_eval=t_eval,
                    method="BDF", rtol=par.rtol, atol=par.atol)
    N = par.Nx
    n_t = sol.y[:N, :]
    p_t = sol.y[N:, :]
    u_t = p_t/(par.m*np.maximum(n_t, par.n_floor))
    return sol.t, n_t, p_t, u_t

def plot_spacetime_p(t, p_t, title="p(x,t) under uniform E (n uniform)"):
    extent = [x.min(), x.max(), t.min(), t.max()]
    plt.figure(figsize=(10,4.6))
    plt.imshow(p_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.colorbar(label="p")
    plt.xlabel("x"); plt.ylabel("t"); plt.title(title)
    plt.tight_layout()
    plt.savefig("spacetime_p.png", dpi=160)

def save_animation_frames_uniform(t, n_t, p_t):
    import os
    os.makedirs(par.frames_dir, exist_ok=True)
    G0 = Gamma(par.n0) if np.isscalar(par.n0) else Gamma(float(np.mean(par.n0)))
    p_ss = par.e * par.n0 * par.E_ext / G0
    p_mean = p_t.mean(axis=0)
    p_mean_analytic = p_ss + (par.p0 - p_ss) * np.exp(-G0 * t)

    for j in range(len(t)):
        fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        axs[0].plot(x, n_t[:, j], lw=2)
        axs[0].set_ylabel("n(x,t)")
        axs[0].set_title(f"t = {t[j]:.2f}")
        axs[0].set_ylim(0.9*par.n0, 1.1*par.n0)

        axs[1].plot(x, p_t[:, j], lw=2, label="p(x,t)")
        axs[1].axhline(p_ss, ls="--", lw=1.6, label=r"$p_{\rm ss}=en_0E/\Gamma(n_0)$")
        axs[1].set_ylabel("p(x,t)"); axs[1].set_xlabel("x")
        axs[1].legend(loc="best")
        inset = axs[1].inset_axes([0.62, 0.15, 0.35, 0.75])
        inset.plot(t[:j+1], p_mean[:j+1], lw=1.8)
        inset.plot(t[:j+1], p_mean_analytic[:j+1], lw=1.2, ls="--")
        inset.set_title(r"$\langle p\rangle(t)$")
        inset.set_xlim(t[0], t[-1])
        inset.set_ylim(min(p_mean_analytic.min(), p_mean.min())*1.05,
                       max(p_mean_analytic.max(), p_mean.max())*1.05)
        inset.tick_params(labelsize=8)

        plt.tight_layout()
        fname = f"{par.frames_dir}/frame_{j:04d}.png"
        plt.savefig(fname, dpi=140); plt.close()

t, n_t, p_t, u_t = run_once()

plot_spacetime_p(t, p_t, title="Uniform density; p relaxes to enE/Γ")

if par.save_frames:
    save_animation_frames_uniform(t, n_t, p_t)

G0 = par.Gamma0 * np.exp(-par.n0/par.w)
p_ss = par.e * par.n0 * par.E_ext / G0
print(f"Gamma(n0) = {G0:.6f},  p_ss = en0E/Gamma = {p_ss:.6f}")
print(f"p_mean(t=0)={p_t.mean(axis=0)[0]:.6f},  p_mean(t_end)≈{p_t.mean(axis=0)[-1]:.6f}")
