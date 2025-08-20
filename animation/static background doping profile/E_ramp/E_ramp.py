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
    U: float = 0.5
    Gamma0: float = 0.05
    w: float = 1.0
    epsilon: float = 15.0
    E0: float = 0.05
    E1: float = 0.90
    t_ramp: float = 80.0
    n1: float = 0.05
    lam: float = 0.005/4
    x0: float = 0.55
    L: float = 100.0
    Nx: int = 256
    t_final: float = 160.0
    n_save: int = 240
    rtol: float = 1e-7
    atol: float = 1e-9
    n_floor: float = 1e-12
    save_frames: bool = True
    frames_dir: str = "frames_E_ramp"
    cmap: str = "inferno"

par = Par()

x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j*k
inv_k2 = np.zeros_like(k, dtype=np.complex128)
nz = (k != 0); inv_k2[nz] = 1.0/(k[nz]**2)

def Dx(f): 
    return (ifft(ik * fft(f))).real

def Gamma(n): return par.Gamma0 * np.exp(-n/par.w)
def Pi0(n):   return 0.5 * par.U * n**2

def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s*s*(3 - 2*s)

def E_of_t(t):
    return par.E0 + (par.E1 - par.E0) * smoothstep(t / par.t_ramp)

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

    dn_dt = -Dx(n_eff * v)

    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)

    nbar = nbar_profile()
    phi  = phi_from_n_with_bg(n, nbar)
    Ex   = E_of_t(t) - Dx(phi)

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
    return sol.t, n_t, p_t, nbar

def spacetime_p(t, p_t, title="p(x,t) under ramped E(t)"):
    extent=[x.min(), x.max(), t.min(), t.max()]
    plt.figure(figsize=(10,4.6))
    plt.imshow(p_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.colorbar(label="p")
    plt.xlabel("x"); plt.ylabel("t"); plt.title(title)
    plt.tight_layout(); plt.savefig("spacetime_p_Eramp.png", dpi=160); plt.close()

def save_animation_frames(t, n_t, p_t, nbar):
    os.makedirs(par.frames_dir, exist_ok=True)
    Gbar = Gamma(nbar)

    for j in range(len(t)):
        n = n_t[:, j]
        p = p_t[:, j]
        E_now = E_of_t(t[j])
        p_naive = par.e * nbar * E_now / np.maximum(Gbar, 1e-16)

        n_eff = np.maximum(n, par.n_floor)
        Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)
        phi = phi_from_n_with_bg(n, nbar)
        Ex  = E_now - Dx(phi)
        R = Gamma(n_eff)*p + Dx(Pi) - par.e*n_eff*Ex

        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        axs[0].plot(x, n - nbar, lw=2)
        axs[0].axhline(0, ls='--', lw=1)
        axs[0].set_ylabel(r"$\delta n$")
        axs[0].set_title(f"E(t)={E_now:.3f},  t={t[j]:.2f}")

        axs[1].plot(x, p, lw=2, label="p(x,t)")
        axs[1].plot(x, p_naive, lw=1.4, ls='--', label=r"$p_{\rm naive}(x,t)=e\,\bar n\,E(t)/\Gamma(\bar n)$")
        axs[1].set_ylabel("p")
        axs[1].legend(loc="best", fontsize=9)

        axs[2].plot(x, R, lw=1.8)
        axs[2].set_ylabel("residual")
        axs[2].set_xlabel("x")

        plt.tight_layout()
        fname = f"{par.frames_dir}/frame_{j:04d}.png"
        plt.savefig(fname, dpi=140); plt.close()

t, n_t, p_t, nbar = run_once()

spacetime_p(t, p_t, title="Ramped field: p(x,t) increases with E(t)")

if par.save_frames:
    save_animation_frames(t, n_t, p_t, nbar)

G0 = par.Gamma0 * np.exp(-par.n0/par.w)
print(f"[E ramp] E(0)={par.E0:.3f} → E(t_ramp+)={par.E1:.3f},  Γ(n0)≈{G0:.4f}")
p_mean_0  = p_t[:,0].mean()
p_mean_T  = p_t[:,-1].mean()
p_naive_T = (par.e * nbar * par.E1 / np.maximum(Gamma(nbar), 1e-16)).mean()
print(f"p_mean(t=0)≈{p_mean_0:.4f},  p_mean(t_end)≈{p_mean_T:.4f},  ⟨p_naive(E1)⟩≈{p_naive_T:.4f}")
