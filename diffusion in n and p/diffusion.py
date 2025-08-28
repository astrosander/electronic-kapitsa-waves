import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp
import os

@dataclass
class P:
    m: float = 1.0
    e: float = 1.0
    n0: float = 1.0
    U: float = 0.08
    Gamma0: float = 0.08
    w: float = 1.0
    E: float = 0.0
    eps: float = 20.0
    Dn: float = 0.05
    Dp: float = 0.10
    use_full: bool = True
    include_pressure: bool = True
    include_poisson: bool = False
    include_advective_p: bool = True
    L: float = 200.0
    Nx: int = 512
    t_final: float = 160.0
    n_save: int = 320
    rtol: float = 1e-6
    atol: float = 1e-8
    n_floor: float = 1e-9
    A_n: float = 0.10
    A_p: float = 0.0
    x0: float = 60.0
    sigma: float = 6.0
    u_background: float = 0.0
    outdir: str = "out_diffusive"
    cmap: str = "viridis"

par = P()

x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k = 2.0*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j * k
k2 = k**2

def Dx(f):   return (ifft(ik * fft(f))).real
def Dxx(f):  return (ifft((-k2) * fft(f))).real

def Gamma(n):
    return par.Gamma0 * np.exp(-n/par.w)

def Pi0(n):
    return 0.5 * par.U * n**2

def phi_from_n(n):
    rhs_hat = fft((par.e/par.eps) * (n - par.n0))
    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    nonzero = (k2 != 0)
    phi_hat[nonzero] = rhs_hat[nonzero] / (-k2[nonzero])
    phi_hat[~nonzero] = 0.0
    return (ifft(phi_hat)).real

def rhs(t, y):
    N = par.Nx
    n = y[:N]
    p = y[N:]

    n_eff = np.maximum(n, par.n_floor)
    v = p / (par.m * n_eff)

    dn_dt = -Dx(p) + par.Dn * Dxx(n)

    adv_term = (p/n_eff) * Dx(p) if par.include_advective_p else 0.0

    press_grad = 0.0
    if par.use_full and par.include_pressure:
        press_grad = Dx(Pi0(n_eff) + (p**2)/(par.m * n_eff))

    force_poisson = 0.0
    if par.use_full and par.include_poisson:
        phi = phi_from_n(n)
        force_poisson = n_eff * Dx(phi)

    dp_dt = -Gamma(n_eff)*p + par.e * n_eff * par.E + par.Dp * Dxx(p)

    dp_dt -= press_grad
    dp_dt -= adv_term
    dp_dt -= force_poisson

    if not par.use_full:
        dp_dt = -Gamma(n_eff)*p + par.e*n_eff*par.E + par.Dp*Dxx(p)
        if par.include_advective_p:
            dp_dt -= (p/n_eff) * Dx(p)

    return np.concatenate([dn_dt, dp_dt])

def gaussian(d, sigma):
    return np.exp(-0.5*(d/sigma)**2)

def periodic_delta(x, x0, L):
    return (x - x0 + 0.5*L) % L - 0.5*L

def initial_fields_solitary():
    d = periodic_delta(x, par.x0, par.L)
    n = par.n0 + par.A_n * gaussian(d, par.sigma)
    u = par.u_background
    p = par.m * n * u + par.A_p * gaussian(d, par.sigma)
    return n, p

def run_once(n0, p0, tag="solitary"):
    os.makedirs(par.outdir, exist_ok=True)
    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)
    sol = solve_ivp(rhs, (0.0, par.t_final), y0, t_eval=t_eval,
                    method="BDF", rtol=par.rtol, atol=par.atol)

    N = par.Nx
    n_t = sol.y[:N, :]
    p_t = sol.y[N:, :]
    u_t = p_t / (par.m*np.maximum(n_t, par.n_floor))

    plt.figure(figsize=(9.5, 4.3))
    extent = [x.min(), x.max(), sol.t.min(), sol.t.max()]
    plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.xlabel("x"); plt.ylabel("t"); plt.title(f"n(x,t)  [{tag}]")
    plt.colorbar(label="n")
    plt.tight_layout()
    plt.savefig(f"{par.outdir}/spacetime_n_{tag}.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9.5, 3.4))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        j = int(frac*(len(sol.t)-1))
        plt.plot(x, n_t[:, j], label=f"t={sol.t[j]:.1f}")
    plt.legend(); plt.xlabel("x"); plt.ylabel("n"); plt.title(f"Density snapshots  [{tag}]")
    plt.tight_layout(); plt.savefig(f"{par.outdir}/snapshots_n_{tag}.png", dpi=160); plt.close()

    jmax = np.argmax(n_t, axis=0)
    x_max = x[jmax]
    x_unwrap = np.unwrap(2*np.pi*x_max/par.L) * (par.L/(2*np.pi))
    coef = np.polyfit(sol.t, x_unwrap, 1)
    c_crest = coef[0]

    print(f"[{tag}] crest speed ≈ {c_crest:.4f}  (units of x/t)")
    return sol.t, n_t, p_t, u_t, c_crest

def measure_sigma_for_mode(m, A=1e-3, t_short=12.0):
    kx = 2*np.pi*m/par.L
    n0 = par.n0 + A*np.cos(kx*x)
    p0 = par.m * n0 * par.u_background
    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, t_short, 60)
    sol = solve_ivp(rhs, (0.0, t_short), y0, t_eval=t_eval,
                    method="BDF", rtol=par.rtol, atol=par.atol)
    N = par.Nx
    n_t = sol.y[:N, :]
    n_hat_t = fft(n_t, axis=0)[m, :]
    amp = np.abs(n_hat_t)
    s0 = max(2, int(0.1*len(sol.t)))
    s1 = int(0.6*len(sol.t))
    coeff = np.polyfit(sol.t[s0:s1], np.log(amp[s0:s1] + 1e-30), 1)
    sigma = coeff[0]
    return kx, sigma

def scan_increment_over_k(m_list):
    k_vals, sigmas = [], []
    for m in m_list:
        kx, s = measure_sigma_for_mode(m)
        k_vals.append(kx); sigmas.append(s)
        print(f"[k-scan] m={m:3d}, k={kx:.4f}, sigma≈{s:+.4e}")
    k_vals = np.array(k_vals); sigmas = np.array(sigmas)
    plt.figure(figsize=(7.4, 4.2))
    plt.plot(k_vals, sigmas, "o-")
    plt.axhline(0, color="k", lw=1, ls="--")
    plt.xlabel("k"); plt.ylabel("σ(k)  (early-time increment)")
    plt.title("Increment vs wavenumber")
    plt.tight_layout(); plt.savefig(f"{par.outdir}/sigma_vs_k.png", dpi=160); plt.close()
    return k_vals, sigmas

if __name__ == "__main__":
    os.makedirs(par.outdir, exist_ok=True)

    par.use_full = True
    par.include_pressure = True
    par.include_poisson = False
    par.include_advective_p = True

    par.Dn = 0.05
    par.Dp = 0.10
    par.E  = 0.00

    n0, p0 = initial_fields_solitary()
    t, n_t, p_t, u_t, c_crest = run_once(n0, p0, tag="solitary")

    m_list = list(range(1, 24, 2))
    scan_increment_over_k(m_list)

    par.use_full = False
    par.Dn, par.Dp = 0.06, 0.12
    par.E  = 0.10
    n0, p0 = initial_fields_solitary()
    run_once(n0, p0, tag="reduced")
