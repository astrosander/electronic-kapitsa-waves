import os
NTHREADS = int(os.environ.get("NTHREADS", os.cpu_count() or 1))
os.environ["OMP_NUM_THREADS"] = str(NTHREADS)
os.environ["MKL_NUM_THREADS"] = str(NTHREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NTHREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NTHREADS)
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from scipy.fft import fft, ifft, fftfreq, set_workers
from scipy.integrate import solve_ivp
from scipy import linalg as _sla

try:
    _sla.set_blas_num_threads(NTHREADS)
except Exception:
    pass
try:
    _sla.set_lapack_num_threads(NTHREADS)
except Exception:
    pass

try:
    from threadpoolctl import threadpool_limits, threadpool_info
except Exception:
    class _NoOpCtx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def threadpool_limits(*a, **k): return _NoOpCtx()
    def threadpool_info(): return []

for lib in threadpool_info():
    print("[threadpool]", lib.get("internal_api"), "threads=", lib.get("num_threads"),
          "lib=", lib.get("filename"))
@dataclass
class P:
    m: float = 1.0
    e: float = 1.0
    U: float = 0.5
    nbar0: float = 1.0
    Gamma0: float = 2.50
    w: float = 0.5
    include_poisson: bool = False
    eps: float = 20.0

    u_d: float = 0.42
    maintain_drift: str = "field"
    Kp: float = 0.15

    Dn: float = 0.2
    Dp: float = 0.2

    J0: float = 1.0
    sigma_J: float = 2.0**1/2
    x0: float = 5.0
    source_model: str = "as_given"

    use_nbar_gaussian: bool = False
    nbar_amp: float = 0.0
    nbar_sigma: float = 120.0

    L: float = 80.0
    Nx: int = 512
    t_final: float = 25.0
    n_save: int = 3600
    rtol: float = 1e-3
    atol: float = 1e-7
    n_floor: float = 1e-7
    dealias_23: bool = True

    seed_amp_n: float = 2e-1
    seed_mode: int = 3
    seed_amp_p: float = 2e-1

    outdir: str = "out_drift"
    cmap: str = "inferno"

par = P()
x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j*k
k2 = k**2

def Dx(f):
    return (ifft(ik * fft(f))).real

def Dxx(f):
    return (ifft((-k2) * fft(f))).real

def filter_23(f):
    if not par.dealias_23:
        return f
    fh = fft(f)
    kc = par.Nx//3
    fh[kc:-kc] = 0.0
    return (ifft(fh)).real

def Gamma(n):
    return par.Gamma0 * np.exp(-np.maximum(n, par.n_floor)/par.w)

def Pi0(n):
    return 0.5 * par.U * n**2

def phi_from_n(n, nbar):
    rhs_hat = fft((par.e/par.eps) * (n - nbar))
    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    nz = (k2 != 0)
    phi_hat[nz] = rhs_hat[nz] / (-k2[nz])
    return (ifft(phi_hat)).real

def periodic_delta(x, x0, L): return (x - x0 + 0.5*L) % L - 0.5*L

def nbar_profile():
    if par.use_nbar_gaussian and par.nbar_amp != 0.0:
        d = periodic_delta(x, par.x0, par.L)
        return par.nbar0 + par.nbar_amp * np.exp(-0.5*(d/par.nbar_sigma)**2)
    else:
        return np.full_like(x, par.nbar0)

def pbar_profile(nbar):
    return par.m * nbar * par.u_d

def J_profile():
    d = periodic_delta(x, par.x0, par.L)
    return par.J0 * np.exp(-0.5*(d/par.sigma_J)**2)

def gamma_from_J(Jx):
    return np.trapz(Jx, x)/par.L

def S_injection(n, nbar, Jx, gamma):
    if par.source_model == "as_given":
        return Jx * nbar - gamma * (n - nbar)
    elif par.source_model == "balanced":
        return Jx * nbar - gamma * n
    else:
        raise ValueError("source_model must be 'as_given' or 'balanced'")

def E_base_from_drift(nbar):
    return par.m * par.u_d * np.mean(Gamma(nbar)) / par.e / 0.8187307530779819 * 40.0

def rhs(t, y, E_base):
    N = par.Nx
    n = y[:N]
    p = y[N:]

    nbar = nbar_profile()

    n_eff = np.maximum(n, par.n_floor)

    Jx = J_profile()
    gamma = gamma_from_J(Jx)
    SJ = S_injection(n_eff, nbar, Jx, gamma)

    v = p/(par.m*n_eff)
    u_mean = float(np.mean(v))
    if par.maintain_drift == "feedback":
        E_eff = E_base + par.Kp * (par.u_d - u_mean)
    else:
        E_eff = E_base

    dn_dt = -Dx(p) + par.Dn * Dxx(n) + SJ * 0 + par.u_d*Dx(p)
    dn_dt = filter_23(dn_dt)

    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)
    grad_Pi = Dx(Pi)
    force_Phi = 0.0
    if par.include_poisson:
        phi = phi_from_n(n_eff, nbar)
        force_Phi = n_eff * Dx(phi)

    dp_dt = -Gamma(n_eff)*p - grad_Pi + par.e*n_eff*E_eff - force_Phi + par.Dp * Dxx(p) + par.u_d*Dx(n)
    dp_dt = filter_23(dp_dt)

    return np.concatenate([dn_dt, dp_dt])

def initial_fields():
    nbar = nbar_profile()
    pbar = pbar_profile(nbar)
    n0 = nbar.copy()
    p0 = pbar.copy()
    if par.seed_amp_n != 0.0 and par.seed_mode != 0:
        kx = 2*np.pi*par.seed_mode / par.L
        n0 += par.seed_amp_n * np.cos(kx * x)
    if par.seed_amp_p != 0.0 and par.seed_mode != 0:
        kx = 2*np.pi*par.seed_mode / par.L
        p0 += par.seed_amp_p * np.cos(kx * x)
    return n0, p0

def run_once(tag="drift"):
    os.makedirs(par.outdir, exist_ok=True)

    n0, p0 = initial_fields()
    E_base = E_base_from_drift(nbar_profile()) if par.maintain_drift in ("field","feedback") else 0.0
    E_base = 15.0

    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)

    with threadpool_limits(limits=NTHREADS, user_api=("blas", "lapack", "openmp")):
        with set_workers(NTHREADS):
            sol = solve_ivp(lambda t,y: rhs(t,y,E_base),
                            (0.0, par.t_final), y0, t_eval=t_eval,
                            method="Radau", rtol=par.rtol, atol=par.atol)

    N = par.Nx
    n_t = sol.y[:N,:]
    p_t = sol.y[N:,:]

    n_eff_t = np.maximum(n_t, par.n_floor)
    v_t = p_t/(par.m*n_eff_t)
    print(f"[run]  <u>(t=0)={np.mean(v_t[:,0]):.4f},  <u>(t_end)={np.mean(v_t[:,-1]):.4f},  target u_d={par.u_d:.4f}")

    extent=[x.min(), x.max(), sol.t.min(), sol.t.max()]
    plt.figure(figsize=(9.6,4.3))
    plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.xlabel("x"); plt.ylabel("t"); plt.title(f"n(x,t)  [lab]  {tag}")
    plt.colorbar(label="n")
    plt.plot([par.x0, par.x0], [sol.t.min(), sol.t.max()], 'w--', lw=1, alpha=0.7)
    plt.tight_layout(); plt.savefig(f"{par.outdir}/spacetime_n_lab_{tag}.png", dpi=160); plt.close()

    n_co = np.empty_like(n_t)
    for j, tj in enumerate(sol.t):
        shift = (par.u_d * tj) % par.L
        s_idx = int(np.round(shift/dx)) % par.Nx
        n_co[:, j] = np.roll(n_t[:, j], -s_idx)
    plt.figure(figsize=(9.6,4.3))
    plt.imshow(n_co.T, origin="lower", aspect="auto",
               extent=[x.min(), x.max(), sol.t.min(), sol.t.max()], cmap=par.cmap)
    plt.xlabel("ξ = x - u_d t"); plt.ylabel("t"); plt.title(f"n(ξ,t)  [co-moving u_d={par.u_d}]  {tag}")
    plt.colorbar(label="n"); plt.tight_layout()
    plt.savefig(f"{par.outdir}/spacetime_n_comoving_{tag}.png", dpi=160); plt.close()

    plt.figure(figsize=(9.6,3.4))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        j = int(frac*(len(sol.t)-1))
        plt.plot(x, n_t[:,j], label=f"t={sol.t[j]:.1f}")
    plt.legend(); plt.xlabel("x"); plt.ylabel("n"); plt.title(f"Density snapshots  {tag}")
    plt.tight_layout();plt.savefig(f"{par.outdir}/snapshots_n_{tag}.pdf", dpi=160); plt.savefig(f"{par.outdir}/snapshots_n_{tag}.png", dpi=160); plt.close()

    return sol.t, n_t, p_t

def measure_sigma_for_mode(m_pick=3, A=1e-3, t_short=35.0):
    oldA, oldm = par.seed_amp_n, par.seed_mode
    par.seed_amp_n, par.seed_mode = A, m_pick
    t, n_t, _ = run_once(tag=f"sigma_m{m_pick}")
    par.seed_amp_n, par.seed_mode = oldA, oldm

    nhat_t = fft(n_t, axis=0)[m_pick, :]
    amp = np.abs(nhat_t)
    i0 = max(2, int(0.1*len(t))); i1 = int(0.5*len(t))
    slope = np.polyfit(t[i0:i1], np.log(amp[i0:i1] + 1e-30), 1)[0]
    print(f"[sigma] mode m={m_pick}, sigma≈{slope:+.3e}")
    return slope

if __name__ == "__main__":
    os.makedirs(par.outdir, exist_ok=True)

    par.maintain_drift = "feedback"
    par.include_poisson = False
    par.source_model = "as_given"

    run_once(tag="drift_run")
