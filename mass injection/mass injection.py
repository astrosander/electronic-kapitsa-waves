import os

NTHREADS = int(os.environ.get("NTHREADS", "4"))
os.environ["OMP_NUM_THREADS"] = str(NTHREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NTHREADS)
os.environ["MKL_NUM_THREADS"] = str(NTHREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NTHREADS)

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.fft import fft, ifft, fftfreq, set_workers
from scipy.integrate import solve_ivp
import glob
from dataclasses import asdict
try:
    from threadpoolctl import threadpool_limits
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False
    print(f"[Warning] threadpoolctl not available, install with: pip install threadpoolctl")

try:
    from scipy import linalg as _sla
    _sla.set_blas_num_threads(NTHREADS)
    _sla.set_lapack_num_threads(NTHREADS)
except Exception:
    pass

print(f"[Threading] Using {NTHREADS} threads for FFTs and linear algebra")

@dataclass
class P:
    m: float = 1.0
    e: float = 1.0
    U: float = 1.0#0.06
    nbar0: float = 0.2
    Gamma0: float = 2.50#0.08
    w: float = 5.0
    include_poisson: bool = False
    eps: float = 20.0

    u_d: float = 20.0
    maintain_drift: str = "field"
    Kp: float = 0.15

    Dn: float = 0.5#/10#0.03
    Dp: float = 0.1

    J0: float = 1.0#0.04
    sigma_J: float = 2.0**1/2#6.0
    x0: float = 5.0
    source_model: str = "as_given"

    use_nbar_gaussian: bool = False
    nbar_amp: float = 0.0
    nbar_sigma: float = 120.0

    L: float = 10.0
    Nx: int = 812#812
    t_final: float = 20.0
    n_save: int = 360
    # rtol: float = 5e-7
    # atol: float = 5e-9
    rtol = 1e-3
    atol = 1e-7
    n_floor: float = 1e-7
    dealias_23: bool = True

    seed_amp_n: float = 0e-3
    seed_mode: int = 1
    seed_amp_p: float = 0e-3

    outdir: str = "out_drift"
    cmap: str = "inferno"

    # ---- NEW: time-dependent perturbation U0 * nbar(x) = [lambda0 + lambda1*cos(nu*t)] * exp( - (x-x0)^2 / (2*sigma^2) ) ----
    use_static_perturbation: bool = True
    lambda0: float = 0.15          # base amplitude of U0 * nbar(x)
    lambda1: float = 0.1           # modulation amplitude for time-dependent part
    nu: float = 1.0                # driving frequency for time modulation
    sigma_static: float = 1.0     # sigma for the Gaussian in x
    set_static_equilibrium: bool = False  # if True: n0 = (U0/U) * nbar(x) at t=0

par = P()

x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j*k
k2 = k**2

def Dx(f):  return (ifft(ik * fft(f, workers=NTHREADS), workers=NTHREADS)).real
def Dxx(f): return (ifft((-k2) * fft(f, workers=NTHREADS), workers=NTHREADS)).real

def filter_23(f):
    if not par.dealias_23: return f
    fh = fft(f, workers=NTHREADS)
    kc = par.Nx//3
    fh[kc:-kc] = 0.0
    return (ifft(fh, workers=NTHREADS)).real

def Gamma(n):
    return par.Gamma0 * np.exp(-np.maximum(n, par.n_floor)/par.w)

def Pi0(n):
    return 0.5 * par.U * n**2

def phi_from_n(n, nbar):
    rhs_hat = fft((par.e/par.eps) * (n - nbar), workers=NTHREADS)
    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    nz = (k2 != 0)
    phi_hat[nz] = rhs_hat[nz] / (-k2[nz])
    return (ifft(phi_hat, workers=NTHREADS)).real

def periodic_delta(x, x0, L): return (x - x0 + 0.5*L) % L - 0.5*L

def _power_spectrum_1d(n_slice, L):
    """Return one-sided k>0 spectrum of a single spatial slice."""
    N = n_slice.size
    dn = n_slice - np.mean(n_slice)
    nhat = fft(dn, workers=NTHREADS)
    P = (nhat * np.conj(nhat)).real / (N*N)
    m = np.arange(N//2 + 1)
    kpos = 2*np.pi*m / L
    return kpos[1:], P[1:N//2+1]

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

def U0nbar_profile(t=0.0):
    """
    Time-dependent model: U0 * nbar(x) = [lambda0 + lambda1*cos(nu*t)] * exp( - (x-x0)^2 / (2*sigma_static^2) ).
    If use_static_perturbation is False, returns a uniform zero field.
    
    Parameters:
    t : float
        Current time for time-dependent modulation
    """
    if not par.use_static_perturbation:
        return np.zeros_like(x)
    
    d = periodic_delta(x, par.x0, par.L)
    spatial_profile = np.exp(-0.5*(d/par.sigma_static)**2)
    
    # Time-dependent amplitude: lambda(t) = lambda0 + lambda1*cos(nu*t)
    lambda_t = par.lambda0 + par.lambda1 * np.cos(par.nu * t)
    
    return lambda_t * spatial_profile

def S_injection(n, nbar, Jx, gamma):
    if par.source_model == "as_given":
        return Jx * nbar - gamma * (n - nbar)
    elif par.source_model == "balanced":
        return Jx * nbar - gamma * n
    else:
        raise ValueError("source_model must be 'as_given' or 'balanced'")

def E_base_from_drift(nbar):
    return par.m * par.u_d * np.mean(Gamma(nbar)) / par.e

def rhs(t, y, E_base):
    N = par.Nx
    n = y[:N]
    p = y[N:]

    nbar = nbar_profile()
    pbar = pbar_profile(nbar)

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

    dn_dt = -Dx(p) + par.Dn * Dxx(n) + SJ *0
    dn_dt = filter_23(dn_dt)

    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)
    grad_Pi = Dx(Pi)
    force_Phi = 0.0
    if par.include_poisson:
        phi = phi_from_n(n_eff, nbar)
        force_Phi = n_eff * Dx(phi)

    # ---- NEW: screening force n * ∂x[ U0 nbar(x,t) ] ----
    grad_U0nbar = Dx(U0nbar_profile(t))

    dp_dt = -Gamma(n_eff)*p - grad_Pi + par.e*n_eff*E_eff - force_Phi + par.Dp * Dxx(p) + n_eff * grad_U0nbar     # <— Eq. (14) term
    dp_dt = filter_23(dp_dt)

    return np.concatenate([dn_dt, dp_dt])

def initial_fields():
    nbar = nbar_profile()
    pbar = pbar_profile(nbar)

    # default initial fields
    n0 = nbar.copy()
    p0 = pbar.copy()

    # ---- NEW: set n0 from Eq. (13) when requested (best used with no drift) ----
    if par.set_static_equilibrium:
        n0 = (U0nbar_profile(0.0) / max(par.U, 1e-15))  # n(x) = [U0 nbar(x,t=0)] / U
        p0[:] = 0.0                                     # fluid at rest
    if par.seed_amp_n != 0.0 and par.seed_mode != 0:
        if par.seed_mode == 1:
            kx1 = 2*np.pi*3 / par.L
            kx2 = 2*np.pi*5 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x)+np.cos(kx2 * x))
        if par.seed_mode == 2:
            kx1 = 2*np.pi*5 / par.L
            kx2 = 2*np.pi*8 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x)+np.cos(kx2 * x))
        if par.seed_mode == 3:
            kx1 = 2*np.pi*8 / par.L
            kx2 = 2*np.pi*13 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x)+np.cos(kx2 * x))
        if par.seed_mode == 4:
            kx1 = 2*np.pi*13 / par.L
            kx2 = 2*np.pi*21 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x)+np.cos(kx2 * x))
        if par.seed_mode == 5:
            kx1 = 2*np.pi*21 / par.L
            kx2 = 2*np.pi*34 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x)+np.cos(kx2 * x))
        if par.seed_mode == 6:
            kx1 = 2*np.pi*34 / par.L
            kx2 = 2*np.pi*55 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x)+np.cos(kx2 * x))

    if par.seed_amp_p != 0.0 and par.seed_mode != 0:
        if par.seed_mode == 1:
            kx1 = 2*np.pi*3 / par.L
            kx2 = 2*np.pi*5 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x)+np.cos(kx2 * x))
        if par.seed_mode == 2:
            kx1 = 2*np.pi*5 / par.L
            kx2 = 2*np.pi*8 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x)+np.cos(kx2 * x))
        if par.seed_mode == 3:
            kx1 = 2*np.pi*8 / par.L
            kx2 = 2*np.pi*13 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x)+np.cos(kx2 * x))
        if par.seed_mode == 4:
            kx1 = 2*np.pi*13 / par.L
            kx2 = 2*np.pi*21 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x)+np.cos(kx2 * x))
        if par.seed_mode == 5:
            kx1 = 2*np.pi*21 / par.L
            kx2 = 2*np.pi*34 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x)+np.cos(kx2 * x))
        if par.seed_mode == 6:
            kx1 = 2*np.pi*34 / par.L
            kx2 = 2*np.pi*55 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x)+np.cos(kx2 * x))
        # kx = 2*np.pi*par.seed_mode / par.L
        # p0 += par.seed_amp_p * np.cos(kx * x)
    return n0, p0

def save_final_spectra(m, t, n_t, L, tag="", n_avg_time=None):
    k0, P0 = _power_spectrum_1d(n_t[:, 0],  L)
    kf, Pf = _power_spectrum_1d(n_t[:, -1], L)
    
    # Add time-averaged spectrum if provided
    if n_avg_time is not None:
        k_avg, P_avg = _power_spectrum_1d(n_avg_time, L)
    else:
        k_avg, P_avg = None, None

    meta = asdict(par).copy()
    meta['outdir'] = str(par.outdir)

    os.makedirs(par.outdir, exist_ok=True)
    out = os.path.join(par.outdir, f"spec_m{int(m):02d}_{tag}.npz")
    
    save_data = {
        'm': int(m),
        't_final': float(t[-1]),
        'L': float(L),
        'Nx': int(par.Nx),
        'k0': k0, 'P0': P0,
        'k': kf, 'P': Pf,
        'meta': meta
    }
    
    # Add time-averaged data if available
    if k_avg is not None and P_avg is not None:
        save_data['k_avg'] = k_avg
        save_data['P_avg'] = P_avg
    
    np.savez_compressed(out, **save_data)
    print(f"[save] spectra → {out}")

def plot_fft_initial_last(n_t, t, L, tag="compare", k_marks=(), n_avg_time=None):
    """Overlay t=0 and t=t_end spectra; optional vertical k_marks."""
    k0, P0 = _power_spectrum_1d(n_t[:, 0],   L)
    k1, P1 = _power_spectrum_1d(n_t[:, -1],  L)

    i0 = np.argmax(P0); i1 = np.argmax(P1)
    k0_peak, k1_peak = k0[i0], k1[i1]

    plt.figure(figsize=(8.6, 4.2))
    plt.plot(k0, P0, label="t = 0")
    plt.plot(k1, P1, label=f"t = {t[-1]:.2f}")
    
    # Add time-averaged spectrum if provided
    if n_avg_time is not None:
        k_avg, P_avg = _power_spectrum_1d(n_avg_time, L)
        i_avg = np.argmax(P_avg)
        k_avg_peak = k_avg[i_avg]
        plt.plot(k_avg, P_avg, 'k--', lw=2, label="<n>(t=[10,50])")
        plt.plot([k_avg_peak], [P_avg[i_avg]], "d", ms=6, color='k', label=f"peak_avg k={k_avg_peak:.3f}")
    
    plt.plot([k0_peak], [P0[i0]], "o", ms=6, label=f"peak0 k={k0_peak:.3f}")
    plt.plot([k1_peak], [P1[i1]], "s", ms=6, label=f"peak1 k={k1_peak:.3f}")

    for km in k_marks:
        plt.axvline(km, color="k", ls="--", lw=1, alpha=0.6)

    plt.xlabel("$k$")
    plt.ylabel("Power $|\\hat{n}(k)|^2$")
    plt.title("Fourier spectrum of $n(x,t)$: initial vs final vs time-averaged")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False, ncol=2, fontsize=9)
    os.makedirs(par.outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{par.outdir}/fft_compare_{tag}.png", dpi=160)
    plt.savefig(f"{par.outdir}/fft_compare_{tag}.pdf", dpi=160)
    plt.close()

def plot_all_final_spectra(results, L, tag="final_overlay", normalize=False):
    """
    results: list of tuples (m, t, n_t) from run_all_modes_snapshots
    Plots the power spectrum of n(x, t_final) for each run on the SAME axes.
    """
    # Colorblind-friendly palette
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
              '#FF7F00', '#A65628', '#F781BF', '#999999']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    plt.figure(figsize=(7.0, 4.0))
    plt.style.use('default')
    
    for i, (m, t, n_t) in enumerate(results):
        # k, P = _power_spectrum_1d(n_t[:, -1], L)
        k, P = _power_spectrum_1d(n_t[:, 0], L) 
        if normalize and np.max(P) > 0:
            P = P / np.max(P)
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        if m == 1:
            plt.plot(k, P, lw=1.2, color=color, label=f"$\\cos(3x) + \\cos(5x)$")
        elif m == 2:
            plt.plot(k, P, lw=1.2, color=color, label=f"$\\cos(5x) + \\cos(8x)$")
        elif m == 3:
            plt.plot(k, P, lw=1.2, color=color, label=f"$\\cos(8x) + \\cos(15x)$")
        elif m == 4:
            plt.plot(k, P, lw=1.2, color=color, label=f"$\\cos(7x) + \\cos(13x)$")
        else:
            plt.plot(k, P, lw=1.2, color=color, label=f"$\\cos(ax) + \\cos(bx)$")
        
        ip = np.argmax(P)
        plt.plot([k[ip]], [P[ip]], marker=marker, ms=6, color=color,
                     markeredgecolor='white', markeredgewidth=1.0)
    
    plt.xlim(0, 20)
    plt.xlabel("$k$", fontsize=12)
    plt.ylabel("$|\\hat{n}(k)|^2$" + (" (norm.)" if normalize else ""), fontsize=12)
    plt.grid(True, which="both", alpha=0.3, linestyle='--')
    plt.legend(frameon=False, ncol=2, fontsize=9)
    
    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    os.makedirs(par.outdir, exist_ok=True)
    plt.tight_layout()
    plt.title(r"Initial spectra at $t = 0$")
    plt.savefig(f"{par.outdir}/fft_final_overlay_{tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{par.outdir}/fft_final_overlay_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def run_once(tag="seed_mode"):
    os.makedirs(par.outdir, exist_ok=True)

    n0, p0 = initial_fields()
    # E_base = E_base_from_drift(nbar_profile()) if par.maintain_drift in ("field","feedback") else 0.0
    
    E_base=15.0
    print(f"[run] E_base = {E_base:.4f}")

    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)

    if HAS_THREADPOOLCTL:
        with threadpool_limits(limits=NTHREADS, user_api="blas"):
            with set_workers(NTHREADS):
                sol = solve_ivp(lambda t,y: rhs(t,y,E_base),
                                (0.0, par.t_final), y0, t_eval=t_eval,
                                method="BDF", rtol=par.rtol, atol=par.atol)
    else:
        with set_workers(NTHREADS):
            sol = solve_ivp(lambda t,y: rhs(t,y,E_base),
                            (0.0, par.t_final), y0, t_eval=t_eval,
                            method="BDF", rtol=par.rtol, atol=par.atol)

    N = par.Nx
    n_t = sol.y[:N,:]
    p_t = sol.y[N:,:]

    n_eff_t = np.maximum(n_t, par.n_floor)
    v_t = p_t/(par.m*n_eff_t)
    
    # Time averaging over t=[10,50]
    t_start_avg = 10.0
    t_end_avg = 50.0
    
    # Find time indices for averaging window
    i_start = np.argmin(np.abs(sol.t - t_start_avg))
    i_end = np.argmin(np.abs(sol.t - t_end_avg))
    
    # Compute time-averaged quantities
    u_avg_time = np.mean(v_t[:, i_start:i_end+1])  # Average velocity over time window
    n_avg_time = np.mean(n_t[:, i_start:i_end+1], axis=1)  # Time-averaged density profile
    
    print(f"[run]  <u>(t=0)={np.mean(v_t[:,0]):.4f},  <u>(t_end)={np.mean(v_t[:,-1]):.4f},  target u_d={par.u_d:.4f}")
    print(f"[run]  <u>_avg(t=[{t_start_avg},{t_end_avg}])={u_avg_time:.4f}, time window: t[{i_start}:{i_end}] = [{sol.t[i_start]:.1f},{sol.t[i_end]:.1f}]")

    extent=[x.min(), x.max(), sol.t.min(), sol.t.max()]
    plt.figure(figsize=(9.6,4.3))
    plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.xlabel("$x$"); plt.ylabel("$t$"); plt.title(f"$n(x,t)$ [lab] {tag}")
    plt.colorbar(label="$n$")
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
    plt.xlabel("$\\xi = x - u_d t$"); plt.ylabel("$t$"); plt.title(f"$n(\\xi,t)$ [co-moving $u_d={par.u_d}$] {tag}")
    plt.colorbar(label="$n$"); plt.tight_layout()
    plt.savefig(f"{par.outdir}/spacetime_n_comoving_{tag}.png", dpi=160); plt.close()

    plt.figure(figsize=(9.6,3.4))
    
    # Calculate min/max values for display
    n_min_global = n_t.min()
    n_max_global = n_t.max()
    n_min_avg = n_avg_time.min()
    n_max_avg = n_avg_time.max()
    
    for frac in [0.0, 1.0]:
        j = int(frac*(len(sol.t)-1))
        n_snapshot = n_t[:,j]
        n_min_snap = n_snapshot.min()
        n_max_snap = n_snapshot.max()
        plt.plot(x, n_snapshot, label=f"$t={sol.t[j]:.1f}$ [min={n_min_snap:.3f}, max={n_max_snap:.3f}]")
    
    # Add time-averaged profile
    plt.plot(x, n_avg_time, 'k--', lw=2, label=f"$\\langle n\\rangle(t=[{t_start_avg},{t_end_avg}])$ [min={n_min_avg:.3f}, max={n_max_avg:.3f}]")
    
    plt.legend(fontsize=9); plt.xlabel("$x$"); plt.ylabel("$n$"); plt.title(f"Density snapshots {tag}")
    
    # Add global min/max info
    plt.text(0.02, 0.95, f"Global: min={n_min_global:.3f}, max={n_max_global:.3f}", 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.text(0.5, 0.08, f"$D_p={par.Dp}$, $D_n={par.Dn}$, $m={par.seed_mode}$", color="red",
         fontsize=12, ha="right", va="top", transform=plt.gca().transAxes)

    plt.tight_layout(); plt.savefig(f"{par.outdir}/snapshots_n_{tag}.png", dpi=160); plt.close()
    
    # Print values to console
    print(f"[snapshots] Global n: min={n_min_global:.6f}, max={n_max_global:.6f}")
    print(f"[snapshots] t=0: min={n_t[:,0].min():.6f}, max={n_t[:,0].max():.6f}")
    print(f"[snapshots] t_final: min={n_t[:,-1].min():.6f}, max={n_t[:,-1].max():.6f}")
    print(f"[snapshots] Time-averaged: min={n_min_avg:.6f}, max={n_max_avg:.6f}")

    plot_fft_initial_last(n_t, sol.t, par.L, tag=tag, k_marks=(), n_avg_time=n_avg_time)
    
    save_final_spectra(par.seed_mode, sol.t, n_t, par.L, tag=tag, n_avg_time=n_avg_time)

    return sol.t, n_t, p_t

def measure_sigma_for_mode(m_pick=3, A=1e-3, t_short=35.0):
    oldA, oldm = par.seed_amp_n, par.seed_mode
    par.seed_amp_n, par.seed_mode = A, m_pick
    t, n_t, _ = run_once(tag=f"sigma_m{m_pick}")

    par.seed_amp_n, par.seed_mode = oldA, oldm

    nhat_t = fft(n_t, axis=0, workers=NTHREADS)[m_pick, :]
    amp = np.abs(nhat_t)
    i0 = max(2, int(0.1*len(t))); i1 = int(0.5*len(t))
    slope = np.polyfit(t[i0:i1], np.log(amp[i0:i1] + 1e-30), 1)[0]
    print(f"[sigma] mode m={m_pick}, sigma≈{slope:+.3e}")
    return slope


def run_all_modes_snapshots(tag="snapshots_panels"):
    os.makedirs(par.outdir, exist_ok=True)

    modes = range(1,7)
    results = []

    oldA, oldm = par.seed_amp_n, par.seed_mode

    try:
        for m in modes:
            par.seed_mode = m
            t, n_t, _ = run_once(tag=f"m{m}")  
            results.append((m, t, n_t))
            save_final_spectra(m, t, n_t, par.L, tag=f"m{m}")

        fig, axes = plt.subplots(
            len(modes), 1, sharex=True,
            figsize=(10, 12),
            constrained_layout=True
        )
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, (m, t, n_t) in zip(axes, results):
            for frac in [0.0, 1.0]:
                j = int(frac*(len(t)-1))
                ax.plot(x, n_t[:, j], label=f"t={t[j]:.1f}")

            ax.legend(fontsize=8, loc="upper right")
            
            if m == 1:
                ax.set_ylabel(r"$\delta n \sim \cos(3x) + \cos(5x)$")
            elif m == 2:
                ax.set_ylabel(r"$\delta n \sim \cos(5x) + \cos(8x)$")
            elif m == 3:
                ax.set_ylabel(r"$\delta n \sim \cos(8x) + \cos(15x)$")
            elif m == 4:
                ax.set_ylabel(r"$\delta n \sim \cos(7x) + \cos(13x)$")
            # ax.set_ylabel(f"m={m}")
            # ax.text(
            #     -0.02, 0.5, f"m={m}",
            #     transform=ax.transAxes, rotation=90,
            #     va="center", ha="right", color="red", fontsize=11
            # )

        axes[-1].set_xlabel("$x$")

        plt.suptitle(f"Density snapshots for modes $m=1..5$ [{tag}]")
        outpath = f"{par.outdir}/snapshots_panels_{tag}.png"
        plt.savefig(outpath, dpi=160)
        outpath = f"{par.outdir}/snapshots_panels_{tag}.svg"
        plt.savefig(outpath, dpi=160)
        outpath = f"{par.outdir}/snapshots_panels_{tag}.pdf"
        plt.savefig(outpath, dpi=160)
        plt.close()
        print(f"[plot] saved {outpath}")

        plot_all_final_spectra(results, par.L, tag=tag, normalize=False)

    finally:
        par.seed_amp_n, par.seed_mode = oldA, oldm


def run_lambda_panel(lambda0_values, u_d_fixed=2.0, tag="lambda_panel"):
    """
    Run parameter sweep over lambda0 values with fixed u_d
    """
    old_lambda0 = par.lambda0
    old_u_d = par.u_d
    old_outdir = par.outdir
    
    # Set fixed drift velocity
    par.u_d = u_d_fixed
    
    results = []
    
    try:
        for lambda0 in lambda0_values:
            par.lambda0 = lambda0
            par.outdir = f"{old_outdir}_lambda{lambda0:g}_ud{u_d_fixed:g}"
            
            print(f"\n[lambda_panel] Running with lambda0 = {lambda0}, u_d = {u_d_fixed}")
            
            t, n_t, p_t = run_once(tag=f"lambda{lambda0:g}_ud{u_d_fixed:g}")
            results.append((lambda0, t, n_t))
        
        # Create panel plots
        plot_lambda_panel(results, lambda0_values, u_d_fixed, tag=tag)
            
    finally:
        par.lambda0 = old_lambda0
        par.u_d = old_u_d
        par.outdir = old_outdir
    
    return results

def plot_lambda_panel(results, lambda0_values, u_d_fixed, tag="lambda_panel"):
    """
    Create panel plots for different lambda0 values
    """
    n_plots = len(lambda0_values)
    
    # 1. Spacetime panel
    fig, axes = plt.subplots(
        n_plots, 1, 
        figsize=(12, 2.5 * n_plots),
        gridspec_kw={'hspace': 0.15}
    )
    
    if n_plots == 1:
        axes = [axes]
    
    # Find global vmin/vmax for consistent color scaling
    vmin, vmax = float('inf'), float('-inf')
    for lambda0, t, n_t in results:
        vmin = min(vmin, n_t.min())
        vmax = max(vmax, n_t.max())
    
    for i, (lambda0, t, n_t) in enumerate(results):
        ax = axes[i]
        
        extent = [x.min(), x.max(), t.min(), t.max()]
        im = ax.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, 
                      cmap=par.cmap, vmin=vmin, vmax=vmax)
        
        # Add vertical line at x0
        ax.plot([par.x0, par.x0], [t.min(), t.max()], 'w--', lw=1, alpha=0.7)
        
        # Labels
        ax.set_ylabel(f"$\\lambda_0={lambda0}$\n$t$", fontsize=11)
        if i == n_plots - 1:
            ax.set_xlabel("$x$", fontsize=11)
        
        # Add parameter info
        ax.text(0.02, 0.95, f'$\\lambda_0={lambda0}$, $u_d={u_d_fixed}$', 
               transform=ax.transAxes, fontsize=10, alpha=0.9, 
               verticalalignment='top', color='white',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.6))
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, aspect=30, pad=0.02, shrink=0.8)
    cbar.set_label("$n$", fontsize=12)
    
    plt.suptitle(f"Spacetime evolution $n(x,t)$ for different $\\lambda_0$ values ($u_d={u_d_fixed}$)", fontsize=14)
    
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig(f"out_drift/spacetime_lambda_panel_{tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"out_drift/spacetime_lambda_panel_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] saved out_drift/spacetime_lambda_panel_{tag}.png")
    
    # 2. Final density profiles comparison (1x3 panel)
    n_plots = len(lambda0_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4), gridspec_kw={'wspace': 0.3})
    
    if n_plots == 1:
        axes = [axes]
    
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
    
    # Find global y-limits for consistent scaling (including time-averaged data)
    y_min, y_max = float('inf'), float('-inf')
    for lambda0, t, n_t in results:
        n_final = n_t[:, -1]
        y_min = min(y_min, n_final.min())
        y_max = max(y_max, n_final.max())
        
        # Also consider time-averaged data for y-limits
        t_start_avg = 10.0
        t_end_avg = 50.0
        i_start = np.argmin(np.abs(t - t_start_avg))
        i_end = np.argmin(np.abs(t - t_end_avg))
        n_avg_time = np.mean(n_t[:, i_start:i_end+1], axis=1)
        y_min = min(y_min, n_avg_time.min())
        y_max = max(y_max, n_avg_time.max())
    
    for i, (lambda0, t, n_t) in enumerate(results):
        ax = axes[i]
        j = len(t) - 1  # final time
        n_final = n_t[:, j]
        n_min_final = n_final.min()
        n_max_final = n_final.max()
        
        # Compute time-averaged density and its min/max
        t_start_avg = 10.0
        t_end_avg = 50.0
        i_start = np.argmin(np.abs(t - t_start_avg))
        i_end = np.argmin(np.abs(t - t_end_avg))
        n_avg_time = np.mean(n_t[:, i_start:i_end+1], axis=1)
        n_min_avg = n_avg_time.min()
        n_max_avg = n_avg_time.max()
        
        color = colors[i % len(colors)]
        ax.plot(x, n_final, lw=2, color=color, label='Final')
        ax.plot(x, n_avg_time, 'k--', lw=1.5, alpha=0.7, label=f'Avg t=[{t_start_avg},{t_end_avg}]')
        ax.axvline(par.x0, color='k', linestyle='--', alpha=0.5)
        
        # Set consistent y-limits
        ax.set_ylim(y_min * 0.95, y_max * 1.05)
        
        ax.set_xlabel("$x$", fontsize=11)
        if i == 0:
            ax.set_ylabel("$n(x,t)$", fontsize=11)
        
        ax.set_title(f"$\\lambda_0 = {lambda0}$", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add min/max info in bottom-left corner (showing range = max - min)
        range_final = n_max_final - n_min_final
        range_avg = n_max_avg - n_min_avg
        minmax_text = f"Final: $\\Delta n={range_final:.4f}$\nAvg: $\\Delta n={range_avg:.4f}$"
        ax.text(0.02, 0.05, minmax_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='lightblue', alpha=0.8))
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')
        
        # Print to console
        print(f"[panel] λ₀={lambda0}: n_final Δn={range_final:.6f} (min={n_min_final:.6f}, max={n_max_final:.6f})")
        print(f"[panel] λ₀={lambda0}: n_avg Δn={range_avg:.6f} (min={n_min_avg:.6f}, max={n_max_avg:.6f})")
    
    plt.suptitle(f"Final density profiles for different $\\lambda_0$ values ($u_d={u_d_fixed}$)", fontsize=14)
    plt.tight_layout()
    
    plt.savefig(f"out_drift/density_profiles_lambda_panel_{tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"out_drift/density_profiles_lambda_panel_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] saved out_drift/density_profiles_lambda_panel_{tag}.png")
    
    # 3. Fourier spectra comparison
    plt.figure(figsize=(10, 6))
    
    for i, (lambda0, t, n_t) in enumerate(results):
        k, P = _power_spectrum_1d(n_t[:, -1], par.L)  # final spectrum
        color = colors[i % len(colors)]
        plt.plot(k, P, lw=2, color=color, label=f"$\\lambda_0 = {lambda0}$")
    
    plt.xlabel("$k$", fontsize=12)
    plt.ylabel("Power $|\\hat{n}(k)|^2$", fontsize=12)
    plt.title(f"Final Fourier spectra for different $\\lambda_0$ values ($u_d={u_d_fixed}$)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=11)
    plt.xlim(0, 20)
    plt.tight_layout()
    
    plt.savefig(f"out_drift/fft_spectra_lambda_panel_{tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"out_drift/fft_spectra_lambda_panel_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] saved out_drift/fft_spectra_lambda_panel_{tag}.png")

def run_ud_panel(u_d_values, lambda0_fixed=1.0, tag="ud_panel"):
    """
    Run parameter sweep over u_d values with fixed lambda0
    """
    old_lambda0 = par.lambda0
    old_u_d = par.u_d
    old_outdir = par.outdir
    
    # Set fixed lambda0
    par.lambda0 = lambda0_fixed
    
    results = []
    
    try:
        for u_d in u_d_values:
            par.u_d = u_d
            par.outdir = f"{old_outdir}_lambda{lambda0_fixed:g}_ud{u_d:g}"
            
            print(f"\n[ud_panel] Running with u_d = {u_d}, lambda0 = {lambda0_fixed}")
            
            t, n_t, p_t = run_once(tag=f"lambda{lambda0_fixed:g}_ud{u_d:g}")
            results.append((u_d, t, n_t))
        
        # Create panel plots
        plot_ud_panel(results, u_d_values, lambda0_fixed, tag=tag)
            
    finally:
        par.lambda0 = old_lambda0
        par.u_d = old_u_d
        par.outdir = old_outdir
    
    return results

def plot_ud_panel(results, u_d_values, lambda0_fixed, tag="ud_panel"):
    """
    Create panel plots for different u_d values
    """
    n_plots = len(u_d_values)
    
    # 1. Spacetime panel
    fig, axes = plt.subplots(
        n_plots, 1, 
        figsize=(12, 2.5 * n_plots),
        gridspec_kw={'hspace': 0.15}
    )
    
    if n_plots == 1:
        axes = [axes]
    
    # Find global vmin/vmax for consistent color scaling
    vmin, vmax = float('inf'), float('-inf')
    for u_d, t, n_t in results:
        vmin = min(vmin, n_t.min())
        vmax = max(vmax, n_t.max())
    
    for i, (u_d, t, n_t) in enumerate(results):
        ax = axes[i]
        
        extent = [x.min(), x.max(), t.min(), t.max()]
        im = ax.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, 
                      cmap=par.cmap, vmin=vmin, vmax=vmax)
        
        # Add vertical line at x0
        ax.plot([par.x0, par.x0], [t.min(), t.max()], 'w--', lw=1, alpha=0.7)
        
        # Labels
        ax.set_ylabel(f"$u_d={u_d}$\n$t$", fontsize=11)
        if i == n_plots - 1:
            ax.set_xlabel("$x$", fontsize=11)
        
        # Add parameter info
        ax.text(0.02, 0.95, f'$u_d={u_d}$, $\\lambda_0={lambda0_fixed}$', 
               transform=ax.transAxes, fontsize=10, alpha=0.9, 
               verticalalignment='top', color='white',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.6))
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, aspect=30, pad=0.02, shrink=0.8)
    cbar.set_label("$n$", fontsize=12)
    
    plt.suptitle(f"Spacetime evolution $n(x,t)$ for different $u_d$ values ($\\lambda_0={lambda0_fixed}$)", fontsize=14)
    
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig(f"out_drift/spacetime_ud_panel_{tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"out_drift/spacetime_ud_panel_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] saved out_drift/spacetime_ud_panel_{tag}.png")
    
    # 2. Final density profiles comparison (5x1 panel - horizontal layout)
    # Use only first 5 u_d values for cleaner layout
    n_plots_display = min(5, n_plots)
    fig, axes = plt.subplots(1, n_plots_display, figsize=(3.5*n_plots_display, 4), gridspec_kw={'wspace': 0.2})
    
    if n_plots == 1:
        axes = [axes]
    
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628']
    
    # Find global y-limits for consistent scaling (including time-averaged data)
    y_min, y_max = float('inf'), float('-inf')
    for u_d, t, n_t in results:
        n_final = n_t[:, -1]
        y_min = min(y_min, n_final.min())
        y_max = max(y_max, n_final.max())
        
        # Also consider time-averaged data for y-limits
        t_start_avg = 10.0
        t_end_avg = 50.0
        i_start = np.argmin(np.abs(t - t_start_avg))
        i_end = np.argmin(np.abs(t - t_end_avg))
        n_avg_time = np.mean(n_t[:, i_start:i_end+1], axis=1)
        y_min = min(y_min, n_avg_time.min())
        y_max = max(y_max, n_avg_time.max())
    
    for i, (u_d, t, n_t) in enumerate(results):
        ax = axes[i]
        j = len(t) - 1  # final time
        n_final = n_t[:, j]
        n_min_final = n_final.min()
        n_max_final = n_final.max()
        
        # Compute time-averaged density and its min/max
        t_start_avg = 10.0
        t_end_avg = 50.0
        i_start = np.argmin(np.abs(t - t_start_avg))
        i_end = np.argmin(np.abs(t - t_end_avg))
        n_avg_time = np.mean(n_t[:, i_start:i_end+1], axis=1)
        n_min_avg = n_avg_time.min()
        n_max_avg = n_avg_time.max()
        
        color = colors[i % len(colors)]
        ax.plot(x, n_final, lw=2, color=color, label='Final')
        ax.plot(x, n_avg_time, 'k--', lw=1.5, alpha=0.7, label=f'Avg t=[{t_start_avg},{t_end_avg}]')
        ax.axvline(par.x0, color='k', linestyle='--', alpha=0.5)
        
        # Set consistent y-limits
        ax.set_ylim(y_min * 0.95, y_max * 1.05)
        
        ax.set_xlabel("$x$", fontsize=11)
        if i == 0:
            ax.set_ylabel("$n(x,t)$", fontsize=11)
        
        ax.set_title(f"$u_d = {u_d}$", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add min/max info in bottom-left corner (showing range = max - min)
        range_final = n_max_final - n_min_final
        range_avg = n_max_avg - n_min_avg
        minmax_text = f"Final: $\\Delta n={range_final:.4f}$\nAvg: $\\Delta n={range_avg:.4f}$"
        ax.text(0.02, 0.05, minmax_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='lightblue', alpha=0.8))
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')
        
        # Print to console
        print(f"[panel] u_d={u_d}: n_final $\\Delta n={range_final:.6f}$ (min={n_min_final:.6f}, max={n_max_final:.6f})")
        print(f"[panel] u_d={u_d}: n_avg $\\Delta n={range_avg:.6f}$ (min={n_min_avg:.6f}, max={n_max_avg:.6f})")
    
    plt.suptitle(f"Final density profiles for different $u_d$ values ($\\lambda_0={lambda0_fixed}$)", fontsize=14)
    plt.tight_layout()
    
    plt.savefig(f"out_drift/density_profiles_ud_panel_{tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"out_drift/density_profiles_ud_panel_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] saved out_drift/density_profiles_ud_panel_{tag}.png")
    
    # 3. Fourier spectra comparison
    plt.figure(figsize=(10, 6))
    
    for i, (u_d, t, n_t) in enumerate(results):
        k, P = _power_spectrum_1d(n_t[:, -1], par.L)  # final spectrum
        color = colors[i % len(colors)]
        plt.plot(k, P, lw=2, color=color, label=f"$u_d = {u_d}$")
    
    plt.xlabel("$k$", fontsize=12)
    plt.ylabel("Power $|\\hat{n}(k)|^2$", fontsize=12)
    plt.title(f"Final Fourier spectra for different $u_d$ values ($\\lambda_0={lambda0_fixed}$)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=11)
    plt.xlim(0, 20)
    plt.tight_layout()
    
    plt.savefig(f"out_drift/fft_spectra_ud_panel_{tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"out_drift/fft_spectra_ud_panel_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] saved out_drift/fft_spectra_ud_panel_{tag}.png")

def extract_pulse_frequency(n_t, t, x, method="peak_tracking"):
    """
    Extract the characteristic frequency of pulse propagation from simulation data.
    
    Parameters:
    n_t : ndarray
        Density field evolution (space x time)
    t : ndarray
        Time array
    x : ndarray
        Spatial array
    method : str
        Method to extract frequency: "peak_tracking", "spectral", or "correlation"
    
    Returns:
    omega_pulse : float
        Characteristic pulse frequency
    """
    if method == "peak_tracking":
        # Track the motion of density peaks over time
        peak_positions = []
        for j in range(n_t.shape[1]):
            # Find the position of maximum density at each time
            i_max = np.argmax(n_t[:, j])
            peak_positions.append(x[i_max])
        
        peak_positions = np.array(peak_positions)
        
        # Calculate velocity from position differences (accounting for periodicity)
        dt = t[1] - t[0]
        velocities = []
        for j in range(1, len(peak_positions)):
            dx = peak_positions[j] - peak_positions[j-1]
            # Handle periodic boundary conditions
            if dx > par.L/2:
                dx -= par.L
            elif dx < -par.L/2:
                dx += par.L
            velocities.append(dx / dt)
        
        velocities = np.array(velocities)
        # Remove outliers and get average velocity
        v_mean = np.median(velocities)
        
        # Estimate characteristic wavelength from spatial profile
        k_vals, P_vals = _power_spectrum_1d(n_t[:, -1], par.L)
        i_peak = np.argmax(P_vals)
        k_char = k_vals[i_peak]
        lambda_char = 2*np.pi / k_char if k_char > 0 else par.L
        
        # Characteristic frequency: omega = 2*pi*v/lambda
        omega_pulse = 2*np.pi * v_mean / lambda_char
        
    elif method == "spectral":
        # Use temporal Fourier transform at a fixed spatial point
        x_monitor = par.x0  # Monitor point at the perturbation center
        i_monitor = np.argmin(np.abs(x - x_monitor))
        n_temporal = n_t[i_monitor, :]
        
        # Remove mean and window the signal
        n_temporal = n_temporal - np.mean(n_temporal)
        window = np.hanning(len(n_temporal))
        n_windowed = n_temporal * window
        
        # FFT to get frequency spectrum
        dt = t[1] - t[0]
        freqs = fftfreq(len(t), dt)
        fft_vals = fft(n_windowed)
        power = np.abs(fft_vals)**2
        
        # Find dominant frequency (excluding DC component)
        pos_freqs = freqs[freqs > 0]
        pos_power = power[freqs > 0]
        i_max = np.argmax(pos_power)
        omega_pulse = 2*np.pi * pos_freqs[i_max]
        
    elif method == "correlation":
        # Cross-correlation between different spatial points
        x1_idx = np.argmin(np.abs(x - par.x0))
        x2_idx = np.argmin(np.abs(x - (par.x0 + par.L/4) % par.L))
        
        n1 = n_t[x1_idx, :] - np.mean(n_t[x1_idx, :])
        n2 = n_t[x2_idx, :] - np.mean(n_t[x2_idx, :])
        
        # Cross-correlation
        correlation = np.correlate(n1, n2, mode='full')
        lags = np.arange(-len(n1)+1, len(n1))
        
        # Find peak correlation and corresponding lag
        i_peak = np.argmax(np.abs(correlation))
        lag_time = lags[i_peak] * (t[1] - t[0])
        
        # Estimate frequency from phase delay
        distance = par.L/4  # Distance between monitoring points
        if lag_time != 0:
            phase_velocity = distance / np.abs(lag_time)
            # Estimate wavelength and frequency
            k_vals, P_vals = _power_spectrum_1d(n_t[:, -1], par.L)
            i_peak = np.argmax(P_vals)
            k_char = k_vals[i_peak]
            omega_pulse = k_char * phase_velocity
        else:
            omega_pulse = 0.0
    
    return omega_pulse

def analyze_resonance_response(nu_values, lambda1_fixed=0.1, tag="resonance_sweep"):
    """
    Sweep driving frequency nu and analyze the system response to find resonances.
    
    Parameters:
    nu_values : array_like
        Array of driving frequencies to test
    lambda1_fixed : float
        Fixed modulation amplitude
    tag : str
        Tag for output files
    
    Returns:
    results : list
        List of tuples (nu, omega_pulse, response_amplitude, t, n_t)
    """
    old_nu = par.nu
    old_lambda1 = par.lambda1
    old_outdir = par.outdir
    
    par.lambda1 = lambda1_fixed
    
    results = []
    
    try:
        for nu in nu_values:
            par.nu = nu
            par.outdir = f"{old_outdir}_resonance_nu{nu:.3f}"
            
            print(f"\n[resonance] Running with nu = {nu:.3f}, lambda1 = {lambda1_fixed}")
            
            t, n_t, p_t = run_once(tag=f"resonance_nu{nu:.3f}")
            
            # Extract pulse frequency
            omega_pulse = extract_pulse_frequency(n_t, t, x, method="spectral")
            
            # Calculate response amplitude (variance of density fluctuations)
            # Use time window after transients have settled
            t_start = max(5.0, par.t_final * 0.3)
            i_start = np.argmin(np.abs(t - t_start))
            n_late = n_t[:, i_start:]
            response_amplitude = np.var(n_late)
            
            results.append((nu, omega_pulse, response_amplitude, t, n_t))
            
            print(f"[resonance] nu={nu:.3f}: omega_pulse={omega_pulse:.3f}, response_amp={response_amplitude:.6e}")
            
    finally:
        par.nu = old_nu
        par.lambda1 = old_lambda1
        par.outdir = old_outdir
    
    return results

def plot_resonance_analysis(results, tag="resonance"):
    """
    Plot resonance analysis results.
    
    Parameters:
    results : list
        Results from analyze_resonance_response
    tag : str
        Tag for output files
    """
    nu_vals = [r[0] for r in results]
    omega_pulse_vals = [r[1] for r in results]
    response_amps = [r[2] for r in results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Response amplitude vs driving frequency
    ax1.plot(nu_vals, response_amps, 'o-', lw=2, ms=6, color='blue')
    ax1.set_xlabel("Driving frequency $\\nu$")
    ax1.set_ylabel("Response amplitude (var($\\delta n$))")
    ax1.set_title("System Response vs Driving Frequency")
    ax1.grid(True, alpha=0.3)
    
    # Mark the peak response
    i_max = np.argmax(response_amps)
    nu_resonance = nu_vals[i_max]
    ax1.axvline(nu_resonance, color='red', linestyle='--', alpha=0.7, 
                label=f'Max response at $\\nu={nu_resonance:.3f}$')
    ax1.legend()
    
    # 2. Pulse frequency vs driving frequency
    ax2.plot(nu_vals, omega_pulse_vals, 's-', lw=2, ms=6, color='green', label='$\\omega_{\\text{pulse}}$')
    ax2.plot(nu_vals, nu_vals, 'k--', alpha=0.5, label='$\\nu$ (driving)')
    ax2.set_xlabel("Driving frequency $\\nu$")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Pulse Frequency vs Driving Frequency")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Resonance condition: |omega_pulse - nu|
    frequency_mismatch = np.abs(np.array(omega_pulse_vals) - np.array(nu_vals))
    ax3.plot(nu_vals, frequency_mismatch, '^-', lw=2, ms=6, color='purple')
    ax3.set_xlabel("Driving frequency $\\nu$")
    ax3.set_ylabel("$|\\omega_{\\text{pulse}} - \\nu|$")
    ax3.set_title("Frequency Mismatch (Resonance Condition)")
    ax3.grid(True, alpha=0.3)
    
    # Mark minimum mismatch
    i_min = np.argmin(frequency_mismatch)
    nu_best_match = nu_vals[i_min]
    ax3.axvline(nu_best_match, color='red', linestyle='--', alpha=0.7,
                label=f'Best match at $\\nu={nu_best_match:.3f}$')
    ax3.legend()
    
    plt.tight_layout()
    
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig(f"out_drift/resonance_analysis_{tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"out_drift/resonance_analysis_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] saved out_drift/resonance_analysis_{tag}.png")
    print(f"[resonance] Peak response at nu = {nu_resonance:.3f}")
    print(f"[resonance] Best frequency match at nu = {nu_best_match:.3f}")
    
    return nu_resonance, nu_best_match

def plot_temporal_evolution_panel(nu_values, lambda1_fixed=0.1, tag="temporal_panel"):
    """
    Plot temporal evolution at x₀ for various driving frequencies in a N×1 panel format.
    Layout: N rows × 1 column (vertical arrangement).
    
    Parameters:
    nu_values : array_like
        Array of driving frequencies to test
    lambda1_fixed : float
        Fixed modulation amplitude
    tag : str
        Tag for output files
    """
    old_nu = par.nu
    old_lambda1 = par.lambda1
    old_outdir = par.outdir
    
    par.lambda1 = lambda1_fixed
    
    results = []
    
    try:
        print(f"[temporal_panel] Running simulations for nu values: {nu_values}")
        
        for nu in nu_values:
            if nu <= 0.0001:
                par.lambda1 = 0.0
            else:
                par.lambda1 = 0.05

            par.nu = nu
            par.outdir = f"{old_outdir}_temporal_nu{nu:.3f}"
            
            print(f"\n[temporal_panel] Running with nu = {nu:.3f}, lambda1 = {lambda1_fixed}")
            
            t, n_t, p_t = run_once(tag=f"temporal_nu{nu:.3f}")
            results.append((nu, t, n_t))
            
    finally:
        par.nu = old_nu
        par.lambda1 = old_lambda1
        par.outdir = old_outdir
    
    # Create panel plot (n_panels rows x 1 column)
    n_panels = len(nu_values)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.5*n_panels), gridspec_kw={'hspace': 0.25})
    
    if n_panels == 1:
        axes = [axes]
    
    # Find center index for monitoring
    i_center = np.argmin(np.abs(x - par.x0))
    
    # Find global y-limits for consistent scaling
    y_min, y_max = float('inf'), float('-inf')
    for nu, t, n_t in results:
        n_center = n_t[i_center, :]
        y_min = min(y_min, n_center.min())
        y_max = max(y_max, n_center.max())
    
    for i, (nu, t, n_t) in enumerate(results):
        ax = axes[i]
        
        # Extract temporal evolution at x₀
        n_center = n_t[i_center, :]
        
        # Plot density evolution
        ax.plot(t, n_center, 'b-', lw=1.0, label=f"$n(x_0,t)$")
        
        # Add the driving modulation for comparison (only if λ₁ > 0)
        if lambda1_fixed > 0:
            lambda_t = par.lambda0 + lambda1_fixed * np.cos(nu * t)
            # Scale and shift for visibility
            lambda_scaled = par.nbar0 + (lambda_t - par.lambda0)
            if nu > 0.0001:
                ax.plot(t, lambda_scaled, 'r-', lw=0.5, alpha=0.7, label="$\\lambda(t)$ (scaled)")
        # else:
            # For λ₁=0 case, show the constant λ₀ level
            # ax.axhline(par.lambda0, color='r', linestyle='--', lw=1.5, alpha=0.7, label=f"$\\lambda_0 = {par.lambda0}$ (constant)")
        
        # Set consistent y-limits
        # ax.set_ylim(y_min * 0.95, y_max * 1.05)
        
        if i == n_panels - 1:  # Only label x-axis on bottom panel
            ax.set_xlabel("$t$", fontsize=11)
        ax.set_ylabel("$n(x_0,t)$", fontsize=11)
        
        if nu <= 0.0001:
            ax.set_title(f"$\\lambda_1 = 0.0$", fontsize=14)
        else:
            ax.set_title(f"$\\nu = {nu:.3f}$", fontsize=12)

        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')
        
        # Add frequency info
        period = 2*np.pi/nu if nu > 0 else np.inf
        ax.text(0.02, 0.95, f"$T = {period:.2f}$", transform=ax.transAxes, 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # plt.suptitle(f"Temporal Evolution at $x_0$ for Different Driving Frequencies ($\\lambda_1 = {lambda1_fixed}$)", fontsize=13, y=0.995)
    # plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig(f"out_drift/temporal_evolution_panel_{tag}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"out_drift/temporal_evolution_panel_{tag}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"[plot] Temporal evolution panel saved: out_drift/temporal_evolution_panel_{tag}.png")
    
    return results

def plot_spatial_temporal_evolution_panel(nu_values, lambda1_fixed=0.1, tag="spatial_temporal_panel"):
    """
    Plot temporal evolution at multiple spatial locations (left and right of x₀) for various driving frequencies.
    Layout: N rows × 1 column, each showing n(t) at different x positions.
    
    Parameters:
    nu_values : array_like
        Array of driving frequencies to test
    lambda1_fixed : float
        Fixed modulation amplitude
    tag : str
        Tag for output files
    """
    old_nu = par.nu
    old_lambda1 = par.lambda1
    old_outdir = par.outdir
    
    par.lambda1 = lambda1_fixed
    
    results = []
    
    try:
        print(f"[spatial_temporal_panel] Running simulations for nu values: {nu_values}")
        
        for nu in nu_values:
            if nu <= 0.0001:
                par.lambda1 = 0.0
            else:
                par.lambda1 = lambda1_fixed

            par.nu = nu
            par.outdir = f"{old_outdir}_spatial_temporal_nu{nu:.3f}"
            
            print(f"\n[spatial_temporal_panel] Running with nu = {nu:.3f}, lambda1 = {par.lambda1}")
            
            t, n_t, p_t = run_once(tag=f"spatial_temporal_nu{nu:.3f}")
            results.append((nu, t, n_t))
            
    finally:
        par.nu = old_nu
        par.lambda1 = old_lambda1
        par.outdir = old_outdir
    
    # Define spatial monitoring points relative to x₀ (reduced to 3 points)
    x_offsets = [-1.0, 0.0, 1.0]  # Relative to x₀
    x_positions = [(par.x0 + offset) % par.L for offset in x_offsets]  # Handle periodic boundaries
    x_indices = [np.argmin(np.abs(x - x_pos)) for x_pos in x_positions]
    
    colors = ['red', 'blue', 'green']  # Colors for different positions
    labels = [f'$x_0{offset:+.1f}$' if offset != 0 else '$x_0$' for offset in x_offsets]
    
    # Vertical displacement for clarity (small offsets to separate the curves)
    vertical_offsets = [-0.005, 0.0, 0.005]  # Small vertical shifts
    
    # Create panel plot (n_panels rows x 1 column)
    n_panels = len(nu_values)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.8*n_panels), gridspec_kw={'hspace': 0.25})
    
    if n_panels == 1:
        axes = [axes]
    
    # Find global y-limits for consistent scaling across all positions and frequencies
    y_min, y_max = float('inf'), float('-inf')
    for nu, t, n_t in results:
        for i_pos in x_indices:
            n_pos = n_t[i_pos, :]
            y_min = min(y_min, n_pos.min())
            y_max = max(y_max, n_pos.max())
    
    for i, (nu, t, n_t) in enumerate(results):
        ax = axes[i]
        
        # Plot temporal evolution at each spatial position with vertical displacement
        for j, (i_pos, color, label, v_offset) in enumerate(zip(x_indices, colors, labels, vertical_offsets)):
            n_pos = n_t[i_pos, :] + v_offset  # Add vertical displacement
            ax.plot(t, n_pos, color=color, lw=1.2, label=label, alpha=0.8)
        
        # Add the driving modulation for comparison (only if λ₁ > 0)
        if nu > 0.0001 and par.lambda1 > 0:
            lambda_t = par.lambda0 + par.lambda1 * np.cos(nu * t)
            # Scale and shift for visibility
            lambda_scaled = par.nbar0 + 0.1 * (lambda_t - par.lambda0)
            ax.plot(t, lambda_scaled, 'k--', lw=1.0, alpha=0.6, label="$\\lambda(t)$ (scaled)")
        
        # Set consistent y-limits
        ax.set_ylim(y_min * 0.98, y_max * 1.02)
        
        if i == n_panels - 1:  # Only label x-axis on bottom panel
            ax.set_xlabel("$t$", fontsize=11)
        ax.set_ylabel("$n(x,t)$", fontsize=11)
        
        if nu <= 0.0001:
            ax.set_title(f"$\\lambda_1 = 0.0$ (no modulation)", fontsize=12)
        else:
            ax.set_title(f"$\\nu = {nu:.3f}$", fontsize=12)

        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right', ncol=2)
        
        # Add frequency info
        period = 2*np.pi/nu if nu > 0 else np.inf
        period_text = f"$T = {period:.2f}$" if period < 1000 else "$T = \\infty$"
        ax.text(0.02, 0.95, period_text, transform=ax.transAxes, 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle(f"Temporal Evolution at Multiple Spatial Locations ($\\lambda_1 = {lambda1_fixed}$)", fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig(f"out_drift/spatial_temporal_evolution_panel_{tag}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"out_drift/spatial_temporal_evolution_panel_{tag}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"[plot] Spatial-temporal evolution panel saved: out_drift/spatial_temporal_evolution_panel_{tag}.png")
    
    return results

def plot_final_density_profiles_panel(nu_values, lambda1_fixed=0.1, tag="final_density_panel"):
    """
    Plot density profiles at t=t_final for various driving frequencies in a N×1 panel format.
    Layout: N rows × 1 column (vertical arrangement).
    
    Parameters:
    nu_values : array_like
        Array of driving frequencies to test
    lambda1_fixed : float
        Fixed modulation amplitude
    tag : str
        Tag for output files
    """
    old_nu = par.nu
    old_lambda1 = par.lambda1
    old_outdir = par.outdir
    
    par.lambda1 = lambda1_fixed
    
    results = []
    
    try:
        print(f"[final_density_panel] Running simulations for nu values: {nu_values}")
        
        for nu in nu_values:
            if nu <= 0.0001:
                par.lambda1 = 0.0
            else:
                par.lambda1 = lambda1_fixed

            par.nu = nu
            par.outdir = f"{old_outdir}_final_density_nu{nu:.3f}"
            
            print(f"\n[final_density_panel] Running with nu = {nu:.3f}, lambda1 = {par.lambda1}")
            
            t, n_t, p_t = run_once(tag=f"final_density_nu{nu:.3f}")
            results.append((nu, t, n_t))
            
    finally:
        par.nu = old_nu
        par.lambda1 = old_lambda1
        par.outdir = old_outdir
    
    # Create panel plot (n_panels rows x 1 column)
    n_panels = len(nu_values)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.8*n_panels), gridspec_kw={'hspace': 0.25})
    
    if n_panels == 1:
        axes = [axes]
    
    # Find global y-limits for consistent scaling
    y_min, y_max = float('inf'), float('-inf')
    for nu, t, n_t in results:
        n_final = n_t[:, -1]
        n_initial = n_t[:, 0]
        y_min = min(y_min, n_final.min(), n_initial.min())
        y_max = max(y_max, n_final.max(), n_initial.max())
    
    for i, (nu, t, n_t) in enumerate(results):
        ax = axes[i]
        
        # Plot initial and final density profiles
        n_initial = n_t[:, 0]
        n_final = n_t[:, -1]
        
        ax.plot(x, n_initial, 'k--', lw=1.5, alpha=0.7, label=f"$t = 0$")
        ax.plot(x, n_final, 'b-', lw=2.0, label=f"$t = {t[-1]:.1f}$")
        
        # Add vertical line at x₀
        ax.axvline(par.x0, color='red', linestyle=':', alpha=0.7, lw=1.5, label='$x_0$')
        
        # Calculate and display density variation measures
        delta_n_initial = n_initial.max() - n_initial.min()
        delta_n_final = n_final.max() - n_final.min()
        
        # Set consistent y-limits
        ax.set_ylim(y_min * 0.98, y_max * 1.02)
        
        if i == n_panels - 1:  # Only label x-axis on bottom panel
            ax.set_xlabel("$x$", fontsize=11)
        ax.set_ylabel("$n(x)$", fontsize=11)
        
        if nu <= 0.0001:
            ax.set_title(f"$\\lambda_1 = 0.0$ (no modulation)", fontsize=12)
        else:
            ax.set_title(f"$\\nu = {nu:.3f}$", fontsize=12)

        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')
        
        # Add density variation info
        delta_text = f"$\\Delta n_{{final}} = {delta_n_final:.4f}$"
        ax.text(0.02, 0.95, delta_text, transform=ax.transAxes, 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                verticalalignment='top')
        
        # Add frequency info on the right
        period = 2*np.pi/nu if nu > 0 else np.inf
        period_text = f"$T = {period:.2f}$" if period < 1000 else "$T = \\infty$"
        ax.text(0.98, 0.95, period_text, transform=ax.transAxes, 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                verticalalignment='top', horizontalalignment='right')
    
    plt.suptitle(f"Density Profiles: Initial vs Final ($\\lambda_1 = {lambda1_fixed}$)", fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig(f"out_drift/final_density_profiles_panel_{tag}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"out_drift/final_density_profiles_panel_{tag}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"[plot] Final density profiles panel saved: out_drift/final_density_profiles_panel_{tag}.png")
    
    return results

def run_simulations_once_and_analyze(nu_values, lambda1_fixed=0.1, tag="comprehensive_analysis"):
    """
    Efficient analysis: run each simulation only once, then perform all analyses on the stored results.
    
    Parameters:
    nu_values : array_like
        Array of driving frequencies to test
    lambda1_fixed : float
        Fixed modulation amplitude
    tag : str
        Tag for output files
        
    Returns:
    results : list
        List of (nu, t, n_t, p_t) tuples containing simulation results
    """
    old_nu = par.nu
    old_lambda1 = par.lambda1
    old_outdir = par.outdir
    
    results = []
    
    try:
        print(f"[comprehensive_analysis] Running simulations once for nu values: {nu_values}")
        print(f"[comprehensive_analysis] This will generate all analysis plots from the same simulation data")
        
        for i, nu in enumerate(nu_values):
            # Set parameters for this frequency
            if nu <= 0.0001:
                par.lambda1 = 0.0
                print(f"\n[{i+1}/{len(nu_values)}] Running baseline simulation (nu ≈ 0, λ₁=0)")
            else:
                par.lambda1 = lambda1_fixed
                print(f"\n[{i+1}/{len(nu_values)}] Running simulation for nu = {nu:.3f}, λ₁ = {par.lambda1}")

            par.nu = nu
            par.outdir = f"{old_outdir}_comprehensive_nu{nu:.3f}"
            
            # Run simulation once
            t, n_t, p_t = run_once(tag=f"comprehensive_nu{nu:.3f}")
            results.append((nu, t, n_t, p_t))
            
            print(f"    ✓ Simulation completed for nu = {nu:.3f}")
            
    finally:
        par.nu = old_nu
        par.lambda1 = old_lambda1
        par.outdir = old_outdir
    
    print(f"\n[comprehensive_analysis] All simulations completed! Now generating analysis plots...")
    return results

def plot_from_stored_results(results, lambda1_used=0.1, tag="from_stored"):
    """
    Generate all analysis plots from pre-computed simulation results.
    
    Parameters:
    results : list
        List of (nu, t, n_t, p_t) tuples from run_simulations_once_and_analyze
    lambda1_used : float
        The lambda1 value that was used in the simulations
    tag : str
        Tag for output files
    """
    nu_values = [r[0] for r in results]
    
    print(f"[plot_from_stored] Generating comprehensive analysis from {len(results)} stored results...")
    
    # === 1. TEMPORAL EVOLUTION AT x₀ ===
    print(f"[plot_from_stored] 1/3: Temporal evolution at x₀...")
    
    n_panels = len(nu_values)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.5*n_panels), gridspec_kw={'hspace': 0.25})
    
    if n_panels == 1:
        axes = [axes]
    
    # Find center index for monitoring
    i_center = np.argmin(np.abs(x - par.x0))
    
    # Find global y-limits for consistent scaling
    y_min, y_max = float('inf'), float('-inf')
    for nu, t, n_t, p_t in results:
        n_center = n_t[i_center, :]
        y_min = min(y_min, n_center.min())
        y_max = max(y_max, n_center.max())
    
    for i, (nu, t, n_t, p_t) in enumerate(results):
        ax = axes[i]
        
        # Extract temporal evolution at x₀
        n_center = n_t[i_center, :]
        
        # Plot density evolution
        ax.plot(t, n_center, 'b-', lw=1.0, label=f"$n(x_0,t)$")
        
        # Add the driving modulation for comparison (only if λ₁ > 0)
        if nu > 0.0001 and lambda1_used > 0:
            lambda_t = par.lambda0 + lambda1_used * np.cos(nu * t)
            # Scale and shift for visibility
            lambda_scaled = par.nbar0 + 0.1*(lambda_t - par.lambda0)
            ax.plot(t, lambda_scaled, 'r-', lw=0.5, alpha=0.7, label="$\\lambda(t)$ (scaled)")
        
        if i == n_panels - 1:  # Only label x-axis on bottom panel
            ax.set_xlabel("$t$", fontsize=11)
        ax.set_ylabel("$n(x_0,t)$", fontsize=11)
        
        if nu <= 0.0001:
            ax.set_title(f"$\\lambda_1 = 0.0$", fontsize=14)
        else:
            ax.set_title(f"$\\nu = {nu:.3f}$", fontsize=12)

        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')
        
        # Add frequency info
        period = 2*np.pi/nu if nu > 0 else np.inf
        ax.text(0.02, 0.95, f"$T = {period:.2f}$" if period < 1000 else "$T = \\infty$", 
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig(f"out_drift/temporal_evolution_panel_{tag}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"out_drift/temporal_evolution_panel_{tag}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"    ✓ Saved: out_drift/temporal_evolution_panel_{tag}.png")
    
    # === 2. SPATIAL-TEMPORAL EVOLUTION ===
    print(f"[plot_from_stored] 2/3: Spatial-temporal evolution...")
    
    # Define spatial monitoring points relative to x₀ (reduced to 3 points)
    x_offsets = [-1.0, 0.0, 1.0]  # Relative to x₀
    x_positions = [(par.x0 + offset) % par.L for offset in x_offsets]  # Handle periodic boundaries
    x_indices = [np.argmin(np.abs(x - x_pos)) for x_pos in x_positions]
    
    colors = ['red', 'blue', 'green']  # Colors for different positions
    labels = [f'$x_0{offset:+.1f}$' if offset != 0 else '$x_0$' for offset in x_offsets]
    
    # Vertical displacement for clarity (small offsets to separate the curves)
    vertical_offsets = [-0.05, 0.0, 0.05]  # Small vertical shifts
    
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.8*n_panels), gridspec_kw={'hspace': 0.25})
    
    if n_panels == 1:
        axes = [axes]
    
    # Find global y-limits for consistent scaling across all positions and frequencies
    y_min, y_max = float('inf'), float('-inf')
    for nu, t, n_t, p_t in results:
        for i_pos in x_indices:
            n_pos = n_t[i_pos, :]
            y_min = min(y_min, n_pos.min())
            y_max = max(y_max, n_pos.max())
    
    for i, (nu, t, n_t, p_t) in enumerate(results):
        ax = axes[i]
        
        # Plot temporal evolution at each spatial position with vertical displacement
        for j, (i_pos, color, label, v_offset) in enumerate(zip(x_indices, colors, labels, vertical_offsets)):
            n_pos = n_t[i_pos, :] + v_offset  # Add vertical displacement
            ax.plot(t, n_pos, color=color, lw=1.2, label=label, alpha=0.8)
        
        # Add the driving modulation for comparison (only if λ₁ > 0)
        if nu > 0.0001 and lambda1_used > 0:
            lambda_t = par.lambda0 + lambda1_used * np.cos(nu * t)
            # Scale and shift for visibility
            lambda_scaled = par.nbar0 + 0.1 * (lambda_t - par.lambda0)
            ax.plot(t, lambda_scaled, 'k--', lw=1.0, alpha=0.6, label="$\\lambda(t)$ (scaled)")
        
        # Set consistent y-limits
        ax.set_ylim(y_min * 0.98, y_max * 1.02)
        
        if i == n_panels - 1:  # Only label x-axis on bottom panel
            ax.set_xlabel("$t$", fontsize=11)
        ax.set_ylabel("$n(x,t)$", fontsize=11)
        
        if nu <= 0.0001:
            ax.set_title(f"$\\lambda_1 = 0.0$ (no modulation)", fontsize=12)
        else:
            ax.set_title(f"$\\nu = {nu:.3f}$", fontsize=12)

        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right', ncol=2)
        
        # Add frequency info
        period = 2*np.pi/nu if nu > 0 else np.inf
        period_text = f"$T = {period:.2f}$" if period < 1000 else "$T = \\infty$"
        ax.text(0.02, 0.95, period_text, transform=ax.transAxes, 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle(f"Temporal Evolution at Multiple Spatial Locations ($\\lambda_1 = {lambda1_used}$)", fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    
    plt.savefig(f"out_drift/spatial_temporal_evolution_panel_{tag}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"out_drift/spatial_temporal_evolution_panel_{tag}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"    ✓ Saved: out_drift/spatial_temporal_evolution_panel_{tag}.png")
    
    # === 3. FINAL DENSITY PROFILES ===
    print(f"[plot_from_stored] 3/3: Final density profiles...")
    
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.8*n_panels), gridspec_kw={'hspace': 0.25})
    
    if n_panels == 1:
        axes = [axes]
    
    # Find global y-limits for consistent scaling
    y_min, y_max = float('inf'), float('-inf')
    for nu, t, n_t, p_t in results:
        n_final = n_t[:, -1]
        n_initial = n_t[:, 0]
        y_min = min(y_min, n_final.min(), n_initial.min())
        y_max = max(y_max, n_final.max(), n_initial.max())
    
    for i, (nu, t, n_t, p_t) in enumerate(results):
        ax = axes[i]
        
        # Plot initial and final density profiles
        n_initial = n_t[:, 0]
        n_final = n_t[:, -1]
        
        ax.plot(x, n_initial, 'k--', lw=1.5, alpha=0.7, label=f"$t = 0$")
        ax.plot(x, n_final, 'b-', lw=2.0, label=f"$t = {t[-1]:.1f}$")
        
        # Add vertical line at x₀
        ax.axvline(par.x0, color='red', linestyle=':', alpha=0.7, lw=1.5, label='$x_0$')
        
        # Calculate and display density variation measures
        delta_n_initial = n_initial.max() - n_initial.min()
        delta_n_final = n_final.max() - n_final.min()
        
        # Set consistent y-limits
        ax.set_ylim(y_min * 0.98, y_max * 1.02)
        
        if i == n_panels - 1:  # Only label x-axis on bottom panel
            ax.set_xlabel("$x$", fontsize=11)
        ax.set_ylabel("$n(x)$", fontsize=11)
        
        if nu <= 0.0001:
            ax.set_title(f"$\\lambda_1 = 0.0$ (no modulation)", fontsize=12)
        else:
            ax.set_title(f"$\\nu = {nu:.3f}$", fontsize=12)

        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')
        
        # Add density variation info
        delta_text = f"$\\Delta n_{{final}} = {delta_n_final:.4f}$"
        ax.text(0.02, 0.95, delta_text, transform=ax.transAxes, 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                verticalalignment='top')
        
        # Add frequency info on the right
        period = 2*np.pi/nu if nu > 0 else np.inf
        period_text = f"$T = {period:.2f}$" if period < 1000 else "$T = \\infty$"
        ax.text(0.98, 0.95, period_text, transform=ax.transAxes, 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                verticalalignment='top', horizontalalignment='right')
    
    plt.suptitle(f"Density Profiles: Initial vs Final ($\\lambda_1 = {lambda1_used}$)", fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    
    plt.savefig(f"out_drift/final_density_profiles_panel_{tag}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"out_drift/final_density_profiles_panel_{tag}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"    ✓ Saved: out_drift/final_density_profiles_panel_{tag}.png")
    
    print(f"\n[plot_from_stored] All analysis plots completed!")
    print(f"Generated files:")
    print(f"  • out_drift/temporal_evolution_panel_{tag}.png")
    print(f"  • out_drift/spatial_temporal_evolution_panel_{tag}.png") 
    print(f"  • out_drift/final_density_profiles_panel_{tag}.png")

def plot_time_dependent_modulation(t_plot, tag="modulation"):
    """
    Plot the time-dependent modulation function λ(t) = λ₀ + λ₁cos(νt).
    
    Parameters:
    t_plot : ndarray
        Time array for plotting
    tag : str
        Tag for output files
    """
    lambda_t = par.lambda0 + par.lambda1 * np.cos(par.nu * t_plot)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_plot, lambda_t, 'b-', lw=2, label=f'$\\lambda(t) = {par.lambda0} + {par.lambda1}\\cos({par.nu}t)$')
    plt.axhline(par.lambda0, color='k', linestyle='--', alpha=0.5, label=f'$\\lambda_0 = {par.lambda0}$')
    plt.axhline(par.lambda0 + par.lambda1, color='r', linestyle=':', alpha=0.7, label=f'Max: {par.lambda0 + par.lambda1}')
    plt.axhline(par.lambda0 - par.lambda1, color='r', linestyle=':', alpha=0.7, label=f'Min: {par.lambda0 - par.lambda1}')
    plt.xlabel("Time $t$")
    plt.ylabel("$\\lambda(t)$")
    plt.title("Time-dependent Potential Modulation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Show the spatial profile at a few different times
    plt.subplot(2, 1, 2)
    times_sample = [0, np.pi/(2*par.nu), np.pi/par.nu, 3*np.pi/(2*par.nu)]
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, t_sample in enumerate(times_sample):
        if t_sample <= t_plot[-1]:
            U0nbar_profile_t = U0nbar_profile(t_sample)
            phase_desc = f"$\\nu t = {par.nu * t_sample:.2f}$"
            plt.plot(x, U0nbar_profile_t, color=colors[i], lw=2, 
                    label=f't={t_sample:.2f} ({phase_desc})')
    
    plt.xlabel("Position $x$")
    plt.ylabel("$U_0 \\bar{n}(x,t)$")
    plt.title("Spatial Profile at Different Times")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig(f"out_drift/time_modulation_{tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"out_drift/time_modulation_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] saved out_drift/time_modulation_{tag}.png")

def run_resonance_study(tag="resonance_study"):
    """
    Complete resonance study: sweep frequencies and analyze results.
    """
    # First, run a single simulation to estimate the natural pulse frequency
    print("[resonance_study] Running reference simulation to estimate natural frequency...")
    
    old_lambda1 = par.lambda1
    par.lambda1 = 0.0  # No modulation for reference
    
    try:
        t_ref, n_t_ref, _ = run_once(tag="reference_no_modulation")
        omega_natural = extract_pulse_frequency(n_t_ref, t_ref, x, method="spectral")
        print(f"[resonance_study] Estimated natural pulse frequency: ω ≈ {omega_natural:.3f}")
        
    finally:
        par.lambda1 = old_lambda1
    
    # Create frequency sweep around the natural frequency
    if omega_natural > 0:
        nu_center = omega_natural / (2*np.pi)  # Convert to frequency
        nu_values = np.linspace(0.5 * nu_center, 2.0 * nu_center, 15)
    else:
        # Fallback if natural frequency estimation fails
        nu_values = np.linspace(0.1, 2.0, 15)
    
    print(f"[resonance_study] Sweeping frequencies: {nu_values}")
    
    # Plot the modulation function
    t_plot = np.linspace(0, par.t_final, 200)
    plot_time_dependent_modulation(t_plot, tag=tag)
    
    # Run frequency sweep
    results = analyze_resonance_response(nu_values, lambda1_fixed=par.lambda1, tag=tag)
    
    # Plot results
    nu_resonance, nu_best_match = plot_resonance_analysis(results, tag=tag)
    
    return results, nu_resonance, nu_best_match

def create_snapshots_panel_from_images(lambda0_val, u_d_values, tag="snapshots_panel"):
    """
    Create a vertical panel plot from existing snapshot images with delta calculations
    """
    import glob
    from PIL import Image
    import matplotlib.image as mpimg
    
    n_plots = len(u_d_values)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots), gridspec_kw={'hspace': 0.1})
    
    if n_plots == 1:
        axes = [axes]
    
    for i, u_d in enumerate(u_d_values):
        ax = axes[i]
        
        # Look for the snapshot image
        image_pattern = f"out_drift_lambda{lambda0_val}_ud{u_d}/snapshots_n_lambda{lambda0_val}_ud{u_d}.png"
        
        # Look for the corresponding .npz file to get delta values
        npz_pattern = f"out_drift_lambda{lambda0_val}_ud{u_d}/spec_m01_lambda{lambda0_val}_ud{u_d}.npz"
        
        delta_text = ""
        try:
            # Try to load spectral data to get additional info
            if os.path.exists(npz_pattern):
                print(f"[snapshots_panel] Found npz: {npz_pattern}")
                # We don't have the actual n_t data in the npz file, so we'll calculate from results if available
                delta_text = " ($\\Delta n$ from data)"
            else:
                print(f"[snapshots_panel] No npz found: {npz_pattern}")
        except Exception as e:
            print(f"[snapshots_panel] Error loading npz: {e}")
        
        try:
            # Load and display the image
            img = mpimg.imread(image_pattern)
            ax.imshow(img)
            
            # Add delta info if we have it from the recent simulation
            title_text = f"$u_d = {u_d}$, $\\lambda_0 = {lambda0_val}$"
            
            # Try to get delta from recent simulation results if available
            delta_info = ""
            if hasattr(create_snapshots_panel_from_images, 'recent_results'):
                for u_d_res, t_res, n_t_res in create_snapshots_panel_from_images.recent_results:
                    if abs(u_d_res - u_d) < 1e-6:  # Match u_d
                        # Calculate time-averaged delta instead of final delta
                        t_start_avg = 10.0
                        t_end_avg = 20.0  # Use t_final from parameters
                        i_start = np.argmin(np.abs(t_res - t_start_avg))
                        i_end = np.argmin(np.abs(t_res - t_end_avg))
                        n_avg_time = np.mean(n_t_res[:, i_start:i_end+1], axis=1)
                        n_min_avg = n_avg_time.min()
                        n_max_avg = n_avg_time.max()
                        delta_n_avg = n_max_avg - n_min_avg
                        delta_info = f", $\\Delta n_{{avg}} = {delta_n_avg:.4f}$"
                        break
            
            ax.set_title(title_text + delta_info, fontsize=12)
            ax.axis('off')  # Remove axes for clean image display
            
            print(f"[snapshots_panel] Loaded: {image_pattern}")
            
        except (FileNotFoundError, OSError) as e:
            # If image not found, create placeholder
            ax.text(0.5, 0.5, f"Image not found:\nu_d={u_d}, λ₀={lambda0_val}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
            ax.set_title(f"u_d = {u_d}, λ₀ = {lambda0_val} (missing)", fontsize=12)
            ax.axis('off')
            
            print(f"[snapshots_panel] Missing: {image_pattern}")
    
    plt.suptitle(f"Density Snapshots Panel - λ₀={lambda0_val} for different u_d values", fontsize=14)
    plt.tight_layout()
    
    os.makedirs("out_drift", exist_ok=True)
    output_path = f"out_drift/snapshots_panel_lambda{lambda0_val}_{tag}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(f"out_drift/snapshots_panel_lambda{lambda0_val}_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] saved {output_path}")

if __name__ == "__main__":
    # Time-dependent modulation resonance study
    print(f"[main] Starting time-dependent modulation resonance study")
    print(f"[main] Parameters: λ₀={par.lambda0}, λ₁={par.lambda1}, ν={par.nu}")
    
    # Run the complete resonance study
    results, nu_resonance, nu_best_match = run_resonance_study(tag="kapitsa_resonance")
    
    print(f"\n[main] Resonance study completed!")
    print(f"[main] Peak response found at driving frequency ν = {nu_resonance:.3f}")
    print(f"[main] Best frequency match at ν = {nu_best_match:.3f}")
    
    # Generate temporal evolution panel around the resonance
    print(f"\n[main] Generating temporal evolution panel around resonance...")
    nu_center = nu_resonance
    nu_panel_values = np.array([0.7*nu_center, 0.85*nu_center, nu_center, 1.15*nu_center, 1.3*nu_center])
    plot_temporal_evolution_panel(nu_panel_values, lambda1_fixed=par.lambda1, tag="around_resonance")
    print(f"[main] Temporal panel saved: out_drift/temporal_evolution_panel_around_resonance.png")
    
    # Optional: run a detailed simulation at the resonance frequency
    if nu_resonance > 0:
        print(f"\n[main] Running detailed simulation at resonance frequency ν = {nu_resonance:.3f}")
        old_nu = par.nu
        par.nu = nu_resonance
        par.outdir = "out_drift_resonance_detailed"
        
        try:
            t_res, n_t_res, _ = run_once(tag=f"resonance_detailed_nu{nu_resonance:.3f}")
            
            # Create additional analysis plots for the resonance case
            plt.figure(figsize=(12, 8))
            
            # Spacetime plot
            plt.subplot(2, 2, 1)
            extent = [x.min(), x.max(), t_res.min(), t_res.max()]
            plt.imshow(n_t_res.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
            plt.xlabel("$x$")
            plt.ylabel("$t$")
            plt.title(f"$n(x,t)$ at resonance $\\nu={nu_resonance:.3f}$")
            plt.colorbar(label="$n$")
            
            # Temporal evolution at the perturbation center
            plt.subplot(2, 2, 2)
            i_center = np.argmin(np.abs(x - par.x0))
            n_center = n_t_res[i_center, :]
            plt.plot(t_res, n_center, 'b-', lw=2, label="$n(x_0,t)$")
            
            # Add the driving modulation for comparison
            lambda_t = par.lambda0 + par.lambda1 * np.cos(par.nu * t_res)
            plt.plot(t_res, par.nbar0 + 0.1*(lambda_t - par.lambda0), 'r--', lw=2, 
                    label="$\\lambda(t)$ (scaled)")
            plt.xlabel("$t$")
            plt.ylabel("$n$")
            plt.title("Temporal Evolution at $x_0$")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # FFT of temporal signal
            plt.subplot(2, 2, 3)
            dt = t_res[1] - t_res[0]
            freqs = fftfreq(len(t_res), dt)
            n_centered = n_center - np.mean(n_center)
            fft_n = fft(n_centered * np.hanning(len(n_centered)))
            power_n = np.abs(fft_n)**2
            
            pos_freqs = freqs[freqs > 0]
            pos_power = power_n[freqs > 0]
            
            plt.plot(pos_freqs, pos_power, 'b-', lw=2)
            plt.axvline(nu_resonance/(2*np.pi), color='r', linestyle='--', 
                       label=f'Driving freq = {nu_resonance/(2*np.pi):.3f}')
            plt.xlabel("Frequency")
            plt.ylabel("Power")
            plt.title("Frequency Spectrum of $n(x_0,t)$")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Final spatial profile
            plt.subplot(2, 2, 4)
            plt.plot(x, n_t_res[:, 0], 'k--', alpha=0.7, label="$t=0$")
            plt.plot(x, n_t_res[:, -1], 'b-', lw=2, label=f"$t={t_res[-1]:.1f}$")
            plt.axvline(par.x0, color='r', linestyle=':', alpha=0.7, label="$x_0$")
            plt.xlabel("$x$")
            plt.ylabel("$n$")
            plt.title("Initial vs Final Density")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"out_drift/resonance_detailed_analysis.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"out_drift/resonance_detailed_analysis.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[main] Detailed resonance analysis saved to out_drift/resonance_detailed_analysis.png")
            
        finally:
            par.nu = old_nu
            par.outdir = "out_drift"