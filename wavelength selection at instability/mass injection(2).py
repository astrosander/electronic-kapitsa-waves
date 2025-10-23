import os

# For multiprocessing, use single thread per process to avoid conflicts
NTHREADS = 1
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
from multiprocessing import Pool, cpu_count
import copy
import time
from datetime import datetime, timedelta
# Set single thread for BLAS/LAPACK to avoid conflicts in multiprocessing
try:
    from scipy import linalg as _sla
    _sla.set_blas_num_threads(NTHREADS)
    _sla.set_lapack_num_threads(NTHREADS)
except Exception:
    pass

print(f"[Multiprocessing] Using single thread per process for FFTs and linear algebra")

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

    u_d: float = 20.00
    # u_d: float = .0
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
    Nx: int = 812#12
    t_final: float = 100.0
    n_save: int = 360
    # rtol: float = 5e-7
    # atol: float = 5e-9
    rtol = 1e-3
    atol = 1e-7
    n_floor: float = 1e-7
    dealias_23: bool = True

    seed_amp_n: float = 20e-3
    seed_mode: int = 1
    seed_amp_p: float = 20e-3

    outdir: str = "out_drift"
    cmap: str = "inferno"

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

def S_injection(n, nbar, Jx, gamma):
    if par.source_model == "as_given":
        return Jx * nbar - gamma * (n - nbar)
    elif par.source_model == "balanced":
        return Jx * nbar - gamma * n
    else:
        raise ValueError("source_model must be 'as_given' or 'balanced'")

def E_base_from_drift(nbar):
    return par.m * par.u_d * np.mean(Gamma(nbar)) / par.e /0.8187307530779819*40.0

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

    dp_dt = -Gamma(n_eff)*p - grad_Pi + par.e*n_eff*E_eff - force_Phi + par.Dp * Dxx(p)
    dp_dt = filter_23(dp_dt)

    return np.concatenate([dn_dt, dp_dt])

def initial_fields():
    nbar = nbar_profile()
    pbar = pbar_profile(nbar)
    n0 = nbar.copy()
    p0 = pbar.copy()
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

def save_final_spectra(m, t, n_t, L, tag=""):
    k0, P0 = _power_spectrum_1d(n_t[:, 0],  L)
    kf, Pf = _power_spectrum_1d(n_t[:, -1], L)

    meta = asdict(par).copy()
    meta['outdir'] = str(par.outdir)

    os.makedirs(par.outdir, exist_ok=True)
    out = os.path.join(par.outdir, f"spec_m{int(m):02d}_{tag}.npz")
    np.savez_compressed(out,
                        m=int(m),
                        t_final=float(t[-1]),
                        L=float(L),
                        Nx=int(par.Nx),
                        k0=k0, P0=P0,
                        k=kf, P=Pf,
                        meta=meta)
    print(f"[save] spectra → {out}")

def save_complete_data(m, t, n_t, p_t, x, L, tag=""):
    """Save complete simulation data including all fields and parameters."""
    # Calculate power spectra
    k0, P0 = _power_spectrum_1d(n_t[:, 0],  L)
    kf, Pf = _power_spectrum_1d(n_t[:, -1], L)
    
    # Calculate velocity field
    n_eff_t = np.maximum(n_t, par.n_floor)
    v_t = p_t/(par.m*n_eff_t)
    
    # Create comprehensive metadata
    meta = asdict(par).copy()
    meta['outdir'] = str(par.outdir)
    meta['x_min'] = float(x.min())
    meta['x_max'] = float(x.max())
    meta['dx'] = float(x[1] - x[0])
    meta['dt'] = float(t[1] - t[0]) if len(t) > 1 else 0.0
    
    os.makedirs(par.outdir, exist_ok=True)
    out = os.path.join(par.outdir, f"complete_m{int(m):02d}_{tag}.npz")
    
    np.savez_compressed(out,
                        # Mode and simulation info
                        m=int(m),
                        t_final=float(t[-1]),
                        L=float(L),
                        Nx=int(par.Nx),
                        Nt=int(len(t)),
                        
                        # Spatial and temporal grids
                        x=x,
                        t=t,
                        
                        # Complete field data
                        n=n_t,           # density field n(x,t) [Nx x Nt]
                        p=p_t,           # momentum field p(x,t) [Nx x Nt] 
                        v=v_t,           # velocity field v(x,t) [Nx x Nt]
                        
                        # Power spectra
                        k0=k0, P0=P0,   # initial spectrum
                        k=kf, P=Pf,     # final spectrum
                        
                        # All parameters
                        meta=meta)
    print(f"[save] complete data → {out}")

def plot_fft_initial_last(n_t, t, L, tag="compare", k_marks=()):
    """Overlay t=0 and t=t_end spectra; optional vertical k_marks."""
    k0, P0 = _power_spectrum_1d(n_t[:, 0],   L)
    k1, P1 = _power_spectrum_1d(n_t[:, -1],  L)

    i0 = np.argmax(P0); i1 = np.argmax(P1)
    k0_peak, k1_peak = k0[i0], k1[i1]

    plt.figure(figsize=(8.6, 4.2))
    plt.plot(k0, P0, label="t = 0")
    plt.plot(k1, P1, label=f"t = {t[-1]:.2f}")
    plt.plot([k0_peak], [P0[i0]], "o", ms=6, label=f"peak0 k={k0_peak:.3f}")
    plt.plot([k1_peak], [P1[i1]], "s", ms=6, label=f"peak1 k={k1_peak:.3f}")

    for km in k_marks:
        plt.axvline(km, color="k", ls="--", lw=1, alpha=0.6)

    plt.xlabel("$k$")
    plt.ylabel("power $|\\hat{n}(k)|^2$")
    plt.title("Fourier spectrum of $n(x,t)$: initial vs final")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False, ncol=2)
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
    plt.title(f"Initial spectra at t = 0")
    plt.savefig(f"{par.outdir}/fft_final_overlay_{tag}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{par.outdir}/fft_final_overlay_{tag}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def run_once(tag="seed_mode"):
    os.makedirs(par.outdir, exist_ok=True)

    n0, p0 = initial_fields()
    E_base = E_base_from_drift(nbar_profile()) if par.maintain_drift in ("field","feedback") else 0.0

    E_base = 15.0

    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)

    # Use single-threaded FFT for multiprocessing
    with set_workers(NTHREADS):
        sol = solve_ivp(lambda t,y: rhs(t,y,E_base),
                        (0.0, par.t_final), y0, t_eval=t_eval,
                        method="BDF", rtol=par.rtol, atol=par.atol)

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
    for frac in [0.0, 1.0]:
        j = int(frac*(len(sol.t)-1))
        plt.plot(x, n_t[:,j], label=f"t={sol.t[j]:.1f}")
    plt.legend(); plt.xlabel("x"); plt.ylabel("n"); plt.title(f"Density snapshots  {tag}")
    plt.text(0.5, 0.08, f"Dp={par.Dp}, Dn={par.Dn}, m={par.seed_mode}", color="red",
         fontsize=12, ha="right", va="top", transform=plt.gca().transAxes)

    plt.tight_layout(); plt.savefig(f"{par.outdir}/snapshots_n_{tag}.png", dpi=160); plt.close()

    plot_fft_initial_last(n_t, sol.t, par.L, tag=tag, k_marks=())
    
    save_final_spectra(par.seed_mode, sol.t, n_t, par.L, tag=tag)
    save_complete_data(par.seed_mode, sol.t, n_t, p_t, x, par.L, tag=tag)

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


def run_single_mode_worker(args):
    """Worker function for multiprocessing - runs a single mode simulation."""
    mode, par_dict, x_array, L, outdir = args
    
    start_time = time.time()
    print(f"[worker] Starting mode {mode} at {datetime.now().strftime('%H:%M:%S')}")
    
    # Create a local copy of parameters
    local_par = P(**par_dict)
    local_par.seed_mode = mode
    local_par.outdir = outdir
    
    # Set up local variables
    local_x = x_array
    local_dx = local_x[1] - local_x[0]
    local_k = 2*np.pi*fftfreq(local_par.Nx, d=local_dx)
    local_ik = 1j*local_k
    local_k2 = local_k**2
    
    # Define local functions that use the local parameters
    def local_Dx(f):  return (ifft(local_ik * fft(f, workers=NTHREADS), workers=NTHREADS)).real
    def local_Dxx(f): return (ifft((-local_k2) * fft(f, workers=NTHREADS), workers=NTHREADS)).real
    
    def local_filter_23(f):
        if not local_par.dealias_23: return f
        fh = fft(f, workers=NTHREADS)
        kc = local_par.Nx//3
        fh[kc:-kc] = 0.0
        return (ifft(fh, workers=NTHREADS)).real
    
    def local_Gamma(n):
        return local_par.Gamma0 * np.exp(-np.maximum(n, local_par.n_floor)/local_par.w)
    
    def local_Pi0(n):
        return 0.5 * local_par.U * n**2
    
    def local_phi_from_n(n, nbar):
        rhs_hat = fft((local_par.e/local_par.eps) * (n - nbar), workers=NTHREADS)
        phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
        nz = (local_k2 != 0)
        phi_hat[nz] = rhs_hat[nz] / (-local_k2[nz])
        return (ifft(phi_hat, workers=NTHREADS)).real
    
    def local_periodic_delta(x, x0, L): return (x - x0 + 0.5*L) % L - 0.5*L
    
    def local_nbar_profile():
        if local_par.use_nbar_gaussian and local_par.nbar_amp != 0.0:
            d = local_periodic_delta(local_x, local_par.x0, local_par.L)
            return local_par.nbar0 + local_par.nbar_amp * np.exp(-0.5*(d/local_par.nbar_sigma)**2)
        else:
            return np.full_like(local_x, local_par.nbar0)
    
    def local_pbar_profile(nbar):
        return local_par.m * nbar * local_par.u_d
    
    def local_J_profile():
        d = local_periodic_delta(local_x, local_par.x0, local_par.L)
        return local_par.J0 * np.exp(-0.5*(d/local_par.sigma_J)**2)
    
    def local_gamma_from_J(Jx): 
        return np.trapz(Jx, local_x)/local_par.L
    
    def local_S_injection(n, nbar, Jx, gamma):
        if local_par.source_model == "as_given":
            return Jx * nbar - gamma * (n - nbar)
        elif local_par.source_model == "balanced":
            return Jx * nbar - gamma * n
        else:
            raise ValueError("source_model must be 'as_given' or 'balanced'")
    
    def local_E_base_from_drift(nbar):
        return local_par.m * local_par.u_d * np.mean(local_Gamma(nbar)) / local_par.e /0.8187307530779819*40.0
    
    def local_rhs(t, y, E_base):
        N = local_par.Nx
        n = y[:N]
        p = y[N:]

        nbar = local_nbar_profile()
        pbar = local_pbar_profile(nbar)

        n_eff = np.maximum(n, local_par.n_floor)

        Jx = local_J_profile()
        gamma = local_gamma_from_J(Jx)
        SJ = local_S_injection(n_eff, nbar, Jx, gamma)

        v = p/(local_par.m*n_eff)
        u_mean = float(np.mean(v))
        if local_par.maintain_drift == "feedback":
            E_eff = E_base + local_par.Kp * (local_par.u_d - u_mean)
        else:
            E_eff = E_base

        dn_dt = -local_Dx(p) + local_par.Dn * local_Dxx(n) + SJ *0
        dn_dt = local_filter_23(dn_dt)

        Pi = local_Pi0(n_eff) + (p**2)/(local_par.m*n_eff)
        grad_Pi = local_Dx(Pi)
        force_Phi = 0.0
        if local_par.include_poisson:
            phi = local_phi_from_n(n_eff, nbar)
            force_Phi = n_eff * local_Dx(phi)

        dp_dt = -local_Gamma(n_eff)*p - grad_Pi + local_par.e*n_eff*E_eff - force_Phi + local_par.Dp * local_Dxx(p)
        dp_dt = local_filter_23(dp_dt)

        return np.concatenate([dn_dt, dp_dt])
    
    def local_initial_fields():
        nbar = local_nbar_profile()
        pbar = local_pbar_profile(nbar)
        n0 = nbar.copy()
        p0 = pbar.copy()
        if local_par.seed_amp_n != 0.0 and local_par.seed_mode != 0:
            if local_par.seed_mode == 1:
                kx1 = 2*np.pi*3 / local_par.L
                kx2 = 2*np.pi*5 / local_par.L
                n0 += local_par.seed_amp_n * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
            if local_par.seed_mode == 2:
                kx1 = 2*np.pi*5 / local_par.L
                kx2 = 2*np.pi*8 / local_par.L
                n0 += local_par.seed_amp_n * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
            if local_par.seed_mode == 3:
                kx1 = 2*np.pi*8 / local_par.L
                kx2 = 2*np.pi*13 / local_par.L
                n0 += local_par.seed_amp_n * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
            if local_par.seed_mode == 4:
                kx1 = 2*np.pi*13 / local_par.L
                kx2 = 2*np.pi*21 / local_par.L
                n0 += local_par.seed_amp_n * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
            if local_par.seed_mode == 5:
                kx1 = 2*np.pi*21 / local_par.L
                kx2 = 2*np.pi*34 / local_par.L
                n0 += local_par.seed_amp_n * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
            if local_par.seed_mode == 6:
                kx1 = 2*np.pi*34 / local_par.L
                kx2 = 2*np.pi*55 / local_par.L
                n0 += local_par.seed_amp_n * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))

        if local_par.seed_amp_p != 0.0 and local_par.seed_mode != 0:
            if local_par.seed_mode == 1:
                kx1 = 2*np.pi*3 / local_par.L
                kx2 = 2*np.pi*5 / local_par.L
                p0 += local_par.seed_amp_p * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
            if local_par.seed_mode == 2:
                kx1 = 2*np.pi*5 / local_par.L
                kx2 = 2*np.pi*8 / local_par.L
                p0 += local_par.seed_amp_p * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
            if local_par.seed_mode == 3:
                kx1 = 2*np.pi*8 / local_par.L
                kx2 = 2*np.pi*13 / local_par.L
                p0 += local_par.seed_amp_p * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
            if local_par.seed_mode == 4:
                kx1 = 2*np.pi*13 / local_par.L
                kx2 = 2*np.pi*21 / local_par.L
                p0 += local_par.seed_amp_p * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
            if local_par.seed_mode == 5:
                kx1 = 2*np.pi*21 / local_par.L
                kx2 = 2*np.pi*34 / local_par.L
                p0 += local_par.seed_amp_p * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
            if local_par.seed_mode == 6:
                kx1 = 2*np.pi*34 / local_par.L
                kx2 = 2*np.pi*55 / local_par.L
                p0 += local_par.seed_amp_p * (np.cos(kx1 * local_x)+np.cos(kx2 * local_x))
        return n0, p0
    
    def local_save_complete_data(m, t, n_t, p_t, x, L, tag=""):
        """Save complete simulation data including all fields and parameters."""
        # Calculate power spectra
        k0, P0 = _power_spectrum_1d(n_t[:, 0],  L)
        kf, Pf = _power_spectrum_1d(n_t[:, -1], L)
        
        # Calculate velocity field
        n_eff_t = np.maximum(n_t, local_par.n_floor)
        v_t = p_t/(local_par.m*n_eff_t)
        
        # Create comprehensive metadata
        meta = asdict(local_par).copy()
        meta['outdir'] = str(local_par.outdir)
        meta['x_min'] = float(x.min())
        meta['x_max'] = float(x.max())
        meta['dx'] = float(x[1] - x[0])
        meta['dt'] = float(t[1] - t[0]) if len(t) > 1 else 0.0
        
        os.makedirs(local_par.outdir, exist_ok=True)
        out = os.path.join(local_par.outdir, f"complete_m{int(m):02d}_{tag}.npz")
        
        np.savez_compressed(out,
                            # Mode and simulation info
                            m=int(m),
                            t_final=float(t[-1]),
                            L=float(L),
                            Nx=int(local_par.Nx),
                            Nt=int(len(t)),
                            
                            # Spatial and temporal grids
                            x=x,
                            t=t,
                            
                            # Complete field data
                            n=n_t,           # density field n(x,t) [Nx x Nt]
                            p=p_t,           # momentum field p(x,t) [Nx x Nt] 
                            v=v_t,           # velocity field v(x,t) [Nx x Nt]
                            
                            # Power spectra
                            k0=k0, P0=P0,   # initial spectrum
                            k=kf, P=Pf,     # final spectrum
                            
                            # All parameters
                            meta=meta)
        print(f"[save] complete data → {out}")
    
    # Run the simulation
    os.makedirs(local_par.outdir, exist_ok=True)
    
    n0, p0 = local_initial_fields()
    E_base = local_E_base_from_drift(local_nbar_profile()) if local_par.maintain_drift in ("field","feedback") else 0.0
    E_base = 15.0

    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, local_par.t_final, local_par.n_save)

    # Use single-threaded FFT for multiprocessing
    with set_workers(NTHREADS):
        sol = solve_ivp(lambda t,y: local_rhs(t,y,E_base),
                        (0.0, local_par.t_final), y0, t_eval=t_eval,
                        method="BDF", rtol=local_par.rtol, atol=local_par.atol)

    N = local_par.Nx
    n_t = sol.y[:N,:]
    p_t = sol.y[N:,:]

    n_eff_t = np.maximum(n_t, local_par.n_floor)
    v_t = p_t/(local_par.m*n_eff_t)
    
    elapsed_time = time.time() - start_time
    print(f"[worker] Mode {mode} completed in {elapsed_time:.1f}s: <u>(t=0)={np.mean(v_t[:,0]):.4f},  <u>(t_end)={np.mean(v_t[:,-1]):.4f},  target u_d={local_par.u_d:.4f}")

    # Save individual results
    save_final_spectra(mode, sol.t, n_t, local_par.L, tag=f"m{mode}")
    local_save_complete_data(mode, sol.t, n_t, p_t, local_x, local_par.L, tag=f"m{mode}")
    
    return mode, sol.t, n_t, p_t

def run_all_modes_snapshots(tag="snapshots_panels", n_processes=None):
    """Run all modes in parallel using pure multiprocessing (no multithreading)."""
    os.makedirs(par.outdir, exist_ok=True)

    modes = range(1,7)
    
    # Determine number of processes - use all available CPU cores
    if n_processes is None:
        n_processes = min(len(modes), cpu_count())
    
    print(f"[multiprocessing] Using {n_processes} processes for {len(modes)} modes")
    print(f"[multiprocessing] Each process uses single thread to avoid conflicts")
    print(f"[progress] Starting simulation at {datetime.now().strftime('%H:%M:%S')}")
    print(f"[progress] Each mode simulates t=0 to t={par.t_final:.1f} (total: {len(modes)} × {par.t_final:.1f} = {len(modes) * par.t_final:.1f} time units)")
    
    # Prepare arguments for worker processes
    par_dict = asdict(par)
    args_list = [(mode, par_dict, x, par.L, par.outdir) for mode in modes]
    
    # Track progress
    start_time = time.time()
    
    # Run in parallel with progress tracking using pure multiprocessing
    print(f"[progress] Launching {len(modes)} parallel simulations...")
    print(f"[multiprocessing] Each mode runs in separate process with single thread")
    with Pool(processes=n_processes) as pool:
        # Use map_async to get better progress tracking
        async_result = pool.map_async(run_single_mode_worker, args_list)
        
        # Progress tracking variables
        last_progress_time = start_time
        progress_interval = 3  # Update every 3 seconds
        completed_modes = 0
        
        # Wait for completion with progress updates
        while not async_result.ready():
            current_time = time.time()
            
            # Update progress every few seconds
            if current_time - last_progress_time >= progress_interval:
                elapsed = current_time - start_time
                
                # Calculate progress based on simulation time (t_final) and number of modes
                # Each mode runs from t=0 to t=t_final, so total simulation time = modes * t_final
                total_simulation_time = len(modes) * par.t_final
                
                # Estimate progress based on elapsed wall time vs expected simulation time
                # Assume simulation takes roughly 1-2 seconds per simulation time unit
                time_per_sim_unit = 1.2  # seconds per unit of t_final (adjusted for typical PDE solving)
                expected_total_time = total_simulation_time * time_per_sim_unit
                
                # Progress based on elapsed time vs expected total time
                estimated_progress = min(95.0, (elapsed / expected_total_time) * 100)
                
                # Estimate ETA based on expected completion time
                estimated_remaining = max(0, expected_total_time - elapsed)
                eta = datetime.now() + timedelta(seconds=estimated_remaining)
                
                # Create progress bar based on estimated progress
                bar_length = 20
                filled_length = int(bar_length * estimated_progress / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"[progress] {completed_modes}/{len(modes)} modes done |{bar}| {estimated_progress:.1f}% - ETA: {eta.strftime('%H:%M:%S')} (t_final={par.t_final:.1f})")
                
                last_progress_time = current_time
            
            time.sleep(1)  # Check every second
        
        results = async_result.get()
    
    total_time = time.time() - start_time
    print(f"[progress] All simulations completed in {total_time:.1f}s at {datetime.now().strftime('%H:%M:%S')}")
    print(f"[progress] Average time per mode: {total_time/len(modes):.1f}s")
    print(f"[progress] Total simulation time: {len(modes)} modes × {par.t_final:.1f} = {len(modes) * par.t_final:.1f} time units")
    print(f"[progress] Wall time efficiency: {len(modes) * par.t_final / total_time:.2f}x speedup (parallel processing)")
    
    # Sort results by mode number
    results.sort(key=lambda x: x[0])
    
    # Print completion summary
    print(f"[progress] Successfully completed modes: {[r[0] for r in results]}")
    
    # Extract results for plotting
    modes_results = [(m, t, n_t) for m, t, n_t, _ in results]

    # Create the combined plot
    fig, axes = plt.subplots(
        len(modes), 1, sharex=True,
        figsize=(10, 12),
        constrained_layout=True
    )
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ax, (m, t, n_t) in zip(axes, modes_results):
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

    axes[-1].set_xlabel("x")

    plt.suptitle(f"Density snapshots for modes m=1..5  [{tag}]")
    outpath = f"{par.outdir}/snapshots_panels_{tag}.png"
    plt.savefig(outpath, dpi=160)
    outpath = f"{par.outdir}/snapshots_panels_{tag}.svg"
    plt.savefig(outpath, dpi=160)
    outpath = f"{par.outdir}/snapshots_panels_{tag}.pdf"
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"[plot] saved {outpath}")

    plot_all_final_spectra(modes_results, par.L, tag=tag, normalize=False)


if __name__ == "__main__":
    # Test multiprocessing with a smaller example first
    print("[test] Testing multiprocessing implementation...")
    
    # You can specify the number of processes explicitly if needed
    # run_all_modes_snapshots(tag="seed_modes_1to5", n_processes=4)
    
    # Or let it auto-detect the optimal number
    run_all_modes_snapshots(tag="seed_modes_1to5")