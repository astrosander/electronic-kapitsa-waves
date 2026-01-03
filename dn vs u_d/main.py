import os
import time
NTHREADS = int(os.environ.get("NTHREADS", "1"))
os.environ["OMP_NUM_THREADS"] = str(NTHREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NTHREADS)
os.environ["MKL_NUM_THREADS"] = str(NTHREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NTHREADS)

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.fft import fft, ifft, fftfreq, set_workers
from scipy.integrate import solve_ivp
from dataclasses import asdict
import multiprocessing as mp

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
    U: float = 1.0
    nbar0: float = 0.2
    Gamma0: float = 2.50
    w: float = 0.14
    include_poisson: bool = False
    eps: float = 20.0

    u_d: float = 5.245
    maintain_drift: str = "field"
    Kp: float = 0.15

    Dn: float = 0.03
    Dp: float = 0.03

    x0: float = 5

    lambda_diss: float = 0.0
    sigma_diss: float = 2.0
    lambda_gauss: float = 0.0
    sigma_gauss: float = 2.0
    x0_gauss: float = 5

    use_nbar_gaussian: bool = False
    nbar_amp: float = 0.0
    nbar_sigma: float = 120.0

    L: float = 10.0
    Nx: int = 5120
    t_final: float = 50.0
    n_save: int = 1000
    rtol = 1e-4
    atol = 1e-7
    n_floor: float = 1e-7
    dealias_23: bool = True

    seed_amp_n: float = 0#0.030
    seed_mode: int = 7
    seed_amp_p: float = 0#0.030

    I_SD: float = 0.0
    x_source: float = 2.5
    x_drain: float = 7.5
    sigma_contact: float = 0.05

    outdir: str = "out_drift/small_dissipation_perturbation"
    cmap: str = "inferno"

par = P()

dx = None
k = None
ik = None
k2 = None
_x_grid = None
_sd_src = None
_sd_drn = None
_sd_key = None

def _update_global_arrays():
    global dx, k, ik, k2, _nz_mask, _x_grid

    dx = par.L / par.Nx
    k = 2*np.pi*fftfreq(par.Nx, d=dx)
    ik = 1j*k
    k2 = k**2
    _nz_mask = (k2 != 0)
    _x_grid = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    _update_source_drain_profiles()

def Dx(f):  
    if k is None or len(k) != len(f):
        _update_global_arrays()
    return (ifft(ik * fft(f, workers=NTHREADS), workers=NTHREADS)).real

def Dxx(f): 
    if k2 is None or len(k2) != len(f):
        _update_global_arrays()
    return (ifft((-k2) * fft(f, workers=NTHREADS), workers=NTHREADS)).real

def filter_23(f):
    if not par.dealias_23: return f
    fh = fft(f, workers=NTHREADS)
    kc = len(f)//3
    fh[kc:-kc] = 0.0
    return (ifft(fh, workers=NTHREADS)).real

_nz_mask = None

def Gamma(n):
    return par.Gamma0 * np.exp(-np.maximum(n, par.n_floor)/par.w)

def Gamma_spatial(n):
    Gamma_base = Gamma(n)
    if par.lambda_diss != 0.0:

        x_local = np.linspace(0.0, par.L, len(n), endpoint=False)
        d = periodic_delta(x_local, par.x0, par.L)
        perturbation = par.lambda_diss * np.exp(-0.5 * (d / par.sigma_diss)**2)
        return Gamma_base + perturbation
    else:
        return Gamma_base

def Pi0(n):
    return 0.5 * par.U * n**2

def phi_from_n(n, nbar):
    rhs_hat = fft((par.e/par.eps) * (n - nbar), workers=NTHREADS)
    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    phi_hat[_nz_mask] = rhs_hat[_nz_mask] / (-k2[_nz_mask])
    return (ifft(phi_hat, workers=NTHREADS)).real

def periodic_delta(x, x0, L): return (x - x0 + 0.5*L) % L - 0.5*L

def _normalized_delta_profile(x, x0, L, sigma):
    x0 = float(x0) % float(L)
    if sigma is None or sigma <= 0.0:
        g = np.zeros_like(x)
        idx = int(np.round(x0 / dx)) % par.Nx
        g[idx] = 1.0 / dx
        return g
    d = periodic_delta(x, x0, L)
    g = np.exp(-0.5 * (d / sigma)**2)
    g /= (g.sum() * dx)
    return g

def _update_source_drain_profiles():
    global _sd_src, _sd_drn, _sd_key
    key = (par.Nx, float(par.L), float(par.I_SD != 0.0),
           float(par.x_source), float(par.x_drain), float(par.sigma_contact))
    if _sd_key == key:
        return
    if par.I_SD == 0.0:
        _sd_src = None
        _sd_drn = None
    else:
        _sd_src = _normalized_delta_profile(_x_grid, par.x_source, par.L, par.sigma_contact)
        _sd_drn = _normalized_delta_profile(_x_grid, par.x_drain,  par.L, par.sigma_contact)
    _sd_key = key

def _power_spectrum_1d(n_slice, L):
    N = n_slice.size
    dn = n_slice - np.mean(n_slice)
    nhat = fft(dn, workers=NTHREADS)
    P = (nhat * np.conj(nhat)).real / (N*N)
    m = np.arange(N//2 + 1)
    kpos = 2*np.pi*m / L
    return kpos[1:], P[1:N//2+1]

def _fourier_shift_real(f, shift, L):
    N = f.size
    k = 2*np.pi * np.fft.rfftfreq(N, d=L/N)
    F = np.fft.rfft(f)
    return np.fft.irfft(F * np.exp(1j*k*shift), n=N)

def estimate_velocity_fourier(n_t1, n_t2, t1, t2, L, power_floor=1e-3):
    N = n_t1.size

    f1 = n_t1 - n_t1.mean()
    f2 = n_t2 - n_t2.mean()

    k = 2*np.pi * np.fft.rfftfreq(N, d=L/N)
    F1 = np.fft.rfft(f1)
    F2 = np.fft.rfft(f2)
    C  = np.conj(F1) * F2
    phi = np.angle(C)


    k = k[1:]
    phi = phi[1:]
    w = np.abs(C[1:])
    mask = w > (power_floor * w.max())
    k, phi, w = k[mask], phi[mask], w[mask]


    phi = np.unwrap(phi)



    num = np.sum(w * k * phi)
    den = np.sum(w * k**2)
    shift = num / den




    shift = -shift


    shift = (shift + 0.5*L) % L - 0.5*L

    dt = float(t2 - t1)
    u = shift / dt
    return u, shift

def find_modulation_period_by_shift(n_t1, n_t2, t1, t2, L):
    u, shift = estimate_velocity_fourier(n_t1, n_t2, t1, t2, L)

    N = n_t1.size
    f1 = n_t1 - n_t1.mean()
    f2 = n_t2 - n_t2.mean()
    F1 = np.fft.rfft(f1)
    F2 = np.fft.rfft(f2)
    xcorr = np.fft.irfft(np.conj(F1) * F2, n=N)
    dx = L/N
    shifts = dx * (np.arange(N) - (N//2))
    xcorr = np.roll(xcorr, -N//2)

    correlations = (xcorr - xcorr.min())/(xcorr.max()-xcorr.min() + 1e-15)
    corr_max = np.interp(shift, shifts, correlations)

    return u, shift, corr_max, shifts, correlations

def plot_period_detection(n_initial, n_final, t_initial, t_final, L, u_momentum, u_target, tag="period_detection"):
    u_drift, shift_opt, corr_max, shifts, correlations = find_modulation_period_by_shift(
        n_initial, n_final, t_initial, t_final, L
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.linspace(0, L, len(n_final), endpoint=False)
    
    n_initial_shifted = _fourier_shift_real(n_initial, shift_opt, L)
    
    ax1.plot(x, n_initial, 'b-', label=f'Initial n(x,{t_initial:.2f})', alpha=0.6, linewidth=1.5)
    ax1.plot(x, n_final, 'r-', label=f'Final n(x,{t_final:.2f})', alpha=0.8, linewidth=2)
    ax1.plot(x, n_initial_shifted, 'g--', label=f'Initial shifted by {shift_opt:.3f}', alpha=0.8, linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('n(x)')
    ax1.set_title('Velocity Detection: Shifted Initial vs Final')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(shifts, correlations, 'b-', linewidth=2, label='Correlation')
    ax2.axvline(shift_opt, color='r', linestyle='--', linewidth=2, label=f'Optimal shift={shift_opt:.3f}')
    ax2.plot([shift_opt], [corr_max], 'ro', markersize=8, label=f'Max corr={corr_max:.3f}')
    
    ax2.set_xlabel('Spatial shift')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation vs Shift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    delta_t = t_final - t_initial
    fig.suptitle(f'Instantaneous velocity at t={t_final:.3f}: u_drift={u_drift:.3f} (shift={shift_opt:.3f}, Δt={delta_t:.4f}) | u_momentum={u_momentum:.3f}, u_target={u_target:.3f}', 
                 y=0.98, fontsize=10.5)
    
    os.makedirs(par.outdir, exist_ok=True)
    plt.savefig(f"{par.outdir}/period_detection_{tag}.png", dpi=160, bbox_inches='tight')
    plt.savefig(f"{par.outdir}/period_detection_{tag}.pdf", dpi=160, bbox_inches='tight')
    plt.close()

def nbar_profile():
    x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    
    if par.seed_mode == 2 or par.seed_mode == 7:
        return np.full_like(x_local, par.nbar0)
    
    nbar_base = np.full_like(x_local, par.nbar0)
    
    if par.lambda_gauss != 0.0:
        d_gauss = periodic_delta(x_local, par.x0_gauss, par.L)
        gauss_pert = par.lambda_gauss * np.exp(-0.5*(d_gauss/par.sigma_gauss)**2)
        nbar_base += gauss_pert
    
    if par.use_nbar_gaussian and par.nbar_amp != 0.0:
        d = periodic_delta(x_local, par.x0, par.L)
        nbar_base += par.nbar_amp * np.exp(-0.5*(d/par.nbar_sigma)**2)
    
    return nbar_base

def E_base_from_drift(nbar):
    return par.m * par.u_d * np.mean(Gamma(nbar)) / par.e 

def rhs(t, y, E_base):
    N = par.Nx
    n = y[:N]
    p = y[N:]

    nbar = nbar_profile()

    n_eff = np.maximum(n, par.n_floor)

    v = p/(par.m*n_eff)
    u_mean = float(np.mean(v))
    if par.maintain_drift == "feedback":
        E_eff = E_base + par.Kp * (par.u_d - u_mean)
    else:
        E_eff = E_base

    _update_source_drain_profiles()

    dn_dt = -Dx(p) + par.Dn * Dxx(n)
    if par.I_SD != 0.0:
        dn_dt = dn_dt + par.I_SD * (_sd_src - _sd_drn)
    dn_dt = filter_23(dn_dt)

    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)
    grad_Pi = Dx(Pi)
    force_Phi = 0.0
    if par.include_poisson:
        phi = phi_from_n(n_eff, nbar)
        force_Phi = n_eff * Dx(phi)

    dp_dt = -Gamma_spatial(n_eff)*p - grad_Pi + par.e*n_eff*E_eff - force_Phi + par.Dp * Dxx(p)
    if par.I_SD != 0.0:
        dp_dt = dp_dt + (par.m * par.u_d * par.I_SD) * (_sd_src - _sd_drn)
    dp_dt = filter_23(dp_dt)

    return np.concatenate([dn_dt, dp_dt])

def initial_fields():
    x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    
    nbar = np.full_like(x_local, par.nbar0)
    pbar = par.m * nbar * par.u_d
    
    n0 = nbar.copy()
    p0 = pbar.copy()
    
    if par.seed_mode == 2 and (par.seed_amp_n != 0.0 or par.seed_amp_p != 0.0):
        modes = [2, 3, 5, 8, 13, 21, 34, 55]
        
        sine_perturbation = np.zeros_like(x_local)
        for mode in modes:
            kx = 2*np.pi*mode / par.L
            sine_perturbation += np.cos(kx * x_local)
        
        sine_perturbation = sine_perturbation / len(modes)
        
        if par.seed_amp_n != 0.0:
            delta_n = par.seed_amp_n * sine_perturbation
            n0 = nbar + delta_n
        
        if par.seed_amp_p != 0.0:
            delta_p = par.seed_amp_p * sine_perturbation
            p0 = pbar + delta_p
    
    elif par.seed_mode == 7 and (par.seed_amp_n != 0.0 or par.seed_amp_p != 0.0):
        modes = [3, 5, 7]
        
        cosine_perturbation = np.zeros_like(x_local)
        for mode in modes:
            kx = 2*np.pi*mode / par.L
            cosine_perturbation += np.cos(kx * x_local)
        
        if par.seed_amp_n != 0.0:
            delta_n = par.seed_amp_n * cosine_perturbation
            n0 = nbar + delta_n
        
        if par.seed_amp_p != 0.0:
            delta_p = par.seed_amp_p * cosine_perturbation
            p0 = pbar + delta_p
    
    return n0, p0

def save_final_spectra(m, t, n_t, p_t, L, tag=""):
    meta = asdict(par).copy()
    meta['outdir'] = str(par.outdir)
    os.makedirs(par.outdir, exist_ok=True)
    
    u_d_str = f"{par.u_d:.4f}".replace('.', 'p')
    out = os.path.join(par.outdir, f"data_m{int(m):02d}_ud{u_d_str}_{tag}.npz")
    
    np.savez_compressed(out,
                        m=int(m),
                        t=t,
                        n_t=n_t,
                        p_t=p_t,
                        L=float(L),
                        Nx=int(par.Nx),
                        meta=meta)
    print(f"[save] Full data → {out}")

def plot_fft_initial_last(n_t, t, L, tag="compare"):
    k0, P0 = _power_spectrum_1d(n_t[:, 0],   L)
    k1, P1 = _power_spectrum_1d(n_t[:, -1],  L)

    i0 = np.argmax(P0); i1 = np.argmax(P1)
    k0_peak, k1_peak = k0[i0], k1[i1]

    plt.figure(figsize=(8.6, 4.2))
    plt.plot(k0, P0, label="t = 0")
    plt.plot(k1, P1, label=f"t = {t[-1]:.2f}")
    plt.plot([k0_peak], [P0[i0]], "o", ms=6, label=f"peak0 k={k0_peak:.3f}")
    plt.plot([k1_peak], [P1[i1]], "s", ms=6, label=f"peak1 k={k1_peak:.3f}")

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

def rhs_with_progress(t, y, E_base, last_print_time=[0.0], start_time=[0.0], worker_id=0):
    if t - last_print_time[0] > 1.0:
        progress = (t / par.t_final) * 100
        elapsed_wall_time = time.time() - start_time[0]
        
        if progress > 0.1:
            estimated_total_time = elapsed_wall_time / (progress / 100.0)
            estimated_remaining = estimated_total_time - elapsed_wall_time
            est_str = f"EST: {estimated_remaining:6.1f}s"
        else:
            est_str = "EST: ---s"
        
        print(f"\r[Worker {worker_id:2d}] {progress:6.1f}% (t = {t:6.3f}/{par.t_final:.3f}) | Wall: {elapsed_wall_time:6.1f}s | {est_str}", end="", flush=True)
        last_print_time[0] = t
    return rhs(t, y, E_base)

def run_once(tag="seed_mode", worker_id=0):
    os.makedirs(par.outdir, exist_ok=True)

    n0, p0 = initial_fields()
    E_base = E_base_from_drift(nbar_profile()) if par.maintain_drift in ("field","feedback") else 0.0

    print(f"[Worker {worker_id:2d}] E_base={E_base}")
    
    if par.lambda_diss != 0.0:
        print(f"[Worker {worker_id:2d}] Localized dissipation: lambda_diss={par.lambda_diss}, sigma_diss={par.sigma_diss}, x0={par.x0}")
        print(f"[Worker {worker_id:2d}]   (adds {par.lambda_diss:+.3f}*exp(-(x-{par.x0})^2/(2*{par.sigma_diss}^2)) to Gamma)")
    else:
        print(f"[Worker {worker_id:2d}] No localized dissipation perturbation (lambda_diss=0)")
    
    if par.lambda_gauss != 0.0:
        print(f"[Worker {worker_id:2d}] Time-independent Gaussian: lambda_gauss={par.lambda_gauss}, sigma_gauss={par.sigma_gauss}, x0_gauss={par.x0_gauss}")
        print(f"[Worker {worker_id:2d}]   (adds {par.lambda_gauss:+.3f}*exp(-(x-{par.x0_gauss})^2/(2*{par.sigma_gauss}^2)) to nbar)")
    else:
        print(f"[Worker {worker_id:2d}] No time-independent Gaussian perturbation (lambda_gauss=0)")

    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)

    print(f"[Worker {worker_id:2d}] Starting simulation for {tag}...")
    print(f"[Worker {worker_id:2d}] Parameters: t_final={par.t_final}, Nx={par.Nx}, u_d={par.u_d}")
    
    start_wall_time = time.time()
    last_print_time = [0.0]
    start_time = [start_wall_time]

    if HAS_THREADPOOLCTL:
        with threadpool_limits(limits=NTHREADS, user_api="blas"):
            with set_workers(NTHREADS):
                sol = solve_ivp(lambda t,y: rhs_with_progress(t,y,E_base,last_print_time,start_time,worker_id),
                                (0.0, par.t_final), y0, t_eval=t_eval,
                                method="BDF", rtol=par.rtol, atol=par.atol)
    else:
        with set_workers(NTHREADS):
            sol = solve_ivp(lambda t,y: rhs_with_progress(t,y,E_base,last_print_time,start_time,worker_id),
                            (0.0, par.t_final), y0, t_eval=t_eval,
                            method="BDF", rtol=par.rtol, atol=par.atol)

    total_wall_time = time.time() - start_wall_time
    print(f"\n[Worker {worker_id:2d}] Completed successfully!")
    print(f"[Worker {worker_id:2d}] Final time: {sol.t[-1]:.3f}, Success: {sol.success}")
    print(f"[Worker {worker_id:2d}] Total wall time: {total_wall_time:.2f} seconds")

    N = par.Nx
    n_t = sol.y[:N,:]
    p_t = sol.y[N:,:]

    n_eff_t = np.maximum(n_t, par.n_floor)
    v_t = p_t/(par.m*n_eff_t)
    u_momentum_initial = np.mean(v_t[:,0])
    u_momentum_final = np.mean(v_t[:,-1])
    
    idx_t1 = -5
    idx_t2 = -1

    print(f"[Worker {worker_id:2d}] len=",len(sol.t), idx_t1, idx_t2)
    
    u_drift_inst, shift_opt_inst, corr_max_inst, _, _ = find_modulation_period_by_shift(
        n_t[:, idx_t1], n_t[:, idx_t2], sol.t[idx_t1], sol.t[idx_t2], par.L
    )
    
    print(f"[Worker {worker_id:2d}]  <u>(t=0)={u_momentum_initial:.4f},  <u>(t_end)={u_momentum_final:.4f},  target u_d={par.u_d:.4f}")
    print(f"[Worker {worker_id:2d}]  u_drift_instantaneous={u_drift_inst:.4f} (from shift={shift_opt_inst:.3f}, Δt={sol.t[idx_t2]-sol.t[idx_t1]:.3f})")
    print(f"[Worker {worker_id:2d}]  measured at t={sol.t[idx_t2]:.3f}, correlation_max={corr_max_inst:.4f}")
    
    plot_period_detection(n_t[:, idx_t1], n_t[:, idx_t2], sol.t[idx_t1], sol.t[idx_t2], 
                         par.L, u_momentum_final, par.u_d, tag=tag)

    x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    dx_local = x_local[1] - x_local[0]
    
    extent=[x_local.min(), x_local.max(), sol.t.min(), sol.t.max()]
    plt.figure(figsize=(9.6,4.3))
    plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.xlabel("x"); plt.ylabel("t"); plt.title(f"n(x,t)  [lab]  {tag}")
    plt.colorbar(label="n")
    plt.plot([par.x0, par.x0], [sol.t.min(), sol.t.max()], 'w--', lw=1, alpha=0.7)
    plt.tight_layout(); plt.savefig(f"{par.outdir}/spacetime_n_lab_{tag}.png", dpi=160);
    plt.close()

    n_co = np.empty_like(n_t)
    for j, tj in enumerate(sol.t):
        shift = (par.u_d * tj) % par.L
        s_idx = int(np.round(shift/dx_local)) % par.Nx
        n_co[:, j] = np.roll(n_t[:, j], -s_idx)
    plt.figure(figsize=(9.6,4.3))
    plt.imshow(n_co.T, origin="lower", aspect="auto",
               extent=[x_local.min(), x_local.max(), sol.t.min(), sol.t.max()], cmap=par.cmap)
    plt.xlabel("ξ = x - u_d t"); plt.ylabel("t"); plt.title(f"n(ξ,t)  [co-moving u_d={par.u_d}]  {tag}")
    plt.colorbar(label="n"); plt.tight_layout()
    plt.savefig(f"{par.outdir}/spacetime_n_comoving_{tag}.png", dpi=160); plt.close()

    plt.figure(figsize=(9.6,3.4))
    for frac in [0.0, 1.0]:
        j = int(frac*(len(sol.t)-1))
        plt.plot(x_local, n_t[:,j], label=f"t={sol.t[j]:.1f}")
    plt.legend(); plt.xlabel("x"); plt.ylabel("n"); plt.title(f"Density snapshots  {tag}")
    plt.text(0.5, 0.08, f"Dp={par.Dp}, Dn={par.Dn}, m={par.seed_mode}", color="red",
         fontsize=12, ha="right", va="top", transform=plt.gca().transAxes)

    plt.tight_layout(); plt.savefig(f"{par.outdir}/snapshots_n_{tag}.png", dpi=160); plt.close()

    plot_fft_initial_last(n_t, sol.t, par.L, tag=tag)
    
    save_final_spectra(par.seed_mode, sol.t, n_t, p_t, par.L, tag=tag)

    return sol.t, n_t, p_t

def run_single_ud_worker(u_d, base_params, worker_id=0):
    local_par = P()
    for key, value in base_params.items():
        if hasattr(local_par, key):
            setattr(local_par, key, value)
    
    local_par.u_d = u_d
    u_d_str = f"{u_d:.4f}".replace('.', 'p')
    Dn_str = f"{local_par.Dn:.2f}".replace('.', 'p')
    Dp_str = f"{local_par.Dp:.2f}".replace('.', 'p')
    local_par.outdir = f"multiple_u_d/w=0.14_modes_3_5_7_L10(lambda={local_par.lambda_diss}, sigma={local_par.sigma_diss}, seed_amp_n={local_par.seed_amp_n}, seed_amp_p={local_par.seed_amp_p})_Dn={Dn_str}_Dp={Dp_str}/out_drift_ud{u_d_str}"
    
    if u_d > 1e-6:
        local_par.t_final = 20*10.0/u_d
    else:
        local_par.t_final = 50.0
    local_par.n_save = 1024
    
    local_par.Nx = 512*2
    
    global par
    par = local_par
    
    _update_global_arrays()
    
    print(f"\n{'='*50}")
    print(f"[Worker {worker_id:2d}] Running simulation for u_d = {u_d:.4f}")
    print(f"[Worker {worker_id:2d}] Parameters: t_final={par.t_final:.2f}, Nx={par.Nx}")
    print(f"{'='*50}")
    
    try:
        start_time = time.time()
        t, n_t, p_t = run_once(tag=f"ud{u_d}", worker_id=worker_id)
        elapsed = time.time() - start_time
        
        print(f"[Worker {worker_id:2d}] Completed u_d={u_d:.4f} in {elapsed:.1f}s")
        print(f"[Worker {worker_id:2d}] Final time: {t[-1]:.3f}, Data shapes: n_t={n_t.shape}, p_t={p_t.shape}")
        
        return {
            'u_d': u_d,
            'success': True,
            'elapsed_time': elapsed,
            'final_time': t[-1],
            'outdir': par.outdir
        }
        
    except Exception as e:
        print(f"[Worker {worker_id:2d}] Error in simulation for u_d={u_d}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'u_d': u_d,
            'success': False,
            'error': str(e)
        }

def run_multiple_ud():
    u_d_values = [0.5]
    
    print(f"[run_multiple_ud] Running single simulation with u_d = {u_d_values[0]:.4f}")
    
    base_params = asdict(par)
    
    # n_cpus = mp.cpu_count()
    # n_workers = min(len(u_d_values), max(1, n_cpus - 1))
    # 
    # print(f"\n[Parallel] Using {n_workers} parallel workers (out of {n_cpus} CPUs)")
    # print(f"[Parallel] Running {len(u_d_values)} simulations in parallel")
    # print(f"[Parallel] Progress will be shown for each worker simultaneously")
    # print(f"[Parallel] Each worker will update its progress line independently")
    
    overall_start_time = time.time()
    
    # with mp.Pool(processes=n_workers) as pool:
    #     worker_args = [(u_d, base_params, i) for i, u_d in enumerate(u_d_values)]
    #     results = pool.starmap(run_single_ud_worker, worker_args)
    
    result = run_single_ud_worker(u_d_values[0], base_params, worker_id=0)
    results = [result]
    
    overall_elapsed = time.time() - overall_start_time
    
    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETED!")
    print(f"{'='*60}")
    print(f"Total wall time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print(f"\nSummary:")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"  Successful: {len(successful)}/{len(results)}")
    print(f"  Failed: {len(failed)}/{len(results)}")
    
    if successful:
        total_sim_time = sum(r['elapsed_time'] for r in successful)
        avg_time = total_sim_time / len(successful)
        # speedup = total_sim_time / overall_elapsed
        print(f"  Average simulation time: {avg_time:.1f}s")
        print(f"  Total simulation time (sequential equivalent): {total_sim_time:.1f}s")
        # print(f"  Parallel speedup: {speedup:.2f}x")
    
    if failed:
        print(f"\nFailed simulations:")
        for r in failed:
            print(f"  u_d={r['u_d']:.4f}: {r.get('error', 'Unknown error')}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    par.m = 1.0
    par.e = 1.0
    par.U = 1.0
    par.nbar0 = 0.2
    par.Gamma0 = 2.5
    par.w = 1.0/4
    par.include_poisson = False
    par.eps = 20.0
    par.u_d = 0.42
    par.maintain_drift = 'field'
    par.Kp = 0.15
    par.Dn = 0.001
    par.Dp = 0.01
    par.x0 = par.L/2
    par.lambda_diss = 0.0
    par.sigma_diss = -1.0
    par.lambda_gauss = 0.0
    par.sigma_gauss = 2.0
    par.x0_gauss = par.L/2
    par.use_nbar_gaussian = False
    par.nbar_amp = 0.0
    par.nbar_sigma = 120.0
    par.L = 10.0
    par.Nx = 1024
    par.t_final = 350.0
    par.n_save = 700
    par.n_floor = 1e-7
    par.dealias_23 = True
    par.seed_amp_n = 0.0
    par.seed_mode = 7
    par.seed_amp_p = 0.0

    par.I_SD = 3e-4
    par.x_source = 2.5
    par.x_drain = 7.5
    par.sigma_contact = 0.25
    
    run_multiple_ud()