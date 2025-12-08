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
import glob
from dataclasses import asdict
import multiprocessing as mp
from functools import partial
import copy

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 14,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})


try:
    from spectral_analysis import load_spectral_evolution, plot_spectral_evolution, plot_spectral_growth_rates, compare_spectral_evolution
    HAS_SPECTRAL_ANALYSIS = True
except ImportError:
    HAS_SPECTRAL_ANALYSIS = False
    print("[Warning] spectral_analysis module not found. Install with: pip install -e .")
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
    w: float = 0.04
    include_poisson: bool = False
    eps: float = 20.0

    u_d: float = 5.245

    maintain_drift: str = "field"
    Kp: float = 0.15

    Dn: float = 0.1
    Dp: float = 0.1

    J0: float = 1.0
    sigma_J: float = 2.0**1/2
    x0: float = 10
    source_model: str = "as_given"


    lambda_diss: float = 0.0
    sigma_diss: float = 2.0


    lambda_gauss: float = 0.0
    sigma_gauss: float = 2.0
    x0_gauss: float = 10

    use_nbar_gaussian: bool = False
    nbar_amp: float = 0.0
    nbar_sigma: float = 120.0

    L: float = 10.0
    Nx: int = 1212
    t_final: float = 50.0
    n_save: int = 100


    rtol = 3e-2
    atol = 1e-4
    n_floor: float = 1e-4
    dealias_23: bool = True

    bc_type: str = "periodic"

    N_bc: int = 8
    tau_bc_n: float = 0.5
    tau_bc_p: float = 0.5
    use_hard_bc_clamp: bool = False


    seed_amp_n: float = 0.030
    seed_mode: int = 7
    seed_amp_p: float = 0.030

    outdir: str = "out_drift/small_dissipation_perturbation"
    cmap: str = "inferno"

class SimpleSolution:
    def __init__(self, t, y, success=True, message=""):
        self.t = t
        self.y = y
        self.success = success
        self.message = message


par = P()


x = None
dx = None
k = None
ik = None
k2 = None

_fft_cache = {}

def _update_global_arrays():
    global x, dx, k, ik, k2, _kc, _nz_mask

    x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    dx = x[1] - x[0]
    k = 2*np.pi*fftfreq(par.Nx, d=dx)
    ik = 1j*k
    k2 = k**2
    _kc = par.Nx//3
    _nz_mask = (k2 != 0)

def Dx_fd(f):
    global x, dx
    if x is None or len(x) != len(f):
        _update_global_arrays()
    h = dx
    out = np.empty_like(f)

    out[1:-1] = (f[2:] - f[:-2]) / (2.0 * h)

    out[0] = (f[1] - f[0]) / h
    out[-1] = (f[-1] - f[-2]) / h

    return out

def Dxx_fd(f):
    global x, dx
    if x is None or len(x) != len(f):
        _update_global_arrays()
    h = dx
    out = np.empty_like(f)


    out[1:-1] = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / (h * h)


    out[0] = (f[2] - 2.0 * f[1] + f[0]) / (h * h)


    out[-1] = (f[-1] - 2.0 * f[-2] + f[-3]) / (h * h)

    return out

def Dx(f):
    return Dx_fd(f)
    global k, ik
    if k is None or len(k) != len(f):
        _update_global_arrays()
    return (ifft(ik * fft(f, workers=NTHREADS), workers=NTHREADS)).real

def Dxx(f):
    return Dxx_fd(f)
    global k2
    if k2 is None or len(k2) != len(f):
        _update_global_arrays()
    return (ifft((-k2) * fft(f, workers=NTHREADS), workers=NTHREADS)).real

def Dx_phys(f):
    if par.bc_type == "ds_open":
        return Dx_fd(f)
    else:
        return Dx(f)

def Dxx_phys(f):
    if par.bc_type == "ds_open":
        return Dxx_fd(f)
    else:
        return Dxx(f)

def filter_23(f):
    if not par.dealias_23:
        return f

    if par.bc_type != "periodic":
        return f
    fh = fft(f, workers=NTHREADS)
    kc = len(f)//3
    fh[kc:-kc] = 0.0
    return (ifft(fh, workers=NTHREADS)).real


_kc = None
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

def calculate_velocity_from_period(n_initial, n_final, t_initial, t_final, L):
    u_drift, shift_optimal, correlation_max, shifts, correlations = find_modulation_period_by_shift(
        n_initial, n_final, t_initial, t_final, L
    )

    return u_drift, shift_optimal, correlation_max, shifts, correlations

def calculate_velocity_from_shift_refined(n_initial, n_final, t_final, L, search_range=None):
    dx = L / len(n_initial)
    x = np.linspace(0, L, len(n_initial), endpoint=False)


    dn_initial = n_initial - np.mean(n_initial)
    dn_final = n_final - np.mean(n_final)


    u_coarse, shift_coarse, _ = calculate_velocity_from_shift(n_initial, n_final, t_final, L)


    if search_range is None:
        u_search_width = 2.0
        u_min = u_coarse - u_search_width
        u_max = u_coarse + u_search_width
    else:
        u_min, u_max = search_range


    n_search = 201
    u_test = np.linspace(u_min, u_max, n_search)
    correlations = np.zeros(n_search)

    for i, u in enumerate(u_test):
        shift = u * t_final


        x_shifted = (x - shift) % L
        dn_initial_shifted = np.interp(x, x_shifted, dn_initial)


        correlations[i] = np.corrcoef(dn_final, dn_initial_shifted)[0, 1]


    max_idx = np.argmax(correlations)
    u_optimal = u_test[max_idx]
    shift_optimal = u_optimal * t_final
    correlation_max = correlations[max_idx]

    return u_optimal, shift_optimal, correlation_max

def plot_period_detection(n_initial, n_final, t_initial, t_final, L, u_momentum, u_target, tag="period_detection"):
    u_drift, shift_opt, corr_max, shifts, correlations = calculate_velocity_from_period(
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

    return u_drift, shift_opt, corr_max

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

def pbar_profile(nbar):
    return par.m * nbar * par.u_d

def J_profile():
    x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    d = periodic_delta(x_local, par.x0, par.L)
    return par.J0 * np.exp(-0.5*(d/par.sigma_J)**2)

def gamma_from_J(Jx):
    x_local = np.linspace(0.0, par.L, len(Jx), endpoint=False)
    return np.trapz(Jx, x_local)/par.L

def S_injection(n, nbar, Jx, gamma):
    if par.source_model == "as_given":
        return Jx * nbar - gamma * (n - nbar)
    elif par.source_model == "balanced":
        return Jx * nbar - gamma * n
    else:
        raise ValueError("source_model must be 'as_given' or 'balanced'")

def E_base_from_drift(nbar):
    return par.m * par.u_d * np.mean(Gamma(nbar)) / par.e

def apply_ds_open_relaxation(n, p, dn_dt, dp_dt):
    if par.bc_type != "ds_open":
        return dn_dt, dp_dt

    N = len(n)
    N_bc = min(par.N_bc, N // 4)

    left_mask  = np.zeros(N)
    right_mask = np.zeros(N)
    left_mask[:N_bc]  = 1.0
    right_mask[-N_bc:] = 1.0








    n_eq_source  = par.nbar0
    p_eq_drain   = par.m * par.nbar0 * par.u_d

    gamma_n = 1.0 / par.tau_bc_n
    gamma_p = 1.0 / par.tau_bc_p


    dn_dt += gamma_n * left_mask * (n_eq_source - n)



    dp_dt += gamma_p * right_mask * (p_eq_drain - p)

    return dn_dt, dp_dt

def enforce_ds_open_bc_state(n, p):
    if par.bc_type != "ds_open":
        return

    if not par.use_hard_bc_clamp:
        return


    n[0] = par.nbar0


    p[-1] = par.m * par.nbar0 * par.u_d

def rhs(t, y, E_base):
    N = par.Nx
    n = y[:N]
    p = y[N:]


    nbar = nbar_profile()


    n_eff = np.maximum(n, par.n_floor)


    v = p / (par.m * n_eff)
    u_mean = float(np.mean(v))
    if par.maintain_drift == "feedback":
        E_eff = E_base + par.Kp * (par.u_d - u_mean)
    else:
        E_eff = E_base


    dn_dt = -Dx_phys(p) + par.Dn * Dxx_phys(n)


    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)
    grad_Pi = Dx_phys(Pi)

    force_Phi = 0.0
    if par.include_poisson:
        phi = phi_from_n(n_eff, nbar)
        force_Phi = n_eff * Dx_phys(phi)

    dp_dt = -Gamma_spatial(n_eff)*p - grad_Pi + par.e*n_eff*E_eff - force_Phi + par.Dp * Dxx_phys(p)



    dn_dt, dp_dt = apply_ds_open_relaxation(n, p, dn_dt, dp_dt)


    if par.bc_type == "periodic":
        dn_dt = filter_23(dn_dt)
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


    elif par.seed_amp_n != 0.0 and par.seed_mode != 0 and par.seed_mode != 2:
        if par.seed_mode == 1:
            kx1 = 2*np.pi*3 / par.L
            kx2 = 2*np.pi*5 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x_local)+np.cos(kx2 * x_local))
        if par.seed_mode == 3:
            kx1 = 2*np.pi*8 / par.L
            kx2 = 2*np.pi*13 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x_local)+np.cos(kx2 * x_local))
        if par.seed_mode == 4:
            kx1 = 2*np.pi*13 / par.L
            kx2 = 2*np.pi*21 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x_local)+np.cos(kx2 * x_local))
        if par.seed_mode == 5:
            kx1 = 2*np.pi*21 / par.L
            kx2 = 2*np.pi*34 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x_local)+np.cos(kx2 * x_local))
        if par.seed_mode == 6:
            kx1 = 2*np.pi*34 / par.L
            kx2 = 2*np.pi*55 / par.L
            n0 += par.seed_amp_n * (np.cos(kx1 * x_local)+np.cos(kx2 * x_local))

    if par.seed_amp_p != 0.0 and par.seed_mode != 0 and par.seed_mode != 2:
        if par.seed_mode == 1:
            kx1 = 2*np.pi*3 / par.L
            kx2 = 2*np.pi*5 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x_local)+np.cos(kx2 * x_local))
        if par.seed_mode == 3:
            kx1 = 2*np.pi*8 / par.L
            kx2 = 2*np.pi*13 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x_local)+np.cos(kx2 * x_local))
        if par.seed_mode == 4:
            kx1 = 2*np.pi*13 / par.L
            kx2 = 2*np.pi*21 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x_local)+np.cos(kx2 * x_local))
        if par.seed_mode == 5:
            kx1 = 2*np.pi*21 / par.L
            kx2 = 2*np.pi*34 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x_local)+np.cos(kx2 * x_local))
        if par.seed_mode == 6:
            kx1 = 2*np.pi*34 / par.L
            kx2 = 2*np.pi*55 / par.L
            p0 += par.seed_amp_p * (np.cos(kx1 * x_local)+np.cos(kx2 * x_local))



    if par.bc_type != "periodic" and par.use_hard_bc_clamp:

        n0[0] = par.nbar0

        p0[-1] = par.m * par.nbar0 * par.u_d

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


def plot_fft_initial_last(n_t, t, L, tag="compare", k_marks=()):
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
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
              '#FF7F00', '#A65628', '#F781BF', '#999999']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    plt.figure(figsize=(7.0, 4.0))
    plt.style.use('default')

    for i, (m, t, n_t) in enumerate(results):

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

def integrate_explicit_ds_open(y0, t_span, t_eval, E_base, worker_id=0):
    t0, t_final = t_span
    _update_global_arrays()
    N = par.Nx

    x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    dx_local = x_local[1] - x_local[0]


    D_max = max(par.Dn, par.Dp, 1e-12)


    c0 = np.sqrt(max(par.U * par.nbar0, 0.0) / max(par.m, 1e-12))
    u_char = abs(par.u_d) + c0

    CFL_diff = 0.20
    CFL_adv  = 0.40

    dt_diff = CFL_diff * dx_local * dx_local / D_max
    dt_adv  = CFL_adv  * dx_local / max(u_char, 1e-6)


    base_dt = min(dt_diff, dt_adv, (t_final - t0) / 1e4)

    dt_sub  = 0.25 * base_dt

    print(f"[Worker {worker_id:2d}] Explicit RK4 for ds_open: "
          f"dt_sub={dt_sub:.3e}, dt_diff={dt_diff:.3e}, dt_adv={dt_adv:.3e}")


    t_eval = np.asarray(t_eval)
    n_out = t_eval.size
    Y = np.empty((2 * N, n_out))
    T = t_eval.copy()

    y = y0.copy()
    t = t0
    out_idx = 0


    Y[:, out_idx] = y
    out_idx += 1


    n_cap = getattr(par, "n_cap", 5.0 * par.nbar0)

    p_cap = getattr(par, "p_cap", 5.0 * par.m * par.nbar0 * abs(par.u_d) + 1e-6)

    last_print_time = [t0]
    start_wall = [time.time()]

    while t < t_final - 1e-14:

        dt = dt_sub
        if out_idx < n_out:
            dt = min(dt, T[out_idx] - t)
        if t + dt > t_final:
            dt = t_final - t
        if dt <= 0.0:
            break


        k1 = rhs_with_progress(t,           y,                E_base, last_print_time, start_wall, worker_id)
        k2 = rhs_with_progress(t + 0.5*dt, y + 0.5*dt*k1,    E_base, last_print_time, start_wall, worker_id)
        k3 = rhs_with_progress(t + 0.5*dt, y + 0.5*dt*k2,    E_base, last_print_time, start_wall, worker_id)
        k4 = rhs_with_progress(t + dt,     y + dt*k3,        E_base, last_print_time, start_wall, worker_id)

        y += (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t += dt


        n = y[:N]
        p = y[N:]


        np.maximum(n, par.n_floor, out=n)

        np.clip(n, par.n_floor, n_cap, out=n)

        np.clip(p, -p_cap, p_cap, out=p)


        enforce_ds_open_bc_state(n, p)


        y[:N] = n
        y[N:] = p


        while out_idx < n_out and t >= T[out_idx] - 1e-12:
            Y[:, out_idx] = y
            out_idx += 1



    if out_idx < n_out:
        for j in range(out_idx, n_out):
            Y[:, j] = y

    sol = SimpleSolution(t=T, y=Y, success=True,
                         message="explicit RK4 ds_open (fixed dt_sub, caps)")
    print(f"[Worker {worker_id:2d}] Solver success: {sol.success}")
    print(f"[Worker {worker_id:2d}] Final time: {T[-1]:.3f}, target: {t_final:.3f}")
    print(f"[Worker {worker_id:2d}] Total wall time: {time.time()-start_wall[0]:.2f} seconds")
    print(f"[Worker {worker_id:2d}] Solver info: explicit RK4 (no solve_ivp)")

    return sol


def run_once(tag="seed_mode", worker_id=0):
    os.makedirs(par.outdir, exist_ok=True)

    n0, p0 = initial_fields()
    E_base = (
        E_base_from_drift(nbar_profile())
        if par.maintain_drift in ("field", "feedback")
        else 0.0
    )

    print(f"[Worker {worker_id:2d}] E_base={E_base}")


    if par.bc_type == "ds_open":
        bc_mode = "hybrid (soft relaxation + hard Dirichlet)" if par.use_hard_bc_clamp else "purely soft (relaxation only)"
        print(f"[Worker {worker_id:2d}] DS/open BC mode: {bc_mode}")
        print(f"[Worker {worker_id:2d}]   N_bc={par.N_bc}, tau_bc_n={par.tau_bc_n}, tau_bc_p={par.tau_bc_p}")

    if par.lambda_diss != 0.0:
        print(f"[Worker {worker_id:2d}] Localized dissipation: "
              f"lambda_diss={par.lambda_diss}, sigma_diss={par.sigma_diss}, x0={par.x0}")
    else:
        print(f"[Worker {worker_id:2d}] No localized dissipation perturbation (lambda_diss=0)")

    if par.lambda_gauss != 0.0:
        print(f"[Worker {worker_id:2d}] Time-independent Gaussian: "
              f"lambda_gauss={par.lambda_gauss}, sigma_gauss={par.sigma_gauss}, x0_gauss={par.x0_gauss}")
    else:
        print(f"[Worker {worker_id:2d}] No time-independent Gaussian perturbation (lambda_gauss=0)")

    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)


    if np.any(np.isnan(y0)) or np.any(np.isinf(y0)):
        print(f"[Worker {worker_id:2d}] WARNING: Initial conditions contain NaN or Inf values!")
        print(f"[Worker {worker_id:2d}]   n0: NaN={np.any(np.isnan(n0))}, Inf={np.any(np.isinf(n0))}")
        print(f"[Worker {worker_id:2d}]   p0: NaN={np.any(np.isnan(p0))}, Inf={np.any(np.isinf(p0))}")
    if np.any(n0 < 0):
        print(f"[Worker {worker_id:2d}] WARNING: Initial density contains negative values "
              f"(min={np.min(n0):.2e})")


    try:
        rhs_test = rhs(0.0, y0, E_base)
        if np.any(np.isnan(rhs_test)) or np.any(np.isinf(rhs_test)):
            print(f"[Worker {worker_id:2d}] WARNING: RHS at t=0 contains NaN or Inf values!")
            print(f"[Worker {worker_id:2d}]   RHS: NaN={np.any(np.isnan(rhs_test))}, "
                  f"Inf={np.any(np.isinf(rhs_test))}")
    except Exception as e:
        print(f"[Worker {worker_id:2d}] ERROR: RHS evaluation at t=0 failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"[Worker {worker_id:2d}] Starting simulation for {tag}...")
    print(f"[Worker {worker_id:2d}] Parameters: t_final={par.t_final}, Nx={par.Nx}, u_d={par.u_d}")

    start_wall_time = time.time()




    if par.bc_type == "ds_open":

        print(f"[Worker {worker_id:2d}] Using explicit RK4 integrator for DS/open BC")
        sol = integrate_explicit_ds_open(y0, (0.0, par.t_final), t_eval, E_base, worker_id)
        total_wall_time = time.time() - start_wall_time
        success = sol.success
        sol_message = sol.message
        t = sol.t
        Y = sol.y
    else:

        method = "BDF"
        rtol = par.rtol
        atol = par.atol
        print(f"[Worker {worker_id:2d}] Using BDF solver (rtol={rtol:.2e}, atol={atol:.2e}) "
              f"for periodic BC")

        last_print_time = [0.0]
        start_time = [start_wall_time]

        def _rhs_wrapper(t_inner, y_inner):
            return rhs_with_progress(t_inner, y_inner, E_base,
                                     last_print_time, start_time, worker_id)

        if HAS_THREADPOOLCTL:
            with threadpool_limits(limits=NTHREADS, user_api="blas"):
                with set_workers(NTHREADS):
                    sol = solve_ivp(
                        _rhs_wrapper,
                        (0.0, par.t_final),
                        y0,
                        t_eval=t_eval,
                        method=method,
                        rtol=rtol,
                        atol=atol,
                        max_step=0.1,
                        first_step=1e-3,
                    )
        else:
            with set_workers(NTHREADS):
                sol = solve_ivp(
                    _rhs_wrapper,
                    (0.0, par.t_final),
                    y0,
                    t_eval=t_eval,
                    method=method,
                    rtol=rtol,
                    atol=atol,
                    max_step=0.1,
                    first_step=1e-3,
                )

        total_wall_time = time.time() - start_wall_time
        success = sol.success
        sol_message = sol.message
        t = sol.t
        Y = sol.y




    print(f"[Worker {worker_id:2d}] Solver success: {success}")
    print(f"[Worker {worker_id:2d}] Final time: {t[-1]:.3f}, target: {par.t_final:.3f}")
    print(f"[Worker {worker_id:2d}] Total wall time: {total_wall_time:.2f} seconds")
    if not success and par.bc_type != "ds_open":
        print(f"[Worker {worker_id:2d}] Solver failed with message: {sol_message}")
    elif par.bc_type == "ds_open":
        print(f"[Worker {worker_id:2d}] Solver info: {sol_message}")


    N = par.Nx
    n_t = Y[:N, :]
    p_t = Y[N:, :]


    n_eff_t = np.maximum(n_t, par.n_floor)
    v_t = p_t / (par.m * n_eff_t)
    u_momentum_initial = float(np.mean(v_t[:, 0]))
    u_momentum_final   = float(np.mean(v_t[:, -1]))
    print(f"[Worker {worker_id:2d}]  <u>(t=0)={u_momentum_initial:.4f},  "
          f"<u>(t_end)={u_momentum_final:.4f},  target u_d={par.u_d:.4f}")


    if par.bc_type == "ds_open":
        print(f"[Worker {worker_id:2d}]  BC values at t_end: "
              f"n[0]={n_t[0, -1]:.4f} (target={par.nbar0:.4f}), "
              f"p[-1]={p_t[-1, -1]:.4f} (target={par.m*par.nbar0*par.u_d:.4f})")


    if len(t) >= 2:
        idx_t1 = -5 if len(t) >= 5 else 0
        idx_t2 = -1
        print(f"[Worker {worker_id:2d}] Using time points {idx_t1} and {idx_t2} "
              f"out of {len(t)} total")

        u_drift_inst, shift_opt_inst, corr_max_inst, shifts_inst, correlations_inst = \
            calculate_velocity_from_period(
                n_t[:, idx_t1], n_t[:, idx_t2],
                t[idx_t1], t[idx_t2],
                par.L
            )

        print(f"[Worker {worker_id:2d}]  u_drift_instantaneous={u_drift_inst:.4f} "
              f"(from shift={shift_opt_inst:.3f}, Δt={t[idx_t2]-t[idx_t1]:.3f})")
        print(f"[Worker {worker_id:2d}]  measured at t={t[idx_t2]:.3f}, "
              f"correlation_max={corr_max_inst:.4f}")


        plot_period_detection(
            n_t[:, idx_t1], n_t[:, idx_t2],
            t[idx_t1], t[idx_t2],
            par.L, u_momentum_final, par.u_d, tag=tag
        )
    else:
        print(f"[Worker {worker_id:2d}] Not enough time points ({len(t)}) "
              f"for instantaneous velocity measurement; skipping.")


    if len(t) > 1:
        x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
        dx_local = x_local[1] - x_local[0]
        extent = [x_local.min(), x_local.max(), t.min(), t.max()]


        plt.figure(figsize=(9.6, 5.0))
        plt.imshow(n_t.T, origin="lower", aspect="auto",
                   extent=extent, cmap=par.cmap)
        plt.xlabel("$x$")
        plt.ylabel("$t$")
        plt.title(f"$n(x,t)$  [lab]")
        plt.colorbar(label="$n$")
        plt.tight_layout()
        os.makedirs(par.outdir, exist_ok=True)
        fname_lab = os.path.join(par.outdir, f"spacetime_n_lab_{tag}.png")
        plt.savefig(fname_lab, dpi=300)
        plt.show()
        plt.close()
        print(f"[Worker {worker_id:2d}] Saved lab-frame spacetime plot → {os.path.abspath(fname_lab)}")


        n_co = np.empty_like(n_t)
        for j, tj in enumerate(t):
            shift = (par.u_d * tj) % par.L
            s_idx = int(np.round(shift / dx_local)) % par.Nx
            n_co[:, j] = np.roll(n_t[:, j], -s_idx)

        plt.figure(figsize=(9.6, 4.3))
        plt.imshow(n_co.T, origin="lower", aspect="auto",
                   extent=extent, cmap=par.cmap)
        plt.xlabel("ξ = x - u_d t")
        plt.ylabel("t")
        plt.title(f"n(ξ,t)  [co-moving u_d={par.u_d}]  {tag}")
        plt.colorbar(label="n")
        plt.tight_layout()
        fname_co = os.path.join(par.outdir, f"spacetime_n_comoving_{tag}.png")
        plt.savefig(fname_co, dpi=160)
        plt.close()
        print(f"[Worker {worker_id:2d}] Saved co-moving spacetime plot → {os.path.abspath(fname_co)}")


        plt.figure(figsize=(9.6, 5.0))
        percentages = [0, 20, 40, 60, 80, 100]

        colors = plt.cm.tab10(np.linspace(0, 1, len(percentages)))
        for i, pct in enumerate(percentages):
            idx = int((pct / 100) * (len(t) - 1))
            idx = min(idx, len(t) - 1)
            plt.plot(x_local, n_t[:, idx], label=f"$t={t[idx]:.1f}/{par.t_final:.1f}$",
                    color=colors[i], linewidth=2)
        plt.legend()
        plt.xlabel("$x$")
        plt.ylabel("$n$")
        plt.title(f"Density snapshots")
        plt.xlim(0, par.L)
        plt.tight_layout()
        fname_snaps = os.path.join(par.outdir, f"snapshots_n_{tag}.png")
        plt.savefig(fname_snaps, dpi=300)
        plt.close()
        print(f"[Worker {worker_id:2d}] Saved snapshots plot → {os.path.abspath(fname_snaps)}")
    else:
        print(f"[Worker {worker_id:2d}] Skipping spacetime plots: only {len(t)} time point(s) available")


    try:
        plot_fft_initial_last(n_t, t, par.L, tag=tag, k_marks=())
    except Exception as e:
        print(f"[Worker {worker_id:2d}] Warning: FFT plot failed with error: {e}")


    save_final_spectra(par.seed_mode, t, n_t, p_t, par.L, tag=tag)

    return t, n_t, p_t


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

    modes = range(1,2)
    results = []

    oldA, oldm = par.seed_amp_n, par.seed_mode

    print(f"[Multi-mode] Running {len(modes)} modes: {list(modes)}")


    multi_start_time = time.time()

    try:
        for i, m in enumerate(modes, 1):
            print(f"\n[Multi-mode] Running mode {m} ({i}/{len(modes)})")
            par.seed_mode = m
            t, n_t, p_t = run_once(tag=f"m{m}")
            results.append((m, t, n_t))
            save_final_spectra(m, t, n_t, p_t, par.L, tag=f"m{m}")
            print(f"[Multi-mode] Completed mode {m}")

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

        plot_all_final_spectra(results, par.L, tag=tag, normalize=False)

        total_multi_time = time.time() - multi_start_time
        print(f"\n[Multi-mode] All {len(modes)} modes completed successfully!")
        print(f"[Multi-mode] Total wall time: {total_multi_time:.2f} seconds")

    finally:
        par.seed_amp_n, par.seed_mode = oldA, oldm


def run_single_ud_worker(u_d, base_params, worker_id=0):
    import copy
    from dataclasses import dataclass


    local_par = P()
    for key, value in base_params.items():
        if hasattr(local_par, key):
            setattr(local_par, key, value)


    local_par.u_d = u_d
    u_d_str = f"{u_d:.4f}".replace('.', 'p')
    local_par.outdir = f"multiple_u_d/w=0.07_modes_3_5_7_L10(lambda={local_par.lambda_diss}, sigma={local_par.sigma_diss}, seed_amp_n={local_par.seed_amp_n}, seed_amp_p={local_par.seed_amp_p})/out_drift_ud{u_d_str}"


    local_par.t_final = 20*10.0/u_d

    local_par.n_save = 512


    local_par.Nx = 512


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

def run_single_w_worker(w, u_d, base_params, worker_id=0):
    import copy
    from dataclasses import dataclass


    local_par = P()
    for key, value in base_params.items():
        if hasattr(local_par, key):
            setattr(local_par, key, value)


    local_par.w = w
    local_par.u_d = u_d
    w_str = f"{w:.2f}"
    u_d_str = f"{u_d:.1f}".replace('.', 'p')
    local_par.outdir = f"multiple_u_d/multiple_w/w={w_str}_modes_3_5_7_L10(lambda={local_par.lambda_diss}, sigma={local_par.sigma_diss}, seed_amp_n={local_par.seed_amp_n}, seed_amp_p={local_par.seed_amp_p})/out_drift_ud{u_d_str}"


    local_par.t_final = 20*10.0/u_d
    local_par.n_save = 512
    local_par.Nx = 512


    global par
    par = local_par


    _update_global_arrays()

    print(f"\n{'='*50}")
    print(f"[Worker {worker_id:2d}] Running simulation for w = {w:.3f}, u_d = {u_d:.3f}")
    print(f"[Worker {worker_id:2d}] Parameters: t_final={par.t_final:.2f}, Nx={par.Nx}")
    print(f"{'='*50}")

    try:
        start_time = time.time()
        t, n_t, p_t = run_once(tag=f"w{w}_ud{u_d}", worker_id=worker_id)
        elapsed = time.time() - start_time

        print(f"[Worker {worker_id:2d}] Completed w={w:.3f}, u_d={u_d:.3f} in {elapsed:.1f}s")
        print(f"[Worker {worker_id:2d}] Final time: {t[-1]:.3f}, Data shapes: n_t={n_t.shape}, p_t={p_t.shape}")

        return {
            'w': w,
            'u_d': u_d,
            'success': True,
            'elapsed_time': elapsed,
            'final_time': t[-1],
            'outdir': par.outdir
        }

    except Exception as e:
        print(f"[Worker {worker_id:2d}] Error in simulation for w={w}, u_d={u_d}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'w': w,
            'u_d': u_d,
            'success': False,
            'elapsed_time': 0,
            'final_time': 0,
            'outdir': par.outdir,
            'error': str(e)
        }

def run_multiple_w():
    w_values = np.arange(0.01, 0.16, 0.01)

    u_d_values = np.arange(0.1, 2.0, 0.1)

    print(f"[run_multiple_w] Running parameter sweep with {len(w_values)} w values and {len(u_d_values)} u_d values")
    print(f"[run_multiple_w] w range: [{w_values[0]:.2f}, {w_values[-1]:.2f}]")
    print(f"[run_multiple_w] u_d range: [{u_d_values[0]:.1f}, {u_d_values[-1]:.1f}]")
    print(f"[run_multiple_w] Total combinations: {len(w_values) * len(u_d_values)}")


    base_params = asdict(par)


    combinations = [(w, u_d) for w in w_values for u_d in u_d_values]


    n_cpus = mp.cpu_count()
    n_workers = min(len(combinations), max(1, n_cpus - 1))

    print(f"\n[Parallel] Using {n_workers} parallel workers (out of {n_cpus} CPUs)")
    print(f"[Parallel] Running {len(combinations)} simulations in parallel")
    print(f"[Parallel] Progress will be shown for each worker simultaneously")
    print(f"[Parallel] Each worker will update its progress line independently")

    overall_start_time = time.time()


    with mp.Pool(processes=n_workers) as pool:

        worker_args = [(w, u_d, base_params, i) for i, (w, u_d) in enumerate(combinations)]
        results = pool.starmap(run_single_w_worker, worker_args)

    overall_elapsed = time.time() - overall_start_time


    print(f"\n{'='*60}")
    print(f"ALL SIMULATIONS COMPLETED!")
    print(f"{'='*60}")
    print(f"Total wall time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print(f"\nSummary:")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"  Successful: {len(successful)}/{len(results)}")
    print(f"  Failed: {len(failed)}/{len(results)}")

    if successful:
        print(f"\nSuccessful simulations (first 10):")
        for r in successful[:10]:
            print(f"  w={r['w']:.2f}, u_d={r['u_d']:.1f}: {r['elapsed_time']:.1f}s, t_final={r['final_time']:.3f}")
        if len(successful) > 10:
            print(f"  ... and {len(successful) - 10} more")

    if failed:
        print(f"\nFailed simulations:")
        for r in failed:
            print(f"  w={r['w']:.2f}, u_d={r['u_d']:.1f}: {r.get('error', 'Unknown error')}")

    print(f"{'='*60}")

def run_single_diffusion_worker(w, u_d, Dn, Dp, base_params, worker_id=0):
    import copy
    from dataclasses import dataclass


    local_par = P()
    for key, value in base_params.items():
        if hasattr(local_par, key):
            setattr(local_par, key, value)


    local_par.w = w
    local_par.u_d = u_d
    local_par.Dn = Dn
    local_par.Dp = Dp

    w_str = f"{w:.2f}"
    u_d_str = f"{u_d:.1f}".replace('.', 'p')
    Dn_str = f"{Dn:.2f}".replace('.', 'p')
    Dp_str = f"{Dp:.2f}".replace('.', 'p')

    local_par.outdir = f"multiple_u_d/non-periodic/w={w_str}_Dn={Dn_str}_Dp={Dp_str}_L10(lambda={local_par.lambda_diss}, sigma={local_par.sigma_diss}, seed_amp_n={local_par.seed_amp_n}, seed_amp_p={local_par.seed_amp_p})/out_drift_ud{u_d_str}"


    local_par.t_final = 20*10.0/u_d
    local_par.n_save = 512
    local_par.Nx = 512


    global par
    par = local_par


    _update_global_arrays()

    print(f"\n{'='*50}")
    print(f"[Worker {worker_id:2d}] Running simulation for w={w:.3f}, u_d={u_d:.3f}, Dn={Dn:.3f}, Dp={Dp:.3f}")
    print(f"[Worker {worker_id:2d}] Parameters: t_final={par.t_final:.2f}, Nx={par.Nx}")
    print(f"{'='*50}")

    try:
        start_time = time.time()
        t, n_t, p_t = run_once(tag=f"w{w}_ud{u_d}_Dn{Dn}_Dp{Dp}", worker_id=worker_id)
        elapsed = time.time() - start_time

        print(f"[Worker {worker_id:2d}] Completed w={w:.3f}, u_d={u_d:.3f}, Dn={Dn:.3f}, Dp={Dp:.3f} in {elapsed:.1f}s")
        print(f"[Worker {worker_id:2d}] Final time: {t[-1]:.3f}, Data shapes: n_t={n_t.shape}, p_t={p_t.shape}")

        return {
            'w': w,
            'u_d': u_d,
            'Dn': Dn,
            'Dp': Dp,
            'success': True,
            'elapsed_time': elapsed,
            'final_time': t[-1],
            'outdir': par.outdir
        }

    except Exception as e:
        print(f"[Worker {worker_id:2d}] Error in simulation for w={w}, u_d={u_d}, Dn={Dn}, Dp={Dp}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'w': w,
            'u_d': u_d,
            'Dn': Dn,
            'Dp': Dp,
            'success': False,
            'elapsed_time': 0,
            'final_time': 0,
            'outdir': par.outdir,
            'error': str(e)
        }

def run_diffusion_parameter_sweep():
    w_values = np.arange(0.01, 0.26, 0.01)


    u_d_values = np.arange(0.1, 2.0, 0.1)



    base_Dn = 0.5
    base_Dp = 0.1

    diffusion_combinations = [

        {"name": "Dn_half", "Dn": base_Dn / 2, "Dp": base_Dp},

        {"name": "Dp_half", "Dn": base_Dn, "Dp": base_Dp / 2},

        {"name": "both_half", "Dn": base_Dn / 2, "Dp": base_Dp / 2},

        {"name": "Dn_double", "Dn": base_Dn * 2, "Dp": base_Dp},

        {"name": "Dp_double", "Dn": base_Dn, "Dp": base_Dp * 2},

        {"name": "both_double", "Dn": base_Dn * 2, "Dp": base_Dp * 2},
    ]

    print(f"[run_diffusion_parameter_sweep] Running parameter sweep:")
    print(f"  w values: {len(w_values)} from {w_values[0]:.2f} to {w_values[-1]:.2f}")
    print(f"  u_d values: {len(u_d_values)} from {u_d_values[0]:.1f} to {u_d_values[-1]:.1f}")
    print(f"  Diffusion combinations: {len(diffusion_combinations)}")
    print(f"  Total simulations: {len(w_values) * len(u_d_values) * len(diffusion_combinations)}")


    base_params = asdict(par)


    all_combinations = []
    for diff_combo in diffusion_combinations:
        for w in w_values:
            for u_d in u_d_values:
                all_combinations.append((w, u_d, diff_combo["Dn"], diff_combo["Dp"], diff_combo["name"]))


    n_cpus = mp.cpu_count()
    n_workers = min(len(all_combinations), max(1, n_cpus - 1))

    print(f"\n[Parallel] Using {n_workers} parallel workers (out of {n_cpus} CPUs)")
    print(f"[Parallel] Running {len(all_combinations)} simulations in parallel")
    print(f"[Parallel] Progress will be shown for each worker simultaneously")
    print(f"[Parallel] Each worker will update its progress line independently")

    overall_start_time = time.time()


    with mp.Pool(processes=n_workers) as pool:

        worker_args = [(w, u_d, Dn, Dp, base_params, i) for i, (w, u_d, Dn, Dp, name) in enumerate(all_combinations)]
        results = pool.starmap(run_single_diffusion_worker, worker_args)

    overall_elapsed = time.time() - overall_start_time


    print(f"\n{'='*80}")
    print(f"ALL DIFFUSION PARAMETER SWEEP SIMULATIONS COMPLETED!")
    print(f"{'='*80}")
    print(f"Total wall time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print(f"\nSummary:")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"  Successful: {len(successful)}/{len(results)}")
    print(f"  Failed: {len(failed)}/{len(results)}")


    results_by_combo = {}
    for result in successful:
        combo_key = f"Dn={result['Dn']:.2f}, Dp={result['Dp']:.2f}"
        if combo_key not in results_by_combo:
            results_by_combo[combo_key] = []
        results_by_combo[combo_key].append(result)

    print(f"\nResults by diffusion combination:")
    for combo_key, combo_results in results_by_combo.items():
        avg_time = np.mean([r['elapsed_time'] for r in combo_results])
        print(f"  {combo_key}: {len(combo_results)} simulations, avg time: {avg_time:.1f}s")

    if successful:
        print(f"\nSuccessful simulations (first 10):")
        for r in successful[:10]:
            print(f"  w={r['w']:.2f}, u_d={r['u_d']:.1f}, Dn={r['Dn']:.2f}, Dp={r['Dp']:.2f}: {r['elapsed_time']:.1f}s")
        if len(successful) > 10:
            print(f"  ... and {len(successful) - 10} more")

    if failed:
        print(f"\nFailed simulations:")
        for r in failed:
            print(f"  w={r['w']:.2f}, u_d={r['u_d']:.1f}, Dn={r['Dn']:.2f}, Dp={r['Dp']:.2f}: {r.get('error', 'Unknown error')}")

    print(f"{'='*80}")

    return results

def run_multiple_ud():

    u_d_values = np.arange(0.2, 2.0, 0.1)

    print(f"[run_multiple_ud] Running parameter sweep with {len(u_d_values)} u_d values")
    print(f"[run_multiple_ud] Range: [{u_d_values[0]:.4f}, {u_d_values[-1]:.4f}]")
    print(f"[run_multiple_ud] Step size: {u_d_values[1] - u_d_values[0]:.4f}")
    print(f"[run_multiple_ud] u_d values: {u_d_values}")


    print(f"[run_multiple_ud] u_d values to simulate: {u_d_values}")


    base_params = asdict(par)


    n_cpus = mp.cpu_count()
    n_workers = min(len(u_d_values), max(1, n_cpus - 1))

    print(f"\n[Parallel] Using {n_workers} parallel workers (out of {n_cpus} CPUs)")
    print(f"[Parallel] Running {len(u_d_values)} simulations in parallel")
    print(f"[Parallel] Progress will be shown for each worker simultaneously")
    print(f"[Parallel] Each worker will update its progress line independently")

    overall_start_time = time.time()


    with mp.Pool(processes=n_workers) as pool:

        worker_args = [(u_d, base_params, i) for i, u_d in enumerate(u_d_values)]
        results = pool.starmap(run_single_ud_worker, worker_args)

    overall_elapsed = time.time() - overall_start_time


    print(f"\n{'='*60}")
    print(f"ALL SIMULATIONS COMPLETED!")
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
        speedup = total_sim_time / overall_elapsed
        print(f"  Average simulation time: {avg_time:.1f}s")
        print(f"  Total simulation time (sequential equivalent): {total_sim_time:.1f}s")
        print(f"  Parallel speedup: {speedup:.2f}x")

    if failed:
        print(f"\nFailed simulations:")
        for r in failed:
            print(f"  u_d={r['u_d']:.4f}: {r.get('error', 'Unknown error')}")

    print(f"{'='*60}")

def detect_missing_simulations(base_dir="multiple_u_d/non-periodic"):
    w_values = [f"{w:.2f}" for w in np.arange(0.01, 0.26, 0.01)]
    u_d_values = [f"{u_d:.1f}".replace('.', 'p') for u_d in np.arange(0.1, 2.0, 0.4)]

    diffusion_combinations = [
        {"name": "Dn_half", "Dn": "0p25", "Dp": "0p10"},
        {"name": "Dp_half", "Dn": "0p50", "Dp": "0p05"},
        {"name": "both_half", "Dn": "0p25", "Dp": "0p05"},
        {"name": "Dn_double", "Dn": "1p00", "Dp": "0p10"},
        {"name": "Dp_double", "Dn": "0p50", "Dp": "0p20"},
        {"name": "both_double", "Dn": "1p00", "Dp": "0p20"},
    ]

    missing_folders = []
    missing_out_dirs = []
    missing_npz = []
    complete_simulations = 0

    for diff_combo in diffusion_combinations:
        for w in w_values:
            folder_name = f"w={w}_Dn={diff_combo['Dn']}_Dp={diff_combo['Dp']}_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.03, seed_amp_p=0.03)"
            folder_path = os.path.join(base_dir, folder_name)

            if not os.path.exists(folder_path):
                missing_folders.append(folder_name)
                continue

            for u_d in u_d_values:
                out_dir = os.path.join(folder_path, f"out_drift_ud{u_d}")

                if not os.path.exists(out_dir):
                    missing_out_dirs.append(f"out_drift_ud{u_d} in {folder_name}")
                    continue

                npz_files = glob.glob(os.path.join(out_dir, "*.npz"))
                if not npz_files:
                    missing_npz.append(f"No .npz files in {out_dir}")
                else:
                    complete_simulations += 1

    return {
        'missing_folders': missing_folders,
        'missing_out_dirs': missing_out_dirs,
        'missing_npz': missing_npz,
        'complete_simulations': complete_simulations
    }

def check_and_run_missing_simulations():
    print("="*80)
    print("CHECKING FOR MISSING SIMULATIONS")
    print("="*80)


    missing_info = detect_missing_simulations()

    total_missing = len(missing_info['missing_folders']) * 19 + len(missing_info['missing_out_dirs']) + len(missing_info['missing_npz'])

    print(f"Complete simulations: {missing_info['complete_simulations']}")
    print(f"Missing folders: {len(missing_info['missing_folders'])} (affects {len(missing_info['missing_folders']) * 19} simulations)")
    print(f"Missing out dirs: {len(missing_info['missing_out_dirs'])}")
    print(f"Missing npz files: {len(missing_info['missing_npz'])}")
    print(f"Total missing simulations: {total_missing}")

    if total_missing == 0:
        print("No missing simulations found!")
        return

    print(f"\nRunning {total_missing} missing simulations...")


    base_params = asdict(par)


    missing_tasks = []

    for diff_combo in [
        {"name": "Dn_half", "Dn": 0.25, "Dp": 0.10},
        {"name": "Dp_half", "Dn": 0.50, "Dp": 0.05},
        {"name": "both_half", "Dn": 0.25, "Dp": 0.05},
        {"name": "Dn_double", "Dn": 1.00, "Dp": 0.10},
        {"name": "Dp_double", "Dn": 0.50, "Dp": 0.20},
        {"name": "both_double", "Dn": 1.00, "Dp": 0.20},
    ]:
        for w in np.arange(0.01, 0.26, 0.01):
            for u_d in np.arange(0.1, 2.0, 0.4):

                w_str = f"{w:.2f}"
                u_d_str = f"{u_d:.1f}".replace('.', 'p')
                Dn_str = f"{diff_combo['Dn']:.2f}".replace('.', 'p')
                Dp_str = f"{diff_combo['Dp']:.2f}".replace('.', 'p')

                folder_name = f"w={w_str}_Dn={Dn_str}_Dp={Dp_str}_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.03, seed_amp_p=0.03)"
                folder_path = os.path.join("multiple_u_d/non-periodic", folder_name)
                out_dir = os.path.join(folder_path, f"out_drift_ud{u_d_str}")

                is_missing = False
                if not os.path.exists(folder_path):
                    is_missing = True
                elif not os.path.exists(out_dir):
                    is_missing = True
                else:
                    npz_files = glob.glob(os.path.join(out_dir, "*.npz"))
                    if not npz_files:
                        is_missing = True

                if is_missing:
                    missing_tasks.append((w, u_d, diff_combo['Dn'], diff_combo['Dp']))

    print(f"Found {len(missing_tasks)} missing simulation tasks")

    if len(missing_tasks) == 0:
        print("No missing simulations to run!")
        return


    n_cpus = mp.cpu_count()
    n_workers = min(len(missing_tasks), max(1, n_cpus - 1))

    print(f"Using {n_workers} parallel workers for {len(missing_tasks)} missing simulations")

    overall_start_time = time.time()


    with mp.Pool(processes=n_workers) as pool:
        worker_args = [(w, u_d, Dn, Dp, base_params, i) for i, (w, u_d, Dn, Dp) in enumerate(missing_tasks)]
        results = pool.starmap(run_single_diffusion_worker, worker_args)

    overall_elapsed = time.time() - overall_start_time


    print(f"\n{'='*80}")
    print(f"MISSING SIMULATIONS COMPLETED!")
    print(f"{'='*80}")
    print(f"Total wall time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if failed:
        print(f"\nFailed simulations:")
        for r in failed:
            print(f"  w={r['w']:.2f}, u_d={r['u_d']:.1f}, Dn={r['Dn']:.2f}, Dp={r['Dp']:.2f}: {r.get('error', 'Unknown error')}")

    print(f"{'='*80}")

def run_single_Dn_half_simulation():
    print("="*80)
    print("RUNNING SINGLE Dn_half SIMULATION")
    print("="*80)
    print("Configuration: Dn_half (Dn=0.25, Dp=0.10)")
    print("w = 0.14")
    print("u_d range: 0.1 to 2.0 with 0.1 step")


    par.Dn = 0.1
    par.Dp = 0.10
    par.w = 0.14


    u_d_values = np.arange(0.1, 2.1, 0.1)

    print(f"Total simulations to run: {len(u_d_values)}")


    base_params = asdict(par)


    simulation_tasks = []
    for u_d in u_d_values:
        simulation_tasks.append((par.w, u_d, par.Dn, par.Dp))


    n_cpus = mp.cpu_count()
    n_workers = min(len(simulation_tasks), max(1, n_cpus - 1))

    print(f"Using {n_workers} parallel workers for {len(simulation_tasks)} simulations")

    overall_start_time = time.time()


    with mp.Pool(processes=n_workers) as pool:
        worker_args = [(w, u_d, Dn, Dp, base_params, i) for i, (w, u_d, Dn, Dp) in enumerate(simulation_tasks)]
        results = pool.starmap(run_single_diffusion_worker, worker_args)

    overall_elapsed = time.time() - overall_start_time


    print(f"\n{'='*80}")
    print(f"Dn_half SIMULATION COMPLETED!")
    print(f"{'='*80}")
    print(f"Total wall time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        print(f"\nSuccessful simulations:")
        for r in successful:
            print(f"  w={r['w']:.2f}, u_d={r['u_d']:.1f}, Dn={r['Dn']:.2f}, Dp={r['Dp']:.2f}: {r['elapsed_time']:.1f}s")

    if failed:
        print(f"\nFailed simulations:")
        for r in failed:
            print(f"  w={r['w']:.2f}, u_d={r['u_d']:.1f}, Dn={r['Dn']:.2f}, Dp={r['Dp']:.2f}: {r.get('error', 'Unknown error')}")

    print(f"{'='*80}")

if __name__ == "__main__":


    par.bc_type = "ds_open"


    par.L  = 10.0
    par.Nx = 512*4


    par.u_d    = 0.7
    par.m      = 1.0
    par.e      = 1.0
    par.nbar0  = 0.2

    par.Gamma0 = 2.5
    par.w      = 0.04


    par.U      = 1.0


    par.Dn = 0.001
    par.Dp = 0.001


    par.include_poisson   = False
    par.lambda_diss       = 0.0
    par.lambda_gauss      = 0.0
    par.use_nbar_gaussian = False


    par.seed_mode  = 7
    par.seed_amp_n = 0.01
    par.seed_amp_p = 0.01


    par.t_final = 30.0
    par.n_save  = 600


    par.rtol = 1e-3
    par.atol = 1e-6


    par.N_bc     = 8
    par.tau_bc_n = 0.5
    par.tau_bc_p = 0.5


    par.maintain_drift = "feedback"
    par.Kp = 0.15

    par.outdir = "out_ds_open_unstable_M07"

    _update_global_arrays()

    print(f"[Main] Running DS-open UNSTABLE test: Nx={par.Nx}, t_final={par.t_final}")
    print(f"[Main] BC parameters: N_bc={par.N_bc}, tau_bc_n={par.tau_bc_n}, tau_bc_p={par.tau_bc_p}")
    t, n_t, p_t = run_once(tag="ds_open_unstable_M07", worker_id=0)


    print("\n[BC check]")
    if par.bc_type == "ds_open":
        print("  Source (x≈0): n(0, t0)={:.6g}, n(0, t_end)={:.6g} (target={:.6g})"
              .format(n_t[0, 0], n_t[0, -1], par.nbar0))
        print("  Drain (x≈L):  p(L, t0)={:.6g}, p(L, t_end)={:.6g} (target={:.6g})"
              .format(p_t[-1, 0], p_t[-1, -1], par.m * par.nbar0 * par.u_d))
        print("  Note: Source fixes density, Drain fixes momentum/current")
    else:
        print("  Periodic BC: no boundary constraints")
        print("  n(0, t0)={:.6g}, n(0, t_end)={:.6g}"
              .format(n_t[0, 0], n_t[0, -1]))
        print("  p(L, t0)={:.6g}, p(L, t_end)={:.6g}"
              .format(p_t[-1, 0], p_t[-1, -1]))
