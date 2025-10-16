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

# Import spectral analysis helpers
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
    U: float = 1.0#0.06
    nbar0: float = 0.2
    Gamma0: float = 2.50#0.08
    w: float = 0.4#5.0
    include_poisson: bool = False
    eps: float = 20.0

    u_d: float = 5.245
    # u_d: float = .0
    maintain_drift: str = "field"
    Kp: float = 0.15

    Dn: float = 0.5#/10#0.03
    Dp: float = 0.1

    J0: float = 1.0#0.04
    sigma_J: float = 2.0**1/2#6.0
    x0: float = 12.5
    source_model: str = "as_given"
    
    # Localized dissipation perturbation parameters
    lambda_diss: float = 0.0  # Amplitude of dissipation perturbation (can be positive or negative)
    sigma_diss: float = 2.0   # Width of dissipation perturbation
    
    # Time-independent Gaussian density perturbation
    lambda_gauss: float = 0.0  # Amplitude of Gaussian density perturbation
    sigma_gauss: float = 2.0   # Width of Gaussian density perturbation
    x0_gauss: float = 12.5      # Center of Gaussian density perturbation

    use_nbar_gaussian: bool = False
    nbar_amp: float = 0.0
    nbar_sigma: float = 120.0

    L: float = 10.0  # System size
    Nx: int = 1212#2048#1218#1512#2524#1024
    t_final: float = 50.0
    n_save: int = 100  #200#200  # Reduced for speed
    # rtol: float = 5e-7
    # atol: float = 5e-9
    rtol = 1e-4  # Tighter tolerance for accuracy
    atol = 1e-7  # Tighter tolerance for accuracy
    n_floor: float = 1e-7
    dealias_23: bool = True

    seed_amp_n: float = 0.001  # Small amplitude perturbation
    seed_mode: int = 7  # New mode for cos(6πx/L) + cos(10πx/L) + cos(14πx/L)
    seed_amp_p: float = 0.001  # Small amplitude perturbation

    outdir: str = "out_drift/small_dissipation_perturbation"
    cmap: str = "inferno"

par = P()

# Global arrays will be created dynamically based on current par.Nx
x = None
dx = None
k = None
ik = None
k2 = None

# Pre-compile FFT operations for speed
_fft_cache = {}

def _update_global_arrays():
    """Update global arrays based on current par.Nx"""
    global x, dx, k, ik, k2, _kc, _nz_mask
    x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    dx = x[1] - x[0]
    k = 2*np.pi*fftfreq(par.Nx, d=dx)
    ik = 1j*k
    k2 = k**2
    _kc = par.Nx//3
    _nz_mask = (k2 != 0)

def Dx(f):  
    # Ensure arrays are up to date
    if k is None or len(k) != len(f):
        _update_global_arrays()
    return (ifft(ik * fft(f, workers=NTHREADS), workers=NTHREADS)).real

def Dxx(f): 
    # Ensure arrays are up to date
    if k2 is None or len(k2) != len(f):
        _update_global_arrays()
    return (ifft((-k2) * fft(f, workers=NTHREADS), workers=NTHREADS)).real

def filter_23(f):
    if not par.dealias_23: return f
    fh = fft(f, workers=NTHREADS)
    kc = len(f)//3  # Use actual array length instead of par.Nx
    fh[kc:-kc] = 0.0
    return (ifft(fh, workers=NTHREADS)).real

# Pre-calculate constants for speed (will be updated dynamically)
_kc = None
_nz_mask = None

def Gamma(n):
    return par.Gamma0 * np.exp(-np.maximum(n, par.n_floor)/par.w)

def Gamma_spatial(n):
    """
    Spatially-dependent damping coefficient with localized perturbation.
    Gamma_eff(x,n) = Gamma(n) + lambda_diss * exp(-(x-x0)^2/(2*sigma_diss^2))
    """
    Gamma_base = Gamma(n)
    if par.lambda_diss != 0.0:
        # Use the same spatial grid as the input n array
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
    """Return one-sided k>0 spectrum of a single spatial slice."""
    N = n_slice.size
    dn = n_slice - np.mean(n_slice)
    nhat = fft(dn, workers=NTHREADS)
    P = (nhat * np.conj(nhat)).real / (N*N)
    m = np.arange(N//2 + 1)
    kpos = 2*np.pi*m / L
    return kpos[1:], P[1:N//2+1]

def _fourier_shift_real(f, shift, L):
    """Circularly shift a real 1D signal by a non-integer amount using FFT."""
    N = f.size
    k = 2*np.pi * np.fft.rfftfreq(N, d=L/N)      # wavenumbers (>=0)
    F = np.fft.rfft(f)
    return np.fft.irfft(F * np.exp(1j*k*shift), n=N)

def estimate_velocity_fourier(n_t1, n_t2, t1, t2, L, power_floor=1e-3):
    """
    Estimate spatial shift and velocity from two snapshots using the
    cross-spectrum phase: Δφ_k = k * shift  (mod 2π).
    """
    N = n_t1.size
    # remove mean to drop k=0
    f1 = n_t1 - n_t1.mean()
    f2 = n_t2 - n_t2.mean()

    k = 2*np.pi * np.fft.rfftfreq(N, d=L/N)            # [0..Nyquist]
    F1 = np.fft.rfft(f1)
    F2 = np.fft.rfft(f2)
    C  = np.conj(F1) * F2                        # cross-spectrum
    phi = np.angle(C)                            # phase diff

    # discard k=0; keep only energetic modes for robustness
    k = k[1:]
    phi = phi[1:]
    w = np.abs(C[1:])                            # weights ~ power
    mask = w > (power_floor * w.max())
    k, phi, w = k[mask], phi[mask], w[mask]

    # unwrap the phase along k so it's linear: phi ≈ k * shift
    phi = np.unwrap(phi)

    # weighted least-squares slope: shift = argmin ||phi - k*shift||_W
    # shift = (Σ w k φ) / (Σ w k^2)
    num = np.sum(w * k * phi)
    den = np.sum(w * k**2)
    shift = num / den

    # Fix sign convention: positive shift = forward motion (rightward drift)
    # The phase difference phi = angle(F2) - angle(F1) should give the correct sign
    # But we need to ensure positive shift means n_t1 shifted right to match n_t2
    shift = -shift  # Flip sign to match spatial convention

    # map to principal interval [-L/2, L/2] (purely cosmetic)
    shift = (shift + 0.5*L) % L - 0.5*L

    dt = float(t2 - t1)
    u = shift / dt
    return u, shift

def find_modulation_period_by_shift(n_t1, n_t2, t1, t2, L):
    """
    Find the drift velocity by comparing two snapshots at different times.
    Uses robust Fourier-based method that works with multiple modes.
    
    Parameters:
    -----------
    n_t1 : array
        Density profile at time t1
    n_t2 : array
        Density profile at time t2
    t1 : float
        First time
    t2 : float
        Second time
    L : float
        Domain length
        
    Returns:
    --------
    u_drift : float
        Drift velocity
    shift_optimal : float
        Optimal spatial shift
    correlation_max : float
        Maximum correlation achieved
    shifts : array
        Array of tested shifts
    correlations : array
        Correlation for each shift
    """
    u, shift = estimate_velocity_fourier(n_t1, n_t2, t1, t2, L)
    
    # Build a correlation-vs-shift curve via FFT xcorr for visualization
    N = n_t1.size
    f1 = n_t1 - n_t1.mean()
    f2 = n_t2 - n_t2.mean()
    F1 = np.fft.rfft(f1)
    F2 = np.fft.rfft(f2)
    xcorr = np.fft.irfft(np.conj(F1) * F2, n=N)        # circular correlation
    dx = L/N
    shifts = dx * (np.arange(N) - (N//2))
    xcorr = np.roll(xcorr, -N//2)
    
    # Simple normalized "correlation"
    correlations = (xcorr - xcorr.min())/(xcorr.max()-xcorr.min() + 1e-15)
    
    # Use Fourier-estimated shift as "optimal"
    corr_max = np.interp(shift, shifts, correlations)
    
    return u, shift, corr_max, shifts, correlations

def calculate_velocity_from_period(n_initial, n_final, t_initial, t_final, L):
    """
    Calculate velocity by comparing initial and final profiles.
    
    This method:
    1. Takes snapshots at t_initial and t_final
    2. Finds the spatial shift that best matches the two profiles
    3. Calculates velocity = shift / (t_final - t_initial)
    
    Parameters:
    -----------
    n_initial : array
        Initial density profile n(x, t=t_initial)
    n_final : array
        Final density profile n(x, t=t_final)
    t_initial : float
        Initial time
    t_final : float
        Final time
    L : float
        Domain length
        
    Returns:
    --------
    u_drift : float
        Drift velocity
    shift_optimal : float
        Optimal spatial shift
    correlation_max : float
        Maximum correlation
    shifts : array
        Array of tested shifts
    correlations : array
        Correlation for each shift
    """
    u_drift, shift_optimal, correlation_max, shifts, correlations = find_modulation_period_by_shift(
        n_initial, n_final, t_initial, t_final, L
    )
    
    return u_drift, shift_optimal, correlation_max, shifts, correlations

def calculate_velocity_from_shift_refined(n_initial, n_final, t_final, L, search_range=None):
    """
    Refined velocity calculation using sub-pixel shift optimization.
    
    Parameters:
    -----------
    n_initial, n_final : array
        Initial and final density profiles
    t_final : float
        Final time
    L : float
        Domain length  
    search_range : tuple, optional
        (u_min, u_max) to limit search range
        
    Returns:
    --------
    u_optimal : float
        Optimal velocity
    shift_optimal : float  
        Optimal spatial shift
    correlation_max : float
        Maximum correlation achieved
    """
    dx = L / len(n_initial)
    x = np.linspace(0, L, len(n_initial), endpoint=False)
    
    # Remove mean
    dn_initial = n_initial - np.mean(n_initial)
    dn_final = n_final - np.mean(n_final)
    
    # First get coarse estimate
    u_coarse, shift_coarse, _ = calculate_velocity_from_shift(n_initial, n_final, t_final, L)
    
    # Define search range around coarse estimate
    if search_range is None:
        u_search_width = 2.0  # Search ±2 units around coarse estimate
        u_min = u_coarse - u_search_width
        u_max = u_coarse + u_search_width
    else:
        u_min, u_max = search_range
    
    # Fine search with sub-pixel resolution
    n_search = 201  # Number of search points
    u_test = np.linspace(u_min, u_max, n_search)
    correlations = np.zeros(n_search)
    
    for i, u in enumerate(u_test):
        shift = u * t_final
        
        # Create shifted initial profile using interpolation
        x_shifted = (x - shift) % L
        dn_initial_shifted = np.interp(x, x_shifted, dn_initial)
        
        # Calculate correlation coefficient
        correlations[i] = np.corrcoef(dn_final, dn_initial_shifted)[0, 1]
    
    # Find optimal velocity
    max_idx = np.argmax(correlations)
    u_optimal = u_test[max_idx]
    shift_optimal = u_optimal * t_final
    correlation_max = correlations[max_idx]
    
    return u_optimal, shift_optimal, correlation_max

def plot_period_detection(n_initial, n_final, t_initial, t_final, L, u_momentum, u_target, tag="period_detection"):
    """
    Plot the velocity detection method showing correlation vs shift.
    """
    # Calculate velocity by comparing initial and final profiles
    u_drift, shift_opt, corr_max, shifts, correlations = calculate_velocity_from_period(
        n_initial, n_final, t_initial, t_final, L
    )
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Initial and shifted initial vs final
    x = np.linspace(0, L, len(n_final), endpoint=False)
    
    # Create shifted initial profile at detected shift using Fourier method
    n_initial_shifted = _fourier_shift_real(n_initial, shift_opt, L)
    
    ax1.plot(x, n_initial, 'b-', label=f'Initial n(x,{t_initial:.2f})', alpha=0.6, linewidth=1.5)
    ax1.plot(x, n_final, 'r-', label=f'Final n(x,{t_final:.2f})', alpha=0.8, linewidth=2)
    ax1.plot(x, n_initial_shifted, 'g--', label=f'Initial shifted by {shift_opt:.3f}', alpha=0.8, linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('n(x)')
    ax1.set_title('Velocity Detection: Shifted Initial vs Final')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation vs shift
    ax2.plot(shifts, correlations, 'b-', linewidth=2, label='Correlation')
    ax2.axvline(shift_opt, color='r', linestyle='--', linewidth=2, label=f'Optimal shift={shift_opt:.3f}')
    ax2.plot([shift_opt], [corr_max], 'ro', markersize=8, label=f'Max corr={corr_max:.3f}')
    
    ax2.set_xlabel('Spatial shift')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation vs Shift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add text summary
    delta_t = t_final - t_initial
    fig.suptitle(f'Instantaneous velocity at t={t_final:.3f}: u_drift={u_drift:.3f} (shift={shift_opt:.3f}, Δt={delta_t:.4f}) | u_momentum={u_momentum:.3f}, u_target={u_target:.3f}', 
                 y=0.98, fontsize=10.5)
    
    os.makedirs(par.outdir, exist_ok=True)
    plt.savefig(f"{par.outdir}/period_detection_{tag}.png", dpi=160, bbox_inches='tight')
    plt.savefig(f"{par.outdir}/period_detection_{tag}.pdf", dpi=160, bbox_inches='tight')
    # plt.show()
    plt.close()
    
    return u_drift, shift_opt, corr_max

def nbar_profile():
    # Create spatial grid with current Nx
    x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    
    # For seed_mode == 2 or 7, use uniform background without any Gaussian perturbations
    if par.seed_mode == 2 or par.seed_mode == 7:
        return np.full_like(x_local, par.nbar0)
    
    # For other modes, keep original logic with potential Gaussian perturbations
    nbar_base = np.full_like(x_local, par.nbar0)
    
    # Add time-independent Gaussian perturbation
    if par.lambda_gauss != 0.0:
        d_gauss = periodic_delta(x_local, par.x0_gauss, par.L)
        gauss_pert = par.lambda_gauss * np.exp(-0.5*(d_gauss/par.sigma_gauss)**2)
        nbar_base += gauss_pert
    
    # Add original Gaussian profile if enabled
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
    print(np.mean(Gamma(nbar)))
    return par.m * par.u_d * np.mean(Gamma(nbar)) / par.e 

def rhs(t, y, E_base):
    N = par.Nx
    n = y[:N]
    p = y[N:]

    nbar = nbar_profile()
    pbar = pbar_profile(nbar)

    n_eff = np.maximum(n, par.n_floor)

    # Skip injection calculations since SJ term is disabled
    # Jx = J_profile()
    # gamma = gamma_from_J(Jx)
    # SJ = S_injection(n_eff, nbar, Jx, gamma)

    v = p/(par.m*n_eff)
    u_mean = float(np.mean(v))
    if par.maintain_drift == "feedback":
        E_eff = E_base + par.Kp * (par.u_d - u_mean)
    else:
        E_eff = E_base

    dn_dt = -Dx(p) + par.Dn * Dxx(n)  # SJ term disabled for speed
    dn_dt = filter_23(dn_dt)

    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)
    grad_Pi = Dx(Pi)
    force_Phi = 0.0
    if par.include_poisson:
        phi = phi_from_n(n_eff, nbar)
        force_Phi = n_eff * Dx(phi)

    # Use spatially-dependent damping with localized perturbation
    dp_dt = -Gamma_spatial(n_eff)*p - grad_Pi + par.e*n_eff*E_eff - force_Phi + par.Dp * Dxx(p)
    dp_dt = filter_23(dp_dt)

    return np.concatenate([dn_dt, dp_dt])

def initial_fields():
    # Create spatial grid with current Nx
    x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    
    # Get uniform background profiles (no Gaussian perturbations)
    nbar = np.full_like(x_local, par.nbar0)  # Uniform background density
    pbar = par.m * nbar * par.u_d            # Uniform background momentum
    
    # Initialize with background values
    n0 = nbar.copy()
    p0 = pbar.copy()
    
    # Add sum of sine waves as delta n and delta p for seed_mode == 2
    if par.seed_mode == 2 and (par.seed_amp_n != 0.0 or par.seed_amp_p != 0.0):
    # Define sine wave modes: include more harmonics (m=2,3,5) and larger modes
        modes = [2, 3, 5, 8, 13, 21, 34, 55]
        
        # Create sum of sine waves perturbation
        sine_perturbation = np.zeros_like(x_local)
        for mode in modes:
            kx = 2*np.pi*mode / par.L
            sine_perturbation += np.cos(kx * x_local)
        
        # Normalize by number of modes to keep amplitude reasonable
        sine_perturbation = sine_perturbation / len(modes)
        
        # Add delta n perturbation
        if par.seed_amp_n != 0.0:
            delta_n = par.seed_amp_n * sine_perturbation
            n0 = nbar + delta_n
        
        # Add delta p perturbation  
        if par.seed_amp_p != 0.0:
            delta_p = par.seed_amp_p * sine_perturbation
            p0 = pbar + delta_p
    
    # seed_mode == 7: cos(6πx/L) + cos(10πx/L) + cos(14πx/L) (modes m=3, 5, 7)
    elif par.seed_mode == 7 and (par.seed_amp_n != 0.0 or par.seed_amp_p != 0.0):
        # Modes: 6π/L = 3*(2π/L), 10π/L = 5*(2π/L), 14π/L = 7*(2π/L)
        # So we use Fourier modes m = 3, 5, 7
        modes = [3, 5, 7]
        
        # Create sum of cosine perturbation
        cosine_perturbation = np.zeros_like(x_local)
        for mode in modes:
            kx = 2*np.pi*mode / par.L
            cosine_perturbation += np.cos(kx * x_local)
        
        # Add delta n perturbation (no normalization - use amplitude as-is)
        if par.seed_amp_n != 0.0:
            delta_n = par.seed_amp_n * cosine_perturbation
            n0 = nbar + delta_n
        
        # Add delta p perturbation  
        if par.seed_amp_p != 0.0:
            delta_p = par.seed_amp_p * cosine_perturbation
            p0 = pbar + delta_p
    
    # Handle other seed modes (keeping original logic for compatibility)
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
    
    return n0, p0

def save_final_spectra(m, t, n_t, p_t, L, tag=""):
    meta = asdict(par).copy()
    meta['outdir'] = str(par.outdir)
    os.makedirs(par.outdir, exist_ok=True)
    
    # Create filename with floating point significant digits
    u_d_str = f"{par.u_d:.4f}".replace('.', 'p')  # e.g., 7.5000 -> 7p5000
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

def rhs_with_progress(t, y, E_base, last_print_time=[0.0], start_time=[0.0], worker_id=0):
    """RHS function with progress tracking"""
    if t - last_print_time[0] > 1.0:  # Print every 1.0 time units
        progress = (t / par.t_final) * 100
        elapsed_wall_time = time.time() - start_time[0]
        
        # Calculate estimated time remaining
        if progress > 0.1:  # Only estimate after 0.1% progress to avoid division by zero
            estimated_total_time = elapsed_wall_time / (progress / 100.0)
            estimated_remaining = estimated_total_time - elapsed_wall_time
            est_str = f"EST: {estimated_remaining:6.1f}s"
        else:
            est_str = "EST: ---s"
        
        # Use carriage return to overwrite the same line for this worker
        print(f"\r[Worker {worker_id:2d}] {progress:6.1f}% (t = {t:6.3f}/{par.t_final:.3f}) | Wall: {elapsed_wall_time:6.1f}s | {est_str}", end="", flush=True)
        last_print_time[0] = t
    return rhs(t, y, E_base)

def run_once(tag="seed_mode", worker_id=0):
    os.makedirs(par.outdir, exist_ok=True)

    n0, p0 = initial_fields()
    E_base = E_base_from_drift(nbar_profile()) if par.maintain_drift in ("field","feedback") else 0.0

    print(f"[Worker {worker_id:2d}] E_base={E_base}")
    #E_base = 15.0
    
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
    
    # Initialize timing variables
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


    #             sol = solve_ivp(lambda t,y: rhs_with_progress(t,y,E_base,last_print_time,start_time),
    #                             (0.0, par.t_final), y0, t_eval=t_eval,
    #                             method="BDF", rtol=par.rtol, atol=par.atol,
    #                             max_step=0.05, vectorized=False)#BDF
    # else:
    #     with set_workers(NTHREADS):
    #         sol = solve_ivp(lambda t,y: rhs_with_progress(t,y,E_base,last_print_time,start_time),
    #                         (0.0, par.t_final), y0, t_eval=t_eval,
    #                         method="BDF", rtol=par.rtol, atol=par.atol,
    #                         max_step=0.05, vectorized=False)#BDF
    
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
    
    # Calculate INSTANTANEOUS velocity at t=t_final using two close snapshots
    # Use the last two time points for instantaneous velocity measurement
    idx_t1 = -5  # Second-to-last snapshot
    idx_t2 = -1  # Last snapshot

    print(f"[Worker {worker_id:2d}] len=",len(sol.t), idx_t1, idx_t2)
    
    u_drift_inst, shift_opt_inst, corr_max_inst, shifts_inst, correlations_inst = calculate_velocity_from_period(
        n_t[:, idx_t1], n_t[:, idx_t2], sol.t[idx_t1], sol.t[idx_t2], par.L
    )
    
    print(f"[Worker {worker_id:2d}]  <u>(t=0)={u_momentum_initial:.4f},  <u>(t_end)={u_momentum_final:.4f},  target u_d={par.u_d:.4f}")
    print(f"[Worker {worker_id:2d}]  u_drift_instantaneous={u_drift_inst:.4f} (from shift={shift_opt_inst:.3f}, Δt={sol.t[idx_t2]-sol.t[idx_t1]:.3f})")
    print(f"[Worker {worker_id:2d}]  measured at t={sol.t[idx_t2]:.3f}, correlation_max={corr_max_inst:.4f}")
    
    # Create velocity detection plot showing instantaneous measurement
    plot_period_detection(n_t[:, idx_t1], n_t[:, idx_t2], sol.t[idx_t1], sol.t[idx_t2], 
                         par.L, u_momentum_final, par.u_d, tag=tag)

    # Create local spatial grid for plotting
    x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    dx_local = x_local[1] - x_local[0]
    
    extent=[x_local.min(), x_local.max(), sol.t.min(), sol.t.max()]
    plt.figure(figsize=(9.6,4.3))
    plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.xlabel("x"); plt.ylabel("t"); plt.title(f"n(x,t)  [lab]  {tag}")
    plt.colorbar(label="n")
    plt.plot([par.x0, par.x0], [sol.t.min(), sol.t.max()], 'w--', lw=1, alpha=0.7)
    plt.tight_layout(); plt.savefig(f"{par.outdir}/spacetime_n_lab_{tag}.png", dpi=160); 
    # plt.show()
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

    plot_fft_initial_last(n_t, sol.t, par.L, tag=tag, k_marks=())
    
    save_final_spectra(par.seed_mode, sol.t, n_t, p_t, par.L, tag=tag)

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

    modes = range(1,2)#range(1,7)
    results = []

    oldA, oldm = par.seed_amp_n, par.seed_mode

    print(f"[Multi-mode] Running {len(modes)} modes: {list(modes)}")
    
    # Start timing for multi-mode run
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
            # ax.set_ylabel(f"m={m}")
            # ax.text(
            #     -0.02, 0.5, f"m={m}",
            #     transform=ax.transAxes, rotation=90,
            #     va="center", ha="right", color="red", fontsize=11
            # )

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
    """
    Worker function to run a single u_d simulation.
    This function is designed to be called in parallel by multiprocessing.
    
    Parameters:
    -----------
    u_d : float
        Drift velocity value for this simulation
    base_params : dict
        Dictionary containing all the base parameters
    worker_id : int
        Worker ID for progress tracking
        
    Returns:
    --------
    dict : Results dictionary with u_d, success status, and paths
    """
    # Create a local copy of parameters for this worker
    import copy
    from dataclasses import dataclass
    
    # Recreate the parameter object
    local_par = P()
    for key, value in base_params.items():
        if hasattr(local_par, key):
            setattr(local_par, key, value)
    
    # Override with this specific u_d
    local_par.u_d = u_d
    u_d_str = f"{u_d:.4f}".replace('.', 'p')  # e.g., 7.5000 -> 7p5000
    local_par.outdir = f"multiple_u_d/w=0.4_modes_3_5_7_L10(lambda={local_par.lambda_diss}, sigma={local_par.sigma_diss}, seed_amp_n={local_par.seed_amp_n}, seed_amp_p={local_par.seed_amp_p})/out_drift_ud{u_d_str}"
    
    # Keep t_final fixed at 50.0 for all u_d values
    local_par.t_final = 20*10.0/u_d#50.0
    
    local_par.n_save = 1024*4#100  # Reduced for speed, as per user's settings
    
    # Keep Nx from global par (user set it to 1212)
    local_par.Nx = 512*4
    
    # Update global par for this process
    global par
    par = local_par
    
    # Update global arrays for this process
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
    # Generate u_d values for parameter sweep
    u_d_values = [2.1]#np.arange(1.4, 2.8, 0.1)
    
    print(f"[run_multiple_ud] Running parameter sweep with {len(u_d_values)} u_d values")
    print(f"[run_multiple_ud] Range: [{u_d_values[0]:.4f}, {u_d_values[-1]:.4f}]")
    print(f"[run_multiple_ud] Step size: {u_d_values[1] - u_d_values[0]:.4f}")
    print(f"[run_multiple_ud] u_d values: {u_d_values}")

    # Use the generated u_d_values with optimized spacing
    print(f"[run_multiple_ud] u_d values to simulate: {u_d_values}")
    
    # Convert current parameters to dictionary for passing to workers
    base_params = asdict(par)
    
    # Determine number of parallel workers
    n_cpus = mp.cpu_count()
    n_workers = min(len(u_d_values), max(1, n_cpus - 1))  # Leave one CPU free
    
    print(f"\n[Parallel] Using {n_workers} parallel workers (out of {n_cpus} CPUs)")
    print(f"[Parallel] Running {len(u_d_values)} simulations in parallel")
    print(f"[Parallel] Progress will be shown for each worker simultaneously")
    print(f"[Parallel] Each worker will update its progress line independently")
    
    overall_start_time = time.time()
    
    # Run simulations in parallel with worker IDs
    with mp.Pool(processes=n_workers) as pool:
        # Create worker function with base_params and assign worker IDs
        worker_args = [(u_d, base_params, i) for i, u_d in enumerate(u_d_values)]
        results = pool.starmap(run_single_ud_worker, worker_args)
    
    overall_elapsed = time.time() - overall_start_time
    
    # Print summary
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

if __name__ == "__main__":
    # run_all_modes_snapshots(tag="seed_modes_1to5")
    
    # Example: Test localized dissipation perturbation
    # To test increased dissipation at x=x0:
    par.lambda_diss = 0.0      # Positive = increased damping
    par.sigma_diss =  -1.0       # Width of perturbation
    par.x0 = 12.5               # Location of perturbation
    
    # Set seed_mode to 7 for cos(6πx/L) + cos(10πx/L) + cos(14πx/L) perturbations
    par.seed_mode = 7
    par.seed_amp_n = 0.001  # Small amplitude perturbation
    par.seed_amp_p = 0.001  # Small amplitude perturbation (can set to 0 if only perturbing n)
    
    # par.Nx = 10000  # High spatial resolution
    par.L = 10.0    # System size
    par.t_final = 50.0  # Final time
    # par.max_step = 0.0005  # Maximum time step


    # Ensure NO potential perturbations are active for seed_mode == 2
    par.use_nbar_gaussian = False
    par.nbar_amp = 0.0
    par.include_poisson = False
    par.lambda_gauss = 0.0      # Disable time-independent Gaussian perturbation
    par.sigma_gauss = 2.0       # Width of Gaussian (not used when lambda_gauss=0)
    par.x0_gauss = 12.5          # Center at x=12.5 (not used when lambda_gauss=0)
    
    #   run_once(tag="increased_dissipation")
    run_multiple_ud()

    #
    # To test decreased dissipation (can drive instabilities):
    #   par.lambda_diss = -0.5     # Negative = decreased damping
    #   par.sigma_diss = 2.0
    #   run_once(tag="decreased_dissipation")