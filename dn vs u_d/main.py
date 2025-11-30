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
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 14,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
    "axes.labelsize": 18,           # axis label text
    "xtick.labelsize": 16,          # x-tick labels
    "ytick.labelsize": 16,          # y-tick labels
})

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
    w: float = 0.04
    include_poisson: bool = False
    eps: float = 20.0

    u_d: float = 5.245
    # u_d: float = .0
    maintain_drift: str = "field"
    Kp: float = 0.15

    Dn: float = 0.1#/10#0.03
    Dp: float = 0.1

    J0: float = 1.0#0.04
    sigma_J: float = 2.0**1/2#6.0
    x0: float = 10
    source_model: str = "as_given"
    
    # Localized dissipation perturbation parameters
    lambda_diss: float = 0.0  # Amplitude of dissipation perturbation (can be positive or negative)
    sigma_diss: float = 2.0   # Width of dissipation perturbation
    
    # Time-independent Gaussian density perturbation
    lambda_gauss: float = 0.0  # Amplitude of Gaussian density perturbation
    sigma_gauss: float = 2.0   # Width of Gaussian density perturbation
    x0_gauss: float = 10      # Center of Gaussian density perturbation

    use_nbar_gaussian: bool = False
    nbar_amp: float = 0.0
    nbar_sigma: float = 120.0

    L: float = 10.0  # System size
    Nx: int = 1212#2048#1218#1512#2524#1024
    t_final: float = 50.0
    n_save: int = 100  #200#200  # Reduced for speed
    # rtol: float = 5e-7
    # atol: float = 5e-9
    rtol = 3e-2  # Tighter tolerance for accuracy
    atol = 1e-4  # Tighter tolerance for accuracy
    n_floor: float = 1e-4
    dealias_23: bool = True

    # Boundary condition type
    # "periodic" (default) - old behavior with FFT-based periodic derivatives
    # "ds_open" - Dyakonov-Shur-like open boundaries (soft reservoir layer at contacts)
    bc_type: str = "periodic"  # Default to periodic for stability
    
    # DS/open boundary parameters (soft reservoir layer)
    N_bc: int = 8                 # number of grid cells in each contact region
    tau_bc_n: float = 0.5         # relaxation time for n at contacts
    tau_bc_p: float = 0.5         # relaxation time for p at contacts
    use_hard_bc_clamp: bool = False  # If True: hybrid (soft relaxation + hard Dirichlet at edges)
                                     # If False: purely soft (relaxation only, no hard clamping)

    seed_amp_n: float = 0.030  # Small amplitude perturbation
    seed_mode: int = 7  # New mode for cos(6πx/L) + cos(10πx/L) + cos(14πx/L)
    seed_amp_p: float = 0.030  # Small amplitude perturbation

    outdir: str = "out_drift/small_dissipation_perturbation"
    cmap: str = "inferno"

class SimpleSolution:
    """Minimal stand-in for solve_ivp's return object."""
    def __init__(self, t, y, success=True, message=""):
        self.t = t
        self.y = y
        self.success = success
        self.message = message


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
    # Use periodic grid (endpoint=False) for compatibility with both spectral and FD derivatives
    # For ds_open BC, finite differences use one-sided stencils at boundaries (no wrapping)
    # For periodic BC, spectral derivatives use FFT (wraps around)
    x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    dx = x[1] - x[0]
    k = 2*np.pi*fftfreq(par.Nx, d=dx)
    ik = 1j*k
    k2 = k**2
    _kc = par.Nx//3
    _nz_mask = (k2 != 0)

def Dx_fd(f):
    """
    First derivative with simple 2nd-order finite differences
    on a non-periodic interval [0, L). At the boundaries we use
    one-sided first-order approximations.
    """
    global x, dx
    if x is None or len(x) != len(f):
        _update_global_arrays()
    h = dx
    out = np.empty_like(f)

    # interior: central differences
    out[1:-1] = (f[2:] - f[:-2]) / (2.0 * h)

    # left boundary (x ≈ 0): forward difference
    out[0] = (f[1] - f[0]) / h

    # right boundary (x ≈ L): backward difference
    out[-1] = (f[-1] - f[-2]) / h

    return out

def Dxx_fd(f):
    """
    Second derivative with 2nd-order finite differences.
    At boundaries we use one-sided 2nd-order approximations.
    """
    global x, dx
    if x is None or len(x) != len(f):
        _update_global_arrays()
    h = dx
    out = np.empty_like(f)

    # interior: central 2nd derivative
    out[1:-1] = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / (h * h)

    # left boundary: 2nd-order one-sided (uses points 0,1,2)
    out[0] = (f[2] - 2.0 * f[1] + f[0]) / (h * h)

    # right boundary: symmetric at the right end (N-3,N-2,N-1)
    out[-1] = (f[-1] - 2.0 * f[-2] + f[-3]) / (h * h)

    return out

def Dx(f):  
    return Dx_fd(f)
    """First derivative using spectral (FFT) method - always periodic"""
    global k, ik
    if k is None or len(k) != len(f):
        _update_global_arrays()
    return (ifft(ik * fft(f, workers=NTHREADS), workers=NTHREADS)).real

def Dxx(f): 
    return Dxx_fd(f)
    """Second derivative using spectral (FFT) method - always periodic"""
    global k2
    if k2 is None or len(k2) != len(f):
        _update_global_arrays()
    return (ifft((-k2) * fft(f, workers=NTHREADS), workers=NTHREADS)).real

def Dx_phys(f):
    """Physical first derivative: spectral for periodic, FD for ds_open."""
    if par.bc_type == "ds_open":
        return Dx_fd(f)
    else:
        return Dx(f)

def Dxx_phys(f):
    """Physical second derivative: spectral for periodic, FD for ds_open."""
    if par.bc_type == "ds_open":
        return Dxx_fd(f)
    else:
        return Dxx(f)

def filter_23(f):
    """Spectral dealiasing filter - only applied for periodic BC"""
    if not par.dealias_23: 
        return f
    # Only apply dealiasing for periodic BC (DS/open BC skip it)
    if par.bc_type != "periodic":
        return f
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
    return par.m * par.u_d * np.mean(Gamma(nbar)) / par.e

def apply_ds_open_relaxation(n, p, dn_dt, dp_dt):
    """
    DS/open boundaries via soft reservoir layers:

    - Left (source, x≈0):     n -> nbar0  (density contact)
    - Right (drain, x≈L):     p -> m nbar0 u_d  (current/momentum contact)

    Implemented as relaxation in a thin layer of N_bc cells.
    Optional tapering makes the transition smoother.
    
    This is the "soft" part of the BC. If par.use_hard_bc_clamp=True,
    this works together with enforce_ds_open_bc_state() for a hybrid scheme.
    If par.use_hard_bc_clamp=False, this is the only BC enforcement (purely soft).
    """
    if par.bc_type != "ds_open":
        return dn_dt, dp_dt

    N = len(n)
    N_bc = min(par.N_bc, N // 4)  # safety

    left_mask  = np.zeros(N)
    right_mask = np.zeros(N)
    left_mask[:N_bc]  = 1.0
    right_mask[-N_bc:] = 1.0

    # Optional taper – disabled for a stronger, flat reservoir
    # Uncomment below if you want smoother transitions
    # if N_bc >= 2:
    #     w = np.linspace(1.0, 0.0, N_bc)
    #     left_mask[:N_bc]  *= w
    #     right_mask[-N_bc:] *= w[::-1]

    n_eq_source  = par.nbar0
    p_eq_drain   = par.m * par.nbar0 * par.u_d

    gamma_n = 1.0 / par.tau_bc_n
    gamma_p = 1.0 / par.tau_bc_p

    # Source: pin density, leave momentum mostly free
    dn_dt += gamma_n * left_mask * (n_eq_source - n)
    # (you *can* add a tiny p relaxation here if you want extra damping)

    # Drain: pin momentum/current, leave density mostly free
    dp_dt += gamma_p * right_mask * (p_eq_drain - p)

    return dn_dt, dp_dt

def enforce_ds_open_bc_state(n, p):
    """
    Hard Dirichlet boundary conditions for DS/open case.
    This is applied directly to the *state* arrays (n, p),
    typically after each explicit RK step.

    - Source (x ≈ 0): fix density to nbar0
    - Drain (x ≈ L): fix momentum/current to m * nbar0 * u_d
    
    Can be disabled by setting par.use_hard_bc_clamp = False for purely soft BCs.
    """
    if par.bc_type != "ds_open":
        return
    
    if not par.use_hard_bc_clamp:
        return  # Purely soft BC: no hard clamping

    # Left: density contact
    n[0] = par.nbar0

    # Right: momentum/current contact
    p[-1] = par.m * par.nbar0 * par.u_d

def rhs(t, y, E_base):
    N = par.Nx
    n = y[:N]
    p = y[N:]

    # Background density profile (uniform for seed_mode=7)
    nbar = nbar_profile()

    # Floor to avoid division by tiny densities
    n_eff = np.maximum(n, par.n_floor)

    # Mean drift velocity for feedback field control
    v = p / (par.m * n_eff)
    u_mean = float(np.mean(v))
    if par.maintain_drift == "feedback":
        E_eff = E_base + par.Kp * (par.u_d - u_mean)
    else:
        E_eff = E_base

    # --- continuity: uses BC-aware derivatives ---
    dn_dt = -Dx_phys(p) + par.Dn * Dxx_phys(n)

    # pressure & momentum
    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)
    grad_Pi = Dx_phys(Pi)

    force_Phi = 0.0
    if par.include_poisson:
        phi = phi_from_n(n_eff, nbar)
        force_Phi = n_eff * Dx_phys(phi)

    dp_dt = -Gamma_spatial(n_eff)*p - grad_Pi + par.e*n_eff*E_eff - force_Phi + par.Dp * Dxx_phys(p)


    # Apply DS/open reservoir relaxation in contact layers (no-op for periodic)
    dn_dt, dp_dt = apply_ds_open_relaxation(n, p, dn_dt, dp_dt)

    # Dealias only in purely periodic spectral case
    if par.bc_type == "periodic":
        dn_dt = filter_23(dn_dt)
        dp_dt = filter_23(dp_dt)

    return np.concatenate([dn_dt, dp_dt])


def initial_fields():
    # Create spatial grid with current Nx
    # Always use periodic grid (endpoint=False) - DS/open boundaries handled via soft relaxation
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
    
    # Enforce DS-like boundary conditions in initial fields
    # Only apply hard clamping if enabled (for hybrid BC mode)
    if par.bc_type != "periodic" and par.use_hard_bc_clamp:
        # Source: fix density at x≈0
        n0[0] = par.nbar0
        # Drain: fix current j(L,t) = j0 -> p(L,t) = m * nbar0 * u_d
        p0[-1] = par.m * par.nbar0 * par.u_d
    
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

def integrate_explicit_ds_open(y0, t_span, t_eval, E_base, worker_id=0):
    """
    Very conservative explicit RK4 time stepper for ds_open runs.

    - Uses rhs() which already applies DS/open relaxation and FD derivatives
      (via Dx_phys / Dxx_phys once you've patched rhs as above).
    - Enforces a strong CFL limit for both diffusion and advection.
    - Enforces floors and soft caps on n and p to avoid numerical overflow.
    """
    t0, t_final = t_span
    _update_global_arrays()
    N = par.Nx

    x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
    dx_local = x_local[1] - x_local[0]

    # --- CFL-based base step ---
    D_max = max(par.Dn, par.Dp, 1e-12)

    # very rough characteristic speed: drift + "sound"
    c0 = np.sqrt(max(par.U * par.nbar0, 0.0) / max(par.m, 1e-12))
    u_char = abs(par.u_d) + c0

    CFL_diff = 0.20
    CFL_adv  = 0.40

    dt_diff = CFL_diff * dx_local * dx_local / D_max
    dt_adv  = CFL_adv  * dx_local / max(u_char, 1e-6)

    # also enforce at least ~1e4 substeps over the full window
    base_dt = min(dt_diff, dt_adv, (t_final - t0) / 1e4)
    # strong extra safety – this is what really kills the blow-up
    dt_sub  = 0.25 * base_dt

    print(f"[Worker {worker_id:2d}] Explicit RK4 for ds_open: "
          f"dt_sub={dt_sub:.3e}, dt_diff={dt_diff:.3e}, dt_adv={dt_adv:.3e}")

    # --- output arrays ---
    t_eval = np.asarray(t_eval)
    n_out = t_eval.size
    Y = np.empty((2 * N, n_out))
    T = t_eval.copy()

    y = y0.copy()
    t = t0
    out_idx = 0

    # store initial condition
    Y[:, out_idx] = y
    out_idx += 1

    # soft caps (tune if you like)
    n_cap = getattr(par, "n_cap", 5.0 * par.nbar0)
    # 5× the uniform momentum, in absolute value
    p_cap = getattr(par, "p_cap", 5.0 * par.m * par.nbar0 * abs(par.u_d) + 1e-6)

    last_print_time = [t0]
    start_wall = [time.time()]

    while t < t_final - 1e-14:
        # do not step past final time or next output sample
        dt = dt_sub
        if out_idx < n_out:
            dt = min(dt, T[out_idx] - t)
        if t + dt > t_final:
            dt = t_final - t
        if dt <= 0.0:
            break

        # 4-stage RK4, with progress printing via rhs_with_progress
        k1 = rhs_with_progress(t,           y,                E_base, last_print_time, start_wall, worker_id)
        k2 = rhs_with_progress(t + 0.5*dt, y + 0.5*dt*k1,    E_base, last_print_time, start_wall, worker_id)
        k3 = rhs_with_progress(t + 0.5*dt, y + 0.5*dt*k2,    E_base, last_print_time, start_wall, worker_id)
        k4 = rhs_with_progress(t + dt,     y + dt*k3,        E_base, last_print_time, start_wall, worker_id)

        y += (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t += dt

        # enforce floors & caps to prevent overflows / NaNs
        n = y[:N]
        p = y[N:]

        # floor density
        np.maximum(n, par.n_floor, out=n)
        # soft ceiling on density
        np.clip(n, par.n_floor, n_cap, out=n)
        # soft ceiling on momentum
        np.clip(p, -p_cap, p_cap, out=p)

        # Hard DS/open Dirichlet BC on the *state*
        enforce_ds_open_bc_state(n, p)

        # write back into y (n and p are views, so this updates y)
        y[:N] = n
        y[N:] = p

        # write output if we just hit / passed an output time
        while out_idx < n_out and t >= T[out_idx] - 1e-12:
            Y[:, out_idx] = y
            out_idx += 1

    # If for numerical reasons we didn't hit all t_eval points exactly,
    # just fill the remainder with the last available state.
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
    
    # BC mode diagnostics
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

    # Validate initial conditions
    if np.any(np.isnan(y0)) or np.any(np.isinf(y0)):
        print(f"[Worker {worker_id:2d}] WARNING: Initial conditions contain NaN or Inf values!")
        print(f"[Worker {worker_id:2d}]   n0: NaN={np.any(np.isnan(n0))}, Inf={np.any(np.isinf(n0))}")
        print(f"[Worker {worker_id:2d}]   p0: NaN={np.any(np.isnan(p0))}, Inf={np.any(np.isinf(p0))}")
    if np.any(n0 < 0):
        print(f"[Worker {worker_id:2d}] WARNING: Initial density contains negative values "
              f"(min={np.min(n0):.2e})")

    # Test RHS at t=0 to catch early NaNs/Infs
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

    # ------------------------------------------------------------------
    # Solver selection
    # ------------------------------------------------------------------
    if par.bc_type == "ds_open":
        # Use homebrew explicit RK4 integrator (no step-size underflow)
        print(f"[Worker {worker_id:2d}] Using explicit RK4 integrator for DS/open BC")
        sol = integrate_explicit_ds_open(y0, (0.0, par.t_final), t_eval, E_base, worker_id)
        total_wall_time = time.time() - start_wall_time
        success = sol.success
        sol_message = sol.message
        t = sol.t
        Y = sol.y
    else:
        # Keep SciPy's BDF for periodic case
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

    # ------------------------------------------------------------------
    # Post-solve diagnostics and plotting
    # ------------------------------------------------------------------
    print(f"[Worker {worker_id:2d}] Solver success: {success}")
    print(f"[Worker {worker_id:2d}] Final time: {t[-1]:.3f}, target: {par.t_final:.3f}")
    print(f"[Worker {worker_id:2d}] Total wall time: {total_wall_time:.2f} seconds")
    if not success and par.bc_type != "ds_open":
        print(f"[Worker {worker_id:2d}] Solver failed with message: {sol_message}")
    elif par.bc_type == "ds_open":
        print(f"[Worker {worker_id:2d}] Solver info: {sol_message}")

    # Extract fields
    N = par.Nx
    n_t = Y[:N, :]
    p_t = Y[N:, :]

    # Basic velocity diagnostics
    n_eff_t = np.maximum(n_t, par.n_floor)
    v_t = p_t / (par.m * n_eff_t)
    u_momentum_initial = float(np.mean(v_t[:, 0]))
    u_momentum_final   = float(np.mean(v_t[:, -1]))
    print(f"[Worker {worker_id:2d}]  <u>(t=0)={u_momentum_initial:.4f},  "
          f"<u>(t_end)={u_momentum_final:.4f},  target u_d={par.u_d:.4f}")

    # BC diagnostics
    if par.bc_type == "ds_open":
        print(f"[Worker {worker_id:2d}]  BC values at t_end: "
              f"n[0]={n_t[0, -1]:.4f} (target={par.nbar0:.4f}), "
              f"p[-1]={p_t[-1, -1]:.4f} (target={par.m*par.nbar0*par.u_d:.4f})")

    # --- Instantaneous drift velocity (if enough snapshots) ---
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

        # Diagnostic plot
        plot_period_detection(
            n_t[:, idx_t1], n_t[:, idx_t2],
            t[idx_t1], t[idx_t2],
            par.L, u_momentum_final, par.u_d, tag=tag
        )
    else:
        print(f"[Worker {worker_id:2d}] Not enough time points ({len(t)}) "
              f"for instantaneous velocity measurement; skipping.")

    # --- Spacetime plots (if we have more than one time slice) ---
    if len(t) > 1:
        x_local = np.linspace(0.0, par.L, par.Nx, endpoint=False)
        dx_local = x_local[1] - x_local[0]
        extent = [x_local.min(), x_local.max(), t.min(), t.max()]

        # Lab frame
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

        # Co-moving frame
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

        # Snapshots at 0%, 20%, 40%, 60%, 80%, 100%
        plt.figure(figsize=(9.6, 5.0))
        percentages = [0, 20, 40, 60, 80, 100]
        # Modern color palette (using tab10 colormap)
        colors = plt.cm.tab10(np.linspace(0, 1, len(percentages)))
        for i, pct in enumerate(percentages):
            idx = int((pct / 100) * (len(t) - 1))
            idx = min(idx, len(t) - 1)  # Ensure we don't go out of bounds
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

    # FFT comparison
    try:
        plot_fft_initial_last(n_t, t, par.L, tag=tag, k_marks=())
    except Exception as e:
        print(f"[Worker {worker_id:2d}] Warning: FFT plot failed with error: {e}")

    # Save full time series
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
    local_par.outdir = f"multiple_u_d/w=0.07_modes_3_5_7_L10(lambda={local_par.lambda_diss}, sigma={local_par.sigma_diss}, seed_amp_n={local_par.seed_amp_n}, seed_amp_p={local_par.seed_amp_p})/out_drift_ud{u_d_str}"
    
    # Keep t_final fixed at 50.0 for all u_d values
    local_par.t_final = 20*10.0/u_d#50.0
    #<=1.4 -- 20 periods
    local_par.n_save = 512#1024#100  # Reduced for speed, as per user's settings
    
    # Keep Nx from global par (user set it to 1212)
    local_par.Nx = 512
    
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

def run_single_w_worker(w, u_d, base_params, worker_id=0):
    """
    Worker function to run a single w and u_d simulation.
    
    Parameters:
    -----------
    w : float
        Width parameter value for this simulation
    u_d : float
        Drift velocity value for this simulation
    base_params : dict
        Dictionary containing all the base parameters
    worker_id : int
        Worker ID for progress tracking
        
    Returns:
    --------
    dict : Results dictionary with w, u_d, success status, and paths
    """
    # Create a local copy of parameters for this worker
    import copy
    from dataclasses import dataclass
    
    # Recreate the parameter object
    local_par = P()
    for key, value in base_params.items():
        if hasattr(local_par, key):
            setattr(local_par, key, value)
    
    # Override with this specific w and u_d
    local_par.w = w
    local_par.u_d = u_d
    w_str = f"{w:.2f}"  # e.g., 0.05 -> 0.05
    u_d_str = f"{u_d:.1f}".replace('.', 'p')  # e.g., 1.2 -> 1p2
    local_par.outdir = f"multiple_u_d/multiple_w/w={w_str}_modes_3_5_7_L10(lambda={local_par.lambda_diss}, sigma={local_par.sigma_diss}, seed_amp_n={local_par.seed_amp_n}, seed_amp_p={local_par.seed_amp_p})/out_drift_ud{u_d_str}"
    
    # Keep t_final fixed
    local_par.t_final = 20*10.0/u_d
    local_par.n_save = 512
    local_par.Nx = 512
    
    # Update global par for this process
    global par
    par = local_par
    
    # Update global arrays for this process
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
    """Run simulations for multiple w and u_d values."""
    # Generate w values for parameter sweep
    w_values = np.arange(0.01, 0.16, 0.01)  # 0.01, 0.02, ..., 0.15
    # Generate u_d values for parameter sweep
    u_d_values = np.arange(0.1, 2.0, 0.1)   # 0.1, 0.2, ..., 1.9
    
    print(f"[run_multiple_w] Running parameter sweep with {len(w_values)} w values and {len(u_d_values)} u_d values")
    print(f"[run_multiple_w] w range: [{w_values[0]:.2f}, {w_values[-1]:.2f}]")
    print(f"[run_multiple_w] u_d range: [{u_d_values[0]:.1f}, {u_d_values[-1]:.1f}]")
    print(f"[run_multiple_w] Total combinations: {len(w_values) * len(u_d_values)}")

    # Convert current parameters to dictionary for passing to workers
    base_params = asdict(par)
    
    # Create all combinations of w and u_d
    combinations = [(w, u_d) for w in w_values for u_d in u_d_values]
    
    # Determine number of parallel workers
    n_cpus = mp.cpu_count()
    n_workers = min(len(combinations), max(1, n_cpus - 1))  # Leave one CPU free
    
    print(f"\n[Parallel] Using {n_workers} parallel workers (out of {n_cpus} CPUs)")
    print(f"[Parallel] Running {len(combinations)} simulations in parallel")
    print(f"[Parallel] Progress will be shown for each worker simultaneously")
    print(f"[Parallel] Each worker will update its progress line independently")
    
    overall_start_time = time.time()
    
    # Run simulations in parallel with worker IDs
    with mp.Pool(processes=n_workers) as pool:
        # Create worker function with base_params and assign worker IDs
        worker_args = [(w, u_d, base_params, i) for i, (w, u_d) in enumerate(combinations)]
        results = pool.starmap(run_single_w_worker, worker_args)
    
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
        print(f"\nSuccessful simulations (first 10):")
        for r in successful[:10]:  # Show first 10 to avoid too much output
            print(f"  w={r['w']:.2f}, u_d={r['u_d']:.1f}: {r['elapsed_time']:.1f}s, t_final={r['final_time']:.3f}")
        if len(successful) > 10:
            print(f"  ... and {len(successful) - 10} more")
    
    if failed:
        print(f"\nFailed simulations:")
        for r in failed:
            print(f"  w={r['w']:.2f}, u_d={r['u_d']:.1f}: {r.get('error', 'Unknown error')}")
    
    print(f"{'='*60}")

def run_single_diffusion_worker(w, u_d, Dn, Dp, base_params, worker_id=0):
    """
    Worker function to run a single w, u_d, Dn, Dp simulation.
    
    Parameters:
    -----------
    w : float
        Width parameter value for this simulation
    u_d : float
        Drift velocity value for this simulation
    Dn : float
        Density diffusion coefficient
    Dp : float
        Momentum diffusion coefficient
    base_params : dict
        Dictionary containing all the base parameters
    worker_id : int
        Worker ID for progress tracking
        
    Returns:
    --------
    dict : Results dictionary with w, u_d, Dn, Dp, success status, and paths
    """
    # Create a local copy of parameters for this worker
    import copy
    from dataclasses import dataclass
    
    # Recreate the parameter object
    local_par = P()
    for key, value in base_params.items():
        if hasattr(local_par, key):
            setattr(local_par, key, value)
    
    # Override with this specific w, u_d, Dn, Dp
    local_par.w = w
    local_par.u_d = u_d
    local_par.Dn = Dn
    local_par.Dp = Dp
    
    w_str = f"{w:.2f}"  # e.g., 0.05 -> 0.05
    u_d_str = f"{u_d:.1f}".replace('.', 'p')  # e.g., 1.2 -> 1p2
    Dn_str = f"{Dn:.2f}".replace('.', 'p')  # e.g., 0.25 -> 0p25
    Dp_str = f"{Dp:.2f}".replace('.', 'p')  # e.g., 0.05 -> 0p05
    
    local_par.outdir = f"multiple_u_d/non-periodic/w={w_str}_Dn={Dn_str}_Dp={Dp_str}_L10(lambda={local_par.lambda_diss}, sigma={local_par.sigma_diss}, seed_amp_n={local_par.seed_amp_n}, seed_amp_p={local_par.seed_amp_p})/out_drift_ud{u_d_str}"
    
    # Keep t_final fixed
    local_par.t_final = 20*10.0/u_d
    local_par.n_save = 512
    local_par.Nx = 512
    
    # Update global par for this process
    global par
    par = local_par
    
    # Update global arrays for this process
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
    """Run simulations for different diffusion parameter combinations."""
    # Generate w values for parameter sweep: 0.01 to 0.25 with 0.01 step
    w_values = np.arange(0.01, 0.26, 0.01)  # 0.01, 0.02, ..., 0.25
    
    # Generate u_d values for parameter sweep
    u_d_values = np.arange(0.1, 2.0, 0.1)   # 0.1, 0.2, ..., 1.9
    
    # Define diffusion parameter combinations
    # Base values from par: Dn = 0.5, Dp = 0.1
    base_Dn = 0.5
    base_Dp = 0.1
    
    diffusion_combinations = [
        # 1. Dn reduced two times, Dp the same
        {"name": "Dn_half", "Dn": base_Dn / 2, "Dp": base_Dp},
        # 2. Dp reduced two times, Dn the same  
        {"name": "Dp_half", "Dn": base_Dn, "Dp": base_Dp / 2},
        # 3. Both Dp and Dn reduced two times
        {"name": "both_half", "Dn": base_Dn / 2, "Dp": base_Dp / 2},
        # 4. Dn increased two times, Dp the same
        {"name": "Dn_double", "Dn": base_Dn * 2, "Dp": base_Dp},
        # 5. Dp increased two times, Dn the same
        {"name": "Dp_double", "Dn": base_Dn, "Dp": base_Dp * 2},
        # 6. Both Dp and Dn increased two times
        {"name": "both_double", "Dn": base_Dn * 2, "Dp": base_Dp * 2},
    ]
    
    print(f"[run_diffusion_parameter_sweep] Running parameter sweep:")
    print(f"  w values: {len(w_values)} from {w_values[0]:.2f} to {w_values[-1]:.2f}")
    print(f"  u_d values: {len(u_d_values)} from {u_d_values[0]:.1f} to {u_d_values[-1]:.1f}")
    print(f"  Diffusion combinations: {len(diffusion_combinations)}")
    print(f"  Total simulations: {len(w_values) * len(u_d_values) * len(diffusion_combinations)}")
    
    # Convert current parameters to dictionary for passing to workers
    base_params = asdict(par)
    
    # Create all combinations
    all_combinations = []
    for diff_combo in diffusion_combinations:
        for w in w_values:
            for u_d in u_d_values:
                all_combinations.append((w, u_d, diff_combo["Dn"], diff_combo["Dp"], diff_combo["name"]))
    
    # Determine number of parallel workers
    n_cpus = mp.cpu_count()
    n_workers = min(len(all_combinations), max(1, n_cpus - 1))  # Leave one CPU free
    
    print(f"\n[Parallel] Using {n_workers} parallel workers (out of {n_cpus} CPUs)")
    print(f"[Parallel] Running {len(all_combinations)} simulations in parallel")
    print(f"[Parallel] Progress will be shown for each worker simultaneously")
    print(f"[Parallel] Each worker will update its progress line independently")
    
    overall_start_time = time.time()
    
    # Run simulations in parallel with worker IDs
    with mp.Pool(processes=n_workers) as pool:
        # Create worker function with base_params and assign worker IDs
        worker_args = [(w, u_d, Dn, Dp, base_params, i) for i, (w, u_d, Dn, Dp, name) in enumerate(all_combinations)]
        results = pool.starmap(run_single_diffusion_worker, worker_args)
    
    overall_elapsed = time.time() - overall_start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"ALL DIFFUSION PARAMETER SWEEP SIMULATIONS COMPLETED!")
    print(f"{'='*80}")
    print(f"Total wall time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
    print(f"\nSummary:")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"  Successful: {len(successful)}/{len(results)}")
    print(f"  Failed: {len(failed)}/{len(results)}")
    
    # Group results by diffusion combination
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
        for r in successful[:10]:  # Show first 10 to avoid too much output
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
    # Generate u_d values for parameter sweep
    u_d_values = np.arange(0.2, 2.0, 0.1)
    
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

def detect_missing_simulations(base_dir="multiple_u_d/non-periodic"):
    """
    Detect missing simulations in the diffusion sweep directory.
    
    Returns:
    --------
    dict : Dictionary containing lists of missing folders, missing out dirs, and missing npz files
    """
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
    """
    Check for missing simulations and run only the missing ones.
    """
    print("="*80)
    print("CHECKING FOR MISSING SIMULATIONS")
    print("="*80)
    
    # Detect missing simulations
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
    
    # Convert current parameters to dictionary for passing to workers
    base_params = asdict(par)
    
    # Create missing simulation tasks
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
                # Check if this simulation is missing
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
    
    # Determine number of parallel workers
    n_cpus = mp.cpu_count()
    n_workers = min(len(missing_tasks), max(1, n_cpus - 1))
    
    print(f"Using {n_workers} parallel workers for {len(missing_tasks)} missing simulations")
    
    overall_start_time = time.time()
    
    # Run missing simulations in parallel
    with mp.Pool(processes=n_workers) as pool:
        worker_args = [(w, u_d, Dn, Dp, base_params, i) for i, (w, u_d, Dn, Dp) in enumerate(missing_tasks)]
        results = pool.starmap(run_single_diffusion_worker, worker_args)
    
    overall_elapsed = time.time() - overall_start_time
    
    # Print summary
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
    """
    Run single simulation for Dn_half configuration with w=0.14 and u_d from 0.1 to 2.0.
    """
    print("="*80)
    print("RUNNING SINGLE Dn_half SIMULATION")
    print("="*80)
    print("Configuration: Dn_half (Dn=0.25, Dp=0.10)")
    print("w = 0.14")
    print("u_d range: 0.1 to 2.0 with 0.1 step")
    
    # Set Dn_half parameters
    par.Dn = 0.1
    par.Dp = 0.10
    par.w = 0.14
    
    # Generate u_d values
    u_d_values = np.arange(0.1, 2.1, 0.1)  # 0.1, 0.2, ..., 2.0
    
    print(f"Total simulations to run: {len(u_d_values)}")
    
    # Convert current parameters to dictionary for passing to workers
    base_params = asdict(par)
    
    # Create simulation tasks
    simulation_tasks = []
    for u_d in u_d_values:
        simulation_tasks.append((par.w, u_d, par.Dn, par.Dp))
    
    # Determine number of parallel workers
    n_cpus = mp.cpu_count()
    n_workers = min(len(simulation_tasks), max(1, n_cpus - 1))
    
    print(f"Using {n_workers} parallel workers for {len(simulation_tasks)} simulations")
    
    overall_start_time = time.time()
    
    # Run simulations in parallel
    with mp.Pool(processes=n_workers) as pool:
        worker_args = [(w, u_d, Dn, Dp, base_params, i) for i, (w, u_d, Dn, Dp) in enumerate(simulation_tasks)]
        results = pool.starmap(run_single_diffusion_worker, worker_args)
    
    overall_elapsed = time.time() - overall_start_time
    
    # Print summary
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
    # --- DS-open, clearly unstable but numerically safe ---

    par.bc_type = "ds_open"

    # Geometry
    par.L  = 10.0
    par.Nx = 512*4        # this is what you used for ds_open_unstable_M07

    # Drift & damping
    par.u_d    = 0.7
    par.m      = 1.0
    par.e      = 1.0
    par.nbar0  = 0.2

    par.Gamma0 = 2.5
    par.w      = 0.04   # this matches the 0.01637... Gamma(n̄) you saw

    # Pressure / interaction
    par.U      = 1.0

    # Diffusion (a bit more smoothing for FD ds_open)
    par.Dn = 0.001
    par.Dp = 0.001

    # No extra spatial inhomogeneities
    par.include_poisson   = False
    par.lambda_diss       = 0.0
    par.lambda_gauss      = 0.0
    par.use_nbar_gaussian = False

    # Seeding: unstable but not crazy
    par.seed_mode  = 7           # your cos(6πx/L) + cos(10πx/L) + cos(14πx/L)
    par.seed_amp_n = 0.01
    par.seed_amp_p = 0.01

    # Time
    par.t_final = 30.0          # known to work
    par.n_save  = 600            # ~ every 0.5 time units

    # Solver tolerances – moderate (base; DS-open branch will relax further if needed)
    par.rtol = 1e-3
    par.atol = 1e-6

    # DS/open reservoir layer: gentle but not too slow
    par.N_bc     = 8         # 8 cells at each end
    par.tau_bc_n = 0.5       # relax n on timescale ~0.5
    par.tau_bc_p = 0.5       # relax p on timescale ~0.5

    # Drift maintenance: use feedback to keep <u> ≈ u_d
    par.maintain_drift = "feedback"
    par.Kp = 0.15   # feedback gain (tweak if needed)

    par.outdir = "out_ds_open_unstable_M07"

    _update_global_arrays()

    print(f"[Main] Running DS-open UNSTABLE test: Nx={par.Nx}, t_final={par.t_final}")
    print(f"[Main] BC parameters: N_bc={par.N_bc}, tau_bc_n={par.tau_bc_n}, tau_bc_p={par.tau_bc_p}")
    t, n_t, p_t = run_once(tag="ds_open_unstable_M07", worker_id=0)

    # BC sanity check
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
