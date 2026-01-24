#!/usr/bin/env python3
"""
Plot generalized decay rates gamma_m(T) for angular harmonics.

We use the generalized eigenproblem:
    (-M) eta = gamma * W eta,   W = diag(f(1-f))
and approximate gamma_m by the generalized Rayleigh quotient for trial functions:
    eta_m ~ cos(m theta), sin(m theta)

This script supports:
- active-only matrices saved as CSR (shape Nactive x Nactive)
- legacy full matrices saved dense (will be restricted to active)

Outputs:
- Scaled plot: (gamma_m - gamma_0)/T^2  vs T  (log-log)
"""

import os
import pickle
import csv
import numpy as np
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from scipy.sparse import csr_matrix, isspmatrix_csr, diags
from scipy.sparse.linalg import eigsh, LinearOperator

# Optional numba for performance
USE_NUMBA = False
try:
    import numba
    from numba import njit, prange
    USE_NUMBA = True
    # Configure numba to use all threads (can be overridden by NUMBA_NUM_THREADS env var)
    numba_threads = int(os.environ.get('NUMBA_NUM_THREADS', multiprocessing.cpu_count() or 4))
    if hasattr(numba, 'set_num_threads'):
        numba.set_num_threads(numba_threads)
    print(f"[Numba] Using {numba_threads} threads for parallel execution")
except ImportError:
    USE_NUMBA = False
    print("[Numba] Not available - install numba for better performance")

# Number of parallel workers for temperature processing
N_WORKERS = int(os.environ.get('N_WORKERS', multiprocessing.cpu_count() or 4))
print(f"[Parallel] Using {N_WORKERS} workers for temperature processing")

# --- plot style (keep your choices) ---
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20

IN_DIR = "Matrixes_bruteforce"
Thetas_req = np.geomspace(0.0025, 1.28, 30).astype(float).tolist()

# physical modes m to plot
ms = list(range(9))  # m = 0..12

OUT_PNG = "Eigenvals_bruteforce_generalized.png"
OUT_SVG = "Eigenvals_bruteforce_generalized.svg"
OUT_CSV = "Eigenvals_bruteforce_generalized.csv"
OUT_EIGENVECTORS_NPZ = "Eigenvectors_bruteforce_generalized.npz"  # For storing eigenvectors

# Batch processing to avoid memory issues
BATCH_SIZE = 5  # Process this many temperatures at a time
MAX_PARALLEL_FILES = 3  # Maximum number of files to process in parallel (reduced to save memory)

# --- NEW: eigenmode selection knobs (same spirit as your diagnostics code) ---
N_EIG_CANDIDATES = 120#120   # compute this many eigenpairs near sigma~0 (reduced to avoid memory issues)
SIGMA = 1e-8             # shift-invert target (near slow modes, increased to help convergence without reg)
ZERO_TOL = 1e-10         # used only if you decide to drop conserved; we keep them by default
INCLUDE_CONSERVED = True # keep near-zero modes so m=0 and m=1 show up

INV_MIN_CORR = 0.75      # inversion parity test strength (relax if missing m)
M_TOL = 0.40             # |m_est - m| tolerance (relax if missing m)

# sign-switch estimator parameters
RING_WIDTH_FACTOR = 2.5
MIN_RING_POINTS   = 200
N_ANGLE_BINS_MIN  = 256

# --- NEW: plotting knobs for ring clarity + speed ---
PLOT_PWIN = 1.6              # zoom window in px/py around 0 (ring fills panel)
PLOT_RING_ONLY = True        # mask everything except |p|-1 within a small band
PLOT_RING_W = None           # if None, uses RING_WIDTH_FACTOR*dp


def _as_csr(X):
    if isspmatrix_csr(X):
        return X
    return csr_matrix(X)


def make_index_map(nx: np.ndarray, ny: np.ndarray, Nmax: int, half: int) -> np.ndarray:
    idx_map = -np.ones((Nmax, Nmax), dtype=np.int32)
    idx_map[nx + half, ny + half] = np.arange(nx.size, dtype=np.int32)
    return idx_map


def build_inv_map(nx: np.ndarray, ny: np.ndarray, idx_map: np.ndarray, Nmax: int, half: int,
                  shift_x: float = 0.0, shift_y: float = 0.0) -> np.ndarray:
    """
    Build inversion map for p -> -p.
    For p = dp*(n+shift), inversion requires:
      n_inv + shift = -(n + shift)  =>  n_inv = -n - 2*shift
    This is only an exact integer map when 2*shift is (near) an integer (e.g. 0.0 or 0.5).
    """
    sx2 = int(np.rint(2.0 * float(shift_x)))
    sy2 = int(np.rint(2.0 * float(shift_y)))
    if abs(2.0 * float(shift_x) - sx2) > 1e-12:
        sx2 = 0
    if abs(2.0 * float(shift_y) - sy2) > 1e-12:
        sy2 = 0
    
    inv_ix = (-nx - sx2 + half) % Nmax
    inv_iy = (-ny - sy2 + half) % Nmax
    inv = idx_map[inv_ix, inv_iy].astype(np.int32)
    return inv


if USE_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def w_corr_numba(v: np.ndarray, u: np.ndarray, w: np.ndarray) -> float:
        """Weighted correlation <v,u>_W / <v,v>_W (numba-accelerated, parallel)."""
        n = v.size
        num = 0.0
        den = 0.0
        # Parallel reduction
        for i in prange(n):
            wv = w[i] * v[i]
            num += wv * u[i]
            den += wv * v[i]
        return num / (den + 1e-30)

def w_corr(v: np.ndarray, u: np.ndarray, w: np.ndarray) -> float:
    """Weighted correlation <v,u>_W / <v,v>_W."""
    if USE_NUMBA:
        return w_corr_numba(v, u, w)
    num = float(np.dot(v, w * u))
    den = float(np.dot(v, w * v)) + 1e-30
    return num / den


if USE_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def estimate_sign_switches_on_ring_numba(v_full, active, px, py, P, dp_val, 
                                             ring_width_factor, min_ring_points, n_angle_bins_min):
        """Numba-accelerated sign switch estimation (parallel)."""
        ring_w = ring_width_factor * dp_val
        n_active = active.size
        
        # Collect indices directly (more efficient than mask)
        # Sequential collection (small overhead, but needed for correctness)
        idx_list = np.zeros(n_active, dtype=np.int32)
        idx_count = 0
        for i in range(n_active):
            if np.abs(P[active[i]] - 1.0) <= ring_w:
                idx_list[idx_count] = active[i]
                idx_count += 1
        
        while idx_count < min_ring_points and ring_w < 0.5:
            ring_w *= 1.5
            idx_count = 0
            for i in range(n_active):
                if np.abs(P[active[i]] - 1.0) <= ring_w:
                    idx_list[idx_count] = active[i]
                    idx_count += 1
        
        if idx_count < 16:
            return 0, 0.0
        
        # Compute angles (parallel)
        ang = np.zeros(idx_count, dtype=np.float64)
        for i in prange(idx_count):
            a = np.arctan2(py[idx_list[i]], px[idx_list[i]])
            ang[i] = (a + 2.0 * np.pi) % (2.0 * np.pi)
        
        nbin = max(n_angle_bins_min, int((2.0 * np.pi / max(dp_val, 1e-12)) * 2))
        bins = np.floor(ang / (2.0 * np.pi) * nbin).astype(np.int64)
        
        prof = np.zeros(nbin, dtype=np.float64)
        cnt  = np.zeros(nbin, dtype=np.int64)
        # Sequential accumulation (parallel would have race condition on prof[b])
        # But we can parallelize the value lookup
        for i in range(idx_count):
            b = bins[i]
            prof[b] += v_full[idx_list[i]]
            cnt[b]  += 1
        
        has_any = False
        for i in range(nbin):
            if cnt[i] > 0:
                prof[i] /= cnt[i]
                has_any = True
        
        if not has_any:
            return 0, 0.0
        
        max_prof = 0.0
        for i in range(nbin):
            if cnt[i] > 0:
                abs_prof = np.abs(prof[i])
                if abs_prof > max_prof:
                    max_prof = abs_prof
        
        thr = 1e-10 * (max_prof + 1e-300)
        s = np.zeros(nbin, dtype=np.float64)
        for i in range(nbin):
            if np.abs(prof[i]) < thr:
                s[i] = 0.0
            else:
                s[i] = 1.0 if prof[i] > 0.0 else -1.0
        
        last = 0.0
        for i in range(nbin):
            if s[i] == 0.0:
                s[i] = last
            else:
                last = s[i]
        
        if s[0] == 0.0:
            j = -1
            for i in range(nbin):
                if s[i] != 0.0:
                    j = i
                    break
            if j >= 0:
                for i in range(j):
                    s[i] = s[j]
        
        all_zero = True
        for i in range(nbin):
            if s[i] != 0.0:
                all_zero = False
                break
        if all_zero:
            return 0, 0.0
        
        switches = 0
        for i in range(nbin):
            a = s[i]
            b = s[(i + 1) % nbin]
            if a * b < 0.0:
                switches += 1
        
        m_est = 0.5 * float(switches)
        return switches, m_est

def estimate_sign_switches_on_ring(v_full, active, px, py, P, dp_val, theta_val):
    """
    Estimate angular momentum m using FFT on angular profile (same as Iee_decay_rates_harmonics_plot.py):
      - restrict to active indices near |p|≈1
      - bin by polar angle and perform FFT to find dominant frequency
      - refine using correlation with cos(m*phi) and sin(m*phi) templates
    Returns: (switches:int, m_est:float)
      with m_est from FFT analysis + correlation refinement
    """
    import math
    
    ring_w = RING_WIDTH_FACTOR * float(dp_val)
    idx = active[np.abs(P[active] - 1.0) <= ring_w]
    while idx.size < MIN_RING_POINTS and ring_w < 0.5:
        ring_w *= 1.5
        idx = active[np.abs(P[active] - 1.0) <= ring_w]
    if idx.size < 16:
        return 0, 0.0

    ang = np.arctan2(py[idx], px[idx])
    ang = (ang + 2.0 * math.pi) % (2.0 * math.pi)
    nbin = max(N_ANGLE_BINS_MIN, int((2.0 * math.pi / max(dp_val, 1e-12)) * 2))
    bins = np.floor(ang / (2.0 * math.pi) * nbin).astype(np.int64)

    prof = np.zeros(nbin, dtype=np.float64)
    cnt  = np.zeros(nbin, dtype=np.int64)
    for b, val in zip(bins, v_full[idx]):
        prof[b] += float(val)
        cnt[b]  += 1
    has = cnt > 0
    if not np.any(has):
        return 0, 0.0
    prof[has] /= cnt[has]

    # Fill empty bins by interpolation
    if not np.all(has):
        valid_indices = np.where(has)[0]
        if valid_indices.size > 0:
            valid_values = prof[valid_indices]
            all_indices = np.arange(nbin)
            prof = np.interp(all_indices, valid_indices, valid_values)
        else:
            return 0, 0.0

    # Save original profile for m=0 correlation check
    prof_original = prof.copy()
    
    # Check for m=0 (constant/nearly constant profile) BEFORE removing DC
    prof_std = np.std(prof)
    prof_mean = np.mean(prof)
    prof_abs_max = np.max(np.abs(prof - prof_mean))
    
    if abs(prof_mean) > 1e-10:
        rel_std = prof_std / abs(prof_mean)
    else:
        rel_std = prof_std
    
    # If the profile is nearly constant, return m=0
    if prof_abs_max < 1e-8 or rel_std < 0.01:
        s = np.sign(prof - prof_mean)
        switches = 0
        for i in range(nbin):
            a = s[i]
            b = s[(i + 1) % nbin]
            if a * b < 0.0:
                switches += 1
        return switches, 0.0

    # Remove DC component (mean) for non-constant profiles
    prof_centered = prof - prof_mean
    
    # Normalize
    prof_max = np.max(np.abs(prof_centered))
    if prof_max < 1e-10:
        switches = 0
        return switches, 0.0
    prof = prof_centered / prof_max

    # Perform FFT to find dominant angular frequency
    fft_result = np.fft.fft(prof_original)
    fft_power = np.abs(fft_result)
    
    # Check DC component (index 0) for m=0
    dc_power = fft_power[0]
    
    max_freq = min(nbin // 2, 50)
    if max_freq < 1:
        return 0, 0.0
    
    # Find dominant frequency (excluding DC at index 0)
    freq_range = fft_power[1:max_freq+1]
    if len(freq_range) == 0:
        switches = 0
        return switches, 0.0
    
    dominant_idx = np.argmax(freq_range) + 1
    dominant_power = fft_power[dominant_idx]
    
    # If DC component is dominant or comparable, likely m=0
    if dc_power > 0.5 * dominant_power and dc_power > np.sum(fft_power[1:]) * 0.3:
        switches = 0
        return switches, 0.0
    
    # The frequency index k in FFT corresponds to k periods in 2π, so m = k
    m_est = float(dominant_idx)
    
    # Refine: check nearby frequencies
    if dominant_idx > 1:
        prev_power = fft_power[dominant_idx - 1]
        if prev_power > 0.7 * dominant_power and prev_power > dominant_power * 0.9:
            m_est = float(dominant_idx - 1)
    
    if dominant_idx < max_freq:
        next_power = fft_power[dominant_idx + 1]
        if next_power > 1.1 * dominant_power:
            m_est = float(dominant_idx + 1)
    
    # Refinement: use correlation with cos(m*phi) and sin(m*phi) templates
    phi_bins = np.linspace(0, 2.0 * math.pi, nbin, endpoint=False)
    m_fft_rounded = int(np.round(m_est))
    m_candidates = list(range(max(0, m_fft_rounded - 3), min(max_freq + 1, m_fft_rounded + 4)))
    if 0 not in m_candidates:
        m_candidates.append(0)
    m_candidates = list(set(m_candidates))
    m_candidates.sort()
    
    best_m = int(np.round(m_est))
    best_corr = -1.0
    
    for m_test in m_candidates:
        if m_test == 0:
            prof_orig_norm = prof_original.copy()
            prof_orig_mean = np.mean(prof_orig_norm)
            prof_orig_std = np.std(prof_orig_norm)
            if prof_orig_std > 1e-10:
                prof_orig_norm = (prof_orig_norm - prof_orig_mean) / prof_orig_std
            else:
                prof_orig_norm = prof_orig_norm - prof_orig_mean
            
            template = np.ones_like(prof_orig_norm)
            template = template / (np.linalg.norm(template) + 1e-10)
            corr = abs(np.dot(prof_orig_norm, template))
        else:
            template_cos = np.cos(m_test * phi_bins)
            template_sin = np.sin(m_test * phi_bins)
            template_cos = template_cos / (np.linalg.norm(template_cos) + 1e-10)
            template_sin = template_sin / (np.linalg.norm(template_sin) + 1e-10)
            corr_cos = abs(np.dot(prof, template_cos))
            corr_sin = abs(np.dot(prof, template_sin))
            corr = max(corr_cos, corr_sin)
        
        if corr > best_corr:
            best_corr = corr
            best_m = m_test
    
    # If correlation is low, expand search range
    if best_corr < 0.5:
        expanded_range = list(range(0, min(31, max_freq + 1)))
        for m_test in expanded_range:
            if m_test in m_candidates:
                continue
            if m_test == 0:
                prof_orig_norm = prof_original.copy()
                prof_orig_mean = np.mean(prof_orig_norm)
                prof_orig_std = np.std(prof_orig_norm)
                if prof_orig_std > 1e-10:
                    prof_orig_norm = (prof_orig_norm - prof_orig_mean) / prof_orig_std
                else:
                    prof_orig_norm = prof_orig_norm - prof_orig_mean
                template = np.ones_like(prof_orig_norm)
                template = template / (np.linalg.norm(template) + 1e-10)
                corr = abs(np.dot(prof_orig_norm, template))
            else:
                template_cos = np.cos(m_test * phi_bins)
                template_sin = np.sin(m_test * phi_bins)
                template_cos = template_cos / (np.linalg.norm(template_cos) + 1e-10)
                template_sin = template_sin / (np.linalg.norm(template_sin) + 1e-10)
                corr_cos = abs(np.dot(prof, template_cos))
                corr_sin = abs(np.dot(prof, template_sin))
                corr = max(corr_cos, corr_sin)
            
            if corr > best_corr:
                best_corr = corr
                best_m = m_test
    
    # Use the refined m if correlation is reasonable
    if best_corr > 0.2:
        m_est = float(best_m)
    
    # Compute sign switches for backward compatibility
    s = np.sign(prof)
    switches = 0
    for i in range(nbin):
        a = s[i]
        b = s[(i + 1) % nbin]
        if a * b < 0.0:
            switches += 1

    return int(switches), float(m_est)


# Cache for file lookups to avoid repeated directory scans
_file_cache = None

def find_matrix_file(theta_req: float):
    """
    Pick the *_T{...}.pkl file whose T in the filename is closest to theta_req.
    Uses caching to avoid repeated directory scans.
    """
    global _file_cache
    
    if not os.path.isdir(IN_DIR):
        raise FileNotFoundError(f"Directory not found: {IN_DIR}")

    # Build cache if not exists
    if _file_cache is None:
        files = [fn for fn in os.listdir(IN_DIR) if fn.endswith(".pkl") and "_T" in fn]
        if not files:
            raise FileNotFoundError(f"No .pkl files with '_T' found in {IN_DIR}")

        Ts = []
        paths = []
        for fn in files:
            try:
                tpart = fn.split("_T", 1)[1].rsplit(".pkl", 1)[0]
                Tval = float(tpart)
            except Exception:
                continue
            Ts.append(Tval)
            paths.append(os.path.join(IN_DIR, fn))

        if not Ts:
            raise FileNotFoundError(f"Could not parse any temperatures from filenames in {IN_DIR}")

        _file_cache = (np.array(Ts, dtype=float), paths)

    Ts, paths = _file_cache
    idx = int(np.argmin(np.abs(Ts - theta_req)))
    return float(Ts[idx]), paths[idx]


def load_matrix_nearest(theta_req: float):
    """
    Load matrix closest to theta_req.
    Returns (M, meta, path, theta_used).
    """
    theta_used, path = find_matrix_file(theta_req)
    with open(path, "rb") as fp:
        M, meta = pickle.load(fp)
    return M, meta, path, theta_used


def load_matrix_with_size(path: str):
    """
    Load matrix and return (M, meta, size) where size is active matrix size.
    """
    with open(path, "rb") as fp:
        M, meta = pickle.load(fp)
    active = meta.get("active", None)
    if active is not None and len(active) > 0:
        size = int(len(active))
    elif hasattr(M, 'shape'):
        size = int(M.shape[0])
    else:
        size = 0
    return M, meta, size


def build_centered_lattice(Nmax: int):
    """Build centered lattice indices."""
    half = Nmax // 2
    ns = np.arange(-half, half, dtype=np.int32)
    nx, ny = np.meshgrid(ns, ns, indexing="ij")
    return nx.reshape(-1), ny.reshape(-1), half


def reconstruct_full_arrays(meta):
    """
    Reconstruct full-lattice arrays from minimal meta.
    Needed because we no longer store nx, ny, px, py, P, eps, f in meta.
    """
    import math
    Nmax = int(meta["Nmax"])
    dp   = float(meta["dp"])
    Theta = float(meta["Theta"])
    shift_x = float(meta.get("shift_x", 0.0))
    shift_y = float(meta.get("shift_y", 0.0))
    nx, ny, half = build_centered_lattice(Nmax)
    px = dp * (nx.astype(np.float64) + shift_x)
    py = dp * (ny.astype(np.float64) + shift_y)
    P  = np.sqrt(px * px + py * py)
    eps = P * P

    invT = 1.0 / Theta
    em = math.exp(-invT)
    a  = 1.0 - em
    x = np.clip((eps - 1.0) * invT, -700.0, 700.0)
    f = a / (np.exp(x) + a)
    w = f * (1.0 - f)
    return nx, ny, half, px, py, P, eps, f, w


def get_active_operator(M, meta):
    """
    Returns:
      Ma : csr_matrix, shape (Nactive, Nactive)
      active : np.ndarray of global indices
    """
    active = meta.get("active", None)
    if active is None or len(active) == 0:
        # fallback: treat everything active
        Nstates = int(meta["nx"].size)
        active = np.arange(Nstates, dtype=np.int32)
    else:
        active = np.asarray(active, dtype=np.int32)

    Nactive = int(active.size)

    # New format: active-only CSR saved directly
    if bool(meta.get("active_only", False)) and getattr(M, "shape", None) == (Nactive, Nactive):
        Ma = _as_csr(M)
        return Ma, active

    # Legacy format: full matrix (dense), restrict
    # NOTE: this is expensive for large Nstates
    Ma = _as_csr(M[np.ix_(active, active)])
    return Ma, active


def select_physical_eigs_per_m(Ma: csr_matrix, meta, active: np.ndarray, ms, theta_val: float):
    """
    Solve (-Ma) v = gamma * W v near gamma~0, classify by (i) inversion parity and (ii) m_est from sign-switches,
    then return {m: (gamma_m, v_full_m)} for requested ms, where v_full_m is the eigenvector embedded in full lattice.
    """
    # Reconstruct arrays if needed
    if "nx" not in meta or "px" not in meta:
        nx, ny, half, px, py, P, eps, f, w = reconstruct_full_arrays(meta)
        meta["nx"] = nx
        meta["ny"] = ny
        meta["half"] = half
        meta["px"] = px
        meta["py"] = py
        meta["P"] = P
        meta["eps"] = eps
        meta["f"] = f
    
    Nstates = int(meta["nx"].size)
    nx = np.asarray(meta["nx"], dtype=np.int32)
    ny = np.asarray(meta["ny"], dtype=np.int32)
    Nmax = int(meta["Nmax"])
    half = int(meta.get("half", Nmax // 2))

    idx_map = make_index_map(nx, ny, Nmax, half)
    shift_x = float(meta.get("shift_x", 0.0))
    shift_y = float(meta.get("shift_y", 0.0))
    inv_map = build_inv_map(nx, ny, idx_map, Nmax, half, shift_x=shift_x, shift_y=shift_y)

    # full arrays for classification
    f_full = np.asarray(meta["f"], dtype=np.float64)
    w_full = np.clip(f_full * (1.0 - f_full), 0.0, None)

    # Handle grid shift (from previous patch)
    shift_x = float(meta.get("shift_x", 0.0))
    shift_y = float(meta.get("shift_y", 0.0))
    dp_val = float(meta["dp"])
    if "px" in meta and "py" in meta:
        px = np.asarray(meta["px"], dtype=np.float64)
        py = np.asarray(meta["py"], dtype=np.float64)
    else:
        px = dp_val * (nx.astype(np.float64) + shift_x)
        py = dp_val * (ny.astype(np.float64) + shift_y)
    P  = np.asarray(meta.get("P", np.sqrt(px * px + py * py)), dtype=np.float64)

    # active-space generalized eigenproblem
    w_act = np.clip(w_full[active], 0.0, None)
    w_eps = 1e-30
    w_safe = np.where(w_act > 0.0, w_act, w_eps)
    W = diags(w_safe, 0, format="csr")

    A = _as_csr(-Ma)

    # Memory-safe: avoid shift-invert (sigma) and avoid forming B explicitly.
    # Work with B = D^{-1/2} A D^{-1/2} as a LinearOperator and ask for smallest eigenvalues.
    d = (1.0 / np.sqrt(w_safe)).astype(np.float64)  # diagonal of D^{-1/2}

    def _matvec_B(x):
        # B x = D^{-1/2} A D^{-1/2} x
        # Optimized: use in-place multiplication and scipy's optimized CSR dot
        y = d * x
        z = A.dot(y)  # scipy's CSR dot is already highly optimized
        return d * z

    n = A.shape[0]
    Bop = LinearOperator((n, n), matvec=_matvec_B, dtype=np.float64)

    k_calc = min(n - 2, int(N_EIG_CANDIDATES))
    if k_calc <= 0:
        return {m: np.nan for m in ms}

    # Smallest algebraic eigenvalues ≈ slow modes near 0 (no sigma => no CSC factorization)
    # Optimized: use smaller tolerance for faster convergence, but still accurate enough
    # Reduce k_calc if we only need a few modes (ms is small)
    k_needed = max(len(ms) * 2, 20)  # Need at least 2x modes for safety, but minimum 20
    k_calc = min(k_calc, max(k_needed, n - 2))
    
    vals, y = eigsh(
        Bop,
        k=k_calc,
        which="SA",
        tol=1e-6,  # Further relaxed for speed (was 1e-7)
        maxiter=10000,  # Reduced maxiter
        ncv=min(max(2 * k_calc + 1, 20), n)  # Optimal subspace size for ARPACK
    )
    vals = np.real(vals)
    y = np.real(y)

    # Recover generalized eigenvectors v = D^{-1/2} y
    vecs = (d[:, None] * y).astype(np.float64)

    # sort by gamma (slowest first)
    # (clip tiny negative numerical noise if present)
    vals = np.where(vals < 0.0, 0.0, vals)
    order = np.argsort(vals)
    vals = np.real(vals[order])
    vecs = np.real(vecs[:, order])

    if not INCLUDE_CONSERVED:
        keep = np.where(np.abs(vals) > ZERO_TOL)[0]
        vals = vals[keep]
        vecs = vecs[:, keep]

    # normalize in W-inner-product on active (vectorized)
    wv = w_safe[:, None] * vecs
    norms2 = np.sum(wv * vecs, axis=0)
    norms2 = np.maximum(norms2, 1e-30)  # avoid division by zero
    vecs = vecs / np.sqrt(norms2[None, :])

    # collect candidates per m
    best = {m: None for m in ms}
    dp_val = float(meta["dp"])
    
    # Process eigenvectors in parallel (numba releases GIL, so threading works well)
    def process_eigenvector(i):
        """Process a single eigenvector and return result if valid."""
        gamma = float(vals[i])
        v_active = vecs[:, i].copy()
        
        # embed into full lattice for symmetry + ring counting
        v_full = np.zeros(Nstates, dtype=np.float64)
        v_full[active] = v_active
        
        # inversion parity correlation
        v_full_inv = v_full[inv_map]
        c_inv = w_corr(v_full, v_full_inv, w_full)
        
        switches, m_est = estimate_sign_switches_on_ring(
            v_full=v_full, active=active, px=px, py=py, P=P, dp_val=dp_val, theta_val=theta_val
        )
        m_round = int(np.rint(m_est))
        
        if m_round not in ms:
            return None
        if abs(m_est - float(m_round)) > M_TOL:
            return None
        
        # parity sanity: even m => c_inv ~ +1, odd m => c_inv ~ -1
        if (m_round % 2) == 0:
            if c_inv < INV_MIN_CORR:
                return None
        else:
            if c_inv > -INV_MIN_CORR:
                return None
        
        return (i, m_round, gamma, c_inv, m_est, switches, v_full)
    
    # Parallel processing of eigenvectors using threading (numba releases GIL)
    n_vecs = vecs.shape[1]
    results = []
    if n_vecs > 3 and N_WORKERS > 1:
        # Use threading - numba functions release GIL so this works well
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(process_eigenvector, i) for i in range(n_vecs)]
            results = [f.result() for f in as_completed(futures)]
    else:
        # Sequential for small number of vectors
        results = [process_eigenvector(i) for i in range(n_vecs)]
    
    # Collect best results (keep slowest gamma for each m)
    for result in results:
        if result is None:
            continue
        i, m_round, gamma, c_inv, m_est, switches, v_full = result
        cur = best[m_round]
        if (cur is None) or (gamma < cur["gamma"]):
            best[m_round] = {"gamma": gamma, "c_inv": c_inv, "m_est": m_est, "switches": switches, "v_full": v_full}

    out = {}
    for m in ms:
        if best[m] is None:
            out[m] = (np.nan, None)
        else:
            out[m] = (float(best[m]["gamma"]), best[m]["v_full"])
    return out


def save_csv_incremental(Ts_new, Ts_req_new, gammas_new, existing_Ts, existing_Ts_requested, existing_gammas, ms):
    """
    Save CSV incrementally by merging new results with existing ones.
    This function is called after each batch to save progress.
    """
    # Combine all temperatures
    all_Ts_combined = list(existing_Ts) + list(Ts_new)
    all_Ts_req_combined = list(existing_Ts_requested if len(existing_Ts_requested) == len(existing_Ts) else existing_Ts) + list(Ts_req_new)
    all_gammas_combined = {m: list(existing_gammas[m]) + list(gammas_new[m]) for m in ms}
    
    # Create sorted unique list
    unique_Ts = sorted(set(float(T) for T in all_Ts_combined), key=lambda x: float(x))
    T_to_index = {float(T): i for i, T in enumerate(unique_Ts)}
    
    # Build combined data
    combined_gammas = {m: [np.nan] * len(unique_Ts) for m in ms}
    combined_Ts_req = [None] * len(unique_Ts)
    
    # Fill existing data
    for i, T_existing in enumerate(existing_Ts):
        T_float = float(T_existing)
        idx = T_to_index.get(T_float)
        if idx is not None:
            for m in ms:
                if i < len(existing_gammas[m]):
                    combined_gammas[m][idx] = existing_gammas[m][i]
            if i < len(existing_Ts_requested):
                combined_Ts_req[idx] = existing_Ts_requested[i]
            else:
                combined_Ts_req[idx] = T_existing
    
    # Fill new data
    for i, T_new in enumerate(Ts_new):
        T_float = float(T_new)
        idx = T_to_index.get(T_float)
        if idx is not None:
            for m in ms:
                if i < len(gammas_new[m]):
                    combined_gammas[m][idx] = gammas_new[m][i]
            if i < len(Ts_req_new):
                combined_Ts_req[idx] = Ts_req_new[i]
            else:
                combined_Ts_req[idx] = T_new
    
    # Write CSV
    with open(OUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['T', 'T_requested'] + [f'gamma_{m}' for m in ms]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, T in enumerate(unique_Ts):
            row = {
                'T': f'{T:.10g}',
                'T_requested': f'{combined_Ts_req[i]:.10g}' if combined_Ts_req[i] is not None else f'{T:.10g}'
            }
            for m in ms:
                gm = combined_gammas[m][i]
                if np.isfinite(gm):
                    row[f'gamma_{m}'] = f'{gm:.10e}'
                else:
                    row[f'gamma_{m}'] = ''
            writer.writerow(row)


def main():
    gammas = {m: [] for m in ms}
    eigenvectors = {m: [] for m in ms}  # Store eigenvectors for plotting
    Ts_used = []     # actual loaded temperatures
    Ts_req_used = [] # requested temperatures (for reference)

    # Load existing results if available
    existing_Ts = []
    existing_Ts_requested = []  # Also track T_requested for matching
    existing_gammas = {m: [] for m in ms}
    existing_eigenvectors = {m: [] for m in ms}
    
    if os.path.exists(OUT_CSV):
        print(f"[Loading] Found existing results in {OUT_CSV}")
        try:
            with open(OUT_CSV, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    T_val = float(row.get('T', 0))
                    T_req_val = float(row.get('T_requested', T_val))  # Use T as fallback if T_requested missing
                    existing_Ts.append(T_val)
                    existing_Ts_requested.append(T_req_val)
                    # Load gammas for each mode (one value per temperature)
                    for m in ms:
                        gamma_key = f'gamma_{m}'
                        if gamma_key in row and row[gamma_key].strip():
                            try:
                                gamma_val = float(row[gamma_key])
                                existing_gammas[m].append(gamma_val)
                            except (ValueError, TypeError):
                                existing_gammas[m].append(np.nan)
                        else:
                            existing_gammas[m].append(np.nan)
                # Note: eigenvectors are not saved in CSV (too large), so we'll recompute them for plotting
            print(f"[Loading] Found {len(existing_Ts)} existing temperatures: {existing_Ts}")
        except Exception as e:
            print(f"[Warning] Could not load existing results: {e}")
            import traceback
            traceback.print_exc()
            existing_Ts = []
            existing_Ts_requested = []
            existing_gammas = {m: [] for m in ms}
    
    # Scan all .pkl files in the folder to get all available temperatures
    print(f"[Scanning] Looking for all .pkl files in {IN_DIR}...")
    all_files = []
    if os.path.isdir(IN_DIR):
        files = [fn for fn in os.listdir(IN_DIR) if fn.endswith(".pkl") and "_T" in fn]
        for fn in files:
            try:
                tpart = fn.split("_T", 1)[1].rsplit(".pkl", 1)[0]
                Tval = float(tpart)
                all_files.append((Tval, os.path.join(IN_DIR, fn)))
            except Exception as e:
                print(f"[Warning] Could not parse temperature from {fn}: {e}")
                continue
    
    if len(all_files) == 0:
        print(f"[Error] No .pkl files found in {IN_DIR}")
        return
    
    # Sort by temperature
    all_files.sort(key=lambda x: x[0])
    all_Ts_from_files = [T for T, _ in all_files]
    print(f"[Scanning] Found {len(all_files)} .pkl files with temperatures: {len(all_Ts_from_files)} unique")
    
    # Filter to only include temperatures not already computed
    # Use relative tolerance for floating point comparison (more robust)
    # Check against both T and T_requested columns
    Thetas_to_process = []
    for T_file, path in all_files:
        T_file_float = float(T_file)
        # Check if this temperature is already computed
        is_computed = False
        matched_T = None
        matched_type = None
        for i, T_existing in enumerate(existing_Ts):
            T_existing_float = float(T_existing)
            T_req_existing_float = float(existing_Ts_requested[i]) if i < len(existing_Ts_requested) else T_existing_float
            # Check against both T and T_requested
            # Use more lenient tolerance: 1% relative or 1e-3 absolute, whichever is larger
            rel_tol = 0.01 * max(abs(T_file_float), abs(T_existing_float))
            tol = max(1e-3, rel_tol)  # At least 0.001 absolute tolerance
            if abs(T_file_float - T_existing_float) < tol:
                is_computed = True
                matched_T = T_existing_float
                matched_type = "T"
                break
            # Also check against T_requested with same tolerance
            rel_tol_req = 0.01 * max(abs(T_file_float), abs(T_req_existing_float))
            tol_req = max(1e-3, rel_tol_req)
            if abs(T_file_float - T_req_existing_float) < tol_req:
                is_computed = True
                matched_T = T_req_existing_float
                matched_type = "T_requested"
                break
        
        if not is_computed:
            Thetas_to_process.append((T_file, path))
        else:
            print(f"[Skip] Theta={T_file:.6g} already computed (matches {matched_type}={matched_T:.6g}), skipping...")
    
    # Summary of what will be processed
    n_skipped = len(all_files) - len(Thetas_to_process)
    if n_skipped > 0:
        print(f"[Summary] Skipped {n_skipped} already-computed temperatures, {len(Thetas_to_process)} new temperatures to process")
    
    if len(Thetas_to_process) == 0:
        print(f"[Info] All {len(all_files)} temperatures from folder are already computed in CSV. Skipping computation, using existing data for plotting...")
        # Use existing data (no new computation needed)
        Ts_used = []
        Ts_req_used = []
        gammas = {m: [] for m in ms}
        eigenvectors = {m: [] for m in ms}  # Will be empty for existing data (eigenvectors not saved in CSV)
    else:
        print(f"=== Computing eigenvalues gamma_m(T) for {len(Thetas_to_process)} new temperatures ===")
        print(f"[Batch processing] Processing in batches of {BATCH_SIZE} files to avoid memory issues...")
        
        # Process files in batches to avoid memory issues
        # First, collect file info without loading matrices
        file_info = []  # (T_file, path, size)
        for T_file, path in Thetas_to_process:
            try:
                # Just get file size without loading matrix
                file_size = os.path.getsize(path)
                # Estimate matrix size from file (rough estimate)
                # For now, we'll load it in batches, but try to avoid loading all at once
                file_info.append((T_file, path, file_size))
            except Exception as e:
                print(f"[Warning] Could not get info for file {path}: {e}")
        
        # Sort by file size (largest first) to process big files first
        file_info.sort(key=lambda x: -x[2])
        
        # Process in batches
        Ts_used = []
        Ts_req_used = []
        gammas = {m: [] for m in ms}
        
        # Load existing eigenvectors from NPZ if available
        existing_eigenvectors_npz = {}
        if os.path.exists(OUT_EIGENVECTORS_NPZ):
            try:
                print(f"[Loading] Found existing eigenvectors in {OUT_EIGENVECTORS_NPZ}")
                data = np.load(OUT_EIGENVECTORS_NPZ, allow_pickle=True)
                for m in ms:
                    key = f'eigenvectors_m{m}'
                    if key in data:
                        existing_eigenvectors_npz[m] = data[key].item()  # .item() to convert numpy array to dict
                print(f"[Loading] Loaded eigenvectors for {len(existing_eigenvectors_npz.get(ms[0], {}))} temperatures")
            except Exception as e:
                print(f"[Warning] Could not load existing eigenvectors: {e}")
        
        def process_single_file_batch(T_file, path):
            """Process a single file and return results."""
            try:
                # Load file
                with open(path, "rb") as fp:
                    M, meta = pickle.load(fp)
                
                Ma, active = get_active_operator(M, meta)
                print(f"[process] Theta={T_file:.6g}  |  {os.path.basename(path)}  |  shape={Ma.shape}")
                
                # Process
                sel = select_physical_eigs_per_m(Ma, meta, active, ms, theta_val=float(T_file))
                
                # Extract results
                result_gammas = {}
                result_eigenvectors = {}
                for m in ms:
                    gamma, v_full = sel[m]
                    result_gammas[m] = gamma
                    result_eigenvectors[m] = v_full if not np.isnan(gamma) else None
                
                # Extract grid metadata before cleaning up
                grid_meta = {
                    "Nmax": int(meta["Nmax"]),
                    "dp": float(meta["dp"]),
                    "shift_x": float(meta.get("shift_x", 0.0)),
                    "shift_y": float(meta.get("shift_y", 0.0)),
                }
                
                # Clean up memory
                del M, meta, Ma, active, sel
                
                return (float(T_file), float(T_file), result_gammas, result_eigenvectors, grid_meta)
            except Exception as e:
                print(f"[Error] Processing file {path}: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Process in batches
        num_batches = (len(file_info) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(file_info))
            batch_files = file_info[start_idx:end_idx]
            
            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing {len(batch_files)} files...")
            
            # Process batch (sequential to avoid memory issues)
            batch_results = []
            for T_file, path, _ in batch_files:
                result = process_single_file_batch(T_file, path)
                if result is not None:
                    batch_results.append(result)
            
            # Collect batch results
            for theta_used, T_file, result_gammas, result_eigenvectors, grid_meta in batch_results:
                Ts_used.append(theta_used)
                Ts_req_used.append(T_file)
                for m in ms:
                    gammas[m].append(result_gammas[m])
            
            # Save eigenvectors to NPZ incrementally
            print(f"[Saving] Saving eigenvectors for batch {batch_idx + 1} to {OUT_EIGENVECTORS_NPZ}...")
            batch_eigenvectors_dict = {}
            batch_grid_meta = {}
            for m in ms:
                if m not in batch_eigenvectors_dict:
                    batch_eigenvectors_dict[m] = {}
                for i, (theta_used, _, _, result_eigenvectors, grid_meta) in enumerate(batch_results):
                    batch_eigenvectors_dict[m][theta_used] = result_eigenvectors[m]
                    batch_grid_meta[theta_used] = grid_meta
            
            # Merge with existing and save
            if os.path.exists(OUT_EIGENVECTORS_NPZ):
                try:
                    existing_data = np.load(OUT_EIGENVECTORS_NPZ, allow_pickle=True)
                    for m in ms:
                        key = f'eigenvectors_m{m}'
                        if key in existing_data:
                            existing_dict = existing_data[key].item()
                            existing_dict.update(batch_eigenvectors_dict[m])
                            batch_eigenvectors_dict[m] = existing_dict
                    if "grid_meta" in existing_data:
                        gm = existing_data["grid_meta"].item()
                        gm.update(batch_grid_meta)
                        batch_grid_meta = gm
                except:
                    pass
            
            # Save updated eigenvectors
            save_dict = {f'eigenvectors_m{m}': batch_eigenvectors_dict[m] for m in ms}
            save_dict["grid_meta"] = batch_grid_meta
            np.savez_compressed(OUT_EIGENVECTORS_NPZ, **save_dict)
            
            # Save CSV incrementally
            print(f"[Saving] Updating CSV with batch {batch_idx + 1} results...")
            save_csv_incremental(Ts_used, Ts_req_used, gammas, existing_Ts, existing_Ts_requested, existing_gammas, ms)
            
            # Clean up memory
            del batch_results, batch_eigenvectors_dict
            import gc
            gc.collect()
            
            print(f"[Batch {batch_idx + 1}/{num_batches}] Completed. Memory cleaned.")
    
    # Load final data from CSV (which was saved incrementally)
    print(f"\n[Loading] Loading final data from {OUT_CSV} for plotting...")
    Ts = []
    gammas = {m: [] for m in ms}
    
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                T_val = float(row.get('T', 0))
                Ts.append(T_val)
                for m in ms:
                    gamma_key = f'gamma_{m}'
                    if gamma_key in row and row[gamma_key].strip():
                        try:
                            gamma_val = float(row[gamma_key])
                            gammas[m].append(gamma_val)
                        except (ValueError, TypeError):
                            gammas[m].append(np.nan)
                    else:
                        gammas[m].append(np.nan)
    
    Ts = np.array(Ts, dtype=np.float64)
    print(f"[Loading] Loaded {len(Ts)} temperatures from CSV")
    print("Ts=", Ts)
    if Ts.size == 0:
        print("Error: no data loaded from CSV.")
        return

    # --- plot raw gamma_m(T) ---
    fig, ax = plt.subplots(figsize=(8 * 0.9, 6 * 0.9))

    for m in ms:
        gm = np.array(gammas[m], dtype=np.float64)
        mask = np.isfinite(gm) & (gm > 0.0)
        if np.any(mask):
            ax.loglog(Ts[mask], gm[mask], label=fr"$m={m}$", linewidth=1.5)

    ax.set_xlabel(r"Temperature, $T/T_F$")
    ax.set_ylabel(r"Decay rate (eigenvalue), $\gamma_m$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    print(f"Saved: {OUT_SVG}")
    print(f"Saved: {OUT_PNG}")

    print(f"[Info] CSV already saved incrementally (contains {len(Ts)} temperatures total)")

    # --- Load eigenvectors from NPZ for eigenfunction plotting (in batches to avoid memory issues) ---
    print(f"\n[Loading] Loading eigenvectors from {OUT_EIGENVECTORS_NPZ} for plotting...")
    
    if not os.path.exists(OUT_EIGENVECTORS_NPZ):
        print(f"[Warning] {OUT_EIGENVECTORS_NPZ} not found. Skipping eigenfunction plotting.")
    else:
        # Load eigenvectors from NPZ (they're stored as dict: T -> v_full)
        eigenvectors_npz = {}
        grid_meta_npz = {}
        try:
            data = np.load(OUT_EIGENVECTORS_NPZ, allow_pickle=True)
            for m in ms:
                key = f'eigenvectors_m{m}'
                if key in data:
                    eigenvectors_npz[m] = data[key].item()
            if "grid_meta" in data:
                grid_meta_npz = data["grid_meta"].item()
            print(f"[Loading] Loaded eigenvectors for {len(eigenvectors_npz.get(ms[0], {}))} temperatures")
        except Exception as e:
            print(f"[Warning] Could not load eigenvectors from NPZ: {e}")
            eigenvectors_npz = {}
        
        # Convert to list format for plotting (matching Ts order)
        eigenvectors = {m: [] for m in ms}
        for T in Ts:
            T_float = float(T)
            for m in ms:
                # Find matching eigenvector in NPZ dict
                v_full = None
                if m in eigenvectors_npz:
                    # Try exact match first
                    if T_float in eigenvectors_npz[m]:
                        v_full = eigenvectors_npz[m][T_float]
                    else:
                        # Try tolerance-based match
                        for T_npz, v in eigenvectors_npz[m].items():
                            if abs(T_float - float(T_npz)) < max(1e-3, 0.01 * max(abs(T_float), abs(float(T_npz)))):
                                v_full = v
                                break
                eigenvectors[m].append(v_full)
        
        # Build a per-T grid meta list aligned with Ts (tolerant match)
        grid_meta_list = []
        for T in Ts:
            T_float = float(T)
            gm = None
            if grid_meta_npz:
                if T_float in grid_meta_npz:
                    gm = grid_meta_npz[T_float]
                else:
                    for T_npz, g in grid_meta_npz.items():
                        if abs(T_float - float(T_npz)) < max(1e-3, 0.01 * max(abs(T_float), abs(float(T_npz)))):
                            gm = g
                            break
            grid_meta_list.append(gm)
        
        # --- plot eigenfunction diagrams in table format (rows = m, columns = T) ---
        # Process in batches to avoid memory issues
        plot_eigenfunction_table_batched(Ts, eigenvectors, ms, grid_meta_list=grid_meta_list)


def values_to_grid(values_flat: np.ndarray, nx: np.ndarray, ny: np.ndarray, half: int, Nmax: int):
    """Convert flat array to 2D grid."""
    grid = np.full((Nmax, Nmax), np.nan, dtype=np.float64)
    grid[nx + half, ny + half] = values_flat
    return grid


def create_pcolormesh_coords(half: int, dp: float, Nmax: int):
    """Create coordinate arrays for pcolormesh (needs edge coordinates, not centers)."""
    p_edges = np.linspace(-half * dp - dp/2, (half - 1) * dp + dp/2, Nmax + 1)
    X, Y = np.meshgrid(p_edges, p_edges, indexing='ij')
    return X, Y


_plot_geom_cache = {}

def _plot_geom(Nmax: int, dp: float, shift_x: float, shift_y: float,
               pwin: float, ring_only: bool, ring_w: float):
    """
    Precompute slice indices + extent (+ optional ring mask) for fast repeated plotting at fixed grid.
    """
    key = (int(Nmax), float(dp), float(shift_x), float(shift_y), float(pwin), bool(ring_only),
           None if ring_w is None else float(ring_w))
    if key in _plot_geom_cache:
        return _plot_geom_cache[key]

    half = Nmax // 2
    # center coordinates: px(i) = dp*((i-half)+shift_x)
    # choose i-range so px in [-pwin, +pwin]
    i0 = int(np.floor(half - shift_x - (pwin / dp)))
    i1 = int(np.ceil (half - shift_x + (pwin / dp))) + 1
    j0 = int(np.floor(half - shift_y - (pwin / dp)))
    j1 = int(np.ceil (half - shift_y + (pwin / dp))) + 1
    i0 = max(0, min(Nmax, i0))
    i1 = max(0, min(Nmax, i1))
    j0 = max(0, min(Nmax, j0))
    j1 = max(0, min(Nmax, j1))

    xs = dp * ((np.arange(i0, i1) - half) + shift_x)
    ys = dp * ((np.arange(j0, j1) - half) + shift_y)
    extent = [xs[0] - dp/2, xs[-1] + dp/2, ys[0] - dp/2, ys[-1] + dp/2]

    ring_mask = None
    if ring_only:
        rw = float(ring_w) if ring_w is not None else (RING_WIDTH_FACTOR * dp)
        Xc, Yc = np.meshgrid(xs, ys, indexing="ij")
        P = np.sqrt(Xc*Xc + Yc*Yc)
        ring_mask = (np.abs(P - 1.0) > rw)

    out = (slice(i0, i1), slice(j0, j1), extent, ring_mask)
    _plot_geom_cache[key] = out
    return out


def plot_eigenfunction_table_batched(Ts, eigenvectors, ms, max_temps_plot=20, grid_meta_list=None):
    """
    Plot eigenfunction table, limiting number of temperatures to avoid memory issues.
    If there are more temperatures than max_temps_plot, select evenly spaced ones.
    """
    print(f"\n[Plotting eigenfunctions] Processing {len(Ts)} temperatures...")
    
    # Limit number of temperatures to plot to avoid memory issues
    if len(Ts) > max_temps_plot:
        print(f"[Plotting] Limiting to {max_temps_plot} temperatures (evenly spaced) to avoid memory issues...")
        selected_indices = [int(i * (len(Ts) - 1) / (max_temps_plot - 1)) for i in range(max_temps_plot)]
    else:
        selected_indices = list(range(len(Ts)))
    
    # Use the existing function
    plot_eigenfunction_table(Ts, eigenvectors, ms, selected_T_indices=selected_indices, max_Ts=None,
                             grid_meta_list=grid_meta_list)


def plot_eigenfunction_table(Ts, eigenvectors, ms, selected_T_indices=None, max_Ts=None, grid_meta_list=None):
    """
    Plot eigenfunction diagrams in a table: rows = m values, columns = selected temperatures.
    
    Args:
        Ts: array of temperatures
        eigenvectors: dict {m: [v_full_1, v_full_2, ...]} where each v_full is for corresponding T
        ms: list of m values to plot
        selected_T_indices: list of indices into Ts to plot (if None, use all temperatures)
        max_Ts: maximum number of temperatures to plot (if None, use all; ignored if selected_T_indices is provided)
        grid_meta_list: list of grid metadata dicts (one per T in Ts), or None to use slow fallback
    """
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Circle
    
    # Select temperatures to plot - use ALL by default
    if selected_T_indices is None:
        if max_Ts is None or len(Ts) <= max_Ts:
            # Use all temperatures
            selected_T_indices = list(range(len(Ts)))
        else:
            # Select evenly spaced temperatures if max_Ts is specified and we have more
            selected_T_indices = [int(i * (len(Ts) - 1) / (max_Ts - 1)) for i in range(max_Ts)]
    
    selected_Ts = Ts[selected_T_indices]
    print(f"\n[Plotting eigenfunctions] Using {len(selected_Ts)} temperatures: {selected_Ts}")
    
    # Find valid m values (those with at least one valid eigenvector)
    valid_ms = []
    for m in ms:
        if any(v is not None for v in eigenvectors[m]):
            valid_ms.append(m)
    
    if len(valid_ms) == 0:
        print("[Warning] No valid eigenvectors found for plotting.")
        return
    
    # Use grid_meta from NPZ if available (fast, no disk scans).
    # Fallback: if grid_meta missing, we can still try old slow behavior, but fast path is preferred.
    if grid_meta_list is None:
        grid_meta_list = [None] * len(Ts)
    
    # Compute global min/max across all eigenmodes for a common colorbar (in the ZOOMED window only)
    global_vmax_abs = 0.0
    for row_idx, m in enumerate(valid_ms):
        for col_idx, T_idx in enumerate(selected_T_indices):
            v_full = eigenvectors[m][T_idx]
            if v_full is not None:
                gm = grid_meta_list[T_idx]
                if gm is None:
                    continue
                Nmax_T = int(gm["Nmax"])
                dp_T = float(gm["dp"])
                shift_x = float(gm.get("shift_x", 0.0))
                shift_y = float(gm.get("shift_y", 0.0))
                if len(v_full) != Nmax_T * Nmax_T:
                    continue
                grid = np.asarray(v_full, dtype=np.float64).reshape((Nmax_T, Nmax_T))
                sx, sy, extent, ring_mask = _plot_geom(
                    Nmax_T, dp_T, shift_x, shift_y,
                    pwin=PLOT_PWIN, ring_only=PLOT_RING_ONLY, ring_w=PLOT_RING_W
                )
                g = grid[sx, sy]
                if ring_mask is not None:
                    g = np.ma.array(g, mask=ring_mask)
                    if g.count() == 0:
                        continue
                    vmax_abs = float(np.max(np.abs(g.compressed())))
                else:
                    vmax_abs = float(np.max(np.abs(g)))
                global_vmax_abs = max(global_vmax_abs, vmax_abs)
    
    if global_vmax_abs > 0:
        global_vmin = -global_vmax_abs
        global_vmax = global_vmax_abs
    else:
        global_vmin, global_vmax = -0.1, 0.1
    
    global_norm = TwoSlopeNorm(vmin=global_vmin, vcenter=0, vmax=global_vmax)
    
    # Create table layout: rows = m values, columns = selected temperatures
    nrows = len(valid_ms)
    ncols = len(selected_T_indices)
    
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=(4.5 * ncols, 4.5 * nrows), 
                             constrained_layout=True)
    
    # Handle case where there's only one row or one column
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.reshape(nrows, ncols)
    
    im_common = None
    
    # Fill table: rows = m values, columns = selected temperatures
    for row_idx, m in enumerate(valid_ms):
        for col_idx, T_idx in enumerate(selected_T_indices):
            ax = axes[row_idx, col_idx]
            v_full = eigenvectors[m][T_idx]
            
            if v_full is not None:
                gm = grid_meta_list[T_idx]
                if gm is None:
                    ax.axis("off")
                    continue
                Nmax_T = int(gm["Nmax"])
                dp_T = float(gm["dp"])
                shift_x = float(gm.get("shift_x", 0.0))
                shift_y = float(gm.get("shift_y", 0.0))
                if len(v_full) != Nmax_T * Nmax_T:
                    print(f"[Warning] Eigenvector size mismatch for T={Ts[T_idx]:.6g}, m={m}: "
                          f"expected {Nmax_T*Nmax_T}, got {len(v_full)}. Skipping plot.")
                    ax.axis("off")
                    continue

                grid = np.asarray(v_full, dtype=np.float64).reshape((Nmax_T, Nmax_T))
                sx, sy, extent, ring_mask = _plot_geom(
                    Nmax_T, dp_T, shift_x, shift_y,
                    pwin=PLOT_PWIN, ring_only=PLOT_RING_ONLY, ring_w=PLOT_RING_W
                )
                g = grid[sx, sy]  # (x,y) indexing
                if ring_mask is not None:
                    g = np.ma.array(g, mask=ring_mask)

                # imshow expects [y,x] layout, so transpose
                im = ax.imshow(
                    g.T,
                    extent=extent,
                    origin="lower",
                    cmap="seismic",
                    norm=global_norm,
                    aspect="equal",
                )
                im_common = im

                # Overlay the Fermi circle for reference (ring pops visually)
                ax.add_patch(Circle((0.0, 0.0), 1.0, fill=False, linewidth=0.6, alpha=0.6, color='black'))

                ax.set_xlim(-PLOT_PWIN, PLOT_PWIN)
                ax.set_ylim(-PLOT_PWIN, PLOT_PWIN)

                # Title: show m value on left column, T on top row
                if col_idx == 0:
                    ax.set_ylabel(f"$m={m}$", fontsize=14, rotation=0, labelpad=20)
                if row_idx == 0:
                    ax.set_title(f"$T={selected_Ts[col_idx]:.4g}$", fontsize=14)
                ax.set_xlabel("$p_x$")
                if col_idx > 0:
                    ax.set_ylabel("$p_y$")
            else:
                # No eigenvector available - turn off axis
                ax.axis("off")
    
    if im_common is not None:
        cbar = fig.colorbar(im_common, ax=axes, fraction=0.03, pad=0.02, aspect=30)
        cbar.ax.tick_params(labelsize=10)
    
    out_png = "eigenfunctions_table.png"
    out_svg = "eigenfunctions_table.svg"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()
