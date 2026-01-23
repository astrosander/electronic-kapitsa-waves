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
OUT_NPZ = "Eigenvals_bruteforce_generalized.npz"

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


def _as_csr(X):
    if isspmatrix_csr(X):
        return X
    return csr_matrix(X)


def make_index_map(nx: np.ndarray, ny: np.ndarray, Nmax: int, half: int) -> np.ndarray:
    idx_map = -np.ones((Nmax, Nmax), dtype=np.int32)
    idx_map[nx + half, ny + half] = np.arange(nx.size, dtype=np.int32)
    return idx_map


def build_inv_map(nx: np.ndarray, ny: np.ndarray, idx_map: np.ndarray, Nmax: int, half: int) -> np.ndarray:
    inv_ix = (-nx + half) % Nmax
    inv_iy = (-ny + half) % Nmax
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
    nx, ny, half = build_centered_lattice(Nmax)
    px = dp * nx.astype(np.float64)
    py = dp * ny.astype(np.float64)
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
    inv_map = build_inv_map(nx, ny, idx_map, Nmax, half)

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


def main():
    gammas = {m: [] for m in ms}
    eigenvectors = {m: [] for m in ms}  # Store eigenvectors for plotting
    Ts_used = []     # actual loaded temperatures
    Ts_req_used = [] # requested temperatures (for reference)

    print("=== Computing eigenvalues gamma_m(T) by eigenmode classification (m=0..12) ===")
    
    # Pre-load all files to get sizes and sort by matrix size (largest first)
    print("[Pre-scan] Loading files to determine processing order...")
    file_tasks = []
    seen_paths = {}  # path -> (theta_used, M, meta, size, Treq_list)
    
    for Treq in Thetas_req:
        try:
            theta_used, path = find_matrix_file(float(Treq))
            # Check if we've already loaded this file
            if path not in seen_paths:
                # Load once and get size
                M, meta, size = load_matrix_with_size(path)
                seen_paths[path] = (theta_used, M, meta, size, [Treq])
            else:
                # File already loaded, just add this Treq to the list
                seen_paths[path][4].append(Treq)
        except Exception as e:
            print(f"[Warning] Could not load Theta={Treq:.6g}: {e}")
    
    # Convert to list of tasks, one per unique file
    for path, (theta_used, M, meta, size, Treq_list) in seen_paths.items():
        # Use the first Treq as representative, but keep all for tracking
        file_tasks.append((Treq_list[0], theta_used, path, M, meta, size, Treq_list))
    
    # Sort by size (largest first), then by temperature for stability
    file_tasks.sort(key=lambda x: (-x[5], x[0]))
    
    valid_files = [t for t in file_tasks if t[5] > 0]
    print(f"[Pre-scan] Processing {len(valid_files)} unique files (from {len(Thetas_req)} requests), "
          f"size range: {min([t[5] for t in valid_files]) or 0} - "
          f"{max([t[5] for t in valid_files]) or 0}")

    # Process files in parallel (starting with largest matrices)
    def process_single_file(task):
        """Process a single file and return results for all requested temperatures."""
        # Task structure: (Treq_first, theta_used, path, M, meta, size, Treq_list)
        Treq_first, theta_used, path, M, meta, size, Treq_list = task
        if M is None or meta is None:
            return None
            
        Ma, active = get_active_operator(M, meta)
        print(f"[process] Theta={theta_used:.6g}  |  {os.path.basename(path)}  |  shape={Ma.shape}")
        
        # Process once and reuse for all requested temperatures
        sel = select_physical_eigs_per_m(Ma, meta, active, ms, theta_val=theta_used)
        
        # Return results for all requested temperatures (they all map to same file)
        results = []
        for Treq in Treq_list:
            results.append((theta_used, Treq, sel))
        return results
    
    # Parallel processing of files using threading
    # (numba releases GIL, so threading is efficient)
    processed_results = []
    if len(valid_files) > 1 and N_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(process_single_file, task): task for task in file_tasks if task[5] > 0}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        processed_results.extend(result)
                except Exception as e:
                    task = futures[future]
                    print(f"[Error] Processing file {task[2]}: {e}")
    else:
        # Sequential processing
        for task in file_tasks:
            if task[5] > 0:
                result = process_single_file(task)
                if result is not None:
                    processed_results.extend(result)
    
    # Sort results by temperature to maintain order
    processed_results.sort(key=lambda x: x[0])
    
    # Collect results, avoiding duplicates
    seen_temps = set()
    for theta_used, Treq, sel in processed_results:
        if not any(np.isclose(theta_used, t, rtol=0, atol=0) for t in seen_temps):
            for m in ms:
                gamma, v_full = sel[m]
                gammas[m].append(gamma)
                # Store eigenvector (None if gamma is NaN)
                eigenvectors[m].append(v_full if not np.isnan(gamma) else None)
            Ts_used.append(theta_used)
            Ts_req_used.append(Treq)
            seen_temps.add(theta_used)

    Ts = np.array(Ts_used, dtype=np.float64)
    print("Ts=", Ts)
    if Ts.size == 0:
        print("Error: no matrices loaded.")
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

    # --- save data to NPZ for reproducibility ---
    save_dict = {
        "T": Ts,
        "T_requested": np.array(Ts_req_used, dtype=np.float64),
        "modes": np.array(ms, dtype=np.int32),
    }
    # Save gamma_m for each mode m
    for m in ms:
        gm = np.array(gammas[m], dtype=np.float64)
        save_dict[f"gamma_{m}"] = gm

    np.savez(OUT_NPZ, **save_dict)
    print(f"Saved: {OUT_NPZ}")

    # --- plot eigenfunction diagrams in table format (rows = m, columns = T) ---
    plot_eigenfunction_table(Ts, eigenvectors, ms)


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


def plot_eigenfunction_table(Ts, eigenvectors, ms, selected_T_indices=None, max_Ts=5):
    """
    Plot eigenfunction diagrams in a table: rows = m values, columns = selected temperatures.
    
    Args:
        Ts: array of temperatures
        eigenvectors: dict {m: [v_full_1, v_full_2, ...]} where each v_full is for corresponding T
        ms: list of m values to plot
        selected_T_indices: list of indices into Ts to plot (if None, select evenly spaced)
        max_Ts: maximum number of temperatures to plot
    """
    from matplotlib.colors import TwoSlopeNorm
    
    # Select temperatures to plot
    if selected_T_indices is None:
        if len(Ts) <= max_Ts:
            selected_T_indices = list(range(len(Ts)))
        else:
            # Select evenly spaced temperatures
            selected_T_indices = [int(i * (len(Ts) - 1) / (max_Ts - 1)) for i in range(max_Ts)]
    
    selected_Ts = Ts[selected_T_indices]
    print(f"\n[Plotting eigenfunctions] Selected {len(selected_Ts)} temperatures: {selected_Ts}")
    
    # Find valid m values (those with at least one valid eigenvector)
    valid_ms = []
    for m in ms:
        if any(v is not None for v in eigenvectors[m]):
            valid_ms.append(m)
    
    if len(valid_ms) == 0:
        print("[Warning] No valid eigenvectors found for plotting.")
        return
    
    # Load matrices for each selected temperature to get correct grid parameters
    grid_params = {}  # T_idx -> (nx, ny, half, Nmax, dp)
    for T_idx in selected_T_indices:
        try:
            theta_used, path = find_matrix_file(float(Ts[T_idx]))
            with open(path, "rb") as fp:
                M, meta = pickle.load(fp)
            
            # Reconstruct arrays if needed
            if "nx" not in meta or "px" not in meta:
                nx, ny, half, px, py, P, eps, f, w = reconstruct_full_arrays(meta)
                meta["nx"] = nx
                meta["ny"] = ny
                meta["half"] = half
            
            nx = np.asarray(meta["nx"], dtype=np.int32)
            ny = np.asarray(meta["ny"], dtype=np.int32)
            Nmax = int(meta["Nmax"])
            half = int(meta.get("half", Nmax // 2))
            dp = float(meta["dp"])
            
            grid_params[T_idx] = (nx, ny, half, Nmax, dp)
        except Exception as e:
            print(f"[Error] Could not load matrix for T={Ts[T_idx]:.6g}: {e}")
            return
    
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
    
    # Compute global min/max across all eigenmodes for common colorbar
    global_vmax_abs = 0.0
    for row_idx, m in enumerate(valid_ms):
        for col_idx, T_idx in enumerate(selected_T_indices):
            v_full = eigenvectors[m][T_idx]
            if v_full is not None:
                nx_T, ny_T, half_T, Nmax_T, dp_T = grid_params[T_idx]
                grid = values_to_grid(v_full, nx_T, ny_T, half_T, Nmax_T)
                grid_valid = grid[~np.isnan(grid)]
                if grid_valid.size > 0:
                    vmax_abs = float(np.max(np.abs(grid_valid)))
                    global_vmax_abs = max(global_vmax_abs, vmax_abs)
    
    if global_vmax_abs > 0:
        global_vmin = -global_vmax_abs
        global_vmax = global_vmax_abs
    else:
        global_vmin, global_vmax = -0.1, 0.1
    
    global_norm = TwoSlopeNorm(vmin=global_vmin, vcenter=0, vmax=global_vmax)
    
    # Plotting scale factor
    PLOT_SCALE_FACTOR = 1.0
    
    im_common = None
    
    # Fill table: rows = m values, columns = selected temperatures
    for row_idx, m in enumerate(valid_ms):
        for col_idx, T_idx in enumerate(selected_T_indices):
            ax = axes[row_idx, col_idx]
            v_full = eigenvectors[m][T_idx]
            
            if v_full is not None:
                # Use grid parameters for this specific temperature
                nx_T, ny_T, half_T, Nmax_T, dp_T = grid_params[T_idx]
                grid = values_to_grid(v_full, nx_T, ny_T, half_T, Nmax_T)
                
                # Create coordinates for this grid
                X, Y = create_pcolormesh_coords(half_T, dp_T, Nmax_T)
                
                # Compute plot limits for this grid
                pmin = (-half_T) * dp_T
                pmax = (half_T - 1) * dp_T
                pcenter = (pmin + pmax) / 2.0
                prange = (pmax - pmin) / 2.0
                pmin_plot = pcenter - prange * PLOT_SCALE_FACTOR
                pmax_plot = pcenter + prange * PLOT_SCALE_FACTOR
                
                # Use pcolormesh for smoother ring visualization
                im = ax.pcolormesh(X, Y, grid.T, cmap="seismic", norm=global_norm, 
                                  shading='flat', rasterized=True)
                im_common = im
                
                ax.set_xlim(pmin_plot, pmax_plot)
                ax.set_ylim(pmin_plot, pmax_plot)
                
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
