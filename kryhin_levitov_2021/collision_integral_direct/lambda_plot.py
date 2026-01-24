#!/usr/bin/env python3
import os
import pickle
import csv
import numpy as np
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from scipy.sparse import csr_matrix, isspmatrix_csr, diags
from scipy.sparse.linalg import eigsh, LinearOperator

USE_NUMBA = False
try:
    import numba
    from numba import njit, prange
    USE_NUMBA = True
    numba_threads = int(os.environ.get('NUMBA_NUM_THREADS', multiprocessing.cpu_count() or 4))
    if hasattr(numba, 'set_num_threads'):
        numba.set_num_threads(numba_threads)
    print(f"[Numba] Using {numba_threads} threads for parallel execution")
except ImportError:
    USE_NUMBA = False
    print("[Numba] Not available - install numba for better performance")

N_WORKERS = int(os.environ.get('N_WORKERS', multiprocessing.cpu_count() or 4))
print(f"[Parallel] Using {N_WORKERS} workers for temperature processing")
# plt.rcParams['text.usetex'] = True
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

ms = list(range(9))

OUT_PNG = "Eigenvals_bruteforce_generalized.png"
OUT_SVG = "Eigenvals_bruteforce_generalized.svg"
OUT_CSV = "Eigenvals_bruteforce_generalized.csv"
OUT_EIGENVECTORS_NPZ = "Eigenvectors_bruteforce_generalized.npz"

BATCH_SIZE = 5

USE_HARMONIC_PROJECTION = True
N_EIG_CANDIDATES = 120
ZERO_TOL = 1e-10
INCLUDE_CONSERVED = True

INV_MIN_CORR = 0.75
M_TOL = 0.40

RING_WIDTH_FACTOR = 2.5
MIN_RING_POINTS = 200
N_ANGLE_BINS_MIN = 256

PLOT_PWIN = 1.6
PLOT_RING_ONLY = True
PLOT_RING_W = None

DUMP_SINGLE_EIGENFUNCTIONS = True
DUMP_WHITELIST = []
SINGLE_OUTDIR = "eigenfunctions_single"

ENFORCE_SYMMETRY = True
SEPARATE_M01 = True


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
        n = v.size
        num = 0.0
        den = 0.0
        for i in prange(n):
            wv = w[i] * v[i]
            num += wv * u[i]
            den += wv * v[i]
        return num / (den + 1e-30)

def w_corr(v: np.ndarray, u: np.ndarray, w: np.ndarray) -> float:
    if USE_NUMBA:
        return w_corr_numba(v, u, w)
    num = float(np.dot(v, w * u))
    den = float(np.dot(v, w * v)) + 1e-30
    return num / den


if USE_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def estimate_sign_switches_on_ring_numba(v_full, active, px, py, P, dp_val, 
                                             ring_width_factor, min_ring_points, n_angle_bins_min):
        ring_w = ring_width_factor * dp_val
        n_active = active.size
        
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
        
        ang = np.zeros(idx_count, dtype=np.float64)
        for i in prange(idx_count):
            a = np.arctan2(py[idx_list[i]], px[idx_list[i]])
            ang[i] = (a + 2.0 * np.pi) % (2.0 * np.pi)
        
        nbin = max(n_angle_bins_min, int((2.0 * np.pi / max(dp_val, 1e-12)) * 2))
        bins = np.floor(ang / (2.0 * np.pi) * nbin).astype(np.int64)
        
        prof = np.zeros(nbin, dtype=np.float64)
        cnt  = np.zeros(nbin, dtype=np.int64)
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

    if not np.all(has):
        valid_indices = np.where(has)[0]
        if valid_indices.size > 0:
            valid_values = prof[valid_indices]
            all_indices = np.arange(nbin)
            prof = np.interp(all_indices, valid_indices, valid_values)
        else:
            return 0, 0.0

    prof_original = prof.copy()
    
    prof_std = np.std(prof)
    prof_mean = np.mean(prof)
    prof_abs_max = np.max(np.abs(prof - prof_mean))
    
    if abs(prof_mean) > 1e-10:
        rel_std = prof_std / abs(prof_mean)
    else:
        rel_std = prof_std
    
    if prof_abs_max < 1e-8 or rel_std < 0.01:
        s = np.sign(prof - prof_mean)
        switches = 0
        for i in range(nbin):
            a = s[i]
            b = s[(i + 1) % nbin]
            if a * b < 0.0:
                switches += 1
        return switches, 0.0

    prof_centered = prof - prof_mean
    
    prof_max = np.max(np.abs(prof_centered))
    if prof_max < 1e-10:
        switches = 0
        return switches, 0.0
    prof = prof_centered / prof_max

    fft_result = np.fft.fft(prof_original)
    fft_power = np.abs(fft_result)
    
    dc_power = fft_power[0]
    
    max_freq = min(nbin // 2, 50)
    if max_freq < 1:
        return 0, 0.0
    
    freq_range = fft_power[1:max_freq+1]
    if len(freq_range) == 0:
        switches = 0
        return switches, 0.0
    
    dominant_idx = np.argmax(freq_range) + 1
    dominant_power = fft_power[dominant_idx]
    
    if dc_power > 0.5 * dominant_power and dc_power > np.sum(fft_power[1:]) * 0.3:
        switches = 0
        return switches, 0.0
    
    m_est = float(dominant_idx)
    
    if dominant_idx > 1:
        prev_power = fft_power[dominant_idx - 1]
        if prev_power > 0.7 * dominant_power and prev_power > dominant_power * 0.9:
            m_est = float(dominant_idx - 1)
    
    if dominant_idx < max_freq:
        next_power = fft_power[dominant_idx + 1]
        if next_power > 1.1 * dominant_power:
            m_est = float(dominant_idx + 1)
    
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
    
    if best_corr > 0.2:
        m_est = float(best_m)
    
    s = np.sign(prof)
    switches = 0
    for i in range(nbin):
        a = s[i]
        b = s[(i + 1) % nbin]
        if a * b < 0.0:
            switches += 1

    return int(switches), float(m_est)


_file_cache = None

def find_matrix_file(theta_req: float):
    global _file_cache
    
    if not os.path.isdir(IN_DIR):
        raise FileNotFoundError(f"Directory not found: {IN_DIR}")

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


def build_centered_lattice(Nmax: int):
    half = Nmax // 2
    ns = np.arange(-half, half, dtype=np.int32)
    nx, ny = np.meshgrid(ns, ns, indexing="ij")
    return nx.reshape(-1), ny.reshape(-1), half


def reconstruct_full_arrays(meta):
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
    active = meta.get("active", None)
    if active is None or len(active) == 0:
        Nstates = int(meta["nx"].size)
        active = np.arange(Nstates, dtype=np.int32)
    else:
        active = np.asarray(active, dtype=np.int32)

    Nactive = int(active.size)

    if bool(meta.get("active_only", False)) and getattr(M, "shape", None) == (Nactive, Nactive):
        Ma = _as_csr(M)
        return Ma, active

    Ma = _as_csr(M[np.ix_(active, active)])
    return Ma, active


def _w_inner(a, b, w):
    return float(np.dot(a, w * b))


def _w_orthonormalize(vecs, w):
    out = []
    for v in vecs:
        v = v.astype(np.float64, copy=True)
        for q in out:
            v -= _w_inner(q, v, w) * q
        n2 = _w_inner(v, v, w)
        if n2 > 1e-30:
            out.append(v / np.sqrt(n2))
    return out


def enforce_w_self_adjoint(A, W, enforce=True):
    if hasattr(W, 'diagonal'):
        w_diag = W.diagonal()
    else:
        w_diag = np.asarray(W.sum(axis=0)).flatten()
    
    w_safe = np.where(w_diag > 0, w_diag, 1e-30)
    d = (1.0 / np.sqrt(w_safe)).astype(np.float64)
    
    if not enforce:
        return A, d
    
    n = A.shape[0]
    
    if n < 10000:
        test_vec = np.random.randn(n).astype(np.float64)
        Wtest = w_safe * test_vec
        AWtest = A.dot(Wtest)
        ATtest = A.T.dot(test_vec)
        WATtest = w_safe * ATtest
        defect = np.linalg.norm(AWtest - WATtest) / (np.linalg.norm(AWtest) + 1e-30)
        
        if defect > 1e-8:
            A_dense = A.toarray()
            W_dense = np.diag(w_safe)
            W_inv = np.diag(1.0 / w_safe)
            A_sym_dense = 0.5 * (A_dense + W_inv @ A_dense.T @ W_dense)
            A_sym = csr_matrix(A_sym_dense)
            return A_sym, d
    
    return A, d


def projected_gamma(A, w_safe, inv_orth, cand_cols, return_basis=False):
    """
    Compute gamma from a set of candidate columns by:
    1. Removing invariants
    2. W-orthonormalizing (so U^T W U = I)
    3. Solving standard symmetric eigenproblem C = U^T A U
    
    Returns:
        gamma_min: minimum eigenvalue
        u_min: full-space eigenfunction (active space)
        U: basis matrix (if return_basis=True)
        alpha: eigenvector coefficients in basis (if return_basis=True)
    """
    # Remove invariants
    cols = []
    for u in cand_cols:
        u = u.astype(np.float64, copy=True)
        for q in inv_orth:
            u -= _w_inner(q, u, w_safe) * q
        cols.append(u)

    # W-orthonormalize => U^T W U = I
    U_list = _w_orthonormalize(cols, w_safe)
    if len(U_list) == 0:
        if return_basis:
            return np.nan, None, None, None
        return np.nan, None

    U = np.column_stack(U_list)  # n x r
    AU = A.dot(U)                # n x r
    C = U.T @ AU                 # r x r

    # Defensive symmetrization (should be symmetric if A is correct)
    C = 0.5 * (C + C.T)

    evals, evecs = np.linalg.eigh(C)
    evals = evals[np.isfinite(evals)]
    valid_mask = np.isfinite(evals) & (evals >= -1e-12)
    if not np.any(valid_mask):
        if return_basis:
            return np.nan, None, None, None
        return np.nan, None
    
    evals = evals[valid_mask]
    evecs = evecs[:, valid_mask]
    
    gamma_min = float(np.min(np.maximum(evals, 0.0)))
    
    # Recover the minimizing eigenvector for diagnostics
    j = np.argmin(evals)
    alpha = evecs[:, j]  # eigenvector coefficients in basis
    u_min = U @ alpha  # full-space eigenfunction (active space)
    
    # Normalize in W-norm
    wu = w_safe * u_min
    norm_w = np.sqrt(np.dot(u_min, wu))
    if norm_w > 1e-30:
        u_min = u_min / norm_w
        alpha = alpha / norm_w
    
    if return_basis:
        return gamma_min, u_min, U, alpha
    return gamma_min, u_min


def projected_ritz_pairs(A, w_safe, inv_orth, cand_cols, n_pairs=12):
    """
    Like projected_gamma, but returns several lowest Ritz pairs (gamma, u_min).
    Returns list of (gamma, u) tuples.
    """
    # Remove invariants
    cols = []
    for u in cand_cols:
        u = u.astype(np.float64, copy=True)
        for q in inv_orth:
            u -= _w_inner(q, u, w_safe) * q
        cols.append(u)

    # W-orthonormalize
    U_list = _w_orthonormalize(cols, w_safe)
    if len(U_list) == 0:
        return []

    U = np.column_stack(U_list)          # n x r
    C = U.T @ (A.dot(U))                 # r x r
    C = 0.5 * (C + C.T)

    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    out = []
    for j in range(min(n_pairs, len(evals))):
        lam = float(evals[j])
        if not np.isfinite(lam):
            continue
        gamma = max(0.0, lam)

        alpha = evecs[:, j]
        u = U @ alpha

        # Normalize in W-norm
        wu = w_safe * u
        nrm = np.sqrt(np.dot(u, wu))
        if nrm > 1e-30:
            u = u / nrm
        out.append((gamma, u))
    return out


def odd_sector_candidates(A, w_safe, inv_orth, theta, z, Kodd=25, n_pairs=12):
    """
    Build odd-sector basis and return several lowest Ritz pairs.
    """
    eps = 1e-3
    radial = [
        np.ones_like(z),
        z,
        z * np.log(np.abs(z) + eps),
        z * z,
    ]

    cols = []
    for k in range(1, Kodd + 1, 2):
        ck = np.cos(k * theta)
        sk = np.sin(k * theta)
        for r in radial:
            cols.append(r * ck)
            cols.append(r * sk)

    return projected_ritz_pairs(A, w_safe, inv_orth, cols, n_pairs=n_pairs)


def compute_fft_harmonic_spectrum(u_min, theta, P_act, w_safe, dp_val, Theta, max_harmonic=15):
    """
    Compute FFT harmonic spectrum of u_min on the Fermi ring.
    Returns dictionary with harmonic weights for m=1,3,5,...,max_harmonic.
    """
    # Build angular profile on the ring
    ring_mask = np.abs(P_act - 1.0) <= max(3.0 * dp_val, 2.0 * Theta)
    if np.count_nonzero(ring_mask) < 16:
        ring_mask = np.ones(len(u_min), dtype=bool)
    
    if np.count_nonzero(ring_mask) < 16:
        return {}
    
    theta_ring = theta[ring_mask]
    u_ring = u_min[ring_mask]
    w_ring = w_safe[ring_mask]
    
    # Sort by angle for FFT
    sort_idx = np.argsort(theta_ring)
    theta_sorted = theta_ring[sort_idx]
    u_sorted = u_ring[sort_idx]
    w_sorted = w_ring[sort_idx]
    
    # Bin into uniform angular grid for FFT
    n_bins = max(64, min(256, len(u_sorted)))
    # Fix binning: use floor instead of digitize to avoid shift
    bin_idx = np.floor((theta_sorted % (2.0 * np.pi)) / (2.0 * np.pi) * n_bins).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    # Weighted average in each bin
    u_prof = np.zeros(n_bins, dtype=np.float64)
    w_prof = np.zeros(n_bins, dtype=np.float64)
    for i, b in enumerate(bin_idx):
        u_prof[b] += w_sorted[i] * u_sorted[i]
        w_prof[b] += w_sorted[i]
    
    # Normalize
    mask_prof = w_prof > 1e-30
    if np.count_nonzero(mask_prof) < 8:
        return {}
    
    u_prof[mask_prof] /= w_prof[mask_prof]
    u_prof[~mask_prof] = 0.0
    
    # FFT to get harmonic content
    u_fft = np.fft.fft(u_prof)
    power = np.abs(u_fft) ** 2
    
    # Extract harmonic weights: for each m, sum power at +m and -m
    spectrum = {}
    for m in range(1, min(max_harmonic + 1, n_bins // 2 + 1)):
        if m < n_bins:
            power_m = power[m] + power[n_bins - m] if m > 0 else power[0]
            spectrum[m] = power_m
    
    return spectrum


def harmonic_spectrum_on_fs(u, theta, P_act, w_safe, dp_val, Theta,
                           max_harmonic=31, radial_parity="even"):
    """
    Robust angular harmonic spectrum near the Fermi surface.
    
    Args:
        u: eigenfunction vector
        theta: angular coordinates
        P_act: momentum magnitude
        w_safe: weights
        dp_val: momentum grid spacing
        Theta: temperature
        max_harmonic: maximum harmonic to compute
        radial_parity: "even" (use u as-is) or "odd" (multiply by sign(P-1) so inner/outer add)
    
    Returns:
        dict mapping harmonic m to power
    """
    two_pi = 2.0 * np.pi

    # Smooth radial window around P=1
    dr = P_act - 1.0
    sigma = max(1.5 * dp_val, 0.5 * Theta, 1e-6)
    wr = np.exp(-(dr / sigma) ** 2)

    if radial_parity == "odd":
        u_eff = u * np.sign(dr + 1e-300)   # avoid exact zeros
    else:
        u_eff = u

    weights = w_safe * wr

    # Choose FFT grid (power of 2 for efficiency)
    n_bins = int(2 ** np.ceil(np.log2(max(128, min(1024, len(u_eff))))))
    phi = np.mod(theta, two_pi)
    bins = (phi / two_pi * n_bins).astype(np.int64)
    bins = np.clip(bins, 0, n_bins - 1)

    u_prof = np.zeros(n_bins, dtype=np.float64)
    w_prof = np.zeros(n_bins, dtype=np.float64)

    # Use np.add.at for safe accumulation
    np.add.at(u_prof, bins, weights * u_eff)
    np.add.at(w_prof, bins, weights)

    mask = w_prof > 1e-30
    if np.count_nonzero(mask) < 8:
        return {}

    u_prof[mask] /= w_prof[mask]

    # Fill empty bins by interpolation (prevents FFT garbage)
    if not np.all(mask):
        idx = np.where(mask)[0]
        if len(idx) >= 2:
            u_prof = np.interp(np.arange(n_bins), idx, u_prof[idx])
        else:
            return {}

    # Remove DC
    u_prof -= np.mean(u_prof)

    # rFFT power spectrum
    U = np.fft.rfft(u_prof)
    power = np.abs(U) ** 2

    spectrum = {}
    mmax = min(max_harmonic, len(power) - 1)
    for m in range(1, mmax + 1):
        spectrum[m] = float(power[m])
    return spectrum


def harmonic_powers_direct(u, theta, P_act, w_safe, dp_val, Theta, mmax=31, radial_parity="odd"):
    """
    Robust harmonic content without FFT/binning.
    radial_parity="odd" applies sign(P-1) to undo inner/outer mirroring.
    """
    dr = P_act - 1.0
    sigma = max(1.5 * dp_val, 0.5 * Theta, 1e-6)
    wr = np.exp(-(dr / sigma) ** 2)          # radial window near FS
    w = w_safe * wr

    if radial_parity == "odd":
        u_eff = u * np.sign(dr + 1e-300)
    else:
        u_eff = u

    # Remove weighted mean (kills DC leakage)
    wsum = np.sum(w)
    if wsum < 1e-30:
        return {}
    u_eff = u_eff - (np.sum(w * u_eff) / wsum)

    powers = {}
    for m in range(1, mmax + 1):
        c = np.cos(m * theta)
        s = np.sin(m * theta)
        a = float(np.dot(w * u_eff, c))
        b = float(np.dot(w * u_eff, s))
        powers[m] = a * a + b * b
    return powers


def m_harmonic_powers_direct(u, theta, P_act, w_safe, dp_val, Theta, mmax=31, apply_sign=False):
    """
    Direct weighted Fourier powers (no binning, no FFT).
    If apply_sign=True, multiply u by sign(P-1) (energy-odd compensation).
    """
    dr = P_act - 1.0
    # Smooth radial window centered at FS
    sigma = max(2.0 * dp_val, 0.75 * Theta, 1e-6)
    wr = np.exp(-(dr / sigma) ** 2)
    w = w_safe * wr

    if apply_sign:
        u_eff = u * np.sign(dr + 1e-300)
    else:
        u_eff = u

    wsum = float(np.sum(w))
    if wsum < 1e-30:
        return {}

    # Remove weighted mean to reduce DC leakage
    u_eff = u_eff - float(np.sum(w * u_eff)) / wsum

    powers = {}
    for m in range(1, mmax + 1):
        c = np.cos(m * theta)
        s = np.sin(m * theta)
        a = float(np.dot(w * u_eff, c))
        b = float(np.dot(w * u_eff, s))
        powers[m] = a * a + b * b
    return powers


def pick_best_parity_for_m1(u, theta, P_act, w_safe, dp_val, Theta, mmax=31):
    """
    Returns (apply_sign, purity1, powers) where apply_sign chooses the better parity:
      apply_sign=False -> energy-even
      apply_sign=True  -> energy-odd (inside/outside mirrored)
    purity1 = P1 / sum_{odd} Pm
    """
    pw_even = m_harmonic_powers_direct(u, theta, P_act, w_safe, dp_val, Theta, mmax=mmax, apply_sign=False)
    pw_odd  = m_harmonic_powers_direct(u, theta, P_act, w_safe, dp_val, Theta, mmax=mmax, apply_sign=True)

    def purity(pw):
        odd_total = sum(pw.get(k, 0.0) for k in range(1, mmax + 1, 2)) + 1e-30
        return pw.get(1, 0.0) / odd_total

    pur_even = purity(pw_even)
    pur_odd  = purity(pw_odd)

    if pur_odd > pur_even:
        return True, pur_odd, pw_odd
    else:
        return False, pur_even, pw_even


def project_component_on_ring(v_full, grid_meta, T, m, phase_fix=False):
    """
    Return a FULL-LATTICE vector that contains only the m-th angular harmonic
    on the Fermi ring (smooth radial window). This is for CLEAN plotting / robust m-ID.

    phase_fix=True removes the arbitrary rotation by forcing the complex coefficient real+.
    For m=1 this makes the node line fixed (horizontal, like cos(theta)).
    """
    import math
    
    Nmax = int(grid_meta["Nmax"])
    dp = float(grid_meta["dp"])
    Theta = float(T)  # T is already Theta
    shift_x = float(grid_meta.get("shift_x", 0.0))
    shift_y = float(grid_meta.get("shift_y", 0.0))
    
    # Reconstruct arrays
    nx, ny, half = build_centered_lattice(Nmax)
    px = dp * (nx.astype(np.float64) + shift_x)
    py = dp * (ny.astype(np.float64) + shift_y)
    P = np.sqrt(px * px + py * py)
    
    # Compute weights
    eps = P * P
    invT = 1.0 / Theta
    em = math.exp(-invT)
    a = 1.0 - em
    x = np.clip((eps - 1.0) * invT, -700.0, 700.0)
    f = a / (np.exp(x) + a)
    w_full = np.clip(f * (1.0 - f), 0.0, None)
    
    # Get active indices (where weight is significant)
    active = np.where(w_full > 1e-30)[0]
    if len(active) == 0:
        return np.zeros_like(v_full, dtype=np.float64)
    
    idx = active
    theta = np.arctan2(py[idx], px[idx])
    P_act = P[idx]
    
    # Smooth radial window around P=1
    dr = P_act - 1.0
    sigma = max(2.0 * dp, 0.75 * Theta, 1e-6)
    wr = np.exp(-(dr / sigma) ** 2)
    
    w = w_full[idx] * wr
    u = v_full[idx].astype(np.float64, copy=False)
    
    wsum = float(np.sum(w))
    out = np.zeros_like(v_full, dtype=np.float64)
    if wsum < 1e-30:
        return out
    
    # Remove weighted mean (so m=0 doesn't leak into m=1 etc.)
    u = u - float(np.sum(w * u)) / wsum
    
    if m == 0:
        a0 = float(np.sum(w * u)) / wsum
        out[idx] = a0
        return out
    
    # Complex Fourier coefficient for harmonic m (rotation-invariant magnitude)
    c = np.sum(w * u * np.exp(-1j * m * theta)) / wsum
    
    if phase_fix:
        # Kill the arbitrary rotation (phase): enforce coefficient real positive
        c = abs(c)
    
    # Reconstruct only the m-harmonic
    out[idx] = np.real(c * np.exp(1j * m * theta))
    return out


def gamma_m1_k1_adaptive(A, w_safe, inv_orth, theta, z, P_act, dp_val, Theta):
    """
    Adaptive k=1-only solver that tries both energy-even and energy-odd and picks the best.
    Forces dipole (k=1) structure while allowing parity to adapt with temperature.
    """
    eps = 1e-12

    # Smooth window (stabilizes low-T and avoids off-shell junk)
    dr = P_act - 1.0
    sigma = max(2.0 * dp_val, 0.75 * Theta, 1e-6)
    wr = np.exp(-(dr / sigma) ** 2)

    c1 = np.cos(theta)
    s1 = np.sin(theta)
    sgn = np.sign(dr + 1e-300)

    # Radial bases (include both even-ish and odd-ish pieces)
    radial = [
        np.ones_like(z),
        z,
        z * np.log(np.abs(z) + eps),
        z * z,
    ]

    def solve(apply_sign):
        cols = []
        for r in radial:
            rr = wr * r
            if apply_sign:
                rr = rr * sgn
            cols.append(rr * c1)
            cols.append(rr * s1)
        return projected_gamma(A, w_safe, inv_orth, cols)

    gamma_even, u_even = solve(apply_sign=False)
    gamma_odd,  u_odd  = solve(apply_sign=True)

    # Choose the one that is most m=1-like (purity test)
    best = None
    for gamma, u, apply_sign in [(gamma_even, u_even, False), (gamma_odd, u_odd, True)]:
        if u is None or not np.isfinite(gamma):
            continue
        apply_sign2, pur1, _ = pick_best_parity_for_m1(u, theta, P_act, w_safe, dp_val, Theta, mmax=31)
        # we want m=1 to be strong; allow either internal parity, so use pur1 directly
        score = pur1
        if best is None or score > best[0]:
            best = (score, gamma, u, apply_sign)

    if best is None:
        return np.nan, None

    _, gamma_min, u_min, _ = best
    return gamma_min, u_min


def gamma_m1_targeted(A, w_safe, inv_orth, theta, z, gcol=None, gant=None):
    """
    Target the dipole-like odd mode:
      - angular k=1 only (cos(theta), sin(theta))
      - energy-odd radial basis (z changes sign across FS)
      - optionally include collinear logs to capture structure
    """
    eps = 1e-12

    # IMPORTANT: do NOT include "1" radial factor, it is momentum-like on the FS and causes numerical pathology.
    radial = [
        z,
        z * np.log(np.abs(z) + eps),
        z**3,
    ]
    if gcol is not None:
        radial.append(z * (gcol - np.mean(gcol)))
    if gant is not None:
        radial.append(z * (gant - np.mean(gant)))

    cols = []
    c1 = np.cos(theta)
    s1 = np.sin(theta)
    for r in radial:
        cols.append(r * c1)
        cols.append(r * s1)

    return projected_gamma(A, w_safe, inv_orth, cols)


def gamma_odd_sector(A, w_safe, inv_orth, theta, z, Kodd=25, return_basis=False):
    """
    Compute the minimum eigenvalue in the odd-parity sector (KL's "m=1" slow odd mode).
    Uses a multi-odd-harmonic Galerkin subspace: k=1,3,5,...,Kodd with radial basis.
    """
    # Radial basis
    eps = 1e-3
    radial = [
        np.ones_like(z),
        z,
        z * np.log(np.abs(z) + eps),
        z * z,
    ]

    cols = []
    for k in range(1, Kodd + 1, 2):  # odd harmonics only: 1, 3, 5, ..., Kodd
        ck = np.cos(k * theta)
        sk = np.sin(k * theta)
        for r in radial:
            cols.append(r * ck)
            cols.append(r * sk)

    if return_basis:
        gamma, u_min, U, alpha = projected_gamma(A, w_safe, inv_orth, cols, return_basis=True)
        return gamma, u_min, U, alpha
    else:
        gamma, u_min = projected_gamma(A, w_safe, inv_orth, cols, return_basis=False)
        return gamma, u_min


def gammas_by_harmonic_projection(Ma, meta, active, ms, ring_only=False):
    nx, ny, half, px, py, P, eps, f, w_full = reconstruct_full_arrays(meta)
    w_act = np.clip(w_full[active], 0.0, None)
    w_safe = np.where(w_act > 0, w_act, 1e-30)

    A = _as_csr(-Ma)
    px_act = px[active]
    py_act = py[active]
    P_act = P[active]

    # Energy coordinate for radial basis (compute once)
    Theta = float(meta.get("Theta", 0.0))

    if ring_only:
        dp_val = float(meta["dp"])
        # Tight ring restriction: avoid off-shell garbage at low T
        # Use a few dp, not 2*Theta which becomes too thick at low T
        rw = max(RING_WIDTH_FACTOR * dp_val, min(2.0 * Theta, 12.0 * dp_val))
        mask = np.abs(P_act - 1.0) <= rw
        if np.count_nonzero(mask) > 32:
            idx = np.where(mask)[0]
            A = A[idx, :][:, idx]
            w_safe = w_safe[idx]
            px_act = px_act[idx]
            py_act = py_act[idx]
            P_act = P_act[idx]

    theta = np.arctan2(py_act, px_act)

    # Angular-log factors for collinear scattering structure (odd m modes)
    # These capture the log-like angular dependence from near-collinear scattering
    dp_val = float(meta.get("dp", 0.0))
    delta = max(2.0 * dp_val, 8.0 * Theta, 1e-6)
    gcol = np.log(1.0 / (np.abs(np.sin(0.5 * theta)) + delta))
    gant = np.log(1.0 / (np.abs(np.cos(0.5 * theta)) + delta))  # optional, catches other collinear set

    inv_vecs = [
        np.ones_like(theta),
        px_act.copy(),
        py_act.copy(),
    ]
    inv_orth = _w_orthonormalize(inv_vecs, w_safe)

    # Energy coordinate for radial basis
    eps_act = eps[active]
    if ring_only:
        # If we already restricted by idx, recompute eps_act consistently from current px_act, py_act
        eps_act = px_act * px_act + py_act * py_act

    # Energy deviation from Fermi surface
    de = eps_act - 1.0
    
    # Scale so numbers are O(1)
    Escale = max(Theta, dp_val * dp_val, 1e-6)
    z = de / Escale

    out = {}
    eigenvectors_out = {}
    Nstates = int(nx.size)  # Total number of states in full lattice
    
    for m in ms:
        if m == 0:
            out[m] = 0.0
            eigenvectors_out[m] = None
            continue

        if m == 1:
            # Adaptive k=1-only solver: tries both energy-even and energy-odd, picks the best
            gamma_min, u_min = gamma_m1_k1_adaptive(A, w_safe, inv_orth, theta, z, P_act, dp_val, Theta)
            
            # Diagnostics: use parity-adaptive harmonic measurement
            if u_min is not None:
                apply_sign, pur1, pw = pick_best_parity_for_m1(u_min, theta, P_act, w_safe, dp_val, Theta, mmax=31)
                
                # Report top odd harmonics
                odd = [(k, pw.get(k, 0.0)) for k in range(1, 16, 2)]
                odd.sort(key=lambda x: -x[1])
                tot = sum(pw.values()) + 1e-30
                top5 = ", ".join([f"m={k}:{v/tot:.3f}" for k, v in odd[:5]])
                print(f"    [m=1] parity={'odd' if apply_sign else 'even'}  purity1={pur1:.3f} (gamma={gamma_min:.6e})")
                print(f"    [m=1] direct harmonic spectrum (top 5 odd): {top5}")
                
                # Optional: gauge-fix orientation to prevent random sign flips
                a1 = float(np.dot(w_safe * u_min, np.cos(theta)))
                b1 = float(np.dot(w_safe * u_min, np.sin(theta)))
                # Pick a convention: make the larger component positive
                if abs(a1) >= abs(b1):
                    if a1 < 0:
                        u_min *= -1
                else:
                    if b1 < 0:
                        u_min *= -1
            else:
                print(f"    [m=1] No eigenvector (gamma={gamma_min:.6e})")
            
            # Store eigenvector in full space
            if u_min is not None:
                v_full = np.zeros(Nstates, dtype=np.float64)
                v_full[active] = u_min
                eigenvectors_out[m] = v_full
            else:
                eigenvectors_out[m] = None

            out[m] = gamma_min
            continue

        # For m >= 2: use simple angular basis (can enrich with radial later if needed)
        c = np.cos(m * theta)
        s = np.sin(m * theta)
        cand = [c, s]

        # Compute gamma using stable projection method
        gamma_min, u_min = projected_gamma(A, w_safe, inv_orth, cand)
        
        # Sanity check: print norms of first two candidate columns AFTER projection
        # (projected_gamma already does the projection, so we need to check manually)
        if len(cand) >= 2:
            # Manually project first two to check norms
            v0 = cand[0].astype(np.float64, copy=True)
            v1 = cand[1].astype(np.float64, copy=True)
            for q in inv_orth:
                v0 -= _w_inner(q, v0, w_safe) * q
                v1 -= _w_inner(q, v1, w_safe) * q
            norm0 = np.sqrt(_w_inner(v0, v0, w_safe))
            norm1 = np.sqrt(_w_inner(v1, v1, w_safe))
            print(f"    [m={m}] Basis norms after projection: [{norm0:.6e}, {norm1:.6e}]")
        
        # Diagnostic: check angular harmonic purity using FFT-based harmonic content
        if u_min is not None:
            # Build angular profile on the ring: bin u_min by angle
            # Use ring points (weighted by w_safe to focus on Fermi surface)
            ring_mask = np.abs(P_act - 1.0) <= max(3.0 * dp_val, 2.0 * Theta)
            if np.count_nonzero(ring_mask) < 16:
                # If ring is too sparse, use all points
                ring_mask = np.ones(len(u_min), dtype=bool)
            
            if np.count_nonzero(ring_mask) >= 16:
                theta_ring = theta[ring_mask]
                u_ring = u_min[ring_mask]
                w_ring = w_safe[ring_mask]
                
                # Sort by angle for FFT
                sort_idx = np.argsort(theta_ring)
                theta_sorted = theta_ring[sort_idx]
                u_sorted = u_ring[sort_idx]
                w_sorted = w_ring[sort_idx]
                
                # Bin into uniform angular grid for FFT
                n_bins = max(64, min(256, len(u_sorted)))
                phi_bins = np.linspace(0, 2.0 * np.pi, n_bins, endpoint=False)
                bin_idx = np.digitize(theta_sorted, phi_bins) % n_bins
                
                # Weighted average in each bin
                u_prof = np.zeros(n_bins, dtype=np.float64)
                w_prof = np.zeros(n_bins, dtype=np.float64)
                for i, b in enumerate(bin_idx):
                    u_prof[b] += w_sorted[i] * u_sorted[i]
                    w_prof[b] += w_sorted[i]
                
                # Normalize
                mask_prof = w_prof > 1e-30
                if np.count_nonzero(mask_prof) >= 8:
                    u_prof[mask_prof] /= w_prof[mask_prof]
                    u_prof[~mask_prof] = 0.0
                    
                    # FFT to get harmonic content
                    u_fft = np.fft.fft(u_prof)
                    power = np.abs(u_fft) ** 2
                    
                    # Compute purity: power at harmonic m (sum of +m and -m) relative to total power (excluding DC)
                    k_max = min(n_bins // 2, 20)  # Check up to harmonic 20
                    if k_max >= m and m > 0:
                        # For harmonic m, sum power at +m and -m (for real signals, these are equal)
                        power_m = power[m] + power[n_bins - m] if m < n_bins else power[m]
                        # Total power: sum over all harmonics (excluding DC)
                        power_total = np.sum(power[1:k_max+1]) + np.sum(power[n_bins-k_max:n_bins]) if k_max < n_bins // 2 else np.sum(power[1:])
                        
                        if power_total > 1e-30:
                            purity = power_m / power_total
                            
                            if purity < 0.2:
                                print(f"    [m={m}] WARNING: Low FFT purity {purity:.3f} (gamma={gamma_min:.6e}) - "
                                      f"subspace minimum may not be the intended m={m} harmonic")
                            else:
                                print(f"    [m={m}] FFT harmonic purity: {purity:.3f} (gamma={gamma_min:.6e})")
                        else:
                            print(f"    [m={m}] Could not compute FFT purity (no power), gamma={gamma_min:.6e}")
                    elif m == 0:
                        # DC component
                        power_total = np.sum(power[1:])
                        if power_total > 1e-30:
                            purity = power[0] / (power[0] + power_total)
                            print(f"    [m={m}] FFT DC purity: {purity:.3f} (gamma={gamma_min:.6e})")
                        else:
                            print(f"    [m={m}] Could not compute FFT purity (no power), gamma={gamma_min:.6e}")
                    else:
                        print(f"    [m={m}] Could not compute FFT purity (m={m} > k_max={k_max}), gamma={gamma_min:.6e}")
                else:
                    print(f"    [m={m}] Could not compute FFT purity (insufficient ring points), gamma={gamma_min:.6e}")
            else:
                print(f"    [m={m}] Could not compute FFT purity (ring too sparse), gamma={gamma_min:.6e}")
        else:
            print(f"    [m={m}] No valid eigenvector, gamma={gamma_min:.6e}")
        
        # Store eigenvector in full space
        if u_min is not None:
            v_full = np.zeros(Nstates, dtype=np.float64)
            v_full[active] = u_min
            eigenvectors_out[m] = v_full
        else:
            eigenvectors_out[m] = None
        
        out[m] = gamma_min

    return out, eigenvectors_out


def select_physical_eigs_per_m(Ma: csr_matrix, meta, active: np.ndarray, ms, theta_val: float, progress_callback=None):
    import time
    
    if "nx" not in meta or "px" not in meta:
        if progress_callback:
            progress_callback("    Setting up arrays...")
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

    f_full = np.asarray(meta["f"], dtype=np.float64)
    w_full = np.clip(f_full * (1.0 - f_full), 0.0, None)

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

    w_act = np.clip(w_full[active], 0.0, None)
    w_eps = 1e-30
    w_safe = np.where(w_act > 0.0, w_act, w_eps)
    W = diags(w_safe, 0, format="csr")

    A = _as_csr(-Ma)
    
    if ENFORCE_SYMMETRY:
        if progress_callback:
            progress_callback("    Enforcing W-self-adjointness (symmetrizing operator)...")
        A, d = enforce_w_self_adjoint(A, W, enforce=True)
    else:
        d = (1.0 / np.sqrt(w_safe)).astype(np.float64)

    def _matvec_B(x):
        y = d * x
        z = A.dot(y)
        return d * z

    n = A.shape[0]
    Bop = LinearOperator((n, n), matvec=_matvec_B, dtype=np.float64)

    sqrtw = np.sqrt(w_safe)
    px_act = px[active]
    py_act = py[active]
    
    m01_results = {}
    if SEPARATE_M01:
        if 0 in ms:
            v0 = sqrtw / (np.linalg.norm(sqrtw) + 1e-30)
            Av0 = A.dot(v0)
            gamma0 = float(np.dot(v0, Av0)) / (np.dot(v0, w_safe * v0) + 1e-30)
            gamma0 = max(0.0, gamma0)
            v0_full = np.zeros(Nstates, dtype=np.float64)
            v0_full[active] = v0
            m01_results[0] = (gamma0, v0_full)
            if progress_callback:
                progress_callback(f"    m=0 (density): gamma={gamma0:.6e} (via Rayleigh quotient)")
        
        if 1 in ms:
            v1x = sqrtw * px_act
            v1y = sqrtw * py_act
            v1x_norm = np.linalg.norm(v1x)
            if v1x_norm > 1e-30:
                v1x = v1x / v1x_norm
            v1y = v1y - np.dot(v1y, v1x) * v1x
            v1y_norm = np.linalg.norm(v1y)
            if v1y_norm > 1e-30:
                v1y = v1y / v1y_norm
            
            Av1x = A.dot(v1x)
            gamma1x = float(np.dot(v1x, Av1x)) / (np.dot(v1x, w_safe * v1x) + 1e-30)
            Av1y = A.dot(v1y)
            gamma1y = float(np.dot(v1y, Av1y)) / (np.dot(v1y, w_safe * v1y) + 1e-30)
            gamma1 = max(0.0, min(gamma1x, gamma1y))
            
            v1 = v1x if gamma1x <= gamma1y else v1y
            v1_full = np.zeros(Nstates, dtype=np.float64)
            v1_full[active] = v1
            m01_results[1] = (gamma1, v1_full)
            if progress_callback:
                progress_callback(f"    m=1 (momentum): gamma={gamma1:.6e} (via Rayleigh quotient)")
    
    ms_to_solve = [m for m in ms if m not in m01_results] if SEPARATE_M01 else ms
    
    if len(ms_to_solve) > 0:
        Qcols = [
            sqrtw,
            sqrtw * px_act,
            sqrtw * py_act,
        ]
        Q = np.column_stack(Qcols).astype(np.float64)
        Q, _ = np.linalg.qr(Q)

        def _proj(x):
            return x - Q @ (Q.T @ x)

        def _matvec_Bproj(x):
            x = _proj(x)
            y = d * x
            z = A.dot(y)
            out = d * z
            return _proj(out)
    else:
        out = {}
        for m in ms:
            if m in m01_results:
                out[m] = m01_results[m]
            else:
                out[m] = (np.nan, None)
        return out

    k_calc = min(n - 2, int(N_EIG_CANDIDATES))
    if k_calc <= 0:
        out = {}
        for m in ms:
            if m in m01_results:
                out[m] = m01_results[m]
            else:
                out[m] = (np.nan, None)
        return out

    k_needed = max(len(ms) * 2, 20)
    k_calc = min(k_calc, max(k_needed, n - 2))
    ncv_est = min(max(2 * k_calc + 1, 20), n)
    
    matvec_count = [0]
    eigsh_start = [time.time()]
    last_progress_time = [time.time()]
    
    def _matvec_Bproj_with_progress(x):
        matvec_count[0] += 1
        current_time = time.time()
        
        if progress_callback and (current_time - last_progress_time[0] > 3 or matvec_count[0] % 500 == 0):
            estimated_iterations = max(k_calc * ncv_est, 100)
            estimated_matvecs = estimated_iterations * ncv_est
            progress_pct = min(100, (matvec_count[0] / estimated_matvecs) * 100) if estimated_matvecs > 0 else 0
            elapsed = current_time - eigsh_start[0]
            rate = matvec_count[0] / elapsed if elapsed > 0 else 0
            
            if rate > 0 and estimated_matvecs > matvec_count[0]:
                remaining_matvecs = estimated_matvecs - matvec_count[0]
                est_remaining = remaining_matvecs / rate
                est_total = estimated_matvecs / rate
                progress_callback(f"    [{elapsed:.2f}s / est. {est_total:.1f}s] {matvec_count[0]} matvecs ({progress_pct:.3f}% est., {rate:.0f} matvecs/s, ~{est_remaining:.1f}s remaining)...")
            else:
                progress_callback(f"    [{elapsed:.2f}s] {matvec_count[0]} matvecs ({progress_pct:.3f}% est., {rate:.0f} matvecs/s)...")
            
            last_progress_time[0] = current_time
        
        return _matvec_Bproj(x)
    
    if progress_callback:
        progress_callback(f"    Solving eigenvalue problem (k={k_calc}, n={n}, ncv={ncv_est})...")
    
    Bop = LinearOperator((n, n), matvec=_matvec_Bproj_with_progress, dtype=np.float64)
    
    matvec_count[0] = 0
    eigsh_start[0] = time.time()
    last_progress_time[0] = time.time()
    
    vals, y = eigsh(
        Bop,
        k=k_calc,
        which="SA",
        tol=1e-6,
        maxiter=10000,
        ncv=ncv_est
    )
    eigsh_time = time.time() - eigsh_start[0]
    if progress_callback:
        progress_callback(f"    Eigenvalue solve complete: {matvec_count[0]} matvecs in {eigsh_time:.2f}s ({matvec_count[0]/eigsh_time:.0f} matvecs/s), processing eigenvectors...")
    
    vals = np.real(vals)
    y = np.real(y)

    vecs = (d[:, None] * y).astype(np.float64)

    vals = np.where(vals < 0.0, 0.0, vals)
    order = np.argsort(vals)
    vals = np.real(vals[order])
    vecs = np.real(vecs[:, order])

    if not INCLUDE_CONSERVED:
        keep = np.where(np.abs(vals) > ZERO_TOL)[0]
        vals = vals[keep]
        vecs = vecs[:, keep]

    wv = w_safe[:, None] * vecs
    norms2 = np.sum(wv * vecs, axis=0)
    norms2 = np.maximum(norms2, 1e-30)
    vecs = vecs / np.sqrt(norms2[None, :])

    best = {m: None for m in ms_to_solve}
    dp_val = float(meta["dp"])
    
    def process_eigenvector(i):
        gamma = float(vals[i])
        v_active = vecs[:, i].copy()
        
        v_full = np.zeros(Nstates, dtype=np.float64)
        v_full[active] = v_active
        
        v_full_inv = v_full[inv_map]
        c_inv = w_corr(v_full, v_full_inv, w_full)
        
        switches, m_est = estimate_sign_switches_on_ring(
            v_full=v_full, active=active, px=px, py=py, P=P, dp_val=dp_val, theta_val=theta_val
        )
        m_round = int(np.rint(m_est))
        
        if m_round not in ms_to_solve:
            return None
        if abs(m_est - float(m_round)) > M_TOL:
            return None
        
        if (m_round % 2) == 0:
            if c_inv < INV_MIN_CORR:
                return None
        else:
            if c_inv > -INV_MIN_CORR:
                return None
        
        return (i, m_round, gamma, c_inv, m_est, switches, v_full)
    
    n_vecs = vecs.shape[1]
    results = []
    if n_vecs > 3 and N_WORKERS > 1:
        if progress_callback:
            progress_callback(f"    Processing {n_vecs} eigenvectors in parallel ({N_WORKERS} workers)...")
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(process_eigenvector, i) for i in range(n_vecs)]
            completed = 0
            for f in as_completed(futures):
                results.append(f.result())
                completed += 1
                if progress_callback and completed % max(1, n_vecs // 10) == 0:
                    progress_callback(f"    Processed {completed}/{n_vecs} eigenvectors...")
    else:
        if progress_callback:
            progress_callback(f"    Processing {n_vecs} eigenvectors sequentially...")
        for i in range(n_vecs):
            results.append(process_eigenvector(i))
            if progress_callback and (i + 1) % max(1, n_vecs // 10) == 0:
                    progress_callback(f"    Processed {i + 1}/{n_vecs} eigenvectors...")
    
    for result in results:
        if result is None:
            continue
        i, m_round, gamma, c_inv, m_est, switches, v_full = result
        cur = best[m_round]
        if (cur is None) or (gamma < cur["gamma"]):
            best[m_round] = {"gamma": gamma, "c_inv": c_inv, "m_est": m_est, "switches": switches, "v_full": v_full}

    out = {}
    for m in ms:
        if m in m01_results:
            out[m] = m01_results[m]
        elif m in best and best[m] is not None:
            out[m] = (float(best[m]["gamma"]), best[m]["v_full"])
        else:
            out[m] = (np.nan, None)
    return out


def save_csv_incremental(Ts_new, Ts_req_new, gammas_new, existing_Ts, existing_Ts_requested, existing_gammas, ms):
    all_Ts_combined = list(existing_Ts) + list(Ts_new)
    all_Ts_req_combined = list(existing_Ts_requested if len(existing_Ts_requested) == len(existing_Ts) else existing_Ts) + list(Ts_req_new)
    all_gammas_combined = {m: list(existing_gammas[m]) + list(gammas_new[m]) for m in ms}
    
    unique_Ts = sorted(set(float(T) for T in all_Ts_combined), key=lambda x: float(x))
    T_to_index = {float(T): i for i, T in enumerate(unique_Ts)}
    
    combined_gammas = {m: [np.nan] * len(unique_Ts) for m in ms}
    combined_Ts_req = [None] * len(unique_Ts)
    
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
    eigenvectors = {m: [] for m in ms}
    Ts_used = []
    Ts_req_used = []

    existing_Ts = []
    existing_Ts_requested = []
    existing_gammas = {m: [] for m in ms}
    
    if os.path.exists(OUT_CSV):
        print(f"[Loading] Found existing results in {OUT_CSV}")
        try:
            with open(OUT_CSV, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    T_val = float(row.get('T', 0))
                    T_req_val = float(row.get('T_requested', T_val))
                    existing_Ts.append(T_val)
                    existing_Ts_requested.append(T_req_val)
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
            print(f"[Loading] Found {len(existing_Ts)} existing temperatures: {existing_Ts}")
        except Exception as e:
            print(f"[Warning] Could not load existing results: {e}")
            import traceback
            traceback.print_exc()
            existing_Ts = []
            existing_Ts_requested = []
            existing_gammas = {m: [] for m in ms}
    
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
    
    all_files.sort(key=lambda x: x[0])
    print(f"[Scanning] Found {len(all_files)} .pkl files with temperatures")
    
    Thetas_to_process = []
    for T_file, path in all_files:
        T_file_float = float(T_file)
        is_computed = False
        matched_T = None
        matched_type = None
        for i, T_existing in enumerate(existing_Ts):
            T_existing_float = float(T_existing)
            T_req_existing_float = float(existing_Ts_requested[i]) if i < len(existing_Ts_requested) else T_existing_float
            rel_tol = 0.01 * max(abs(T_file_float), abs(T_existing_float))
            tol = max(1e-3, rel_tol)
            if abs(T_file_float - T_existing_float) < tol:
                is_computed = True
                matched_T = T_existing_float
                matched_type = "T"
                break
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
    
    n_skipped = len(all_files) - len(Thetas_to_process)
    if n_skipped > 0:
        print(f"[Summary] Skipped {n_skipped} already-computed temperatures, {len(Thetas_to_process)} new temperatures to process")
    
    if len(Thetas_to_process) == 0:
        print(f"[Info] All {len(all_files)} temperatures from folder are already computed in CSV. Skipping computation, using existing data for plotting...")
        Ts_used = []
        Ts_req_used = []
        gammas = {m: [] for m in ms}
        eigenvectors = {m: [] for m in ms}
    else:
        print(f"=== Computing eigenvalues gamma_m(T) for {len(Thetas_to_process)} new temperatures ===")
        if USE_HARMONIC_PROJECTION:
            print(f"[Method] Using fast harmonic projection (bypasses expensive eigsh)")
        else:
            print(f"[Method] Using eigsh solver (slower but provides eigenvectors)")
        print(f"[Batch processing] Processing in batches of {BATCH_SIZE} files to avoid memory issues...")
        
        file_info = []
        for T_file, path in Thetas_to_process:
            try:
                file_size = os.path.getsize(path)
                file_info.append((T_file, path, file_size))
            except Exception as e:
                print(f"[Warning] Could not get info for file {path}: {e}")
        
        file_info.sort(key=lambda x: -x[2])
        
        Ts_used = []
        Ts_req_used = []
        gammas = {m: [] for m in ms}
        
        def process_single_file_batch(T_file, path, file_idx=None, total_files=None):
            import time
            start_time = time.time()
            try:
                if file_idx is not None and total_files is not None:
                    print(f"\n[Progress] File {file_idx + 1}/{total_files} | Theta={T_file:.6g} | {os.path.basename(path)}", flush=True)
                else:
                    print(f"\n[Progress] Theta={T_file:.6g} | {os.path.basename(path)}", flush=True)
                
                print(f"  [1/4] Loading file...", end='', flush=True)
                with open(path, "rb") as fp:
                    M, meta = pickle.load(fp)
                load_time = time.time() - start_time
                print(f" ✓ ({load_time:.2f}s)", flush=True)
                
                print(f"  [2/4] Computing active operator...", end='', flush=True)
                Ma, active = get_active_operator(M, meta)
                active_time = time.time() - start_time
                print(f" ✓ shape={Ma.shape} ({active_time:.2f}s)", flush=True)
                
                if USE_HARMONIC_PROJECTION:
                    print(f"  [3/4] Computing eigenvalues (harmonic projection method)...", flush=True)
                else:
                    print(f"  [3/4] Computing eigenvalues (eigsh method)...", flush=True)
                eig_sub_start = time.time()
                
                if USE_HARMONIC_PROJECTION:
                    result_gammas, result_eigenvectors = gammas_by_harmonic_projection(Ma, meta, active, ms, ring_only=False)
                else:
                    def eig_progress(msg):
                        print(msg, flush=True)
                    
                    sel = select_physical_eigs_per_m(Ma, meta, active, ms, theta_val=float(T_file), progress_callback=eig_progress)
                    result_gammas = {}
                    result_eigenvectors = {}
                    for m in ms:
                        gamma, v_full = sel[m]
                        result_gammas[m] = gamma
                        result_eigenvectors[m] = v_full if not np.isnan(gamma) else None
                
                eig_time = time.time() - start_time
                eig_sub_time = time.time() - eig_sub_start
                print(f"  [3/4] ✓ Complete! ({eig_sub_time:.2f}s)", flush=True)
                
                print(f"  [4/4] Extracting results...", end='', flush=True)
                
                grid_meta = {
                    "Nmax": int(meta["Nmax"]),
                    "dp": float(meta["dp"]),
                    "shift_x": float(meta.get("shift_x", 0.0)),
                    "shift_y": float(meta.get("shift_y", 0.0)),
                }
                
                del M, meta, Ma, active
                
                total_time = time.time() - start_time
                print(f" ✓ Complete! Total time: {total_time:.2f}s", flush=True)
                
                return (float(T_file), float(T_file), result_gammas, result_eigenvectors, grid_meta)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f" ✗ Error after {elapsed:.2f}s: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return None
        
        num_batches = (len(file_info) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(file_info))
            batch_files = file_info[start_idx:end_idx]
            
            print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing {len(batch_files)} files...")
            
            batch_results = []
            total_files = len(file_info)
            global_file_idx = start_idx
            for T_file, path, _ in batch_files:
                result = process_single_file_batch(T_file, path, file_idx=global_file_idx, total_files=total_files)
                global_file_idx += 1
                if result is not None:
                    batch_results.append(result)
            
            for theta_used, T_file, result_gammas, result_eigenvectors, grid_meta in batch_results:
                Ts_used.append(theta_used)
                Ts_req_used.append(T_file)
                for m in ms:
                    gammas[m].append(result_gammas[m])
            
            print(f"[Saving] Saving eigenvectors for batch {batch_idx + 1} to {OUT_EIGENVECTORS_NPZ}...")
            batch_eigenvectors_dict = {}
            batch_grid_meta = {}
            for m in ms:
                if m not in batch_eigenvectors_dict:
                    batch_eigenvectors_dict[m] = {}
                for i, (theta_used, _, _, result_eigenvectors, grid_meta) in enumerate(batch_results):
                    batch_eigenvectors_dict[m][theta_used] = result_eigenvectors[m]
                    batch_grid_meta[theta_used] = grid_meta
            
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
            
            save_dict = {f'eigenvectors_m{m}': batch_eigenvectors_dict[m] for m in ms}
            save_dict["grid_meta"] = batch_grid_meta
            np.savez_compressed(OUT_EIGENVECTORS_NPZ, **save_dict)
            
            print(f"[Saving] Updating CSV with batch {batch_idx + 1} results...")
            save_csv_incremental(Ts_used, Ts_req_used, gammas, existing_Ts, existing_Ts_requested, existing_gammas, ms)
            
            del batch_results, batch_eigenvectors_dict
            import gc
            gc.collect()
            
            print(f"[Batch {batch_idx + 1}/{num_batches}] Completed. Memory cleaned.")
    
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

    print(f"\n[Loading] Loading eigenvectors from {OUT_EIGENVECTORS_NPZ} for plotting...")
    
    if not os.path.exists(OUT_EIGENVECTORS_NPZ):
        print(f"[Warning] {OUT_EIGENVECTORS_NPZ} not found. Skipping eigenfunction plotting.")
    else:
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
        
        eigenvectors = {m: [] for m in ms}
        for T in Ts:
            T_float = float(T)
            for m in ms:
                v_full = None
                if m in eigenvectors_npz:
                    if T_float in eigenvectors_npz[m]:
                        v_full = eigenvectors_npz[m][T_float]
                    else:
                        for T_npz, v in eigenvectors_npz[m].items():
                            if abs(T_float - float(T_npz)) < max(1e-3, 0.01 * max(abs(T_float), abs(float(T_npz)))):
                                v_full = v
                                break
                eigenvectors[m].append(v_full)
        
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
        
        plot_eigenfunction_table_batched(Ts, eigenvectors, ms, grid_meta_list=grid_meta_list)
        
        if DUMP_SINGLE_EIGENFUNCTIONS:
            print(f"\n[Dumping single eigenfunctions] Processing...")
            wl = {}
            for (Tt, mlist) in DUMP_WHITELIST:
                wl[float(Tt)] = set(int(x) for x in mlist)
            
            dumped_count = 0
            for i, T in enumerate(Ts):
                gm = grid_meta_list[i] if i < len(grid_meta_list) else None
                if gm is None:
                    continue
                for m in ms:
                    v_full = eigenvectors[m][i] if i < len(eigenvectors[m]) else None
                    if v_full is None:
                        continue
                    
                    if len(DUMP_WHITELIST) > 0:
                        keys = list(wl.keys())
                        if len(keys) == 0:
                            continue
                        k = min(keys, key=lambda x: abs(x - float(T)))
                        if abs(k - float(T)) > max(1e-3, 0.01 * max(abs(k), abs(float(T)))):
                            continue
                        if m not in wl[k]:
                            continue
                    
                    plot_single_eigenfunction(v_full, gm, float(T), int(m))
                    dumped_count += 1
            
            print(f"[Saved] {dumped_count} single eigenfunctions -> {SINGLE_OUTDIR}/eig_T*_m*.png")


_plot_geom_cache = {}

def _plot_geom(Nmax: int, dp: float, shift_x: float, shift_y: float,
               pwin: float, ring_only: bool, ring_w: float):
    key = (int(Nmax), float(dp), float(shift_x), float(shift_y), float(pwin), bool(ring_only),
           None if ring_w is None else float(ring_w))
    if key in _plot_geom_cache:
        return _plot_geom_cache[key]

    half = Nmax // 2
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
    print(f"\n[Plotting eigenfunctions] Processing {len(Ts)} temperatures...")
    
    if len(Ts) > max_temps_plot:
        print(f"[Plotting] Limiting to {max_temps_plot} temperatures (evenly spaced) to avoid memory issues...")
        selected_indices = [int(i * (len(Ts) - 1) / (max_temps_plot - 1)) for i in range(max_temps_plot)]
    else:
        selected_indices = list(range(len(Ts)))
    
    plot_eigenfunction_table(Ts, eigenvectors, ms, selected_T_indices=selected_indices, max_Ts=None,
                             grid_meta_list=grid_meta_list)


def plot_eigenfunction_table(Ts, eigenvectors, ms, selected_T_indices=None, max_Ts=None, grid_meta_list=None):
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Circle
    
    if selected_T_indices is None:
        if max_Ts is None or len(Ts) <= max_Ts:
            selected_T_indices = list(range(len(Ts)))
        else:
            selected_T_indices = [int(i * (len(Ts) - 1) / (max_Ts - 1)) for i in range(max_Ts)]
    
    selected_Ts = Ts[selected_T_indices]
    print(f"\n[Plotting eigenfunctions] Using {len(selected_Ts)} temperatures: {selected_Ts}")
    
    valid_ms = []
    for m in ms:
        if any(v is not None for v in eigenvectors[m]):
            valid_ms.append(m)
    
    if len(valid_ms) == 0:
        print("[Warning] No valid eigenvectors found for plotting.")
        return
    
    if grid_meta_list is None:
        grid_meta_list = [None] * len(Ts)
    
    # No global normalization - each panel will be normalized individually
    
    nrows = len(valid_ms)
    ncols = len(selected_T_indices)
    
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=(4.5 * ncols, 4.5 * nrows), 
                             constrained_layout=True)
    
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.reshape(nrows, ncols)
    
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

                # Project onto the m-th harmonic on the ring for clean plotting
                T_val = Ts[T_idx]
                v_plot = v_full
                if m == 0:
                    v_plot = project_component_on_ring(v_full, gm, T_val, m=0, phase_fix=False)
                elif m == 1:
                    v_plot = project_component_on_ring(v_full, gm, T_val, m=1, phase_fix=True)
                # For m >= 2, we could also project, but keeping raw for now
                grid = np.asarray(v_plot, dtype=np.float64).reshape((Nmax_T, Nmax_T))
                sx, sy, extent, ring_mask = _plot_geom(
                    Nmax_T, dp_T, shift_x, shift_y,
                    pwin=PLOT_PWIN, ring_only=PLOT_RING_ONLY, ring_w=PLOT_RING_W
                )
                g = grid[sx, sy]
                if ring_mask is not None:
                    g = np.ma.array(g, mask=ring_mask)
                    if g.count() == 0:
                        ax.axis("off")
                        continue
                    vmax_abs = float(np.max(np.abs(g.compressed())))
                else:
                    vmax_abs = float(np.max(np.abs(g)))
                
                # Individual normalization for this panel
                vmax_abs = max(vmax_abs, 1e-14)
                panel_norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)

                im = ax.imshow(
                    g.T,
                    extent=extent,
                    origin="lower",
                    cmap="seismic",
                    norm=panel_norm,
                    aspect="equal",
                )

                ax.add_patch(Circle((0.0, 0.0), 1.0, fill=False, linewidth=0.6, alpha=0.6, color='black'))

                ax.set_xlim(-PLOT_PWIN, PLOT_PWIN)
                ax.set_ylim(-PLOT_PWIN, PLOT_PWIN)

                if col_idx == 0:
                    ax.set_ylabel(f"$m={m}$", fontsize=14, rotation=0, labelpad=20)
                if row_idx == 0:
                    ax.set_title(f"$T={selected_Ts[col_idx]:.4g}$", fontsize=14)
                ax.set_xlabel("$p_x$")
                if col_idx > 0:
                    ax.set_ylabel("$p_y$")
            else:
                ax.axis("off")
    
    out_png = "eigenfunctions_table.png"
    out_svg = "eigenfunctions_table.svg"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")


def plot_single_eigenfunction(v_full, grid_meta, T, m,
                              outdir=SINGLE_OUTDIR,
                              pwin=PLOT_PWIN,
                              ring_only=PLOT_RING_ONLY,
                              ring_w=PLOT_RING_W):
    if v_full is None or grid_meta is None:
        return
    Nmax = int(grid_meta["Nmax"])
    dp   = float(grid_meta["dp"])
    sx   = float(grid_meta.get("shift_x", 0.0))
    sy   = float(grid_meta.get("shift_y", 0.0))
    if len(v_full) != Nmax * Nmax:
        return

    # Project onto the m-th harmonic on the ring for clean plotting
    v_plot = v_full
    if m == 0:
        v_plot = project_component_on_ring(v_full, grid_meta, T, m=0, phase_fix=False)
    elif m == 1:
        v_plot = project_component_on_ring(v_full, grid_meta, T, m=1, phase_fix=True)
    # For m >= 2, we could also project, but keeping raw for now
    
    grid = np.asarray(v_plot, dtype=np.float64).reshape((Nmax, Nmax))
    slx, sly, extent, ring_mask = _plot_geom(
        Nmax, dp, sx, sy,
        pwin=float(pwin),
        ring_only=bool(ring_only),
        ring_w=ring_w
    )
    g = grid[slx, sly]
    if ring_mask is not None:
        g = np.ma.array(g, mask=ring_mask)
        if g.count() == 0:
            return

    vmax = float(np.max(np.abs(g.compressed() if hasattr(g, "compressed") else g)))
    vmax = max(vmax, 1e-14)
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(
        g.T,
        extent=extent,
        origin="lower",
        cmap="seismic",
        norm=norm,
        aspect="equal",
    )
    from matplotlib.patches import Circle
    ax.add_patch(Circle((0.0, 0.0), 1.0, fill=False, linewidth=0.8, alpha=0.8, color="black"))
    ax.set_xlim(-pwin, pwin)
    ax.set_ylim(-pwin, pwin)
    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_y$")
    ax.set_title(fr"$T={T:.4g},\ m={m}$")
    fig.tight_layout()

    fn = os.path.join(outdir, f"eig_T{T:.6g}_m{m}.png")
    # fig.savefig(fn, dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()