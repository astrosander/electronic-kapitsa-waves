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

def estimate_sign_switches_on_ring(v_full, active, px, py, P, dp_val):
    """
    Returns (switches:int, m_est:float).
    Uses your binned-angle sign-switch logic.
    """
    if USE_NUMBA:
        return estimate_sign_switches_on_ring_numba(
            v_full, active, px, py, P, float(dp_val),
            float(RING_WIDTH_FACTOR), int(MIN_RING_POINTS), int(N_ANGLE_BINS_MIN)
        )
    
    ring_w = RING_WIDTH_FACTOR * float(dp_val)
    idx = active[np.abs(P[active] - 1.0) <= ring_w]
    while idx.size < MIN_RING_POINTS and ring_w < 0.5:
        ring_w *= 1.5
        idx = active[np.abs(P[active] - 1.0) <= ring_w]
    if idx.size < 16:
        return 0, 0.0

    ang = np.arctan2(py[idx], px[idx])
    ang = (ang + 2.0 * np.pi) % (2.0 * np.pi)
    nbin = max(N_ANGLE_BINS_MIN, int((2.0 * np.pi / max(dp_val, 1e-12)) * 2))
    bins = np.floor(ang / (2.0 * np.pi) * nbin).astype(np.int64)

    prof = np.zeros(nbin, dtype=np.float64)
    cnt  = np.zeros(nbin, dtype=np.int64)
    for b, val in zip(bins, v_full[idx]):
        prof[b] += float(val)
        cnt[b]  += 1
    has = cnt > 0
    if not np.any(has):
        return 0, 0.0
    prof[has] /= cnt[has]

    s = np.sign(prof)
    thr = 1e-10 * (float(np.max(np.abs(prof[has]))) + 1e-300)
    s[np.abs(prof) < thr] = 0.0

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
            s[:j] = s[j]
    if np.all(s == 0.0):
        return 0, 0.0

    switches = 0
    for i in range(nbin):
        a = s[i]
        b = s[(i + 1) % nbin]
        if a * b < 0.0:
            switches += 1
    m_est = 0.5 * float(switches)
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


def select_physical_eigs_per_m(Ma: csr_matrix, meta, active: np.ndarray, ms):
    """
    Solve (-Ma) v = gamma * W v near gamma~0, classify by (i) inversion parity and (ii) m_est from sign-switches,
    then return {m: gamma_m} for requested ms.
    """
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
            v_full=v_full, active=active, px=px, py=py, P=P, dp_val=dp_val
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
        
        return (i, m_round, gamma, c_inv, m_est, switches)
    
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
        i, m_round, gamma, c_inv, m_est, switches = result
        cur = best[m_round]
        if (cur is None) or (gamma < cur["gamma"]):
            best[m_round] = {"gamma": gamma, "c_inv": c_inv, "m_est": m_est, "switches": switches}

    out = {}
    for m in ms:
        out[m] = np.nan if best[m] is None else float(best[m]["gamma"])
    return out


def main():
    gammas = {m: [] for m in ms}
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
        sel = select_physical_eigs_per_m(Ma, meta, active, ms)
        
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
                gammas[m].append(sel[m])
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


if __name__ == "__main__":
    main()
