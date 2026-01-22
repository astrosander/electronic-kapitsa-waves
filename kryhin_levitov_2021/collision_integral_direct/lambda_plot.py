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

from scipy.sparse import csr_matrix, isspmatrix_csr, diags
from scipy.sparse.linalg import eigsh, LinearOperator

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

# --- NEW: eigenmode selection knobs (same spirit as your diagnostics code) ---
N_EIG_CANDIDATES = 120   # compute this many eigenpairs near sigma~0 (reduced to avoid memory issues)
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


def w_corr(v: np.ndarray, u: np.ndarray, w: np.ndarray) -> float:
    """Weighted correlation <v,u>_W / <v,v>_W."""
    num = float(np.dot(v, w * u))
    den = float(np.dot(v, w * v)) + 1e-30
    return num / den


def estimate_sign_switches_on_ring(v_full, active, px, py, P, dp_val):
    """
    Returns (switches:int, m_est:float).
    Uses your binned-angle sign-switch logic.
    """
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


def find_matrix_file(theta_req: float):
    """
    Pick the *_T{...}.pkl file whose T in the filename is closest to theta_req.
    """
    if not os.path.isdir(IN_DIR):
        raise FileNotFoundError(f"Directory not found: {IN_DIR}")

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

    Ts = np.array(Ts, dtype=float)
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

    px = np.asarray(meta.get("px", meta["dp"] * nx), dtype=np.float64)
    py = np.asarray(meta.get("py", meta["dp"] * ny), dtype=np.float64)
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
        y = d * x
        z = A.dot(y)
        return d * z

    n = A.shape[0]
    Bop = LinearOperator((n, n), matvec=_matvec_B, dtype=np.float64)

    k_calc = min(n - 2, int(N_EIG_CANDIDATES))
    if k_calc <= 0:
        return {m: np.nan for m in ms}

    # Smallest algebraic eigenvalues ≈ slow modes near 0 (no sigma => no CSC factorization)
    vals, y = eigsh(
        Bop,
        k=k_calc,
        which="SA",
        tol=1e-8,
        maxiter=20000
    )
    vals = np.real(vals)
    y = np.real(y)

    # Recover generalized eigenvectors v = D^{-1/2} y
    vecs = (d[:, None] * y)

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

    # normalize in W-inner-product on active
    for i in range(vecs.shape[1]):
        v = vecs[:, i]
        n2 = float(np.dot(v, w_safe * v))
        if n2 > 0.0:
            vecs[:, i] = v / np.sqrt(n2)

    # collect candidates per m
    best = {m: None for m in ms}
    dp_val = float(meta["dp"])

    for i in range(vecs.shape[1]):
        gamma = float(vals[i])
        v_active = vecs[:, i].copy()

        # embed into full lattice for symmetry + ring counting
        v_full = np.zeros(Nstates, dtype=np.float64)
        v_full[active] = v_active

        # inversion parity correlation
        c_inv = w_corr(v_full, v_full[inv_map], w_full)

        switches, m_est = estimate_sign_switches_on_ring(
            v_full=v_full, active=active, px=px, py=py, P=P, dp_val=dp_val
        )
        m_round = int(np.rint(m_est))

        if m_round not in best:
            continue
        if abs(m_est - float(m_round)) > M_TOL:
            continue

        # parity sanity: even m => c_inv ~ +1, odd m => c_inv ~ -1
        if (m_round % 2) == 0:
            if c_inv < INV_MIN_CORR:
                continue
        else:
            if c_inv > -INV_MIN_CORR:
                continue

        # keep slowest (smallest gamma) for that m
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

    for Treq in Thetas_req:
        M, meta, path, Tused = load_matrix_nearest(float(Treq))
        if len(Ts_used) > 0 and np.isclose(Tused, Ts_used[-1], rtol=0, atol=0):
            continue
        Ma, active = get_active_operator(M, meta)

        print(f"[load] requested Theta={Treq:.6g}, using nearest Theta={Tused:.6g}  |  {os.path.basename(path)}  |  shape={Ma.shape}")

        sel = select_physical_eigs_per_m(Ma, meta, active, ms)
        for m in ms:
            gammas[m].append(sel[m])

        Ts_used.append(Tused)
        Ts_req_used.append(Treq)

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


if __name__ == "__main__":
    main()
