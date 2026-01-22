#!/usr/bin/env python3
"""
Iee_grid_diagnostics.py

1) Plot f(1-f) over the 2D momentum lattice for T/E_F = 1, 0.1, 0.01.
2) Plot several eigenmodes (eigenfunctions) over the same lattice.

Eigenmodes are computed from the ACTIVE subspace only (meta["active"]) to reduce cost.
You need SciPy for eigenmodes:
  pip install scipy

IMPORTANT:
The paper's spectrum is a GENERALIZED eigenproblem:
  (-M) v = gamma * W v,   W = diag(f(1-f))
not the ordinary eigenproblem of (-M).
"""

import os
import math
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# --- Only needed for eigenmodes ---
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs, eigsh, norm as spnorm

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
# plt.rcParams['font.size'] = 20
# plt.rcParams['axes.labelsize'] = 20
# plt.rcParams['axes.titlesize'] = 20
# plt.rcParams['xtick.labelsize'] = 20
# plt.rcParams['ytick.labelsize'] = 20
# plt.rcParams['legend.fontsize'] = 20
# plt.rcParams['figure.titlesize'] = 20



# ---------------- USER SETTINGS ----------------
# Grid (must match your generator choices if you want to compare to saved matrices)
Nmax = 100
dp   = 0.08

# For part (1)
Thetas_weight = [1.0, 0.1, 0.01]

# For part (2): choose one temperature that you have already generated a matrix for
Theta_eigs = 0.02
TARGET_MS = list(range(12)) # plot m = 0..12 (one representative per m)
N_EIG_PLOT = len(TARGET_MS)
ZERO_TOL = 1e-10            # treat |lambda|<ZERO_TOL as "conserved/zero" mode

# ---- NEW: select "physical" C4-symmetric modes with m = 4*n ----
# We compute MORE eigenpairs than we plot, then FILTER them.
N_EIG_CANDIDATES = 250      # need more to reliably cover m=0..12

# C4 classification via 90° rotation R:  (px,py)->(-py,px) on the discrete torus
# For m = 4*n, we expect R v ≈ +v  (in W-inner-product)
C4_MIN_CORR  = 0.85         # keep if <v, Rv>_W / <v,v>_W >= this
INV_MIN_CORR = 0.85         # also enforce inversion evenness: v(p)≈v(-p) for even m

# "n is integer number of sign switches": on the FS ring,
# sign_switches ≈ 2*m  => m_est ≈ sign_switches/2.
# Requiring m multiple of 4 means sign_switches must be multiple of 8.
REQUIRE_M_MULTIPLE_OF_4 = False
REQUIRE_N_ODD = False       # if True: keep only n odd => m = 4*(odd) = 4,12,20,...

# NEW: how tightly m_est must match an integer m
M_TOL = 0.35                # relax if you miss some m's

# NEW: whether to keep the conserved/near-zero modes (needed to see m=1, and usually one of m=0)
INCLUDE_CONSERVED = True

# ring used to estimate sign switches around the Fermi surface
RING_WIDTH_FACTOR = 2.5     # ring half-width ≈ factor * dp in |p|
MIN_RING_POINTS   = 200     # if too few points, widen ring automatically
N_ANGLE_BINS_MIN  = 256     # angular binning makes sign-switch count robust on a Cartesian grid

# Plotting scale: > 1.0 zooms out (shows more), < 1.0 zooms in (shows less)
PLOT_SCALE_FACTOR = 0.25     # increase to see more of the momentum space

# Where your generator saved matrices
IN_DIR = "Matrixes_bruteforce"
# ----------------------------------------------


# ---------- stable f(P,Theta) ----------
def f_scalar(P: float, Theta: float) -> float:
    invT = 1.0 / Theta
    em = math.exp(-invT)
    a = 1.0 - em
    x = (P * P - 1.0) * invT
    if x > 700.0:
        return 0.0
    if x < -700.0:
        return 1.0
    ex = math.exp(x)
    return a / (ex + a)


def build_centered_lattice(Nmax: int):
    half = Nmax // 2
    ns = np.arange(-half, half, dtype=np.int32)
    nx, ny = np.meshgrid(ns, ns, indexing="ij")
    return nx.reshape(-1), ny.reshape(-1), half


def make_index_map(nx: np.ndarray, ny: np.ndarray, Nmax: int, half: int) -> np.ndarray:
    """
    Map discrete (n_x, n_y) -> flattened index i.
    IMPORTANT: The grid uses FFT-style indices n in [-N/2, ..., N/2-1] so we treat
    symmetry operations with modulo-N wrap (discrete torus).
    """
    idx_map = -np.ones((Nmax, Nmax), dtype=np.int32)
    idx_map[nx + half, ny + half] = np.arange(nx.size, dtype=np.int32)
    return idx_map


def build_symmetry_maps(nx: np.ndarray, ny: np.ndarray, idx_map: np.ndarray, Nmax: int, half: int):
    """
    Return permutations (arrays of indices) implementing:
      inv: (nx,ny)->(-nx,-ny)  with modulo wrap
      rot90: (nx,ny)->(-ny,nx) with modulo wrap
    """
    # array-index coordinates in 0..Nmax-1
    ix = (nx + half) % Nmax
    iy = (ny + half) % Nmax

    inv_ix = (-nx + half) % Nmax
    inv_iy = (-ny + half) % Nmax
    inv = idx_map[inv_ix, inv_iy].astype(np.int32)

    rot_ix = (-ny + half) % Nmax
    rot_iy = (nx + half) % Nmax
    rot90 = idx_map[rot_ix, rot_iy].astype(np.int32)

    return inv, rot90


def w_corr(v: np.ndarray, u: np.ndarray, w: np.ndarray) -> float:
    """Weighted correlation <v,u>_W / <v,v>_W."""
    num = float(np.dot(v, w * u))
    den = float(np.dot(v, w * v)) + 1e-30
    return num / den


def estimate_sign_switches_on_ring(
    v_full: np.ndarray,
    active: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    P: np.ndarray,
    dp_val: float,
    theta_val: float,
):
    """
    Estimate number of sign switches around the FS:
      - restrict to active indices near |p|≈1
      - bin by polar angle for robustness
    Returns: (sign_switches:int, m_est:float, n_est:float)
      with m_est ≈ sign_switches/2, and n_est ≈ m_est/4
    """
    ring_w = RING_WIDTH_FACTOR * float(dp_val)

    idx = active[np.abs(P[active] - 1.0) <= ring_w]
    # widen ring until we have enough points (important at small Theta or coarse dp)
    while idx.size < MIN_RING_POINTS and ring_w < 0.5:
        ring_w *= 1.5
        idx = active[np.abs(P[active] - 1.0) <= ring_w]

    if idx.size < 16:
        return 0, 0.0, 0.0

    ang = np.arctan2(py[idx], px[idx])
    ang = (ang + 2.0 * math.pi) % (2.0 * math.pi)

    # choose angular bins based on dp (coarser dp => fewer distinct angles)
    nbin = max(N_ANGLE_BINS_MIN, int((2.0 * math.pi / max(dp_val, 1e-12)) * 2))
    bins = np.floor(ang / (2.0 * math.pi) * nbin).astype(np.int64)

    prof = np.zeros(nbin, dtype=np.float64)
    cnt  = np.zeros(nbin, dtype=np.int64)
    for b, val in zip(bins, v_full[idx]):
        prof[b] += float(val)
        cnt[b]  += 1

    has = cnt > 0
    if not np.any(has):
        return 0, 0.0, 0.0
    prof[has] /= cnt[has]

    # fill empty bins by forward fill then backward fill to make a closed loop
    s = np.sign(prof)
    # ignore tiny amplitudes as "no data"
    thr = 1e-10 * (float(np.max(np.abs(prof[has]))) + 1e-300)
    s[np.abs(prof) < thr] = 0.0

    # forward fill zeros
    last = 0.0
    for i in range(nbin):
        if s[i] == 0.0:
            s[i] = last
        else:
            last = s[i]
    # backward fill if the start was zero
    if s[0] == 0.0:
        # find first nonzero
        j = -1
        for i in range(nbin):
            if s[i] != 0.0:
                j = i
                break
        if j >= 0:
            s[:j] = s[j]

    if np.all(s == 0.0):
        return 0, 0.0, 0.0

    # count sign changes around the loop
    switches = 0
    for i in range(nbin):
        a = s[i]
        b = s[(i + 1) % nbin]
        if a * b < 0.0:
            switches += 1

    m_est = 0.5 * float(switches)
    n_est = 0.25 * m_est
    return int(switches), float(m_est), float(n_est)


def precompute_for_grid(Nmax: int, dp: float, Theta: float):
    nx, ny, half = build_centered_lattice(Nmax)
    px = dp * nx.astype(np.float64)
    py = dp * ny.astype(np.float64)
    P  = np.sqrt(px * px + py * py)
    f  = np.array([f_scalar(float(Pi), float(Theta)) for Pi in P], dtype=np.float64)
    return (nx, ny, half, px, py, P, f)


def values_to_grid(values_flat: np.ndarray, nx: np.ndarray, ny: np.ndarray, half: int, Nmax: int):
    grid = np.full((Nmax, Nmax), np.nan, dtype=np.float64)
    grid[nx + half, ny + half] = values_flat
    return grid


def format_scientific_latex(value: float, precision: int = 3) -> str:
    """Format a number in LaTeX scientific notation: a.bcd \times 10^{e}"""
    if value == 0.0:
        return "0"
    
    mantissa, exponent = f"{value:.{precision}e}".split('e')
    exponent = int(exponent)
    
    if exponent == 0:
        return mantissa
    else:
        return rf"{mantissa} \cdot 10^{{{exponent}}}"


def plot_f1mf():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, T in zip(axes, Thetas_weight):
        nx, ny, half, px, py, P, f = precompute_for_grid(Nmax, dp, T)
        w = f * (1.0 - f)

        grid = values_to_grid(w, nx, ny, half, Nmax)

        # extent in momentum units
        pmin = (-half) * dp
        pmax = (half - 1) * dp
        pcenter = (pmin + pmax) / 2.0
        prange = (pmax - pmin) / 2.0
        pmin_plot = pcenter - prange * PLOT_SCALE_FACTOR
        pmax_plot = pcenter + prange * PLOT_SCALE_FACTOR
        im = ax.imshow(grid.T, origin="lower", extent=[pmin, pmax, pmin, pmax], aspect="equal", cmap='rainbow')
        # Increase visible range
        ax.set_xlim(pmin_plot, pmax_plot)
        ax.set_ylim(pmin_plot, pmax_plot)
        ax.set_title(f"$T/E_F = {T}$")
        ax.set_xlabel("$p_x$")
        ax.set_ylabel("$p_y$")
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, aspect=30, label=r"$f(1-f)$")
        cbar.ax.tick_params(labelsize=10)

    # fig.suptitle(r"Weight $f_0(1-f_0)$ over momentum lattice", y=1.02)
    fig.savefig("f0_weight_grid.png", dpi=300)
    fig.savefig("f0_weight_grid.svg")
    print("Saved: f0_weight_grid.png")
    print("Saved: f0_weight_grid.svg")
    # plt.show()


def find_matrix_file(theta: float):
    """
    Find the matrix file whose encoded T is closest to 'theta'.
    This avoids strict string matching issues with np.geomspace.
    """
    if not os.path.isdir(IN_DIR):
        raise FileNotFoundError(f"Directory not found: {IN_DIR}")

    # expected pattern: ..._T{T}.pkl
    files = [fn for fn in os.listdir(IN_DIR) if fn.endswith(".pkl") and "_T" in fn]
    if not files:
        raise FileNotFoundError(f"No .pkl matrix files found in {IN_DIR}")

    Ts = []
    paths = []
    for fn in files:
        # extract substring between "_T" and ".pkl"
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
    idx = int(np.argmin(np.abs(Ts - theta)))
    chosen_T = Ts[idx]
    chosen_path = paths[idx]

    print(f"[load] requested Theta={theta:.6g}, using nearest Theta={chosen_T:.6g}")
    return chosen_path


def load_matrix(theta: float):
    path = find_matrix_file(theta)
    with open(path, "rb") as fp:
        M, meta = pickle.load(fp)
    return M, meta, path

def _as_csr(X):
    return X if hasattr(X, "tocsr") else csr_matrix(X)

def compute_and_plot_eigenmodes(theta=None):
    if theta is None:
        theta = Theta_eigs
    M, meta, path = load_matrix(theta)
    print(f"Loaded matrix: {path}")
    print(f"Matrix shape: {M.shape}")

    nx = meta["nx"]
    ny = meta["ny"]
    half = int(meta.get("half", meta["Nmax"] // 2))
    Nstates = nx.size

    active = meta.get("active", None)
    if active is None or len(active) == 0:
        active = np.arange(Nstates, dtype=np.int32)
    active = np.asarray(active, dtype=np.int32)
    Nactive = int(active.size)

    # symmetry maps on the FULL lattice (discrete torus)
    idx_map = make_index_map(nx.astype(np.int32), ny.astype(np.int32), int(meta["Nmax"]), half)
    inv_map, rot90_map = build_symmetry_maps(nx.astype(np.int32), ny.astype(np.int32), idx_map, int(meta["Nmax"]), half)

    # arrays needed for ring + symmetry diagnostics (use saved meta if present)
    px = meta.get("px", None)
    py = meta.get("py", None)
    P  = meta.get("P", None)
    if px is None or py is None or P is None:
        # fallback: rebuild from nx,ny and dp
        dp_val = float(meta["dp"])
        px = dp_val * nx.astype(np.float64)
        py = dp_val * ny.astype(np.float64)
        P  = np.sqrt(px * px + py * py)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    P  = np.asarray(P, dtype=np.float64)

    # Restrict to active subspace
    # New format: generator may save the active-only operator directly (CSR, shape Nactive x Nactive).
    if getattr(M, "shape", None) == (Nactive, Nactive) and bool(meta.get("active_only", False)):
        Maa = _as_csr(M)
    else:
        # Legacy full-matrix format (may be huge)
        Maa = _as_csr(M[np.ix_(active, active)])

    # Build generalized weight matrix W = diag(f(1-f)) on the active subspace
    f_full = meta.get("f", None)
    if f_full is None:
        raise KeyError("meta['f'] not found. Regenerate matrices with f stored in meta.")

    w = f_full[active] * (1.0 - f_full[active])
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 0.0, None)

    # Avoid exact zeros in the generalized mass matrix (ARPACK dislikes singular M)
    w_eps = 1e-30
    w_safe = np.where(w > 0.0, w, w_eps)
    W = diags(w_safe, 0, format="csr")

    # Paper spectrum: (-Maa) v = gamma * W v
    A = _as_csr(-Maa)

    # Tiny regularization on A helps convergence near conserved modes
    regA = 1e-14
    A = A + diags([regA] * A.shape[0], 0, format="csr")

    n = A.shape[0]
    # compute extra candidates, then filter down to the "physical" m=4*n family
    k_calc = min(n - 2, max(N_EIG_CANDIDATES, N_EIG_PLOT + 8))
    if k_calc <= 0:
        raise RuntimeError("Active subspace too small to compute eigenmodes.")

    # Symmetry check in the weighted inner product:
    # A is W-self-adjoint if A^T W ≈ W A  (equivalently B = D^{-1/2} A D^{-1/2} symmetric)
    Dinv_sqrt = diags(1.0 / np.sqrt(w_safe), 0, format="csr")
    B = Dinv_sqrt @ A @ Dinv_sqrt
    sym_num = spnorm(B - B.T)
    sym_den = spnorm(B) + 1e-30
    sym_err = float(sym_num / sym_den)
    print(f"[W-symmetry] ||B-B^T||/||B|| = {sym_err:.3e}")

    sigma = 1e-10

    if sym_err < 1e-8:
        # Use symmetric solver -> returns orthonormal y, which implies W-orthonormal v
        vals, y = eigsh(
            B,
            k=k_calc,
            sigma=sigma,
            which="LM",
            tol=1e-8,
            maxiter=20000
        )
        # Recover v = D^{-1/2} y
        vecs = (Dinv_sqrt @ y).astype(np.float64)
    else:
        # Non-symmetric fallback (you should expect failure of simple W-orthogonality here)
        vals, vecs = eigs(
            A,
            M=W,
            k=k_calc,
            sigma=sigma,
            which="LM",
            tol=1e-8,
            maxiter=20000
        )

    # Warn if eigenvalues have significant imaginary parts (non-normal / numerical noise)
    max_im = float(np.max(np.abs(np.imag(vals)))) if vals.size else 0.0
    max_re = float(np.max(np.abs(np.real(vals)))) if vals.size else 0.0
    if max_im > 1e-8 * (max_re + 1e-14):
        print(f"WARNING: eigenvalues have non-negligible imaginary parts: max|Im|={max_im:.3e}")

    vals = np.real(vals)
    vecs = np.real(vecs)

    # Sort by gamma (ascending = slowest first)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    # Optionally drop near-zero (conserved) modes
    if not INCLUDE_CONSERVED:
        keep = np.where(np.abs(vals) > ZERO_TOL)[0]
        if keep.size == 0:
            raise RuntimeError("All computed modes are ~zero. Increase k_calc or change ZERO_TOL.")
        vals = vals[keep]
        vecs = vecs[:, keep]

    # Normalize all candidate eigenvectors in the generalized inner product: v^T W v = 1
    for i in range(vecs.shape[1]):
        v = vecs[:, i]
        norm2 = float(np.dot(v, w_safe * v))
        if norm2 > 0.0:
            vecs[:, i] = v / math.sqrt(norm2)

    # Embed candidates back into full lattice and classify by C4 + inversion + sign-switch count
    w_full = np.asarray(meta["f"], dtype=np.float64) * (1.0 - np.asarray(meta["f"], dtype=np.float64))
    candidates = []
    for i in range(vecs.shape[1]):
        v_active = vecs[:, i].copy()

        # sign convention: make max-abs component positive (visual stability)
        j = int(np.argmax(np.abs(v_active)))
        if v_active[j] < 0:
            v_active = -v_active

        v_full = np.zeros(Nstates, dtype=np.float64)
        v_full[active] = v_active

        # symmetry correlations in W-inner-product on FULL lattice
        c_inv = w_corr(v_full, v_full[inv_map],  w_full)
        c_c4  = w_corr(v_full, v_full[rot90_map], w_full)

        switches, m_est, n_est = estimate_sign_switches_on_ring(
            v_full=v_full,
            active=active,
            px=px, py=py, P=P,
            dp_val=float(meta["dp"]),
            theta_val=float(theta),
        )

        # convert to the "physical" label m = 4*n (n integer if switches multiple of 8)
        m_round = int(np.rint(m_est))
        n_round = int(np.rint(n_est))

        candidates.append({
            "gamma": float(vals[i]),
            "v_full": v_full,
            "c_inv": float(c_inv),
            "c_c4": float(c_c4),
            "switches": int(switches),
            "m_est": float(m_est),
            "m_round": int(m_round),
            "n_round": int(n_round),
        })

    # NEW: pick one representative per desired m using sign-switch m_est
    by_m = {m: [] for m in TARGET_MS}
    for c in candidates:
        m = int(c["m_round"])
        if m not in by_m:
            continue
        if abs(c["m_est"] - float(m)) > M_TOL:
            continue
        # parity sanity check under inversion: even m => +1, odd m => -1 (approximately)
        if (m % 2) == 0:
            if c["c_inv"] < INV_MIN_CORR:
                continue
        else:
            if c["c_inv"] > -INV_MIN_CORR:
                continue
        by_m[m].append(c)

    chosen = []
    missing = []
    for m in TARGET_MS:
        lst = by_m[m]
        if not lst:
            missing.append(m)
            continue
        lst.sort(key=lambda d: d["gamma"])  # slowest (smallest gamma) first
        chosen.append(lst[0])

    if missing:
        print(f"WARNING: missing m={missing}. Try increasing N_EIG_CANDIDATES or relaxing M_TOL/INV_MIN_CORR.")

    # Print a compact table so you can verify you are getting m=0..12
    print("\n[mode selection]")
    print("  idx   gamma         cC4     cInv    switches   m_est   m_round")
    for k, c in enumerate(chosen):
        print(f"  {k:3d}  {c['gamma']:.6e}  {c['c_c4']:+.3f}  {c['c_inv']:+.3f}   "
              f"{c['switches']:4d}     {c['m_est']:6.2f}    {c['m_round']:4d}")

    # Final arrays for plotting
    vals = np.array([c["gamma"] for c in chosen], dtype=np.float64)
    full_modes = [c["v_full"] for c in chosen]
    mode_labels = [(c["m_round"], c["switches"], c["c_c4"], c["c_inv"]) for c in chosen]
    n_show = len(full_modes)

    # Plot on the same grid
    ncols = 3
    nrows = int(math.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    pmin = (-half) * meta["dp"]
    pmax = (half - 1) * meta["dp"]
    pcenter = (pmin + pmax) / 2.0
    prange = (pmax - pmin) / 2.0
    pmin_plot = pcenter - prange * PLOT_SCALE_FACTOR
    pmax_plot = pcenter + prange * PLOT_SCALE_FACTOR

    # Compute global min/max across all eigenmodes for common colorbar
    global_vmax_abs = 0.0
    for k in range(n_show):
        grid = values_to_grid(full_modes[k], nx, ny, half, meta["Nmax"])
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
    im_common = None

    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        if k >= n_show:
            ax.axis("off")
            continue

        grid = values_to_grid(full_modes[k], nx, ny, half, meta["Nmax"])
        im = ax.imshow(
            grid.T,
            origin="lower",
            extent=[pmin, pmax, pmin, pmax],
            aspect="equal",
            cmap="seismic",
            norm=global_norm
        )
        im_common = im

        # Increase visible range
        ax.set_xlim(pmin_plot, pmax_plot)
        ax.set_ylim(pmin_plot, pmax_plot)

        gamma_str = format_scientific_latex(vals[k], precision=3)
        m_round, switches, c4, cinv = mode_labels[k]
        ax.set_title(rf"$m={m_round}$", fontsize=20)
        ax.set_xlabel("$p_x$")
        ax.set_ylabel("$p_y$")

    if im_common is not None:
        cbar = fig.colorbar(im_common, ax=axes, fraction=0.03, pad=0.02, aspect=30)
        cbar.ax.tick_params(labelsize=10)

    fig.savefig(f"eigenmodes_T{theta:.6g}.png", dpi=300)
    fig.savefig(f"eigenmodes_T{theta:.6g}.svg")
    print(f"Saved: eigenmodes_T{theta:.6g}.png")
    print(f"Saved: eigenmodes_T{theta:.6g}.svg")


if __name__ == "__main__":
    # Part (1): always works
    plot_f1mf()

    # Part (2): requires saved matrix for Theta_eigs
    # Run for all specified theta values
    theta_list = [
        # 0.0025,
        # 0.00310001,
        # 0.00384403,
        # 0.00476661,
        # 0.00591061,
        # 0.00732918,
        # 0.00908822,
        # 0.0112694,
        # 0.0139741,
        # 0.017328,
        # 0.0214868,
        # 0.0266437,
        # 0.0330383,
        # 0.0409676,
        # 0.0508,
        # 0.0629922,
        # 0.0781105,
        # 0.0968574,
        # 0.120104,
        # 0.148929,
        # 0.184672,
        # 0.228995,
        0.283954,
        # 0.352104,
        # 0.436611,
        # 0.541399,
        # 0.671337,
    ]
    
    for theta in theta_list:
        print(f"\n{'='*60}")
        print(f"Processing Theta = {theta:.6g}")
        print(f"{'='*60}")
        try:
            compute_and_plot_eigenmodes(theta=theta)
        except Exception as e:
            print(f"ERROR processing Theta={theta:.6g}: {e}")
            import traceback
            traceback.print_exc()
            continue
