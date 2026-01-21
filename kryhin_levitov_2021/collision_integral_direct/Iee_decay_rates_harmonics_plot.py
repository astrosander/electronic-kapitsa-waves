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
from scipy.sparse.linalg import eigs

# plt.rcParams['text.usetex'] = True
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
Theta_eigs = 0.2
N_EIG_PLOT = 6              # how many eigenfunctions to plot
ZERO_TOL = 1e-10            # treat |lambda|<ZERO_TOL as "conserved/zero" mode

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
        im = ax.imshow(grid.T, origin="lower", extent=[pmin, pmax, pmin, pmax], aspect="equal", cmap='rainbow')
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
def compute_and_plot_eigenmodes():
    M, meta, path = load_matrix(Theta_eigs)
    print(f"Loaded matrix: {path}")
    print(f"Matrix shape: {M.shape}")

    nx = meta["nx"]
    ny = meta["ny"]
    half = int(meta.get("half", meta["Nmax"] // 2))
    Nstates = nx.size

    active = meta.get("active", None)
    if active is None or len(active) == 0:
        active = np.arange(Nstates, dtype=np.int32)

    # Restrict to active subspace
    Maa = M[np.ix_(active, active)]

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
    A = csr_matrix(-Maa)

    # Tiny regularization on A helps convergence near conserved modes
    regA = 1e-14
    A = A + diags([regA] * A.shape[0], 0, format="csr")

    n = A.shape[0]
    k_calc = min(n - 2, max(N_EIG_PLOT + 8, N_EIG_PLOT))
    if k_calc <= 0:
        raise RuntimeError("Active subspace too small to compute eigenmodes.")

    # Target slow modes (small gamma). Shift-invert around small sigma.
    sigma = 1e-10

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

    # Remove near-zero modes (conserved/zero modes)
    keep = np.where(np.abs(vals) > ZERO_TOL)[0]
    if keep.size == 0:
        raise RuntimeError("All computed modes are ~zero. Increase k_calc or change ZERO_TOL.")

    vals = vals[keep]
    vecs = vecs[:, keep]

    # Take first N_EIG_PLOT
    n_show = min(N_EIG_PLOT, vals.size)
    vals = vals[:n_show]
    vecs = vecs[:, :n_show]

    # Normalize eigenvectors in the generalized inner product: v^T W v = 1
    for i in range(n_show):
        v = vecs[:, i]
        norm2 = float(np.dot(v, w_safe * v))
        if norm2 <= 0.0:
            continue
        vecs[:, i] = v / math.sqrt(norm2)

    # Check generalized orthogonality: G_ij = v_i^T W v_j should be ~delta_ij
    G = vecs.T @ (w_safe[:, None] * vecs)
    diagG = np.diag(G)
    offG = G - np.diag(diagG)
    max_off = float(np.max(np.abs(offG))) if offG.size else 0.0
    max_diag_dev = float(np.max(np.abs(diagG - 1.0))) if diagG.size else 0.0
    print(f"[W-orthogonality] max|offdiag|={max_off:.3e}, max|diag-1|={max_diag_dev:.3e}")

    # Embed eigenvectors back into full lattice for plotting
    full_modes = []
    for i in range(n_show):
        v_active = vecs[:, i].copy()

        # sign convention: make max-abs component positive (visual stability)
        j = int(np.argmax(np.abs(v_active)))
        if v_active[j] < 0:
            v_active = -v_active

        v_full = np.zeros(Nstates, dtype=np.float64)
        v_full[active] = v_active
        full_modes.append(v_full)

    # Plot on the same grid
    ncols = 3
    nrows = int(math.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    pmin = (-half) * meta["dp"]
    pmax = (half - 1) * meta["dp"]

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

        gamma_str = format_scientific_latex(vals[k], precision=3)
        ax.set_title(rf"$m={k+1}$, $\gamma = {gamma_str}$")
        ax.set_xlabel("$p_x$")
        ax.set_ylabel("$p_y$")

    if im_common is not None:
        cbar = fig.colorbar(im_common, ax=axes, fraction=0.03, pad=0.02, aspect=30)
        cbar.ax.tick_params(labelsize=10)

    fig.savefig(f"eigenmodes_T{Theta_eigs:.6g}.png", dpi=300)
    fig.savefig(f"eigenmodes_T{Theta_eigs:.6g}.svg")
    print(f"Saved: eigenmodes_T{Theta_eigs:.6g}.png")
    print(f"Saved: eigenmodes_T{Theta_eigs:.6g}.svg")


if __name__ == "__main__":
    # Part (1): always works
    plot_f1mf()

    # Part (2): requires saved matrix for Theta_eigs
    compute_and_plot_eigenmodes()
