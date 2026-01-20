#!/usr/bin/env python3
"""
Iee_grid_diagnostics.py

1) Plot f(1-f) over the 2D momentum lattice for T/E_F = 1, 0.1, 0.01.
2) Plot several eigenmodes (eigenfunctions) over the same lattice.

Eigenmodes are computed from the ACTIVE subspace only (meta["active"]) to reduce cost.
You need SciPy for eigenmodes:
  pip install scipy
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



# ---------------- USER SETTINGS ----------------
# Grid (must match your generator choices if you want to compare to saved matrices)
Nmax = 100
dp   = 0.08

# For part (1)
Thetas_weight = [1.0, 0.1, 0.01]

# For part (2): choose one temperature that you have already generated a matrix for
Theta_eigs = 0.01
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
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$f(1-f)$")

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
    # half = meta["half"]
    half = int(meta.get("half", meta["Nmax"] // 2))
    Nstates = nx.size

    active = meta.get("active", None)
    if active is None or len(active) == 0:
        active = np.arange(Nstates, dtype=np.int32)

    # Restrict to active subspace
    Maa = M[np.ix_(active, active)]

    # We want decay rates => eigenvalues of A = -M
    A = csr_matrix(-Maa)

    # Regularize a tiny bit to help ARPACK with near-zero modes
    reg = 1e-14
    A = A + diags([reg] * A.shape[0], 0, format="csr")

    # Ask for a few more than needed, then discard near-zero
    k_calc = min(A.shape[0] - 2, max(N_EIG_PLOT + 8, N_EIG_PLOT))
    if k_calc <= 0:
        raise RuntimeError("Active subspace too small to compute eigenmodes.")

    # "SM" = smallest magnitude (will include near-zero modes)
    # vals, vecs = eigs(A, k=k_calc, which="SM")
    sigma = 1e-10

    vals, vecs = eigs(
        A,
        k=k_calc,
        sigma=sigma,        # target eigenvalues near sigma
        which="LM",         # after shift-invert, "LM" is what you want
        tol=1e-8,
        maxiter=20000
    )

    vals = np.real(vals)
    vecs = np.real(vecs)

    vals = np.real(vals)
    vecs = np.real(vecs)

    # sort by eigenvalue
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    # remove near-zero modes
    keep = np.where(np.abs(vals) > ZERO_TOL)[0]
    if keep.size == 0:
        raise RuntimeError("All computed modes are ~zero. Increase k_calc or change ZERO_TOL.")

    vals = vals[keep]
    vecs = vecs[:, keep]

    # take first N_EIG_PLOT
    n_show = min(N_EIG_PLOT, vals.size)
    vals = vals[:n_show]
    vecs = vecs[:, :n_show]

    # Embed eigenvectors back into full lattice for plotting
    full_modes = []
    for i in range(n_show):
        v_active = vecs[:, i]

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

    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        if k >= n_show:
            ax.axis("off")
            continue

        grid = values_to_grid(full_modes[k], nx, ny, half, meta["Nmax"])
        # Compute adaptive scale: use symmetric scaling for diverging colormap
        grid_valid = grid[~np.isnan(grid)]
        if len(grid_valid) > 0:
            vmax_abs = np.max(np.abs(grid_valid))
            # Use symmetric scale for better visualization of eigenmodes
            vmin = -vmax_abs
            vmax = vmax_abs
            # Use TwoSlopeNorm to enhance visibility of small non-zero values
            # This stretches the colormap around zero, making small deviations more visible
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            vmin, vmax = -0.1, 0.1
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        # Use 'seismic' colormap: high contrast, designed for visualizing small deviations from zero
        # Dark red/blue at extremes, white at center - makes small non-zero values very visible
        im = ax.imshow(grid.T, origin="lower", extent=[pmin, pmax, pmin, pmax], aspect="equal", cmap='seismic', norm=norm)
        print(grid.T[grid.T != 0])
        lambda_str = format_scientific_latex(vals[k], precision=3)
        ax.set_title(rf"$m={k+1}$, $\lambda = {lambda_str}$")
        ax.set_xlabel("$p_x$")
        ax.set_ylabel("$p_y$")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # fig.suptitle(rf"Eigenfunctions on momentum lattice (T/E_F={Theta_eigs})", y=1.02)
    fig.savefig(f"eigenmodes_T{Theta_eigs:.6g}.png", dpi=300)
    fig.savefig(f"eigenmodes_T{Theta_eigs:.6g}.svg")
    print(f"Saved: eigenmodes_T{Theta_eigs:.6g}.png")
    print(f"Saved: eigenmodes_T{Theta_eigs:.6g}.svg")
    # plt.show()


if __name__ == "__main__":
    # Part (1): always works
    plot_f1mf()

    # Part (2): requires saved matrix for Theta_eigs
    compute_and_plot_eigenmodes()
