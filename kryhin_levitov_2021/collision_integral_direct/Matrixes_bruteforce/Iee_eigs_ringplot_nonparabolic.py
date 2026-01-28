#!/usr/bin/env python3
"""
Solve the generalized eigenproblem for the linearized ee collision operator
on the ACTIVE Cartesian (px,py) shell produced by Iee_matrix_bruteforce_generate.py:

    Iee[eta] = -gamma * f0*(1-f0) * eta

We solve it numerically as:
    A eta = gamma * W eta
with A = -sym(Iee_matrix) and W = diag(f0*(1-f0)).

Then identify m=0..mmax by overlaps with templates:
  m=0: density / constant
  m=1: CURRENT harmonic (vx, vy)  [NOT momentum]
  m>=2: angular harmonics cos(mθ), sin(mθ)

Finally: scatter plots on the px–py "Fermi ring/shell" (the active set).

Usage examples:
  python Iee_eigs_ringplot_nonparabolic.py --pkl Matrixes_bruteforce/M_Iee_...pkl --mmax 8 --k 60
  python Iee_eigs_ringplot_nonparabolic.py --dir Matrixes_bruteforce --mmax 8 --k 80

Notes:
- If m=1 current is protected (parabolic/Galilean), its gamma should be ~0.
- In non-parabolic (Dirac-like) regime, current is not proportional to momentum => gamma_current > 0.
"""

import argparse
import os
import pickle
import numpy as np

from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt


# ------------------------- band helpers (FULL dispersion) -------------------------

def eps_phys(k: np.ndarray, v: float, U: float) -> np.ndarray:
    """ε(k) = sqrt(v^2 k^2 + U^2) - U"""
    return np.sqrt((v * k) ** 2 + U * U) - U

def vgroup_phys(k: np.ndarray, v: float, U: float) -> np.ndarray:
    """v(k) = dε/dk = v^2 k / sqrt(v^2 k^2 + U^2)"""
    denom = np.sqrt((v * k) ** 2 + U * U)
    denom = np.maximum(denom, 1e-30)
    return (v * v * k) / denom


# ------------------------- lattice reconstruction -------------------------

def build_centered_lattice(Nmax: int):
    half = Nmax // 2
    ns = np.arange(-half, half, dtype=np.int32)
    nx, ny = np.meshgrid(ns, ns, indexing="ij")
    return nx.ravel(), ny.ravel(), half

def reconstruct_active_coords(meta: dict):
    """
    Rebuild (px,py,P,theta,k_phys, vg_phys) for ACTIVE indices using meta.
    """
    Nmax   = int(meta["Nmax"])
    dp     = float(meta["dp"])
    shift_x = float(meta.get("shift_x", 0.0))
    shift_y = float(meta.get("shift_y", 0.0))

    active = np.array(meta["active"], dtype=np.int32)

    nx, ny, half = build_centered_lattice(Nmax)

    nxa = nx[active].astype(np.float64)
    nya = ny[active].astype(np.float64)

    px = dp * (nxa + shift_x)
    py = dp * (nya + shift_y)

    P = np.sqrt(px * px + py * py)
    theta = np.arctan2(py, px)

    # Convert to physical k using k = P * kF0 (kF0 saved by generator)
    kF0 = float(meta.get("kF0", 1.0))
    k_phys = P * kF0

    v_band = float(meta.get("v_band", meta.get("V_BAND", 1.0)))
    U_band = float(meta.get("U_band", meta.get("U_BAND", 1.0)))
    vg_phys = vgroup_phys(k_phys, v_band, U_band)

    return px, py, P, theta, k_phys, vg_phys


# ------------------------- generalized eigen-solve -------------------------

def symmetrize_sparse(M: csr_matrix) -> csr_matrix:
    return (M + M.T) * 0.5

def solve_weighted_modes(M_in, w_active, k=60, symmetrize=True):
    """
    Solve:  (-sym(M)) eta = gamma * diag(w) eta

    Return gamma (ascending) and eta vectors (columns), B-normalized:
      sum_i w_i eta_i^2 = 1
    """
    if not issparse(M_in):
        M = csr_matrix(M_in)
    else:
        M = M_in.tocsr()

    M = M.astype(np.float64)

    if symmetrize:
        M = symmetrize_sparse(M)

    A = (-M).tocsr()  # expect PSD

    w = np.asarray(w_active, dtype=np.float64)
    if np.any(w <= 0):
        raise ValueError("w_active contains non-positive entries; active cutoff should prevent this.")

    inv_sqrt_w = 1.0 / np.sqrt(w)

    # Form K = W^{-1/2} A W^{-1/2} (symmetric) as an explicit sparse matrix:
    # K_ij = inv_sqrt_w[i] * A_ij * inv_sqrt_w[j]
    # Implement by scaling rows/cols.
    # Row scaling:
    A_row = A.multiply(inv_sqrt_w[:, None])
    # Column scaling:
    K = A_row.multiply(inv_sqrt_w[None, :]).tocsr()

    # Compute smallest eigenvalues (closest to 0)
    # For near-singular K, SM can converge slowly; bump maxiter/tol.
    evals, u = eigsh(K, k=k, which="SM", tol=1e-10, maxiter=5000)

    # Sort
    idx = np.argsort(evals)
    gammas = np.array(evals[idx], dtype=np.float64)
    u = u[:, idx]

    # Convert back: eta = W^{-1/2} u
    eta = (inv_sqrt_w[:, None] * u)

    # B-normalize each column: sum w eta^2 = 1
    for j in range(eta.shape[1]):
        nrm = np.sqrt(np.dot(w, eta[:, j] * eta[:, j]))
        if nrm > 0:
            eta[:, j] /= nrm

    return gammas, eta


# ------------------------- template overlaps / mode identification -------------------------

def w_inner(a, b, w):
    return float(np.dot(w, a * b))

def norm_w(a, w):
    return np.sqrt(max(w_inner(a, a, w), 1e-300))

def overlap(a, t, w):
    return w_inner(a, t, w) / (norm_w(a, w) * norm_w(t, w))

def pick_modes_by_overlap(gammas, eta, w, theta, px, py, vg_phys, mmax=8):
    """
    Returns dict m -> chosen eigen-index, plus diagnostics overlaps.
    Special handling for m=1: CURRENT templates vx,vy (not momentum).
    """
    k = eta.shape[1]
    used = set()

    # Templates
    t_density = np.ones_like(theta)

    # Momentum templates (for diagnostics / paper comparison)
    t_momx = px.copy()
    t_momy = py.copy()

    # Current templates (requested definition)
    t_curx = vg_phys * np.cos(theta)
    t_cury = vg_phys * np.sin(theta)

    # Precompute cos/sin templates for m>=2 (and for diagnostics m=1 as angular)
    t_cos = {}
    t_sin = {}
    for m in range(1, mmax + 1):
        t_cos[m] = np.cos(m * theta)
        t_sin[m] = np.sin(m * theta)

    chosen = {}
    diag = {}

    # m=0: density/constant
    best_i, best_s = None, -1.0
    for i in range(k):
        s = abs(overlap(eta[:, i], t_density, w))
        if s > best_s and i not in used:
            best_s, best_i = s, i
    chosen[0] = best_i
    used.add(best_i)

    diag[0] = {
        "gamma": gammas[best_i],
        "ov_density": overlap(eta[:, best_i], t_density, w),
        "ov_momx": overlap(eta[:, best_i], t_momx, w),
        "ov_curx": overlap(eta[:, best_i], t_curx, w),
    }

    # m=1: CURRENT harmonic (vx/vy)
    best_i, best_s = None, -1.0
    for i in range(k):
        if i in used:
            continue
        ovx = overlap(eta[:, i], t_curx, w)
        ovy = overlap(eta[:, i], t_cury, w)
        s = np.sqrt(ovx * ovx + ovy * ovy)
        if s > best_s:
            best_s, best_i = s, i

    chosen[1] = best_i
    used.add(best_i)

    # diagnostics: also show momentum overlap for the chosen current mode
    diag[1] = {
        "gamma": gammas[best_i],
        "ov_cur_amp": best_s,
        "ov_curx": overlap(eta[:, best_i], t_curx, w),
        "ov_cury": overlap(eta[:, best_i], t_cury, w),
        "ov_momx": overlap(eta[:, best_i], t_momx, w),
        "ov_momy": overlap(eta[:, best_i], t_momy, w),
        "ov_ang_amp": np.sqrt(overlap(eta[:, best_i], t_cos[1], w)**2 + overlap(eta[:, best_i], t_sin[1], w)**2),
    }

    # m>=2: angular harmonics
    for m in range(2, mmax + 1):
        best_i, best_s = None, -1.0
        for i in range(k):
            if i in used:
                continue
            ovc = overlap(eta[:, i], t_cos[m], w)
            ovs = overlap(eta[:, i], t_sin[m], w)
            s = np.sqrt(ovc * ovc + ovs * ovs)
            if s > best_s:
                best_s, best_i = s, i

        chosen[m] = best_i
        used.add(best_i)

        diag[m] = {
            "gamma": gammas[best_i],
            "ov_ang_amp": best_s,
            "ov_cos": overlap(eta[:, best_i], t_cos[m], w),
            "ov_sin": overlap(eta[:, best_i], t_sin[m], w),
        }

    return chosen, diag


# ------------------------- plotting -------------------------

def ring_scatter(ax, px, py, val, title=""):
    # Robust color limits
    vmax = np.percentile(np.abs(val), 99.0)
    if vmax <= 0:
        vmax = 1.0
    sc = ax.scatter(px, py, c=val, s=6, vmin=-vmax, vmax=vmax)
    ax.set_aspect("equal", "box")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc

def make_figure(px, py, modes, gammas, mu_phys, U_band, Theta, out_prefix):
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    axes = axes.ravel()

    mlist = list(range(0, 9))
    last_sc = None
    for j, m in enumerate(mlist):
        idx = modes[m]
        val = modes["_eta"][:, idx]
        g = gammas[idx]
        last_sc = ring_scatter(
            axes[j], px, py, val,
            title=f"m={m}   γ={g:.3e}"
        )

    fig.suptitle(f"Non-parabolic Iee modes on active shell | μ={mu_phys:g}, U={U_band:g}, Θ={Theta:g}", fontsize=12)

    # Add one shared colorbar
    cbar = fig.colorbar(last_sc, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("η(px,py)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png = out_prefix + ".png"
    pdf = out_prefix + ".pdf"
    fig.savefig(png, dpi=250)
    fig.savefig(pdf)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# ------------------------- CLI -------------------------

def load_one_pkl(path):
    with open(path, "rb") as f:
        M, meta = pickle.load(f)
    if issparse(M):
        M = M.tocsr()
    else:
        M = csr_matrix(M)
    return M, meta

def find_pkls_in_dir(d):
    out = []
    for fn in os.listdir(d):
        if fn.endswith(".pkl") and fn.startswith("M_Iee_"):
            out.append(os.path.join(d, fn))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pkl", type=str, help="Path to one matrix .pkl produced by generator")
    g.add_argument("--dir", type=str, help="Directory containing M_Iee_*.pkl")
    ap.add_argument("--k", type=int, default=80, help="Number of eigenpairs to compute")
    ap.add_argument("--mmax", type=int, default=8, help="Max m to identify (plots m=0..8)")
    ap.add_argument("--no_sym", action="store_true", help="Disable explicit symmetrization (not recommended)")
    ap.add_argument("--show", action="store_true", help="Show matplotlib windows")
    args = ap.parse_args()

    pkls = [args.pkl] if args.pkl else find_pkls_in_dir(args.dir)
    if not pkls:
        raise RuntimeError("No .pkl files found.")

    for path in pkls:
        print("\n=== Processing:", path)
        M, meta = load_one_pkl(path)

        w = np.asarray(meta["w_active"], dtype=np.float64)

        px, py, P, theta, k_phys, vg_phys = reconstruct_active_coords(meta)

        mu_phys = float(meta.get("mu_phys", np.nan))
        U_band  = float(meta.get("U_band", np.nan))
        Theta   = float(meta.get("Theta", np.nan))

        # Solve
        gammas, eta = solve_weighted_modes(M, w, k=args.k, symmetrize=(not args.no_sym))

        # Identify modes by overlap
        chosen, diag = pick_modes_by_overlap(
            gammas, eta, w, theta, px, py, vg_phys, mmax=args.mmax
        )

        # Print diagnostics (this is where you check μ=0.1 vs 10 expectations)
        print("Mode identification by overlaps (B-inner product with w=f0(1-f0)):")
        for m in range(0, args.mmax + 1):
            i = chosen[m]
            d = diag[m]
            if m == 1:
                print(f"  m=1 (CURRENT): idx={i:3d}  gamma={d['gamma']:.6e}  |ov_current|={d['ov_cur_amp']:.3f}  "
                      f"ov_momx={d['ov_momx']:.3f}")
            else:
                key = "ov_density" if m == 0 else "ov_ang_amp"
                print(f"  m={m}:           idx={i:3d}  gamma={d['gamma']:.6e}  overlap={d.get(key, np.nan):.3f}")

        # Store eta for plotting convenience
        chosen["_eta"] = eta

        # Plot 3x3 (m=0..8)
        base = os.path.splitext(os.path.basename(path))[0]
        out_prefix = f"ring_modes_{base}_mu{mu_phys:g}_U{U_band:g}_T{Theta:g}"
        make_figure(px, py, chosen, gammas, mu_phys, U_band, Theta, out_prefix)

        if args.show:
            plt.show()
        else:
            plt.close("all")


if __name__ == "__main__":
    main()
