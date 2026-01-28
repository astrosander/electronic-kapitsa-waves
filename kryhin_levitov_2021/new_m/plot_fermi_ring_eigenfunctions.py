#!/usr/bin/env python3
"""
plot_fermi_ring_eigenfunctions.py

Solve the generalized eigenproblem for the linearized collision operator:
    Iee[eta] = -gamma * f0(1-f0) * eta
i.e.
    (-I) v = gamma * W v,  W = diag(f0(1-f0))

and plot eigenfunctions on the (px,py) active "Fermi ring" for m=0..8.

Works with files produced by Iee_matrix_bruteforce_generate.py:
  Matrixes_bruteforce/M_Iee_N{Nmax}_dp{dp}_T{Theta}.pkl
which contain:
  (M_to_save, meta)
  meta includes: Nmax, half, dp, shift_x, shift_y, active, w_active, Theta, ...
"""

import os
import re
import glob
import pickle
from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs


# ----------------------- I/O helpers -----------------------

def parse_theta_from_filename(path: str):
    # expects ..._T{Theta}.pkl where Theta can be in scientific notation
    m = re.search(r"_T([0-9eE\+\-\.]+)\.pkl$", os.path.basename(path))
    return float(m.group(1)) if m else None


def pick_file(indir: str, theta: Optional[float]):
    files = sorted(glob.glob(os.path.join(indir, "M_Iee_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No files matching {indir}/M_Iee_*.pkl")

    if theta is None:
        # default: pick the first file
        return files[0]

    thetas = []
    for f in files:
        th = parse_theta_from_filename(f)
        if th is not None:
            thetas.append((abs(th - theta), th, f))
    if not thetas:
        raise RuntimeError("Could not parse Theta from filenames.")
    thetas.sort(key=lambda x: x[0])
    return thetas[0][2]


def load_operator(path: str):
    with open(path, "rb") as fp:
        M, meta = pickle.load(fp)

    if not hasattr(M, "tocsr"):
        M = csr_matrix(M)
    else:
        M = M.tocsr()

    if "active" not in meta or "w_active" not in meta:
        raise KeyError("meta must contain 'active' and 'w_active' to form W=diag(f0(1-f0)).")

    return M, meta


# ----------------------- geometry helpers -----------------------

def active_px_py(meta: dict):
    """
    Reconstruct (px,py) for active states.

    Generator used build_centered_lattice:
      ns = [-half, ..., half-1]
      meshgrid(ns,ns, indexing="ij"), reshape(-1)
    so for global index g:
      ix = g // Nmax, iy = g % Nmax
      nx = ix - half, ny = iy - half
    and physical (dimensionless) momenta:
      px = dp * (nx + shift_x), py = dp * (ny + shift_y)
    """
    Nmax = int(meta["Nmax"])
    half = int(meta["half"])
    dp = float(meta["dp"])
    sx = float(meta.get("shift_x", 0.0))
    sy = float(meta.get("shift_y", 0.0))

    active = meta["active"].astype(np.int64)

    ix = active // Nmax
    iy = active % Nmax
    nx = ix - half
    ny = iy - half

    px = dp * (nx.astype(np.float64) + sx)
    py = dp * (ny.astype(np.float64) + sy)

    P = np.sqrt(px * px + py * py)
    theta = np.arctan2(py, px)

    return px, py, P, theta


# ----------------------- eigen + classification -----------------------

def solve_generalized(Iee: csr_matrix, w_active: np.ndarray, nev: int, sigma: float, tol: float, maxiter: int):
    """
    Solve (-Iee) v = gamma * W v  with W = diag(w_active).

    Uses ARPACK via scipy.sparse.linalg.eigs (general, no symmetry assumed).
    """
    A = (-Iee).astype(np.float64).tocsr()
    w = w_active.astype(np.float64)
    if np.any(w <= 0):
        raise ValueError("w_active must be positive to form a valid mass matrix W.")
    W = diags(w, 0, format="csr")

    # Shift-invert around sigma (default sigma=0 finds smallest |gamma| efficiently)
    vals, vecs = eigs(A, M=W, k=nev, sigma=sigma, which="LM", tol=tol, maxiter=maxiter)

    # Physical expectation: gamma real and >=0 (small imag parts may occur numerically)
    gam = np.real(vals)
    vecs = np.real(vecs)

    # Sort by gamma
    order = np.argsort(gam)
    gam = gam[order]
    vecs = vecs[:, order]
    return gam, vecs


def weighted_inner(w, a, b):
    return float(np.sum(w * a * b))


def best_match_in_degenerate_subspace(vecs_sub, w, target):
    """
    Given eigenvectors spanning (nearly) degenerate subspace: V (N x d),
    find combination v = V c maximizing <v,target>_W / ||v||_W.
    Solution: c ∝ (V^T W V)^{-1} (V^T W target).
    """
    V = vecs_sub
    d = V.shape[1]

    # Gram matrix G = V^T W V
    G = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            G[i, j] = weighted_inner(w, V[:, i], V[:, j])

    u = np.array([weighted_inner(w, V[:, i], target) for i in range(d)], dtype=np.float64)

    # Regularize lightly in case ARPACK vectors are nearly linearly dependent
    G_reg = G + 1e-14 * np.eye(d)

    c = np.linalg.solve(G_reg, u)
    v = V @ c

    # normalize in W-norm
    nrm2 = weighted_inner(w, v, v)
    if nrm2 > 0:
        v = v / np.sqrt(nrm2)
    return v


def pick_mode_for_m(gam, vecs, w, theta, m, score_min=0.35, deg_rel=1e-6, max_deg_dim=4):
    """
    Pick the longest-lived eigenvector in the full generalized spectrum
    that has strong overlap with angular harmonic m.

    We:
      1) score each eigenvector by overlap with cos(mθ) or sin(mθ) (W-weighted),
      2) prefer the smallest gamma among those with score >= score_min,
      3) within a small degenerate cluster around that gamma, build the best
         linear combination matching cos or sin (still an eigenvector if exactly degenerate;
         numerically it's a very good representative of that eigenspace).
    """
    if m == 0:
        bcos = np.ones_like(theta)
        bsin = None
    else:
        bcos = np.cos(m * theta)
        bsin = np.sin(m * theta)

    # precompute target norms
    nbcos = np.sqrt(weighted_inner(w, bcos, bcos))
    nbsin = np.sqrt(weighted_inner(w, bsin, bsin)) if bsin is not None else 0.0

    scores = []
    for j in range(vecs.shape[1]):
        v = vecs[:, j]
        nv = np.sqrt(weighted_inner(w, v, v))
        if nv == 0:
            scores.append(0.0)
            continue

        sc = abs(weighted_inner(w, v, bcos)) / (nv * nbcos + 1e-30)
        if bsin is not None:
            ss = abs(weighted_inner(w, v, bsin)) / (nv * nbsin + 1e-30)
            scores.append(max(sc, ss))
        else:
            scores.append(sc)

    scores = np.array(scores)

    eligible = np.where(scores >= score_min)[0]
    if eligible.size == 0:
        # fallback: best score overall
        j0 = int(np.argmax(scores))
    else:
        # among eligible, pick smallest gamma (longest lived)
        j0 = int(eligible[np.argmin(gam[eligible])])

    g0 = float(gam[j0])

    # collect a small "degenerate" neighborhood
    # (helps produce a clean cos/sin-like representative if ARPACK returned a mixture)
    if g0 == 0:
        close = np.where(np.abs(gam - g0) <= 1e-12)[0]
    else:
        close = np.where(np.abs(gam - g0) <= deg_rel * max(1.0, abs(g0)))[0]

    close = close[:max_deg_dim]  # cap
    Vsub = vecs[:, close]

    # pick best match to cos vs sin
    vcos = best_match_in_degenerate_subspace(Vsub, w, bcos)
    score_cos = abs(weighted_inner(w, vcos, bcos)) / (np.sqrt(weighted_inner(w, vcos, vcos)) * nbcos + 1e-30)

    if bsin is not None:
        vsin = best_match_in_degenerate_subspace(Vsub, w, bsin)
        score_sin = abs(weighted_inner(w, vsin, bsin)) / (np.sqrt(weighted_inner(w, vsin, vsin)) * nbsin + 1e-30)
        if score_sin > score_cos:
            return g0, vsin, score_sin
    return g0, vcos, score_cos


# ----------------------- plotting -----------------------

def plot_modes(px, py, P, modes, theta_val, outpath):
    """
    modes: list of (m, gamma, vec, score)
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    # common limits
    lim = max(np.max(np.abs(px)), np.max(np.abs(py)))
    lim = max(lim, 1.2)

    last_sc = None
    for ax, (m, gamma, v, score) in zip(axes, modes):
        # normalize for display only (does not change the eigenproblem)
        v_disp = v / (np.max(np.abs(v)) + 1e-30)

        norm = TwoSlopeNorm(vcenter=0.0, vmin=np.min(v_disp), vmax=np.max(v_disp))
        sc = ax.scatter(px, py, c=v_disp, s=8, norm=norm, cmap="RdBu_r", linewidths=0)
        last_sc = sc

        # draw the ideal FS circle P=1 for reference
        circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=1.0, alpha=0.6, color="k")
        ax.add_patch(circle)

        ax.set_aspect("equal", "box")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r"$p_x$")
        ax.set_ylabel(r"$p_y$")
        ax.set_title(f"m={m}   γ≈{gamma:.3e}   (score {score:.2f})")

    # colorbar
    cbar = fig.colorbar(last_sc, ax=axes.tolist(), shrink=0.75)
    cbar.set_label("eigenfunction (normalized for display)")

    fig.suptitle(f"Generalized eigenmodes on active Fermi ring   Θ={theta_val}", y=0.92)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    print(f"Saved: {outpath}")


# ----------------------- main -----------------------

def main():
    indir =r"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\kryhin_levitov_2021\collision_integral_direct\Matrixes_bruteforce"# "Matrixes_bruteforce"
    theta = None
    nev = 80
    sigma = 0.0
    tol = 1e-10
    maxiter = 200000
    mmax = 8
    out = "fermi_ring_eigenmodes.png"

    path = pick_file(indir, theta)
    Iee, meta = load_operator(path)

    Theta = float(meta.get("Theta", np.nan))
    print(f"Loaded: {path}")
    print(f"Theta={Theta}  Nmax={meta.get('Nmax')}  dp={meta.get('dp')}  Nactive={len(meta['active'])}")

    w = meta["w_active"].astype(np.float64)

    # Solve (-Iee) v = gamma W v
    gam, vecs = solve_generalized(Iee, w, nev=nev, sigma=sigma, tol=tol, maxiter=maxiter)

    # Reconstruct geometry for plotting + harmonic classification
    px, py, P, th = active_px_py(meta)

    # Pick modes for m=0..mmax
    modes = []
    for m in range(mmax + 1):
        g0, v0, score = pick_mode_for_m(gam, vecs, w, th, m)
        modes.append((m, g0, v0, score))

    plot_modes(px, py, P, modes, Theta, out)


if __name__ == "__main__":
    main()
