#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fermi_ring_eigenfunctions.py

Solve the generalized eigenproblem (NO symmetrization, NO sqrt(W) transforms):
    Iee[eta] = -gamma * f0(1-f0) * eta
discretized on the active shell as:
    (-I) v = gamma * W v,   W = diag(w_active), w_active = f0(1-f0)

Then:
  - build physically meaningful conserved targets (N, Px, Py, and energy-like if desired)
  - build current/velocity targets (Jx, Jy) for your non-parabolic band (from meta U_bar, V_bar)
  - compute Rayleigh-quotient decay rates for any chosen perturbation
  - compute eigenpairs near gamma~0 via ARPACK shift-invert generalized eigensolver
  - label modes m=0..mmax by angular harmonic overlap (cos mθ / sin mθ) AFTER
    projecting out the conserved subspace (optional but recommended)
  - plot eigenfunctions on (px,py) “Fermi ring”

This version fixes your NameError (w_active) and is self-contained.

Requirements:
  numpy, scipy, matplotlib
"""

import os
import re
import glob
import pickle
import numpy as np
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs


# ----------------------- file helpers -----------------------

def parse_theta_from_filename(path: str):
    m = re.search(r"_T([0-9eE\+\-\.]+)\.pkl$", os.path.basename(path))
    return float(m.group(1)) if m else None


def pick_file(indir: str, theta: Optional[float]):
    files = sorted(glob.glob(os.path.join(indir, "M_Iee_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No files matching {indir}/M_Iee_*.pkl")

    if theta is None:
        return files[0]

    candidates = []
    for f in files:
        th = parse_theta_from_filename(f)
        if th is not None:
            candidates.append((abs(th - theta), th, f))
    if not candidates:
        raise RuntimeError("Could not parse Theta from filenames (expected ..._T{Theta}.pkl).")

    candidates.sort(key=lambda x: x[0])
    return candidates[0][2]


def load_operator(path: str):
    with open(path, "rb") as fp:
        M, meta = pickle.load(fp)

    if not hasattr(M, "tocsr"):
        M = csr_matrix(M)
    else:
        M = M.tocsr()

    if "active" not in meta or "w_active" not in meta:
        raise KeyError("meta must contain 'active' and 'w_active'.")

    return M, meta


# ----------------------- geometry reconstruction -----------------------

def active_px_py(meta: dict):
    """
    Reconstruct (px,py) for active indices.

    Generator used build_centered_lattice with indexing="ij" and reshape(-1),
    so global index g maps to:
      ix = g // Nmax
      iy = g %  Nmax
      nx = ix - half
      ny = iy - half

    Physical momenta:
      px = dp * (nx + shift_x)
      py = dp * (ny + shift_y)
    """
    Nmax = int(meta["Nmax"])
    half = int(meta["half"])
    dp = float(meta["dp"])
    sx = float(meta.get("shift_x", 0.0))
    sy = float(meta.get("shift_y", 0.0))

    active = np.asarray(meta["active"], dtype=np.int64)

    ix = active // Nmax
    iy = active % Nmax
    nx = ix - half
    ny = iy - half

    px = dp * (nx.astype(np.float64) + sx)
    py = dp * (ny.astype(np.float64) + sy)

    P = np.sqrt(px * px + py * py)
    theta = np.arctan2(py, px)

    return px, py, P, theta


# ----------------------- weighted inner product and projections -----------------------

def wdot(w: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(w * a * b))


def wnorm(w: np.ndarray, a: np.ndarray) -> float:
    return float(np.sqrt(max(wdot(w, a, a), 0.0)))


def project_out_subspace(w: np.ndarray, v: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Project v to be W-orthogonal to columns of B (N x k).
    Uses normal equations: c = (B^T W B)^{-1} (B^T W v), then v <- v - B c.
    """
    if B is None or B.size == 0:
        return v

    k = B.shape[1]
    G = np.zeros((k, k), dtype=np.float64)
    rhs = np.zeros((k,), dtype=np.float64)

    for i in range(k):
        rhs[i] = wdot(w, B[:, i], v)
        for j in range(k):
            G[i, j] = wdot(w, B[:, i], B[:, j])

    G = G + 1e-14 * np.eye(k)
    c = np.linalg.solve(G, rhs)
    return v - B @ c


def normalize_w(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    n = wnorm(w, v)
    if n <= 0:
        return v
    return v / n


# ----------------------- decay-rate diagnostics (Rayleigh quotient) -----------------------

def gamma_rayleigh(A: csr_matrix, w: np.ndarray, eta: np.ndarray) -> float:
    """
    For the generalized problem A v = gamma W v, with W=diag(w),
    Rayleigh quotient gamma(eta) = (eta^T A eta) / (eta^T W eta).
    Here A = (-Iee).
    """
    num = float(np.real(np.dot(eta, A @ eta)))
    den = float(np.real(np.dot(eta, w * eta)))
    return num / (den + 1e-300)


def residual_rel(A: csr_matrix, w: np.ndarray, v: np.ndarray, gamma: float) -> float:
    r = (A @ v) - gamma * (w * v)
    denom = np.linalg.norm(A @ v) + 1e-300
    return float(np.linalg.norm(r) / denom)


# ----------------------- generalized eigen-solve -----------------------

def solve_generalized_eigs(Iee: csr_matrix,
                           w_active: np.ndarray,
                           nev: int,
                           sigma: float = 0.0,
                           tol: float = 1e-10,
                           maxiter: int = 200000):
    """
    Solve (-Iee) v = gamma W v, with W=diag(w_active).
    No symmetry assumptions; uses ARPACK via scipy.sparse.linalg.eigs.

    Shift-invert with sigma=0.0 typically returns eigenvalues closest to 0 (long-lived).
    """
    A = (-Iee).astype(np.float64).tocsr()
    w = w_active.astype(np.float64)

    if np.any(w <= 0):
        raise ValueError("w_active must be strictly positive for a valid mass matrix W.")

    W = diags(w, 0, format="csr")

    k = int(min(max(2, nev), A.shape[0] - 2))
    vals, vecs = eigs(A, M=W, k=k, sigma=sigma, which="LM", tol=tol, maxiter=maxiter)

    gam = np.real(vals)
    vecs = np.real(vecs)

    order = np.argsort(gam)
    gam = gam[order]
    vecs = vecs[:, order]
    return gam, vecs, A


# ----------------------- targets: conserved and current -----------------------

def build_targets(meta: dict, px, py, P, theta):
    """
    Build W-weighted target vectors for:
      number: 1
      momentum: px, py
      energy-like: eps(P) (dimensionless), optional (if meta contains band info)
      current/velocity: vx, vy computed from band parameters (U_bar, V_bar)
    """
    # number
    eta_N = np.ones_like(px, dtype=np.float64)

    # momentum
    eta_Px = px.astype(np.float64).copy()
    eta_Py = py.astype(np.float64).copy()

    # energy-like (dimensionless eps, if available)
    eta_E = None
    if ("U_bar" in meta) and ("V_bar" in meta):
        U_bar = float(meta["U_bar"])
        V_bar = float(meta["V_bar"])
        # eps(P) = sqrt((V_bar P)^2 + U_bar^2) - U_bar
        eta_E = np.sqrt((V_bar * P) ** 2 + (U_bar ** 2)) - U_bar

    # current/velocity (isotropic band -> v points radially)
    eta_Jx = None
    eta_Jy = None
    if ("U_bar" in meta) and ("V_bar" in meta):
        U_bar = float(meta["U_bar"])
        V_bar = float(meta["V_bar"])
        vP = (V_bar * V_bar * P) / np.sqrt((V_bar * P) ** 2 + (U_bar ** 2))
        eta_Jx = vP * np.cos(theta)
        eta_Jy = vP * np.sin(theta)

    return {
        "N": eta_N,
        "Px": eta_Px,
        "Py": eta_Py,
        "E": eta_E,
        "Jx": eta_Jx,
        "Jy": eta_Jy,
    }


# ----------------------- mode labeling by angular harmonics -----------------------

def harmonic_score(w: np.ndarray, v: np.ndarray, theta: np.ndarray, m: int) -> float:
    """
    Score overlap with cos(mθ) and sin(mθ) under W-inner product.
    Returns max(|<v,cos>|, |<v,sin>|) normalized by norms.
    For m=0 uses only cos(0)=1.
    """
    if m == 0:
        b = np.ones_like(theta, dtype=np.float64)
        return abs(wdot(w, v, b)) / (wnorm(w, v) * wnorm(w, b) + 1e-300)

    c = np.cos(m * theta)
    s = np.sin(m * theta)
    nv = wnorm(w, v)
    nc = wnorm(w, c)
    ns = wnorm(w, s)
    sc = abs(wdot(w, v, c)) / (nv * nc + 1e-300)
    ss = abs(wdot(w, v, s)) / (nv * ns + 1e-300)
    return float(max(sc, ss))


def pick_mode_for_m(gam: np.ndarray,
                    vecs: np.ndarray,
                    A: csr_matrix,
                    w: np.ndarray,
                    theta: np.ndarray,
                    m: int,
                    Bproj: Optional[np.ndarray],
                    score_min: float = 0.35):
    """
    Pick the smallest-gamma mode with good harmonic score at this m,
    after projecting candidates to be W-orthogonal to Bproj (optional).

    Returns: (gamma_est, vec, score, resid)
    gamma_est is recomputed via Rayleigh quotient for stability.
    """
    best_j = None
    best_gamma = None
    best_score = -1.0
    best_v = None

    # rank candidates by increasing gamma, try those first
    order = np.argsort(gam)

    for j in order:
        v = vecs[:, j].copy()
        if Bproj is not None:
            v = project_out_subspace(w, v, Bproj)
        v = normalize_w(w, v)

        sc = harmonic_score(w, v, theta, m)
        if sc < score_min:
            continue

        g_est = gamma_rayleigh(A, w, v)
        # We want the smallest positive decay (closest to 0 but >=0).
        # However allow tiny negative numerical noise.
        if best_j is None or g_est < best_gamma:
            best_j = j
            best_gamma = g_est
            best_score = sc
            best_v = v

    if best_j is None:
        # fallback: pick by max score regardless of gamma
        for j in range(vecs.shape[1]):
            v = vecs[:, j].copy()
            if Bproj is not None:
                v = project_out_subspace(w, v, Bproj)
            v = normalize_w(w, v)
            sc = harmonic_score(w, v, theta, m)
            if sc > best_score:
                best_score = sc
                best_v = v
                best_gamma = gamma_rayleigh(A, w, v)
        resid = residual_rel(A, w, best_v, best_gamma)
        return best_gamma, best_v, best_score, resid

    resid = residual_rel(A, w, best_v, best_gamma)
    return best_gamma, best_v, best_score, resid


def pick_mode_by_target(gam: np.ndarray,
                       vecs: np.ndarray,
                       A: csr_matrix,
                       w: np.ndarray,
                       target: np.ndarray,
                       Bproj: Optional[np.ndarray] = None,
                       score_min: float = 0.5):
    """
    Choose the smallest-gamma eigenvector that has large W-overlap with 'target'
    (after optional projection out of Bproj).

    Returns: (gamma_est, vec, score, resid)
    gamma_est is recomputed via Rayleigh quotient for stability.
    """
    t = normalize_w(w, target.copy())
    best = None

    for j in np.argsort(gam):
        v = vecs[:, j].copy()
        if Bproj is not None:
            v = project_out_subspace(w, v, Bproj)
        v = normalize_w(w, v)

        sc = abs(wdot(w, v, t))  # since both W-normalized
        if sc < score_min:
            continue

        g_est = gamma_rayleigh(A, w, v)
        res = residual_rel(A, w, v, g_est)

        if best is None or g_est < best[0]:
            best = (g_est, v, sc, res)

    if best is None:
        # fallback: maximum overlap
        best_sc = -1.0
        best_tuple = None
        for j in range(vecs.shape[1]):
            v = vecs[:, j].copy()
            if Bproj is not None:
                v = project_out_subspace(w, v, Bproj)
            v = normalize_w(w, v)
            sc = abs(wdot(w, v, t))
            if sc > best_sc:
                g_est = gamma_rayleigh(A, w, v)
                res = residual_rel(A, w, v, g_est)
                best_sc = sc
                best_tuple = (g_est, v, sc, res)
        return best_tuple

    return best


# ----------------------- plotting -----------------------

def plot_modes(px, py, modes, Theta, outpath):
    """
    modes: list of dict with keys: m, gamma, score, vec
    """
    mmax = max(item["m"] for item in modes)
    nplots = mmax + 1
    if nplots != 9:
        # still layout 3x3 if possible
        nrows = int(np.ceil(np.sqrt(nplots)))
        ncols = int(np.ceil(nplots / nrows))
    else:
        nrows, ncols = 3, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
    axes = np.array(axes).ravel()

    lim = float(max(np.max(np.abs(px)), np.max(np.abs(py)), 1.2))

    last_sc = None
    for ax, item in zip(axes, modes):
        v = item["vec"]
        v_disp = v / (np.max(np.abs(v)) + 1e-30)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=float(np.min(v_disp)), vmax=float(np.max(v_disp)))

        sc = ax.scatter(px, py, c=v_disp, s=8, cmap="RdBu_r", norm=norm, linewidths=0)
        last_sc = sc

        circ = plt.Circle((0, 0), 1.0, fill=False, linewidth=1.0, alpha=0.6, color="k")
        ax.add_patch(circ)

        ax.set_aspect("equal", "box")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f"m={item['m']}  γ≈{item['gamma']:.3e}  (score {item['score']:.2f})", fontsize=11)

    for k in range(len(modes), len(axes)):
        axes[k].axis("off")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.tolist(), shrink=0.75)
        cbar.set_label("eigenfunction (normalized for display)")

    fig.suptitle(f"Generalized eigenmodes on active Fermi ring   Θ={Theta}", y=0.92, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=200)
    print(f"Saved: {outpath}")
    plt.close(fig)


# ----------------------- main -----------------------

def main():
    # Configuration parameters (previously argparse arguments)
    file_path = r"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\kryhin_levitov_2021\collision_integral_direct\Matrixes_bruteforce\M_Iee_N320_dp0.032716515_T0.1.pkl"#None  # Path to a single M_Iee_*.pkl, or None to auto-select
    indir = "Matrixes_bruteforce"  # Directory with M_Iee_*.pkl files
    theta = None  # Pick the file closest to this Theta (if file_path not set)
    nev = 200  # Number of generalized eigenpairs to compute (increased to capture J_perp modes)
    sigma = 0.0  # Shift-invert target gamma (0 targets longest-lived, or use 0.004 to target J_perp band)
    tol = 1e-10  # ARPACK tolerance
    maxiter = 200000  # ARPACK max iterations
    mmax = 8  # Plot m=0..mmax
    out = "Eigenfuncs/ring_modes.png"  # Output path for the plot
    project_conserved = True  # Project out conserved subspace {N,Px,Py,(E)} before labeling m>=2
    print_current_diag = True  # Print Rayleigh-quotient rates for diagnostics

    if file_path is not None:
        path = file_path
    else:
        path = pick_file(indir, theta)

    Iee, meta = load_operator(path)

    Theta = float(meta.get("Theta", np.nan))
    Nactive = int(len(meta["active"]))
    print(f"Loaded: {path}")
    print(f"Theta={Theta}  Nmax={meta.get('Nmax')}  dp={meta.get('dp')}  Nactive={Nactive}")

    # FIX: get w_active from meta
    w = np.asarray(meta["w_active"], dtype=np.float64)

    # reconstruct geometry
    px, py, P, th = active_px_py(meta)

    # solve generalized eigenproblem
    gam, vecs, A = solve_generalized_eigs(Iee, w, nev=nev, sigma=sigma, tol=tol, maxiter=maxiter)

    # build targets
    t = build_targets(meta, px, py, P, th)
    etaN = t["N"]
    etaPx = t["Px"]
    etaPy = t["Py"]
    etaE = t["E"]
    etaJx = t["Jx"]
    etaJy = t["Jy"]

    # Build momentum projection basis and J_perp (always needed for m=1 selection)
    etaPx_n = None
    etaPy_n = None
    etaJx_n = None
    etaJy_n = None
    etaJx_perp = None
    etaJy_perp = None
    Bmom = None

    if (etaJx is not None) and (etaJy is not None):
        # Normalize targets in W-norm for readable overlaps
        etaPx_n = normalize_w(w, etaPx)
        etaPy_n = normalize_w(w, etaPy)
        etaJx_n = normalize_w(w, etaJx)
        etaJy_n = normalize_w(w, etaJy)

        # Build momentum projection basis
        Bmom = np.column_stack([etaPx_n, etaPy_n])

        # project current perpendicular to momentum
        ax = wdot(w, etaPx_n, etaJx_n)
        ay = wdot(w, etaPy_n, etaJy_n)
        etaJx_perp = normalize_w(w, etaJx_n - ax * etaPx_n)
        etaJy_perp = normalize_w(w, etaJy_n - ay * etaPy_n)

    # optionally print current-vs-momentum diagnostics
    if print_current_diag and (etaJx is not None) and (etaJy is not None):
        print("\n--- Rayleigh-quotient decay diagnostics (A=-Iee) ---")
        print("gamma(Px)       =", gamma_rayleigh(A, w, etaPx_n))
        print("gamma(Py)       =", gamma_rayleigh(A, w, etaPy_n))
        print("gamma(Jx)       =", gamma_rayleigh(A, w, etaJx_n))
        print("gamma(Jy)       =", gamma_rayleigh(A, w, etaJy_n))
        print("gamma(Jx_perp)  =", gamma_rayleigh(A, w, etaJx_perp))
        print("gamma(Jy_perp)  =", gamma_rayleigh(A, w, etaJy_perp))
        print("overlap |<Px,Jx>| =", abs(ax))
        print("overlap |<Py,Jy>| =", abs(ay))

    # Build conserved projection basis if requested
    Bproj = None
    if project_conserved:
        cols = [normalize_w(w, etaN), normalize_w(w, etaPx), normalize_w(w, etaPy)]
        if etaE is not None:
            cols.append(normalize_w(w, etaE))
        Bproj = np.column_stack(cols)

    # pick modes for m=0..mmax
    modes = []
    for m in range(mmax + 1):
        if m == 0:
            # number-like scalar (or use energy target if you want)
            g_m, v_m, sc_m, res_m = pick_mode_by_target(gam, vecs, A, w, etaN, Bproj=None, score_min=0.7)
        elif m == 1:
            # THIS is the non-parabolic current-relaxing dipole:
            # pick the smallest-gamma mode overlapping with J_perp, after projecting out momentum
            if etaJx_perp is not None and Bmom is not None:
                g_m, v_m, sc_m, res_m = pick_mode_by_target(gam, vecs, A, w, etaJx_perp, Bproj=Bmom, score_min=0.5)
            else:
                # fallback to harmonic-based selection if J_perp not available
                g_m, v_m, sc_m, res_m = pick_mode_for_m(gam, vecs, A, w, th, m, Bproj=None)
        else:
            # harmonic-like modes (optionally project out conserved subspace for m>=2)
            Bm = Bproj if (project_conserved and m >= 2) else None
            g_m, v_m, sc_m, res_m = pick_mode_for_m(gam, vecs, A, w, th, m, Bm)
        modes.append({"m": m, "gamma": g_m, "score": sc_m, "resid": res_m, "vec": v_m})

    # print summary
    print("\n--- Selected modes ---")
    for item in modes:
        print(f"m={item['m']:2d}  γ≈{item['gamma']:.6e}  (score {item['score']:.2f})  resid≈{item['resid']:.2e}")

    # plot
    plot_modes(px, py, modes, Theta, out)


if __name__ == "__main__":
    main()
