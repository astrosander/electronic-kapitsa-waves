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

def wdot(w: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Complex-safe W-weighted inner product: <a, b>_W = a^H (W b)"""
    return np.vdot(a, w * b)


def wnorm(w: np.ndarray, a: np.ndarray) -> float:
    """W-norm: sqrt(<a, a>_W)"""
    return float(np.sqrt(max(wdot(w, a, a).real, 0.0)))


def normalize_w(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Normalize v in W-norm"""
    n = np.sqrt(wdot(w, v, v).real) + 1e-300
    return v / n


def wortho_columns(w: np.ndarray, B: np.ndarray, eps: float = 1e-14):
    """
    W-orthonormalize columns of B using modified Gram-Schmidt.
    Returns Q such that Q^H W Q = I (columns are W-orthonormal).
    """
    if B is None or B.size == 0:
        return np.zeros((B.shape[0] if B is not None else 0, 0), dtype=np.complex128)
    
    Q = []
    for i in range(B.shape[1]):
        v = B[:, i].astype(np.complex128, copy=True)
        # Project out previous columns
        for q in Q:
            v -= wdot(w, q, v) * q  # since q is W-orthonormal
        n = np.sqrt(wdot(w, v, v).real)
        if n > eps:
            Q.append(v / n)
    return np.column_stack(Q) if Q else np.zeros((B.shape[0], 0), dtype=np.complex128)


def project_out_w_orthonormal(w: np.ndarray, v: np.ndarray, Q: Optional[np.ndarray]) -> np.ndarray:
    """
    Project v to be W-orthogonal to columns of Q (which are W-orthonormal).
    If Q^H W Q = I, then projection is: v <- v - Q (Q^H (W v))
    """
    if Q is None or Q.size == 0:
        return v
    return v - Q @ (Q.conj().T @ (w * v))


def project_out_subspace(w: np.ndarray, v: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Project v to be W-orthogonal to columns of B (N x k).
    DEPRECATED: Use wortho_columns + project_out_w_orthonormal for stability.
    Kept for backward compatibility.
    """
    if B is None or B.size == 0:
        return v
    Q = wortho_columns(w, B)
    return project_out_w_orthonormal(w, v, Q)


# ----------------------- decay-rate diagnostics (Rayleigh quotient) -----------------------

def gamma_rayleigh(A: csr_matrix, w: np.ndarray, eta: np.ndarray) -> float:
    """
    For the generalized problem A v = gamma W v, with W=diag(w),
    Rayleigh quotient gamma(eta) = (eta^H A eta) / (eta^H W eta).
    Here A = (-Iee). Uses complex-safe vdot.
    """
    num = np.vdot(eta, A @ eta).real
    den = np.vdot(eta, w * eta).real
    return num / (den + 1e-300)


def residual_rel(A: csr_matrix, w: np.ndarray, v: np.ndarray, gamma: float) -> float:
    """Relative residual for eigenpair: ||A v - gamma W v|| / ||A v||"""
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
    Handles complex eigenvalues properly and sorts by distance to sigma.
    """
    A = (-Iee).astype(np.float64).tocsr()
    w = w_active.astype(np.float64)

    # Mask out very small weights to avoid numerical issues
    wmin = 1e-18
    keep = w > wmin
    if not np.all(keep):
        print(f"Warning: masking {np.sum(~keep)} points with w <= {wmin}")
        A = A[keep][:, keep]
        w = w[keep]

    W = diags(w, 0, format="csr")

    k = int(min(max(2, nev), A.shape[0] - 2))
    vals, vecs = eigs(A, M=W, k=k, sigma=sigma, which="LM", tol=tol, maxiter=maxiter)

    # Check if imaginary parts are negligible
    max_imag_gam = np.max(np.abs(np.imag(vals)))
    max_imag_vecs = np.max([np.linalg.norm(np.imag(vecs[:, j])) for j in range(vecs.shape[1])])
    
    if max_imag_gam < 1e-12 and max_imag_vecs < 1e-10:
        gam = np.real(vals)
        vecs = np.real(vecs)
    else:
        print(f"Warning: non-negligible imaginary parts: max|imag(gam)|={max_imag_gam:.2e}, max|imag(vecs)|={max_imag_vecs:.2e}")
        gam = vals
        # Keep vecs complex

    # Sort by distance to sigma (not by real part)
    order = np.argsort(np.abs(vals - sigma))
    gam = gam[order]
    vecs = vecs[:, order]
    
    # Normalize eigenvectors once in W-norm
    for j in range(vecs.shape[1]):
        n = np.sqrt(wdot(w, vecs[:, j], vecs[:, j]).real) + 1e-300
        vecs[:, j] = vecs[:, j] / n
    
    return gam, vecs, A, keep


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

def harmonic_score(w: np.ndarray, v: np.ndarray, theta: np.ndarray, P: np.ndarray, m: int, 
                   ring_width: Optional[float] = None) -> float:
    """
    Score overlap with cos(mθ) and sin(mθ) under W-inner product.
    Returns max(|<v,cos>|, |<v,sin>|) normalized by norms.
    For m=0 uses only cos(0)=1.
    
    If ring_width is provided, restricts scoring to points with |P-1| < ring_width
    to focus on the Fermi ring and avoid radial structure contamination.
    """
    if ring_width is not None:
        mask_ring = np.abs(P - 1.0) < ring_width
        if np.sum(mask_ring) == 0:
            # Fallback if mask is empty
            mask_ring = np.ones_like(P, dtype=bool)
        w_ring = w[mask_ring]
        v_ring = v[mask_ring]
        theta_ring = theta[mask_ring]
    else:
        w_ring = w
        v_ring = v
        theta_ring = theta
    
    if m == 0:
        b = np.ones_like(theta_ring, dtype=np.complex128)
        return abs(wdot(w_ring, v_ring, b)) / (wnorm(w_ring, v_ring) * wnorm(w_ring, b) + 1e-300)

    c = np.cos(m * theta_ring).astype(np.complex128)
    s = np.sin(m * theta_ring).astype(np.complex128)
    nv = wnorm(w_ring, v_ring)
    nc = wnorm(w_ring, c)
    ns = wnorm(w_ring, s)
    sc = abs(wdot(w_ring, v_ring, c)) / (nv * nc + 1e-300)
    ss = abs(wdot(w_ring, v_ring, s)) / (nv * ns + 1e-300)
    return float(max(sc, ss))


def pick_mode_for_m(gam: np.ndarray,
                    vecs: np.ndarray,
                    A: csr_matrix,
                    w: np.ndarray,
                    theta: np.ndarray,
                    P: np.ndarray,
                    m: int,
                    Qproj: Optional[np.ndarray],
                    score_min: float = 0.35,
                    ring_width: Optional[float] = None):
    """
    Pick the smallest-gamma mode with good harmonic score at this m,
    after projecting candidates to be W-orthogonal to Qproj (optional, W-orthonormal).

    Returns: (gamma, vec, score, resid)
    gamma is the actual eigenvalue (not Rayleigh quotient).
    resid is computed on the original eigenvector (not projected).
    """
    best_j = None
    best_gamma = None
    best_score = -1.0
    best_v = None

    # rank candidates by increasing gamma (real part if complex)
    gam_real = np.real(gam)
    order = np.argsort(gam_real)

    for j in order:
        v_orig = vecs[:, j]  # original eigenvector
        v = v_orig.copy()
        
        # Project for scoring only (don't modify original)
        if Qproj is not None:
            v = project_out_w_orthonormal(w, v, Qproj)
        v = normalize_w(w, v)

        sc = harmonic_score(w, v, theta, P, m, ring_width=ring_width)
        if sc < score_min:
            continue

        # Use actual eigenvalue, not Rayleigh quotient
        g_j = float(np.real(gam[j]))
        
        # We want the smallest positive decay (closest to 0 but >=0).
        # However allow tiny negative numerical noise.
        if best_j is None or g_j < best_gamma:
            best_j = j
            best_gamma = g_j
            best_score = sc
            best_v = v_orig  # Keep original for residual check

    if best_j is None:
        # fallback: pick by max score regardless of gamma
        for j in range(vecs.shape[1]):
            v_orig = vecs[:, j]
            v = v_orig.copy()
            if Qproj is not None:
                v = project_out_w_orthonormal(w, v, Qproj)
            v = normalize_w(w, v)
            sc = harmonic_score(w, v, theta, P, m, ring_width=ring_width)
            if sc > best_score:
                best_score = sc
                best_v = v_orig
                best_gamma = float(np.real(gam[j]))
        # Compute residual on original eigenvector
        resid = residual_rel(A, w, best_v, best_gamma)
        return best_gamma, best_v, best_score, resid

    # Compute residual on original eigenvector (not projected)
    resid = residual_rel(A, w, best_v, best_gamma)
    return best_gamma, best_v, best_score, resid


def pick_mode_by_target(gam: np.ndarray,
                       vecs: np.ndarray,
                       A: csr_matrix,
                       w: np.ndarray,
                       target: np.ndarray,
                       Qproj: Optional[np.ndarray] = None,
                       score_min: float = 0.5):
    """
    Choose the smallest-gamma eigenvector that has large W-overlap with 'target'
    (after optional projection out of Qproj, which is W-orthonormal).

    Returns: (gamma, vec, score, resid)
    gamma is the actual eigenvalue (not Rayleigh quotient).
    resid is computed on the original eigenvector (not projected).
    """
    t = normalize_w(w, target.copy())
    best = None
    gam_real = np.real(gam)

    for j in np.argsort(gam_real):
        v_orig = vecs[:, j]  # original eigenvector
        v = v_orig.copy()
        
        # Project for scoring only
        if Qproj is not None:
            v = project_out_w_orthonormal(w, v, Qproj)
        v = normalize_w(w, v)

        sc = abs(wdot(w, v, t))  # since both W-normalized
        if sc < score_min:
            continue

        g_j = float(np.real(gam[j]))  # use actual eigenvalue
        res = residual_rel(A, w, v_orig, g_j)  # residual on original

        if best is None or g_j < best[0]:
            best = (g_j, v_orig, sc, res)  # return original eigenvector

    if best is None:
        # fallback: maximum overlap
        best_sc = -1.0
        best_tuple = None
        for j in range(vecs.shape[1]):
            v_orig = vecs[:, j]
            v = v_orig.copy()
            if Qproj is not None:
                v = project_out_w_orthonormal(w, v, Qproj)
            v = normalize_w(w, v)
            sc = abs(wdot(w, v, t))
            if sc > best_sc:
                g_j = float(np.real(gam[j]))
                res = residual_rel(A, w, v_orig, g_j)
                best_sc = sc
                best_tuple = (g_j, v_orig, sc, res)
        return best_tuple

    return best


def pick_eigenmode_matching_target(gam: np.ndarray,
                                   vecs: np.ndarray,
                                   w: np.ndarray,
                                   target: np.ndarray,
                                   gamma_min: float = 1e-5):
    """
    Choose the eigenvector v_j (UNMODIFIED) that best matches 'target' in W-inner product,
    skipping near-zero modes with gamma < gamma_min.
    Returns (j, score).
    """
    t = target.copy()
    t /= (np.sqrt(np.sum(w * t * t)) + 1e-300)

    best_j = None
    best_sc = -1.0

    for j in range(vecs.shape[1]):
        if gam[j] < gamma_min:
            continue
        v = vecs[:, j]
        nv = np.sqrt(np.sum(w * v * v)) + 1e-300
        sc = abs(np.sum(w * v * t)) / nv  # since t is W-normalized
        if sc > best_sc:
            best_sc = sc
            best_j = j

    return best_j, best_sc


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
        # Take real part for display (imag should be negligible if checked)
        v = np.real(v)
        v_disp = v / (np.max(np.abs(v)) + 1e-30)
        # Use symmetric percentile-based clipping for stability
        vmax = np.percentile(np.abs(v_disp), 99)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

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
    nev = 120   # Number of generalized eigenpairs to compute (increased to capture J_perp modes)
    sigma = 0.0#0.0043#0.0  # Shift-invert target gamma (0 targets longest-lived, or use 0.004 to target J_perp band)
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
    
    # Sanity check: P should be near 1 for active points
    assert np.median(P) > 0.8 and np.median(P) < 1.2, f"Unexpected P range: median={np.median(P)}"

    # solve generalized eigenproblem (may mask out small weights)
    gam, vecs, A, keep = solve_generalized_eigs(Iee, w, nev=nev, sigma=sigma, tol=tol, maxiter=maxiter)
    
    # Apply mask to geometry and targets if weights were masked
    if not np.all(keep):
        px = px[keep]
        py = py[keep]
        P = P[keep]
        th = th[keep]
        w = w[keep]

    # build targets (after masking)
    t = build_targets(meta, px, py, P, th)
    etaN = t["N"]
    etaPx = t["Px"]
    etaPy = t["Py"]
    etaE = t["E"]
    etaJx = t["Jx"]
    etaJy = t["Jy"]
    
    # Orthogonalize energy against number if both exist
    if etaE is not None:
        etaE = etaE - (wdot(w, etaN, etaE) / (wdot(w, etaN, etaN).real + 1e-300)) * etaN
        etaE = normalize_w(w, etaE)

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

    # Spectral decomposition of J_perp onto computed eigenspace
    if etaJx_perp is not None:
        # Eigenvectors are already W-normalized from solve_generalized_eigs
        V = vecs
        t = normalize_w(w, etaJx_perp.copy())  # your current-perp target

        # coefficients c_j = <v_j, t>_W
        c = np.array([wdot(w, V[:, j], t) for j in range(V.shape[1])], dtype=np.complex128)

        # sort by |c| descending
        idx = np.argsort(np.abs(c))[::-1]

        print("\n--- J_perp spectral decomposition (top 10) ---")
        for r in range(min(10, len(idx))):
            j = idx[r]
            print(f"rank {r:2d}: gamma={np.real(gam[j]):.6e}  |c|={abs(c[j]):.3f}")

        # "reconstructed" Rayleigh from the subspace (if eigenvectors were complete and W-orthonormal)
        # not exact in non-Hermitian case, but still a useful summary:
        print("sum |c|^2 over computed eigs =", float(np.sum(np.abs(c)**2)))

    # Build conserved projection basis if requested (W-orthonormalized)
    Qproj = None
    if project_conserved:
        cols = [normalize_w(w, etaN), normalize_w(w, etaPx), normalize_w(w, etaPy)]
        if etaE is not None:
            cols.append(etaE)  # already normalized and orthogonalized
        Bproj = np.column_stack(cols)
        Qproj = wortho_columns(w, Bproj)
    
    # Build momentum projection basis (W-orthonormalized)
    Qmom = None
    if (etaJx is not None) and (etaJy is not None):
        Bmom = np.column_stack([etaPx_n, etaPy_n])
        Qmom = wortho_columns(w, Bmom)

    # Pre-select m=1 current-like eigenmode using unmodified eigenvectors
    v1_eigen = None
    g1_eigen = None
    sc1_eigen = None
    res1_eigen = None
    if etaJx_perp is not None:
        # combine x/y dipoles to avoid choosing the wrong orientation
        tJ = etaJx_perp  # or use both: choose max over (Jx_perp, Jy_perp)
        j1, sc1 = pick_eigenmode_matching_target(gam, vecs, w, tJ, gamma_min=1e-5)
        if j1 is not None:
            v1_eigen = vecs[:, j1]  # IMPORTANT: NOT projected!
            g1_eigen = float(np.real(gam[j1]))
            sc1_eigen = sc1
            # check eigen-residual of the true eigenpair
            res1_eigen = residual_rel(A, w, v1_eigen, g1_eigen)
            print("m=1 current-like eigenmode:",
                  f"gamma={g1_eigen:.6e}, score={sc1_eigen:.2f}, resid={res1_eigen:.2e}")

    # pick modes for m=0..mmax
    # Use ring_width = 2*dp for harmonic scoring to focus on Fermi ring
    dp = float(meta.get("dp", 0.01))
    ring_width = 2.0 * dp
    
    modes = []
    for m in range(mmax + 1):
        if m == 0:
            # number-like scalar (or use energy target if you want)
            g_m, v_m, sc_m, res_m = pick_mode_by_target(gam, vecs, A, w, etaN, Qproj=None, score_min=0.7)
        elif m == 1:
            # THIS is the non-parabolic current-relaxing dipole:
            # use the pre-selected eigenmode if available, otherwise fallback
            if v1_eigen is not None:
                g_m, v_m, sc_m, res_m = g1_eigen, v1_eigen, sc1_eigen, res1_eigen
            elif etaJx_perp is not None and Qmom is not None:
                # fallback to target-based selection with projection
                g_m, v_m, sc_m, res_m = pick_mode_by_target(gam, vecs, A, w, etaJx_perp, Qproj=Qmom, score_min=0.5)
            else:
                # fallback to harmonic-based selection if J_perp not available
                g_m, v_m, sc_m, res_m = pick_mode_for_m(gam, vecs, A, w, th, P, m, Qproj=None, ring_width=ring_width)
        else:
            # harmonic-like modes (optionally project out conserved subspace for m>=2)
            Qm = Qproj if (project_conserved and m >= 2) else None
            g_m, v_m, sc_m, res_m = pick_mode_for_m(gam, vecs, A, w, th, P, m, Qm, ring_width=ring_width)
        modes.append({"m": m, "gamma": g_m, "score": sc_m, "resid": res_m, "vec": v_m})

    # print summary
    print("\n--- Selected modes ---")
    for item in modes:
        print(f"m={item['m']:2d}  γ≈{item['gamma']:.6e}  (score {item['score']:.2f})  resid≈{item['resid']:.2e}")

    # plot
    plot_modes(px, py, modes, Theta, out)


if __name__ == "__main__":
    main()
