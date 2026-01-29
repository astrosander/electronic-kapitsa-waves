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
import csv
import numpy as np

from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator

import matplotlib.pyplot as plt

# plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 16

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


# ------------------------- projected eigensolver -------------------------

def _qr_orthonormal_columns(B: np.ndarray) -> np.ndarray:
    """Thin QR; returns Q with orthonormal columns spanning cols(B)."""
    Q, _ = np.linalg.qr(B)
    return Q

def _make_projected_operator(K: csr_matrix, Q: np.ndarray) -> LinearOperator:
    """
    Return LinearOperator implementing (I - QQ^T) K (I - QQ^T).
    Q must have orthonormal columns (N x r).
    """
    def matvec(x):
        x = x - Q @ (Q.T @ x)
        y = K @ x
        y = y - Q @ (Q.T @ y)
        return y

    n = K.shape[0]
    return LinearOperator((n, n), matvec=matvec, dtype=np.float64)

def solve_weighted_modes_projected(M_in, w_active, templates_eta, k=60, symmetrize=True):
    """
    Solve generalized eigenproblem in the orthogonal complement of span(templates_eta)
    w.r.t. the W-inner product in eta-space.

    Internally solves symmetric K = W^{-1/2} A W^{-1/2} in u-space with Euclidean metric,
    and projects out u_templates = sqrt(w) * templates_eta.

    Returns (gammas, eta) as usual, B-normalized.
    """
    if not issparse(M_in):
        M = csr_matrix(M_in)
    else:
        M = M_in.tocsr()

    M = M.astype(np.float64)
    if symmetrize:
        M = symmetrize_sparse(M)

    A = (-M).tocsr()

    w = np.asarray(w_active, dtype=np.float64)
    if np.any(w <= 0):
        raise ValueError("w_active contains non-positive entries; active cutoff should prevent this.")

    inv_sqrt_w = 1.0 / np.sqrt(w)
    sqrt_w = np.sqrt(w)

    # K = W^{-1/2} A W^{-1/2}
    A_row = A.multiply(inv_sqrt_w[:, None])
    K = A_row.multiply(inv_sqrt_w[None, :]).tocsr()

    # Build invariant basis in u-space: u = sqrt(w) * t_eta
    Uinv = np.column_stack([sqrt_w * t for t in templates_eta]).astype(np.float64)

    # Orthonormalize basis (Euclidean)
    Q = _qr_orthonormal_columns(Uinv)

    # Projected operator
    Kproj = _make_projected_operator(K, Q)

    # Solve smallest projected eigenvalues
    evals, u = eigsh(Kproj, k=k, which="SM", tol=1e-10, maxiter=8000)
    idx = np.argsort(evals)
    gammas = np.array(evals[idx], dtype=np.float64)
    u = u[:, idx]

    # Back to eta
    eta = inv_sqrt_w[:, None] * u

    # B-normalize
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

def rayleigh_gamma_for_template(M, w, t_eta, symmetrize=True):
    """
    gamma_eff(t) = <t, A t> / <t, W t>, where A=-sym(M), W=diag(w).
    """
    if not issparse(M):
        M = csr_matrix(M)
    M = M.tocsr().astype(np.float64)
    if symmetrize:
        M = symmetrize_sparse(M)
    A = (-M).tocsr()

    num = np.dot(t_eta, A @ t_eta)
    den = w_inner(t_eta, t_eta, w)
    if den <= 1e-300:
        return np.nan
    return float(num / den)

def invariant_residual(M, w, t_eta, symmetrize=True):
    """
    Measure ||A t||_{W^{-1}} / ||t||_W  where A = -sym(M).
    This is 0 iff A t = 0 (exact invariant). Always >= 0.
    """
    if not issparse(M):
        M = csr_matrix(M)
    M = M.tocsr().astype(np.float64)
    if symmetrize:
        M = symmetrize_sparse(M)
    A = (-M).tocsr()

    w = np.asarray(w, dtype=np.float64)
    inv_w = 1.0 / np.maximum(w, 1e-300)

    r = A @ t_eta
    num = float(np.dot(inv_w, r * r))            # r^T W^{-1} r
    den = float(np.dot(w, t_eta * t_eta))        # t^T W t
    return np.sqrt(max(num / max(den, 1e-300), 0.0))

def eps_bar_from_meta(P: np.ndarray, meta: dict) -> np.ndarray:
    """Dimensionless ε(P) in units of MU_PHYS (same as generator)."""
    U_bar = float(meta.get("U_bar", meta.get("U_BAND", 1.0) / meta.get("mu_phys", 1.0)))
    V_bar = float(meta.get("V_bar", 1.0))
    return np.sqrt((V_bar * P) ** 2 + U_bar * U_bar) - U_bar

def vgroup_bar_from_meta(P: np.ndarray, meta: dict) -> np.ndarray:
    """Dimensionless group velocity dε/dP (same as generator's vgroup_scalar but vectorized)."""
    U_bar = float(meta.get("U_bar", meta.get("U_BAND", 1.0) / meta.get("mu_phys", 1.0)))
    V_bar = float(meta.get("V_bar", 1.0))
    denom = np.sqrt((V_bar * P) ** 2 + U_bar * U_bar)
    denom = np.maximum(denom, 1e-30)
    return (V_bar * V_bar * P) / denom

def orthogonalize_to_basis(t: np.ndarray, basis: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    """Gram-Schmidt t ⟂ span(basis) under <.,.>_w."""
    out = t.astype(np.float64).copy()
    for b in basis:
        nb2 = w_inner(b, b, w)
        if nb2 <= 1e-300:
            continue
        out -= (w_inner(out, b, w) / nb2) * b
    return out

def pick_modes_by_overlap(gammas, eta, w, theta, px, py, P, meta, mmax=8, include_energy_invariant=True, have_inv_evecs=True, have_density_evec=True):
    """
    Identify:
      m=0: density invariant
      inv_px, inv_py: momentum invariants (exact zero modes)
      inv_E: energy invariant (optional; often ~zero too)
      m=1: DECAYING current mode via j_perp = v - alpha p (orthogonal to momentum)
      m>=2: angular harmonics (cos mθ / sin mθ), after removing invariant subspace
    
    Args:
        have_inv_evecs: If False, momentum invariants were projected out, so don't try to find them in eta.
        have_density_evec: If False, density was projected out, so don't try to find it in eta.
    """
    k = eta.shape[1]
    used = set()

    # ---------- Invariants ----------
    t_den = np.ones_like(theta)

    # momentum in these coordinates (proportional to physical momentum)
    t_px = px.copy()
    t_py = py.copy()

    # energy invariant (dimensionless)
    eps_bar = eps_bar_from_meta(P, meta)
    t_E = eps_bar.copy()

    invariants = [t_den, t_px, t_py]
    inv_names  = ["den", "px", "py"]
    if include_energy_invariant:
        invariants.append(t_E)
        inv_names.append("E")

    chosen = {}
    diag = {}

    # Build invariant basis for orthogonalization (same as m>=2)
    inv_basis = [t_den, t_px, t_py]
    if include_energy_invariant:
        inv_basis.append(t_E)

    # pick best eigenvector for each invariant by overlap, and mark as used
    if have_inv_evecs:
        # Find momentum invariants (and optionally energy)
        for name, templ in zip(inv_names, invariants):
            if name == "den" and not have_density_evec:
                continue  # Skip density if it was projected
            best_i, best_s = None, -1.0
            for i in range(k):
                if i in used:
                    continue
                s = abs(overlap(eta[:, i], templ, w))
                if s > best_s:
                    best_s, best_i = s, i
            chosen[f"inv_{name}"] = best_i
            used.add(best_i)
            diag[f"inv_{name}"] = {"gamma": float(gammas[best_i]), "overlap": float(overlap(eta[:, best_i], templ, w))}

        # Define "m=0" as density invariant's eigenvector (if available)
        if have_density_evec and "inv_den" in chosen:
            chosen[0] = chosen["inv_den"]
            diag[0] = {
                "gamma": float(gammas[chosen[0]]),
                "ov_density": float(overlap(eta[:, chosen[0]], t_den, w)),
            }
        else:
            # Density was projected; mark m=0 to use template
            chosen[0] = None  # Signal to use template
            diag[0] = {
                "gamma": 0.0,  # By definition when projected
                "ov_density": 1.0,  # Template is exact
                "note": "density projected; using template",
            }
    else:
        # Momentum invariants were projected out; pick m=0 based on density availability
        if have_density_evec:
            # Apply same logic as m>=2: orthogonalize density template against other invariants
            # Orthogonalize against momentum (and energy if included), but not against itself
            t_den_ortho = orthogonalize_to_basis(t_den, [t_px, t_py] + ([t_E] if include_energy_invariant else []), w)
            # Pick m=0 as best match to orthogonalized density template
            best_i, best_s = None, -1.0
            for i in range(k):
                if i in used:
                    continue
                s = abs(overlap(eta[:, i], t_den_ortho, w))
                if s > best_s:
                    best_s, best_i = s, i
            chosen[0] = best_i
            used.add(best_i)
            diag[0] = {
                "gamma": float(gammas[chosen[0]]),
                "ov_density": float(overlap(eta[:, chosen[0]], t_den, w)),
            }
        else:
            # Both momentum and density projected; mark m=0 to use template
            chosen[0] = None  # Signal to use template
            diag[0] = {
                "gamma": 0.0,  # By definition when projected
                "ov_density": 1.0,  # Template is exact
                "note": "density projected; using template",
            }

    # ---------- Current templates ----------
    # Use dimensionless v(P)=dε/dP; physical prefactors don’t affect overlaps
    vg_bar = vgroup_bar_from_meta(P, meta)
    v_x = vg_bar * np.cos(theta)
    v_y = vg_bar * np.sin(theta)

    # Build j_perp = v - alpha p (orthogonalize current against momentum invariants)
    # alpha chosen separately for x and y in w-metric
    ax = w_inner(v_x, t_px, w) / max(w_inner(t_px, t_px, w), 1e-300)
    ay = w_inner(v_y, t_py, w) / max(w_inner(t_py, t_py, w), 1e-300)
    jperp_x = v_x - ax * t_px
    jperp_y = v_y - ay * t_py

    # W-norm fractions: how much of current is non-conserved?
    cur_norm2 = w_inner(v_x, v_x, w) + w_inner(v_y, v_y, w)
    perp_norm2 = w_inner(jperp_x, jperp_x, w) + w_inner(jperp_y, jperp_y, w)
    frac = np.sqrt(perp_norm2 / max(cur_norm2, 1e-300))
    diag["jperp_norm"] = float(np.sqrt(max(perp_norm2, 0.0)))  # keep for backward compatibility
    diag["jperp_frac"] = float(frac)  # store fraction for diagnostics

    # m=1 mode selection: use threshold on fraction to distinguish parabolic vs non-parabolic
    FRAC_THR = 5e-2  # 0.05 works well; parabolic case typically has ~0.015

    if frac < FRAC_THR and have_inv_evecs and "inv_px" in chosen and "inv_py" in chosen:
        # current is essentially momentum -> pick the momentum invariant as "m=1 current"
        # choose the better of px/py by overlap with raw current (not jperp)
        ov_px = np.sqrt(overlap(eta[:, chosen["inv_px"]], v_x, w) ** 2 + overlap(eta[:, chosen["inv_px"]], v_y, w) ** 2)
        ov_py = np.sqrt(overlap(eta[:, chosen["inv_py"]], v_x, w) ** 2 + overlap(eta[:, chosen["inv_py"]], v_y, w) ** 2)
        chosen[1] = chosen["inv_px"] if ov_px >= ov_py else chosen["inv_py"]
        diag[1] = {
            "gamma": float(gammas[chosen[1]]),
            "note": f"frac={frac:.3e} < {FRAC_THR}; current≈momentum",
        }
    else:
        # Apply same logic as m>=2: orthogonalize j_perp against full invariant basis
        jperp_x_ortho = orthogonalize_to_basis(jperp_x, inv_basis, w)
        jperp_y_ortho = orthogonalize_to_basis(jperp_y, inv_basis, w)
        # use decaying-current mode from orthogonalized j_perp
        best_i, best_s = None, -1.0
        for i in range(k):
            if i in used:
                continue
            ovx = overlap(eta[:, i], jperp_x_ortho, w)
            ovy = overlap(eta[:, i], jperp_y_ortho, w)
            s = float(np.sqrt(ovx * ovx + ovy * ovy))
            if s > best_s:
                best_s, best_i = s, i
        chosen[1] = best_i
        used.add(best_i)

        # diagnostics: also show overlap with raw current and momentum amplitudes
        ov_cur = np.sqrt(overlap(eta[:, best_i], v_x, w) ** 2 + overlap(eta[:, best_i], v_y, w) ** 2)
        ov_mom = np.sqrt(overlap(eta[:, best_i], t_px, w) ** 2 + overlap(eta[:, best_i], t_py, w) ** 2)
        diag[1] = {
            "gamma": float(gammas[best_i]),
            "ov_jperp_amp": float(best_s),
            "ov_current_amp": float(ov_cur),
            "ov_momentum_amp": float(ov_mom),
            "ax": float(ax),
            "ay": float(ay),
        }

    # ---------- Angular harmonics m>=2 ----------
    # Remove invariant subspace from angular templates to avoid accidentally selecting conserved modes
    # (inv_basis already defined earlier)

    for m in range(2, mmax + 1):
        tc = orthogonalize_to_basis(np.cos(m * theta), inv_basis, w)
        ts = orthogonalize_to_basis(np.sin(m * theta), inv_basis, w)

        best_i, best_s = None, -1.0
        for i in range(k):
            if i in used:
                continue
            ovc = overlap(eta[:, i], tc, w)
            ovs = overlap(eta[:, i], ts, w)
            s = float(np.sqrt(ovc * ovc + ovs * ovs))
            if s > best_s:
                best_s, best_i = s, i

        chosen[m] = best_i
        used.add(best_i)
        diag[m] = {"gamma": float(gammas[best_i]), "ov_ang_amp": float(best_s)}

    return chosen, diag, jperp_x, jperp_y, v_x, v_y


# ------------------------- plotting -------------------------

def ring_scatter(ax, px, py, val, title=""):
    # Robust color limits, normalized *per panel*
    vmax = float(np.max(np.abs(val))) if val.size > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0
    c_plot = val / vmax

    # Choose square marker size based on number of k-points so the ring looks filled
    n_pts = len(px)
    if n_pts < 200:
        msize = 60  # few points → make squares large
    elif n_pts < 1000:
        msize = 16  # typical resolution for these files
    else:
        msize = 6   # very dense sampling → smaller squares

    # Use square "pixel-like" markers without edges to avoid white gaps between points
    sc = ax.scatter(
        px,
        py,
        c=c_plot,
        s=msize,
        marker="s",
        cmap="RdBu_r",
        linewidths=0,
        edgecolors="none",
    )
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc

def make_figure(px, py, modes, gammas, mu_phys, U_band, Theta, out_prefix, w, eta0_override=None):
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 10))
    axes = axes.ravel()

    mlist = list(range(0, 9))
    last_sc = None
    for j, m in enumerate(mlist):
        if m == 0 and (eta0_override is not None or modes[m] is None):
            # Use density template when density was projected
            val = eta0_override if eta0_override is not None else np.ones_like(px)
            g = 0.0
        else:
            idx = modes[m]
            if idx is None:
                # Fallback: should not happen for m > 0, but handle gracefully
                val = np.ones_like(px)
                g = 0.0
            else:
                val = modes["_eta"][:, idx]
                g = gammas[idx]
        # Multiply by f*(1-f) = w to plot eta*f*(1-f)
        val = val * w
        # LaTeX-safe title (avoid raw Unicode when text.usetex=True)
        # Format gamma as a·10^{b} instead of 1e{b}
        g_str = f"{g:.3e}"
        mant_str, exp_str = g_str.split("e")
        exp_int = int(exp_str)
        title = fr"$m={m},\ \gamma={mant_str}\cdot 10^{{{exp_int}}}$"
        last_sc = ring_scatter(
            axes[j],
            px,
            py,
            val,
            title=title,
        )

    # LaTeX-safe super-title using math mode for Greek symbols
    fig.suptitle(fr"$\mu={mu_phys:g},\ U={U_band:g},\ \Theta={Theta:g}$")

    # Add one shared colorbar
    # cbar = fig.colorbar(last_sc, ax=axes.tolist(), shrink=0.85, pad=0.02)
    # cbar.set_label("η(px,py)")

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    png = out_prefix + ".png"
    pdf = out_prefix + ".pdf"
    fig.savefig(png, dpi=250)
    # fig.savefig(pdf)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# ------------------------- CSV output (optional) -------------------------

# Tolerances for deciding when density invariant diagnostics are "good enough"
DENSITY_RES_TOL = 1e-8          # max allowed invariant_residual for density
DENSITY_REL_GAMMA_TOL = 1e-3    # require |gamma_den| < DENSITY_REL_GAMMA_TOL * |m2|


def _csv_ensure_header(path: str, fieldnames: list[str], overwrite: bool):
    if overwrite and os.path.exists(path):
        os.remove(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

def _csv_append_row(path: str, fieldnames: list[str], row: dict):
    # Fill missing keys with empty string to keep DictWriter happy.
    out = {k: row.get(k, "") for k in fieldnames}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(out)


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
    ap.add_argument("--project_momentum", action="store_true",
                    help="Project out momentum subspace {px,py} before eigensolve. Keeps density (default behavior).")
    ap.add_argument("--project_density", action="store_true",
                    help="Also project out density template {1}. Use if you want gamma0 forced to ~0.")
    ap.add_argument("--project_energy", action="store_true",
                    help="Also project out energy-like template eps(P). (Use only if you want it removed.)")
    ap.add_argument("--show", action="store_true", help="Show matplotlib windows")
    ap.add_argument("--csv_out", type=str, default=None,
                    help="Append one CSV row per processed PKL with physically-defined m0 (density Rayleigh gamma) and m1 (total current decay).")
    ap.add_argument("--csv_overwrite", action="store_true", help="Overwrite --csv_out if it exists.")
    args = ap.parse_args()

    pkls = [args.pkl] if args.pkl else find_pkls_in_dir(args.dir)
    if not pkls:
        raise RuntimeError("No .pkl files found.")

    # Prepare CSV header if requested
    csv_fields = None
    if args.csv_out:
        # Minimal CSV: just T, mu, U, and m0..m_mmax
        csv_fields = ["T", "mu", "U"] + [f"m{m}" for m in range(0, args.mmax + 1)]
        _csv_ensure_header(args.csv_out, csv_fields, overwrite=args.csv_overwrite)

    for path in pkls:
        print("\n=== Processing:", path)
        M, meta = load_one_pkl(path)

        w = np.asarray(meta["w_active"], dtype=np.float64)

        px, py, P, theta, k_phys, vg_phys = reconstruct_active_coords(meta)

        mu_phys = float(meta.get("mu_phys", np.nan))
        U_band  = float(meta.get("U_band", np.nan))
        Theta   = float(meta.get("Theta", np.nan))

        # Build invariant templates in eta-space for optional projection
        t_den = np.ones_like(theta)
        t_px = px.copy()
        t_py = py.copy()

        # Build templates for projection: momentum (and optionally density and energy)
        templates = [t_px, t_py]
        if args.project_density:
            templates = [t_den] + templates
        if args.project_energy:
            templates.append(eps_bar_from_meta(P, meta))

        # Calculate effective k: request extra eigenpairs to account for projected dimensions
        k_eff = args.k
        if args.project_momentum:
            k_eff += 2  # px, py
        if args.project_density:
            k_eff += 1  # density
        if args.project_energy:
            k_eff += 1  # energy

        # Solve
        if args.project_momentum or args.project_density:
            gammas, eta = solve_weighted_modes_projected(
                M, w, templates_eta=templates, k=k_eff, symmetrize=(not args.no_sym)
            )
        else:
            gammas, eta = solve_weighted_modes(
                M, w, k=k_eff, symmetrize=(not args.no_sym)
            )

        # Identify modes by overlap
        # If momentum is projected, we can't find px/py eigenvectors
        # If density is projected, we can't find density eigenvector (will use template instead)
        have_inv_evecs = (not args.project_momentum)
        have_density_evec = (not args.project_density)
        chosen, diag, jperp_x, jperp_y, v_x, v_y = pick_modes_by_overlap(
            gammas, eta, w, theta, px, py, P, meta, mmax=args.mmax, 
            include_energy_invariant=True, have_inv_evecs=have_inv_evecs, have_density_evec=have_density_evec
        )

        # Print invariant diagnostics (Rayleigh gammas and residuals)
        sym = (not args.no_sym)
        print("Invariants (diagnostics):")
        gamma_den = rayleigh_gamma_for_template(M, w, t_den, symmetrize=sym)
        res_den   = invariant_residual(M, w, t_den, symmetrize=sym)
        gamma_px  = rayleigh_gamma_for_template(M, w, t_px,  symmetrize=sym)
        res_px    = invariant_residual(M, w, t_px,  symmetrize=sym)
        gamma_py  = rayleigh_gamma_for_template(M, w, t_py,  symmetrize=sym)
        res_py    = invariant_residual(M, w, t_py,  symmetrize=sym)
        print(f"  den: gamma_eff={gamma_den:+.3e}   residual={res_den:.3e}")
        print(f"  px : gamma_eff={gamma_px:+.3e}   residual={res_px:.3e}")
        print(f"  py : gamma_eff={gamma_py:+.3e}   residual={res_py:.3e}")
        
        if args.project_energy or (have_inv_evecs and "inv_E" in diag):
            tE = eps_bar_from_meta(P, meta)
            gE = rayleigh_gamma_for_template(M, w, tE, symmetrize=sym)
            rE = invariant_residual(M, w, tE, symmetrize=sym)
            print(f"  E  : gamma_eff={gE:+.3e}   residual={rE:.3e}")
        
        # Also print eigenvector-based info if invariants were found
        if have_inv_evecs:
            print("Invariants (eigenvector matches):")
            for key in ["inv_den", "inv_px", "inv_py", "inv_E"]:
                if key in diag:
                    print(f"  {key:7s}: idx={chosen[key]:3d} gamma={diag[key]['gamma']:.3e} ov={diag[key]['overlap']:.3f}")

        print(f"j_perp norm (should be ~0 in parabolic, >0 in non-parabolic): {diag['jperp_norm']:.3e}")

        # W-norm fractions: how much of current is non-conserved?
        frac = diag.get("jperp_frac", np.nan)
        print(f"||j_perp||_W / ||j||_W = {frac:.3e}")

        # Current relaxation rate via Rayleigh quotient on projected current
        A = (-symmetrize_sparse(M)).tocsr() if not args.no_sym else (-M).tocsr()
        num = np.dot(jperp_x, A @ jperp_x) + np.dot(jperp_y, A @ jperp_y)
        den_jperp = w_inner(jperp_x, jperp_x, w) + w_inner(jperp_y, jperp_y, w)
        cur_norm2 = w_inner(v_x, v_x, w) + w_inner(v_y, v_y, w)
        gamma_eff_jperp = np.nan
        gamma_j_total   = np.nan
        if den_jperp > 1e-300:
            gamma_eff_jperp = num / den_jperp
            print(f"gamma_eff(j_perp) = {gamma_eff_jperp:.6e}")
        if cur_norm2 > 1e-300:
            # TOTAL current decay (this is the one that -> 0 in parabolic regime):
            # gamma_j_total = <j,Aj>/<j,Wj> = gamma_eff(j_perp) * (||j_perp||_W/||j||_W)^2
            gamma_j_total = num / cur_norm2

        print("Modes:")
        for m in range(0, args.mmax + 1):
            i = chosen[m]
            d = diag[m]
            if m == 1:
                if "note" in d:
                    idx_str = "N/A" if i is None else f"{i:3d}"
                    frac_val = diag.get("jperp_frac", np.nan)
                    print(f"  m=1 (CURRENT): idx={idx_str}  gamma={d['gamma']:.6e}  frac={frac_val:.3e}  {d['note']}")
                else:
                    idx_str = "N/A" if i is None else f"{i:3d}"
                    frac_val = diag.get("jperp_frac", np.nan)
                    print(f"  m=1 (j_perp):  idx={idx_str}  gamma={d['gamma']:.6e}  frac={frac_val:.3e}  "
                          f"|ov_jperp|={d.get('ov_jperp_amp', np.nan):.6f}  "
                          f"|ov_current|={d.get('ov_current_amp', np.nan):.6f}  "
                          f"|ov_momentum|={d.get('ov_momentum_amp', np.nan):.6f}")
            else:
                key = "ov_density" if m == 0 else "ov_ang_amp"
                idx_str = "N/A" if i is None else f"{i:3d}"
                note_str = f"  {d.get('note', '')}" if 'note' in d else ""
                print(f"  m={m}:           idx={idx_str}  gamma={d['gamma']:.6e}  overlap={d.get(key, np.nan):.3f}{note_str}")

        # Store eta for plotting convenience
        chosen["_eta"] = eta

        # Decide whether density invariant diagnostics are good enough; warn if not
        bad_density = False
        # 1) residual too large
        if np.isfinite(res_den) and res_den > DENSITY_RES_TOL:
            bad_density = True
        # 2) |gamma_den| not much smaller than first physical gamma (m=2)
        gamma_m2 = np.nan
        if 2 in diag and "gamma" in diag[2]:
            gamma_m2 = float(diag[2]["gamma"])
        if (
            np.isfinite(gamma_den)
            and np.isfinite(gamma_m2)
            and abs(gamma_den) > DENSITY_REL_GAMMA_TOL * abs(gamma_m2)
        ):
            bad_density = True

        if bad_density:
            print(
                f"WARNING: density invariant diagnostics failed "
                f"(res_den={res_den:.3e}, gamma_den={gamma_den:.3e}, m2={gamma_m2:.3e}); "
                "CSV row will still be written for this point; check diagnostics."
            )

        # ---------------- CSV row (optional) ----------------
        if args.csv_out and csv_fields:
            row = {
                "T": float(Theta),
                "mu": float(mu_phys),
                "U": float(U_band),
                # m0 is the density invariant Rayleigh gamma (NOT the smallest eigenvalue)
                "m0": float(gamma_den) if np.isfinite(gamma_den) else "",
                # m1 is total current decay rate (goes ~0 in parabolic; finite in non-parabolic)
                "m1": float(gamma_j_total) if np.isfinite(gamma_j_total) else "",
            }
            for m in range(2, args.mmax + 1):
                row[f"m{m}"] = float(diag[m]["gamma"]) if (m in diag and "gamma" in diag[m]) else ""
            _csv_append_row(args.csv_out, csv_fields, row)

        # Prepare density template for plotting if density was projected
        eta0_override = None
        if args.project_density:
            # Create B-normalized density template
            eta0 = t_den.copy()
            nrm = np.sqrt(np.dot(w, eta0 * eta0))
            if nrm > 0:
                eta0 /= nrm
            eta0_override = eta0

        # Plot 3x3 (m=0..8)
        base = os.path.splitext(os.path.basename(path))[0]
        out_prefix = f"ring_modes_{base}_mu{mu_phys:g}_U{U_band:g}_T{Theta:g}"
        make_figure(px, py, chosen, gammas, mu_phys, U_band, Theta, out_prefix, w, eta0_override=eta0_override)

        if args.show:
            plt.show()
        else:
            plt.close("all")


if __name__ == "__main__":
    main()


#python Iee_eigs_ringplot_nonparabolic.py --dir mu10 --k 80 --mmax 8 --csv_out gamma_vs_T_mu1_U1.csv --csv_overwrite
#python Iee_eigs_ringplot_nonparabolic.py --pkl "D:\Рабочая папка\GitHub\electronic-kapitsa-waves\kryhin_levitov_2021\collision_integral_direct\Matrixes_bruteforce\M_Iee_N320_dp0.032716515_T0.1.pkl" --k 80 --mmax 8 --csv_out gamma_vs_T_mu1_U1.csv --csv_overwrite