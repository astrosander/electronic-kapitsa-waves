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

def pick_modes_by_overlap(gammas, eta, w, theta, px, py, P, meta, mmax=8, include_energy_invariant=True):
    """
    Identify:
      m=0: density invariant
      inv_px, inv_py: momentum invariants (exact zero modes)
      inv_E: energy invariant (optional; often ~zero too)
      m=1: DECAYING current mode via j_perp = v - alpha p (orthogonal to momentum)
      m>=2: angular harmonics (cos mθ / sin mθ), after removing invariant subspace
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

    # pick best eigenvector for each invariant by overlap, and mark as used
    for name, templ in zip(inv_names, invariants):
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

    # Define "m=0" as density invariant’s eigenvector
    chosen[0] = chosen["inv_den"]
    diag[0] = {
        "gamma": float(gammas[chosen[0]]),
        "ov_density": float(overlap(eta[:, chosen[0]], t_den, w)),
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

    if frac < FRAC_THR:
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
        # use decaying-current mode from j_perp
        best_i, best_s = None, -1.0
        for i in range(k):
            if i in used:
                continue
            ovx = overlap(eta[:, i], jperp_x, w)
            ovy = overlap(eta[:, i], jperp_y, w)
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
    inv_basis = [t_den, t_px, t_py]
    if include_energy_invariant:
        inv_basis.append(t_E)

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
        chosen, diag, jperp_x, jperp_y, v_x, v_y = pick_modes_by_overlap(
            gammas, eta, w, theta, px, py, P, meta, mmax=args.mmax, include_energy_invariant=True
        )

        print("Invariants:")
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
        den = w_inner(jperp_x, jperp_x, w) + w_inner(jperp_y, jperp_y, w)
        if den > 1e-300:
            gamma_eff = num / den
            print(f"gamma_eff(j_perp) = {gamma_eff:.6e}")

        print("Modes:")
        for m in range(0, args.mmax + 1):
            i = chosen[m]
            d = diag[m]
            if m == 1:
                if "note" in d:
                    print(f"  m=1 (CURRENT): idx={i:3d}  gamma={d['gamma']:.6e}  {d['note']}")
                else:
                    print(f"  m=1 (CURRENT): idx={i:3d}  gamma={d['gamma']:.6e}  |ov_current|={d.get('ov_current_amp', np.nan):.3f}  "
                          f"|ov_momentum|={d.get('ov_momentum_amp', np.nan):.3f}")
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
