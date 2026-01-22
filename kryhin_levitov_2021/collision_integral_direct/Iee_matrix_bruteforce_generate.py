#!/usr/bin/env python3
"""
Iee_matrix_bruteforce_generate.py
Brute-force electron-electron collision operator on a 2D momentum lattice.

Discrete momenta:
  p = Δp * (n_x, n_y), where n_x,n_y ∈ {-Nmax/2, ..., Nmax/2-1}
Number of states:
  N = Nmax^2
Matrix size:
  N x N

Collision operator (discretized):
  I_ee[eta]_1 = Σ_{2,1'} W_{12->1'2'} (eta_{1'} + eta_{2'} - eta_1 - eta_2)
Momentum conservation δ_p is enforced EXACTLY by:
  2' = 1 + 2 - 1'

Energy conservation δ_ε uses Lorentzian broadening:
  δ_ε(Δε) = (1/π) * λ / (Δε^2 + λ^2)

We include Δp^4 as the discrete measure for Σ_{p2, p1'}.

NOTE:
- This is intentionally brute force. Runtime scales roughly O(N_active^3),
  where N_active are states with f(1-f) above a cutoff.
- Use numba if possible.
"""

import os
import math
import pickle
import numpy as np

from typing import Tuple
USE_NUMBA = False
try:
    import numba
    from numba import njit, prange
    USE_NUMBA = True
except ImportError as e:
    print("Numba import failed:", e)

if USE_NUMBA:
    # Optional diagnostics: NEVER disable numba if these fail
    try:
        print("cpu_count:", os.cpu_count())
        print("affinity:", len(os.sched_getaffinity(0)))
        print("numba threads:", numba.get_num_threads())
        print("numba layer:", numba.threading_layer())
    except Exception as e:
        print("Numba OK; diagnostics failed:", repr(e))



# --------------------------- USER PARAMETERS ---------------------------
Nmax = 100          # lattice is Nmax x Nmax -> N = Nmax^2
dp   = 0.08         # Δp

# IMPORTANT:
# For low-T scaling you must resolve the thermal shell: need dp^2 << Theta_min.
# With Theta_min=0.0025 you want dp <= ~0.03 (since dp^2=9e-4).
# dp=0.08 => dp^2=6.4e-3 > Theta_min, which produces a T->0 "floor".

# Energy delta broadening (Lorentzian) --- MUST scale with temperature to avoid T->0 floor
LAMBDA_REL = 0.25    # sets lambda_eff = LAMBDA_REL * Theta
LAMBDA_MIN = 1e-12

V2   = 1.0         # |V|^2
HBAR = 1.0         # ħ (set 1 for dimensionless)

# temperatures (T/T_F). Adjust as you like:
Thetas = np.geomspace(0.0025, 1.28, 30).astype(float).tolist()

# active-shell cutoff: only include states where f(1-f) > cutoff
ACTIVE_CUTOFF = 1e-8

# DO NOT include an explicit 1/Theta prefactor.
# If you want a constant phase-space normalization, keep only the constant (2π)^-4.
INCLUDE_DIMLESS_PREF = True
DIMLESS_CONST = 1.0 / ((2.0 * math.pi) ** 4)

OUT_DIR = "Matrixes_bruteforce"
# ----------------------------------------------------------------------

# Build and save ONLY the closed active-subspace operator (recommended).
# This avoids huge Nstates^2 memory and preserves the weighted symmetry better.
BUILD_ACTIVE_ONLY = True


def f_scalar(P: float, Theta: float) -> float:
    """
    Numerically stable version of your f(P,Theta).

    Original:
      f = (1-e^{-1/Θ}) / ( exp((P^2-1)/Θ) + 1 - e^{-1/Θ} )

    Let x = (P^2-1)/Θ, em = exp(-1/Θ), a = 1 - em.
      f = a / (exp(x) + a)

    Avoid overflow in exp(x) by saturating for large |x|.
    """
    invT = 1.0 / Theta
    em = math.exp(-invT)          # safe (invT>0)
    a = 1.0 - em                  # in (0,1)

    x = (P * P - 1.0) * invT

    # exp(x) overflow threshold for double ~ 709
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
    nx = nx.reshape(-1)
    ny = ny.reshape(-1)
    return nx, ny, half


def make_index_map(nx: np.ndarray, ny: np.ndarray, Nmax: int, half: int) -> np.ndarray:
    idx_map = -np.ones((Nmax, Nmax), dtype=np.int32)
    for i in range(nx.size):
        idx_map[nx[i] + half, ny[i] + half] = i
    return idx_map


def precompute(nx, ny, dp: float, Theta: float):
    px = dp * nx.astype(np.float64)
    py = dp * ny.astype(np.float64)
    P  = np.sqrt(px * px + py * py)
    eps = P * P  # constant shift cancels in Δε anyway
    f = np.array([f_scalar(float(Pi), float(Theta)) for Pi in P], dtype=np.float64)
    return px, py, P, eps, f


def active_indices(f: np.ndarray, eps: np.ndarray, Theta: float, cutoff: float) -> np.ndarray:
    """
    Active set should follow the thermal shell width ~ Theta.
    Using only w=f(1-f) on a coarse Cartesian grid can "freeze" the shell at the lattice energy spacing ~ dp^2.
    """
    w = f * (1.0 - f)
    # Require near the Fermi surface in ENERGY (since eps = p^2 and eps_F = 1 in your units)
    # The multiplier 10 is conservative; tighten/loosen if needed.
    shell = np.abs(eps - 1.0) < (10.0 * Theta)
    return np.where((w > cutoff) & shell)[0].astype(np.int32)


if USE_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def assemble_rows_numba_active(Ma, nx, ny, idx_map, half, eps, f,
                                   active, pos, dp, lam_eff, pref, dimless_scale):
        """
        Assemble CLOSED active-subspace operator Ma (shape Nactive x Nactive).
        NOTE: This implements P I P projection:
          - row index i1 is active (output restricted to active)
          - columns correspond to active eta only
          - if i2' is inactive, we keep the event but drop the +eta_{2'} term (since eta=0 there)

        Parallel over active-row index a1: safe because only writes into row a1.
        """
        dp4 = dp ** 4
        Nactive = active.size

        for a1 in prange(Nactive):
            i1 = active[a1]
            n1x = nx[i1]
            n1y = ny[i1]
            e1  = eps[i1]
            f1  = f[i1]

            for a2 in range(Nactive):
                i2 = active[a2]
                n2x = nx[i2]
                n2y = ny[i2]
                e2  = eps[i2]
                f2  = f[i2]

                for a1p in range(Nactive):
                    i1p = active[a1p]
                    n1px = nx[i1p]
                    n1py = ny[i1p]
                    e1p  = eps[i1p]
                    f1p  = f[i1p]

                    # enforce momentum conservation: n2' = n1 + n2 - n1'
                    n2px = n1x + n2x - n1px
                    n2py = n1y + n2y - n1py
                    ix = n2px + half
                    iy = n2py + half

                    if ix < 0 or ix >= idx_map.shape[0] or iy < 0 or iy >= idx_map.shape[1]:
                        continue

                    i2p = idx_map[ix, iy]
                    if i2p < 0:
                        continue

                    # PIP projection: a2p may be -1 (inactive); that's OK
                    a2p = pos[i2p]

                    e2p = eps[i2p]
                    f2p = f[i2p]

                    dE = (e1 + e2 - e1p - e2p)
                    delta_eps = (1.0 / math.pi) * lam_eff / (dE * dE + lam_eff * lam_eff)

                    # Symmetrize Pauli factor to enforce microreversibility on a discrete/approx grid:
                    # F_sym = 1/2 [ f1 f2 (1-f1')(1-f2') + f1' f2' (1-f1)(1-f2) ]
                    F_fwd = f1 * f2 * (1.0 - f1p) * (1.0 - f2p)
                    F_bwd = f1p * f2p * (1.0 - f1) * (1.0 - f2)
                    F = 0.5 * (F_fwd + F_bwd)

                    W = pref * dimless_scale * F * delta_eps * dp4

                    # (eta_{1'} + eta_{2'} - eta_1 - eta_2)
                    Ma[a1, a1p] += W
                    if a2p >= 0:
                        Ma[a1, a2p] += W
                    Ma[a1, a1]  -= W
                    Ma[a1, a2]  -= W


def build_matrix_for_theta(Theta: float):
    nx, ny, half = build_centered_lattice(Nmax)
    idx_map = make_index_map(nx, ny, Nmax, half)
    px, py, P, eps, f = precompute(nx, ny, dp, Theta)

    # Temperature-following active shell
    active = active_indices(f, eps, Theta, ACTIVE_CUTOFF)
    Nstates = nx.size
    Nactive = int(active.size)
    print(f"Theta={Theta:.6g}  Nstates={Nstates}  Nactive={Nactive}  USE_NUMBA={USE_NUMBA}  ACTIVE_ONLY={BUILD_ACTIVE_ONLY}")

    # Resolution warning (this is *the* main reason your curves don't scale at very low T on a Cartesian grid)
    if (dp * dp) > (0.5 * Theta):
        print(f"WARNING: dp^2={dp*dp:.3e} is not << Theta={Theta:.3e}. "
              f"Low-T scaling will saturate/fail. Consider dp<=sqrt(Theta_min)/3.")

    # map global index -> active subspace index (or -1 if inactive)
    pos = -np.ones(Nstates, dtype=np.int32)
    pos[active] = np.arange(Nactive, dtype=np.int32)

    pref = (2.0 * math.pi / HBAR) * V2
    dimless_scale = 1.0
    if INCLUDE_DIMLESS_PREF:
        # keep ONLY constant (2π)^-4, NOT 1/Theta
        dimless_scale = DIMLESS_CONST

    # Temperature-scaled Lorentzian width to avoid T->0 floor
    lam_eff = max(LAMBDA_REL * Theta, LAMBDA_MIN)

    if BUILD_ACTIVE_ONLY:
        Ma = np.zeros((Nactive, Nactive), dtype=np.float64)

        if USE_NUMBA:
            assemble_rows_numba_active(
                Ma, nx, ny, idx_map, half, eps, f,
                active, pos,
                float(dp), float(lam_eff), float(pref), float(dimless_scale)
            )
        else:
            dp4 = dp ** 4
            for a1, i1 in enumerate(active):
                n1x, n1y = int(nx[i1]), int(ny[i1])
                e1, f1 = float(eps[i1]), float(f[i1])
                for a2, i2 in enumerate(active):
                    n2x, n2y = int(nx[i2]), int(ny[i2])
                    e2, f2 = float(eps[i2]), float(f[i2])
                    for a1p, i1p in enumerate(active):
                        n1px, n1py = int(nx[i1p]), int(ny[i1p])
                        e1p, f1p = float(eps[i1p]), float(f[i1p])

                        n2px = n1x + n2x - n1px
                        n2py = n1y + n2y - n1py
                        ix, iy = n2px + half, n2py + half
                        if ix < 0 or ix >= Nmax or iy < 0 or iy >= Nmax:
                            continue
                        i2p = int(idx_map[ix, iy])
                        if i2p < 0:
                            continue
                        a2p = int(pos[i2p])  # may be -1 (inactive); that's OK for PIP projection

                        e2p, f2p = float(eps[i2p]), float(f[i2p])
                        dE = e1 + e2 - e1p - e2p
                        delta_eps = (1.0 / math.pi) * lam_eff / (dE * dE + lam_eff * lam_eff)

                        # Symmetrized Pauli factor
                        F_fwd = f1 * f2 * (1.0 - f1p) * (1.0 - f2p)
                        F_bwd = f1p * f2p * (1.0 - f1) * (1.0 - f2)
                        F = 0.5 * (F_fwd + F_bwd)
                        W = pref * dimless_scale * F * delta_eps * dp4

                        Ma[a1, a1p] += W
                        if a2p >= 0:
                            Ma[a1, a2p] += W
                        Ma[a1, a1]  -= W
                        Ma[a1, a2]  -= W

        # save sparse to disk
        from scipy.sparse import csr_matrix
        M_to_save = csr_matrix(Ma)
    else:
        # legacy full matrix mode (very large for big Nmax)
        M_to_save = np.zeros((Nstates, Nstates), dtype=np.float64)
        if USE_NUMBA:
            raise RuntimeError("Full-matrix numba assembly removed in ACTIVE_ONLY workflow.")
        else:
            raise RuntimeError("Full-matrix mode disabled. Set BUILD_ACTIVE_ONLY=True.")

    meta = {
        "Theta": float(Theta),
        "Nmax": int(Nmax),
        "half": int(half),
        "dp": float(dp),
        "lambda": float(lam_eff),
        "V2": float(V2),
        "hbar": float(HBAR),
        "include_dimless_pref": bool(INCLUDE_DIMLESS_PREF),
        "active_cutoff": float(ACTIVE_CUTOFF),
        "active_only": bool(BUILD_ACTIVE_ONLY),
        "nx": nx, "ny": ny,
        "px": px, "py": py,
        "P": P, "eps": eps, "f": f,
        "active": active,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    fname = os.path.join(OUT_DIR, f"M_Iee_N{Nmax}_dp{dp:g}_T{Theta:.10g}.pkl")
    with open(fname, "wb") as fp:
        pickle.dump((M_to_save, meta), fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {fname}")


def main():
    if not USE_NUMBA:
        print("WARNING: numba not found. This will be very slow. Install numba for practical runtimes.")
    for T in Thetas:
        build_matrix_for_theta(float(T))


if __name__ == "__main__":
    main()
