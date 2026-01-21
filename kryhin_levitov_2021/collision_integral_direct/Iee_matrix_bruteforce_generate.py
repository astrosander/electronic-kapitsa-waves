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
Nmax = 256#24          # lattice is Nmax x Nmax -> N = Nmax^2
dp   = 0.08        # Δp, choose so p_F~1 is resolved
LAMBDA = 1e-3      # λ in Lorentzian δ_ε
V2   = 1.0         # |V|^2
HBAR = 1.0         # ħ (set 1 for dimensionless)

# temperatures (T/T_F). Adjust as you like:
Thetas = np.geomspace(0.0025, 1.28, 30).astype(float).tolist()

# active-shell cutoff: only include states where f(1-f) > cutoff
ACTIVE_CUTOFF = 1e-8

# include extra dimensionless prefactor like your original kernel code:
# (1 / (Theta * (2π)^4))  -- set False to follow the bare formula strictly
INCLUDE_DIMLESS_PREF = True

OUT_DIR = "Matrixes_bruteforce"
# ----------------------------------------------------------------------


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


def active_indices(f: np.ndarray, cutoff: float) -> np.ndarray:
    w = f * (1.0 - f)
    return np.where(w > cutoff)[0].astype(np.int32)


if USE_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def assemble_rows_numba(M, nx, ny, idx_map, half, eps, f, active,
                            dp, lam, pref, dimless_scale):
        """
        Parallel over i1 rows: safe because only writes into row i1.
        """
        dp4 = dp ** 4
        Nactive = active.size

        for a in prange(Nactive):
            i1 = active[a]
            n1x = nx[i1]
            n1y = ny[i1]
            e1  = eps[i1]
            f1  = f[i1]

            for b in range(Nactive):
                i2 = active[b]
                n2x = nx[i2]
                n2y = ny[i2]
                e2  = eps[i2]
                f2  = f[i2]

                for c in range(Nactive):
                    i1p = active[c]
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

                    e2p = eps[i2p]
                    f2p = f[i2p]

                    dE = (e1 + e2 - e1p - e2p)
                    delta_eps = (1.0 / math.pi) * lam / (dE * dE + lam * lam)

                    # F_{121'2'} = f1 f2 (1-f1')(1-f2')
                    F = f1 * f2 * (1.0 - f1p) * (1.0 - f2p)

                    W = pref * dimless_scale * F * delta_eps * dp4

                    # (eta_{1'} + eta_{2'} - eta_1 - eta_2)
                    M[i1, i1p] += W
                    M[i1, i2p] += W
                    M[i1, i1]  -= W
                    M[i1, i2]  -= W


def build_matrix_for_theta(Theta: float):
    nx, ny, half = build_centered_lattice(Nmax)
    idx_map = make_index_map(nx, ny, Nmax, half)
    px, py, P, eps, f = precompute(nx, ny, dp, Theta)

    active = active_indices(f, ACTIVE_CUTOFF)
    Nstates = nx.size
    print(f"Theta={Theta:.6g}  Nstates={Nstates}  Nactive={active.size}  USE_NUMBA={USE_NUMBA}")

    pref = (2.0 * math.pi / HBAR) * V2
    dimless_scale = 1.0
    if INCLUDE_DIMLESS_PREF:
        dimless_scale = 1.0 / (Theta * (2.0 * math.pi) ** 4)

    M = np.zeros((Nstates, Nstates), dtype=np.float64)

    if USE_NUMBA:
        assemble_rows_numba(M, nx, ny, idx_map, half, eps, f, active,
                            float(dp), float(LAMBDA), float(pref), float(dimless_scale))
    else:
        # Pure python fallback (very slow unless Nactive is tiny)
        dp4 = dp ** 4
        for i1 in active:
            n1x, n1y = int(nx[i1]), int(ny[i1])
            e1, f1 = float(eps[i1]), float(f[i1])
            for i2 in active:
                n2x, n2y = int(nx[i2]), int(ny[i2])
                e2, f2 = float(eps[i2]), float(f[i2])
                for i1p in active:
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
                    e2p, f2p = float(eps[i2p]), float(f[i2p])

                    dE = e1 + e2 - e1p - e2p
                    delta_eps = (1.0 / math.pi) * LAMBDA / (dE * dE + LAMBDA * LAMBDA)
                    F = f1 * f2 * (1.0 - f1p) * (1.0 - f2p)
                    W = pref * dimless_scale * F * delta_eps * dp4

                    M[i1, i1p] += W
                    M[i1, i2p] += W
                    M[i1, i1]  -= W
                    M[i1, i2]  -= W

    meta = {
        "Theta": float(Theta),
        "Nmax": int(Nmax),
        "dp": float(dp),
        "lambda": float(LAMBDA),
        "V2": float(V2),
        "hbar": float(HBAR),
        "include_dimless_pref": bool(INCLUDE_DIMLESS_PREF),
        "active_cutoff": float(ACTIVE_CUTOFF),
        "nx": nx, "ny": ny,
        "px": px, "py": py,
        "P": P, "eps": eps, "f": f,
        "active": active,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    fname = os.path.join(OUT_DIR, f"M_Iee_N{Nmax}_dp{dp:g}_T{Theta:.10g}.pkl")
    with open(fname, "wb") as fp:
        pickle.dump((M, meta), fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {fname}")


def main():
    if not USE_NUMBA:
        print("WARNING: numba not found. This will be very slow. Install numba for practical runtimes.")
    for T in Thetas:
        build_matrix_for_theta(float(T))


if __name__ == "__main__":
    main()
