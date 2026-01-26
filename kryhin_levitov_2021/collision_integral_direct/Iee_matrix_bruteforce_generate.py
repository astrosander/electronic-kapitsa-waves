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
# Base grid (used only if AUTO_GRID=False)
Nmax = 100          # lattice is Nmax x Nmax -> N = Nmax^2
dp   = 0.08         # Δp

# --- NEW: target constant "ring pixels" via Theta/dp^2 constant ---
AUTO_DP_FROM_ANCHOR = True
# Piecewise anchors for different temperature ranges
# Low-T anchor (0.01 < Theta < 0.1): targets ~20MB
THETA_ANCHOR_LOW = 0.02
DP_ANCHOR_LOW    = 0.03
# High-T anchor (0.1 < Theta < 1): targets ~20MB
# Use Theta=0.3 as anchor since it's in the middle of the range
# Current: Theta=0.3, dp=0.116 gives 10.6MB (too small)
# Target: ~20MB requires ~sqrt(20/10.6) ≈ 1.37x more Nactive
# So dp should be ~0.116/1.37 ≈ 0.085 to get ~20MB
THETA_ANCHOR_HIGH = 0.3
DP_ANCHOR_HIGH    = 0.085        # adjusted to target ~20MB at Theta=0.3
THETA_CROSSOVER   = 0.1          # switch between anchors at this temperature

DP_RING_MAX  = 0.03         # preferred cap for ring detail (only applies to low-T)
DP_FLOOR     = 1e-4         # safety
# If True: allow dp to exceed DP_RING_MAX to keep file size constant
# If False: enforce DP_RING_MAX strictly (prioritize ring detail over size)
PRIORITIZE_SIZE_OVER_RING = False

# ------------------------- LOW-T BOX VALIDITY -------------------------
# At very low Theta, your dp(theta) can become so small that Nmax hits NMAX_MAX
# and the momentum box fails to reach the Fermi surface p=1 (=> garbage rates).
# We therefore:
#   (i) use a smaller p-box requirement at ultra-low T (dominant scattering stays near p~1),
#  (ii) enforce dp >= dp_box_min(theta) so choose_Nmax() won't clamp.
PBOX_MIN_HIGH = 2.5
PBOX_MIN_LOW  = 1.8
THETA_PBOX_SWITCH = 1e-3

# PATCH (strong): push much higher ring accuracy.
# Smaller dp is necessary to expose clean T^2 (even m) and especially T^4 (odd m).
# Reduced from 3.0 to 1.5 to keep Nactive manageable (runtime-safe range: 1.2–1.8)
RING_PIXEL_BOOST = 1.5      # reduced from 3.0 to prevent Nactive explosion
RING_SHELL_TIGHTEN = 0.45   # tighter ring (suppresses bulk contamination)
RING_FLOOR_DP_MULT = 0.65   # ring thickness floor multiplier (was effectively 2.5; 0.6–1.0 is safe)

# Compute pixel ratios for both regions
PIXEL_RATIO_LOW  = (THETA_ANCHOR_LOW  / (DP_ANCHOR_LOW  * DP_ANCHOR_LOW )) * RING_PIXEL_BOOST
PIXEL_RATIO_HIGH = (THETA_ANCHOR_HIGH / (DP_ANCHOR_HIGH * DP_ANCHOR_HIGH)) * (0.75 * RING_PIXEL_BOOST)

# Optional: choose Nmax so the momentum box isn't absurdly tiny at low T
# pmax ~= (Nmax/2)*dp. 2.5 is usually safe for low T; use 4.0 if you want the same as your old runs.
PBOX_MIN = PBOX_MIN_HIGH
NMAX_MIN = 320   # Level-2: increased from 200 to allow better high-T accuracy (was clamped before)
NMAX_MAX = 3200  # Increased to allow valid box at ultra-low T (Theta ~ 1e-4)

# --- NEW: grid shift (take half-integers instead of integers) ---
# Physical momenta are p = dp * (n + shift), where n is integer lattice index.
# Use 0.5 for half-integer grid, 1/3 for third-integer grid, etc.
# "top/left" in array sense often means negative shift; choose sign as you want.
#
# PATCH: increase Dirac-ring resolution WITHOUT growing matrix size:
# Use a half-integer lattice. This de-aliases the p≈1 shell against the axes and
# increases effective angular sampling on the ring at essentially fixed Nactive.
SHIFT_X = 0.5
SHIFT_Y = 0.5

# IMPORTANT:
# For low-T scaling you must resolve the thermal shell: need dp^2 << Theta_min.
# With Theta_min=0.0025 you want dp <= ~0.03 (since dp^2=9e-4).
# dp=0.08 => dp^2=6.4e-3 > Theta_min, which produces a T->0 "floor".

# Energy delta broadening (Lorentzian) --- MUST scale with temperature to avoid T->0 floor
# PATCH (strong): sharpen energy conservation to recover proper low-T exponents.
# PATCH: For Dirac dispersion, energy is linear in p, so mismatch scales ~O(dp), not dp^2.
LAMBDA_REL = 0.03        # lambda_T = 0.03 * Theta  (was 0.1)
LAMBDA_DP_REL = 2.0      # lambda_dp = 2.0 * dp (energy spacing scale for Dirac)
LAMBDA_MIN = 1e-16       # reduced from 1e-12 to allow tiny rates at very low T

V2   = 1.0         # |V|^2
HBAR = 1.0         # ħ (set 1 for dimensionless)

# temperatures (T/T_F). Adjust as you like:
# TEST: Generate only for Theta = 0.0508
# Thetas = [0.001]#[0.0508]
# Thetas = [0.001]
# Thetas = [0.0025, 0.0035, 0.005, 0.007, 0.01, 0.014, 0.02, 0.028, 0.04, 0.056, 0.08, 0.112, 0.16, 0.224, 0.32, 0.448, 0.64, 0.896, 1.28]#np.geomspace(0.001, 1.28, 30)
# Thetas = [0.0001, 0.0001585, 0.0002512, 0.0003981, 0.0006310, 0.001, 0.0015849, 0.0025119, 0.0039811, 0.0044668, 0.0063096, 0.0089125, 0.012589, 0.017783, 0.031623, 0.050119, 0.089125, 0.15849, 0.28184, 0.79433, ]
# Thetas=[0.001]
# print(Thetas)
# PATCH: include the asymptotic window (1e-4 to 1e-3) where T^4 scaling should appear
# Also include overlap with higher temperatures for continuity
Thetas = (np.geomspace(1e-4, 1e-3, 12).tolist()
          + [0.0012, 0.0016, 0.002, 0.0025, 0.0035, 0.005, 0.007, 0.01, 0.014, 0.02, 0.028, 0.04,
             0.056, 0.08, 0.112, 0.16, 0.224, 0.32, 0.448, 0.64, 0.896, 1.28])

# active-shell cutoff: only include states where f(1-f) > cutoff
# Increased slightly to help reduce Nactive at ultra-low T (optional: try 1e-7 or 3e-7 if needed)
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


def even_ge(n: int) -> int:
    """Small helper: make n even and >= 2."""
    n = max(int(n), 2)
    return n if (n % 2 == 0) else (n + 1)


def pbox_for_theta(theta: float) -> float:
    """Return appropriate pbox requirement based on temperature."""
    t = float(theta)
    return float(PBOX_MIN_LOW) if (t <= float(THETA_PBOX_SWITCH)) else float(PBOX_MIN_HIGH)


def choose_dp(theta: float) -> float:
    """
    Choose dp to keep Theta/dp^2 constant (constant ring pixels / Nactive).
    Uses piecewise anchors: low-T (0.01-0.1) and high-T (0.1-1) regions.
    If PRIORITIZE_SIZE_OVER_RING is True, allows dp to exceed DP_RING_MAX to maintain size.
    """
    theta_val = float(theta)
    pbox = pbox_for_theta(theta_val)
    
    # Choose anchor based on temperature range
    if theta_val < THETA_CROSSOVER:
        # Low-T region: use low-T anchor
        pixel_ratio = PIXEL_RATIO_LOW
        dp_T = math.sqrt(theta_val / pixel_ratio)
        # Apply ring cap only if prioritizing ring detail over size
        if DP_RING_MAX is not None and not PRIORITIZE_SIZE_OVER_RING:
            dp_T = min(dp_T, float(DP_RING_MAX))
    else:
        # High-T region: use high-T anchor
        pixel_ratio = PIXEL_RATIO_HIGH
        dp_T = math.sqrt(theta_val / pixel_ratio)
        # No ring cap for high-T (dp is already large)
    
    # --- CRITICAL: ensure the box can actually contain the required momentum range ---
    # If dp is too small, choose_Nmax(dp) would demand Nmax > NMAX_MAX, then clamp happens,
    # and pmax becomes too small (can even be < 1), producing nonsense low-T rates.
    dp_box_min = (2.0 * pbox) / float(NMAX_MAX)
    dp_T = max(dp_T, dp_box_min)
    
    return max(dp_T, float(DP_FLOOR))


def choose_Nmax(dp_T: float, theta: float) -> int:
    """Choose Nmax to ensure pmax >= pbox_for_theta(theta)."""
    pbox = pbox_for_theta(theta)
    n = math.ceil(2.0 * float(pbox) / max(dp_T, 1e-30))
    n = max(n, int(NMAX_MIN))
    n = min(n, int(NMAX_MAX))
    return even_ge(n)


def make_index_map(nx: np.ndarray, ny: np.ndarray, Nmax: int, half: int) -> np.ndarray:
    # PATCH: vectorized for performance (critical when Nmax becomes thousands at low T)
    idx_map = -np.ones((Nmax, Nmax), dtype=np.int32)
    idx_map[nx + half, ny + half] = np.arange(nx.size, dtype=np.int32)
    return idx_map


def precompute(nx, ny, dp: float, Theta: float, shift_x: float = 0.0, shift_y: float = 0.0):
    """
    Vectorized precompute for huge speedup when Nmax grows.
    NOTE: indices nx,ny remain integers for exact momentum conservation via idx_map,
    but physical momenta are shifted.
    """
    px = dp * (nx.astype(np.float64) + shift_x)
    py = dp * (ny.astype(np.float64) + shift_y)
    P  = np.sqrt(px * px + py * py)
    eps = P  # Dirac: ε(p)=|p| in ε_F units (was parabolic: eps = P*P)
    
    # Vectorized f computation
    invT = 1.0 / float(Theta)
    em = math.exp(-invT)
    a = 1.0 - em
    x = (eps - 1.0) * invT
    x = np.clip(x, -700.0, 700.0)
    ex = np.exp(x)
    f = a / (ex + a)
    
    return px, py, P, eps, f


def active_indices(f: np.ndarray, eps: np.ndarray, Theta: float, cutoff: float, dp: float) -> np.ndarray:
    """
    Active set should follow the thermal shell width ~ Theta.
    Using only w=f(1-f) on a coarse Cartesian grid can "freeze" the shell at the lattice energy spacing.
    
    PATCH (strong): restrict to a true ring around p≈1 (not just energy window) to reduce
    bulk contamination that pollutes angular harmonics on a Cartesian lattice.
    
    PATCH: For Dirac dispersion, eps = p, so energy mismatch scales ~O(dp), not dp^2.
    """
    w = f * (1.0 - f)
    # Require near the Fermi surface in ENERGY (for Dirac: eps = p, eps_F = 1)
    # PATCH: make the shell Theta-dominated with a dp floor (NOT dp^2 for Dirac).
    SHELL_REL = 20.0
    SHELL_DP_REL = 40.0  # For Dirac: dp floor (was 80.0 * dp^2 for parabolic)
    base_width = max(SHELL_REL * Theta, SHELL_DP_REL * dp)
    shell_width = float(RING_SHELL_TIGHTEN) * base_width

    # For Dirac: eps = p, so energy window is already in p-space
    # |eps-1| = |p-1|, so shell_width directly gives p_rad
    # PATCH: use parameterized floor to control ring thickness (prevents Nactive explosion)
    p_rad = max(0.5 * shell_width, RING_FLOOR_DP_MULT * dp)

    # For Dirac: p = eps (no sqrt needed)
    p = eps
    ring = np.abs(p - 1.0) < p_rad
    shell = np.abs(eps - 1.0) < shell_width
    return np.where((w > cutoff) & shell & ring)[0].astype(np.int32)


if USE_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def assemble_rows_numba_active(Ma, nx, ny, idx_map, half, eps, f,
                                   active, pos, dp, lam_eff, pref, dimless_scale):
        """
        Assemble CLOSED active-subspace operator Ma (shape Nactive x Nactive).
        NOTE: This implements a closed operator on the active subspace:
          - row index i1 is active (output restricted to active)
          - columns correspond to active eta only
          - requires i2' to be active (skip event if i2' is inactive) => closed operator

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

                    # Require 2' active => closed operator on active subspace
                    a2p = pos[i2p]
                    if a2p < 0:
                        continue

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
                    Ma[a1, a2p] += W
                    Ma[a1, a1]  -= W
                    Ma[a1, a2]  -= W


def build_matrix_for_theta(Theta: float, Nmax_T: int, dp_T: float):
    nx, ny, half = build_centered_lattice(Nmax_T)
    idx_map = make_index_map(nx, ny, Nmax_T, half)
    px, py, P, eps, f = precompute(nx, ny, dp_T, Theta, float(SHIFT_X), float(SHIFT_Y))

    # Temperature-following active shell
    active = active_indices(f, eps, Theta, ACTIVE_CUTOFF, dp_T)
    Nstates = nx.size
    Nactive = int(active.size)
    
    # Diagnostic: actual pmax reached by the grid (account for half-step)
    pmax = (float(half) - 0.5) * float(dp_T)
    
    print(f"Theta={Theta:.6g}  Nmax={Nmax_T}  dp={dp_T:.6g}  "
          f"Nstates={Nstates}  Nactive={Nactive}  USE_NUMBA={USE_NUMBA}  ACTIVE_ONLY={BUILD_ACTIVE_ONLY}  pmax={pmax:.4f}")
    
    # Hard stop protection: skip matrices that would take too long
    MAX_NACTIVE = 6000
    if Nactive > MAX_NACTIVE:
        print(f"SKIP Theta={Theta}: Nactive={Nactive} > {MAX_NACTIVE} (too slow, would take days)")
        return
    
    if pmax < 1.05:
        print(f"WARNING: pmax={pmax:.4f} < 1.05. Grid does not properly include the Fermi surface p=1. "
              f"Low-T eigenvalues will be garbage. Increase NMAX_MAX or relax dp(theta) scaling / pbox.")

    # Resolution warning (this is *the* main reason your curves don't scale at very low T on a Cartesian grid)
    if (dp_T * dp_T) > (0.5 * Theta):
        print(f"WARNING: dp^2={dp_T*dp_T:.3e} is not << Theta={Theta:.3e}. "
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
    # PATCH: For Dirac, dp floor for λ (energy mismatch scales ~O(dp), not dp^2).
    lam_T   = LAMBDA_REL * Theta
    lam_dp  = LAMBDA_DP_REL * dp_T
    lam_eff = max(lam_T, lam_dp, LAMBDA_MIN)

    if BUILD_ACTIVE_ONLY:
        Ma = np.zeros((Nactive, Nactive), dtype=np.float64)

        if USE_NUMBA:
            assemble_rows_numba_active(
                Ma, nx, ny, idx_map, half, eps, f,
                active, pos,
                float(dp_T), float(lam_eff), float(pref), float(dimless_scale)
            )
        else:
            dp4 = dp_T ** 4
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
                        if ix < 0 or ix >= Nmax_T or iy < 0 or iy >= Nmax_T:
                            continue
                        i2p = int(idx_map[ix, iy])
                        if i2p < 0:
                            continue
                        a2p = int(pos[i2p])
                        # Require 2' active => closed operator on active subspace
                        if a2p < 0:
                            continue

                        e2p, f2p = float(eps[i2p]), float(f[i2p])
                        dE = e1 + e2 - e1p - e2p
                        delta_eps = (1.0 / math.pi) * lam_eff / (dE * dE + lam_eff * lam_eff)

                        # Symmetrized Pauli factor
                        F_fwd = f1 * f2 * (1.0 - f1p) * (1.0 - f2p)
                        F_bwd = f1p * f2p * (1.0 - f1) * (1.0 - f2)
                        F = 0.5 * (F_fwd + F_bwd)
                        W = pref * dimless_scale * F * delta_eps * dp4

                        Ma[a1, a1p] += W
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

    # Store only minimal meta to keep .pkl sizes sane
    w_full = f * (1.0 - f)
    meta = {
        "Theta": float(Theta),
        "Nmax": int(Nmax_T),
        "half": int(half),
        "dp": float(dp_T),
        "pmax": float(pmax),
        "lambda": float(lam_eff),
        "V2": float(V2),
        "hbar": float(HBAR),
        "include_dimless_pref": bool(INCLUDE_DIMLESS_PREF),
        "active_cutoff": float(ACTIVE_CUTOFF),
        "active_only": bool(BUILD_ACTIVE_ONLY),
        "shift_x": float(SHIFT_X),
        "shift_y": float(SHIFT_Y),
        # Store only what eigen solver needs (active-only arrays):
        "active": active.astype(np.int32),
        "w_active": w_full[active].astype(np.float64),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    fname = os.path.join(OUT_DIR, f"M_Iee_N{Nmax_T}_dp{dp_T:.8g}_T{Theta:.10g}.pkl")
    with open(fname, "wb") as fp:
        pickle.dump((M_to_save, meta), fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {fname}")


def main():
    if not USE_NUMBA:
        print("WARNING: numba not found. This will be very slow. Install numba for practical runtimes.")
    if AUTO_DP_FROM_ANCHOR:
        print(f"[AUTO_GRID] Low-T anchor (Theta<{THETA_CROSSOVER}): Theta={THETA_ANCHOR_LOW}, dp={DP_ANCHOR_LOW}, PIXEL_RATIO={PIXEL_RATIO_LOW:.6f}")
        print(f"[AUTO_GRID] High-T anchor (Theta>={THETA_CROSSOVER}): Theta={THETA_ANCHOR_HIGH}, dp={DP_ANCHOR_HIGH}, PIXEL_RATIO={PIXEL_RATIO_HIGH:.6f}")
        print(f"[AUTO_GRID] PRIORITIZE_SIZE_OVER_RING={PRIORITIZE_SIZE_OVER_RING} (allows dp>{DP_RING_MAX} to keep size constant)")
        print(f"[AUTO_GRID] Low-T pbox: {PBOX_MIN_LOW} for Theta<= {THETA_PBOX_SWITCH}, High-T pbox: {PBOX_MIN_HIGH}")
        print(f"[AUTO_GRID] NMAX_MAX={NMAX_MAX}")
    for T in Thetas:
        Theta = float(T)
        if AUTO_DP_FROM_ANCHOR:
            dp_T = choose_dp(Theta)
            Nmax_T = choose_Nmax(dp_T, Theta)
            # Determine which anchor was used
            if Theta < THETA_CROSSOVER:
                pixel_ratio = PIXEL_RATIO_LOW
                anchor_info = f"low-T anchor"
            else:
                pixel_ratio = PIXEL_RATIO_HIGH
                anchor_info = f"high-T anchor"
            print(f"[AUTO_GRID] Theta={Theta:.6g} -> dp={dp_T:.8g}, Nmax={Nmax_T} ({anchor_info}) "
                  f"(Theta/dp^2={Theta/(dp_T*dp_T):.3f}, target={pixel_ratio:.3f})")
        else:
            dp_T = float(dp)
            Nmax_T = int(Nmax)
        build_matrix_for_theta(Theta, Nmax_T, dp_T)


if __name__ == "__main__":
    main()
