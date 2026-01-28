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
    # Set numba threads from environment (important for Slurm)
    try:
        import os
        num_threads = int(os.environ.get("NUMBA_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        if num_threads > 1:
            numba.set_num_threads(num_threads)
    except Exception:
        pass
    
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
# We will *not* try to keep file sizes constant here; accuracy first.
RING_PIXEL_BOOST = 3.0      # strong; try 2.5–4.0
RING_SHELL_TIGHTEN = 0.45   # tighter ring (suppresses bulk contamination)

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
# PATCH: dp^2 floor (not dp). Under momentum conservation, typical Δε mismatch scales ~O(dp^2).
LAMBDA_REL = 0.03        # lambda_T = 0.03 * Theta  (was 0.1)
LAMBDA_DP2_REL = 5.0      # lambda_dp2 = 5.0 * dp^2 (energy spacing scale)
LAMBDA_MIN = 1e-16       # reduced from 1e-12 to allow tiny rates at very low T

V2   = 1.0         # |V|^2
HBAR = 1.0         # ħ (set 1 for dimensionless)

# ------------------------ BILAYER BAND PATCH ------------------------
# Physical band params (same units as mu_phys)
V_BAND   = 1.0    # v
MU_PHYS  = 0.001#1.0    # chemical potential scale (units)
U_BAND   = 1#10.0 * MU_PHYS   # U = 10 * mu  (strongly gapped / U >> mu)

# If you want fixed-density mu(T) instead of fixed mu, set this True.
# Default False = keep mu fixed (recommended for "U = 10 mu" regime)
FIX_DENSITY = False

# Derived scales so that your dimensionless grid momentum P has FS at P=1 at T=0
# Solve mu = sqrt(v^2 kF^2 + U^2) - U  =>  v^2 kF^2 = mu^2 + 2 mu U
K_F0   = math.sqrt(MU_PHYS * MU_PHYS + 2.0 * MU_PHYS * U_BAND) / max(V_BAND, 1e-30)
if K_F0 <= 0.0 or (not math.isfinite(K_F0)):
    raise RuntimeError(f"Bad K_F0={K_F0}. Check MU_PHYS, U_BAND, V_BAND.")

U_BAR  = U_BAND / MU_PHYS
V_BAR  = (V_BAND * K_F0) / MU_PHYS   # = sqrt(1 + 2 U_BAR)

# Dimensionless target density in your (dimensionless) momentum units:
# At T=0 with FS at P=1:  ∫_0^1 P dP = 1/2
DENSITY_TARGET = 0.5

def eps_bilayer_vec(P: np.ndarray) -> np.ndarray:
    # dimensionless energy in units of MU_PHYS, as a function of dimensionless momentum P=k/kF0
    return np.sqrt((V_BAR * P) ** 2 + (U_BAR * U_BAR)) - U_BAR

def eps_bilayer_scalar(p: float) -> float:
    return math.sqrt((V_BAR * p) * (V_BAR * p) + (U_BAR * U_BAR)) - U_BAR

def vgroup_scalar(p: float) -> float:
    # d eps / dP in dimensionless units
    denom = math.sqrt((V_BAR * p) * (V_BAR * p) + (U_BAR * U_BAR))
    return (V_BAR * V_BAR * p) / max(denom, 1e-30)

def fermi_vec(eps: np.ndarray, mu_bar: float, Theta: float) -> np.ndarray:
    # stable Fermi-Dirac: f = 1/(exp((eps-mu)/Theta)+1)
    if Theta <= 0.0:
        return (eps < mu_bar).astype(np.float64)
    x = (eps - mu_bar) / Theta
    x = np.clip(x, -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)

def density_from_mu(mu_bar: float, Theta: float, pmax: float = 12.0, npts: int = 8192) -> float:
    # returns ∫_0^∞ P dP f(P) (dimensionless density up to the same constant prefactor as T=0)
    P = np.linspace(0.0, pmax, npts)
    eps = eps_bilayer_vec(P)
    f = fermi_vec(eps, mu_bar, Theta)
    return float(np.trapz(P * f, P))

def solve_mu_bar_for_density(Theta: float) -> float:
    # Solve density(mu,Theta)=DENSITY_TARGET via bisection.
    # At low T, mu_bar ~ 1.
    if Theta <= 0.0:
        return 1.0

    # bracket
    mu_lo = -20.0
    mu_hi =  5.0
    d_lo = density_from_mu(mu_lo, Theta)
    d_hi = density_from_mu(mu_hi, Theta)
    # expand if needed
    while d_lo > DENSITY_TARGET:
        mu_lo -= 10.0
        d_lo = density_from_mu(mu_lo, Theta)
    while d_hi < DENSITY_TARGET:
        mu_hi += 5.0
        d_hi = density_from_mu(mu_hi, Theta)

    for _ in range(70):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        d_mid = density_from_mu(mu_mid, Theta)
        if d_mid < DENSITY_TARGET:
            mu_lo = mu_mid
        else:
            mu_hi = mu_mid
    return 0.5 * (mu_lo + mu_hi)
# -------------------------------------------------------------------

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
# Thetas = (np.geomspace(1e-4, 1e-3, 12).tolist()
#           + [0.0012, 0.0016, 0.002, 0.0025, 0.0035, 0.005, 0.007, 0.01, 0.014, 0.02, 0.028, 0.04,
#              0.056, 0.08, 0.112, 0.16, 0.224, 0.32, 0.448, 0.64, 0.896, 1.28])

Thetas = np.geomspace(1e-4, 1.28, 50)
#[1e-2]#np.geomspace(1e-4, 1e-3, 12)

# active-shell cutoff: only include states where f(1-f) > cutoff
ACTIVE_CUTOFF = 1e-8

# DO NOT include an explicit 1/Theta prefactor.
# If you want a constant phase-space normalization, keep only the constant (2π)^-4.
INCLUDE_DIMLESS_PREF = True
DIMLESS_CONST = 1.0 / ((2.0 * math.pi) ** 4)

# ----------------------------------------------------------------------
# Target Nactive control at very low T (explicit energy window for shell)
# These knobs control the active shell width and minimum ring thickness.
# For dp ≈ 1e-3, PRAD_FLOOR_DP=0.5 gives Nactive ~ 2π/dp ≈ 5e3 at ultra-low T.
EWIN_THETA_MULT = 7     # energy window in units of Theta
EWIN_DP_MULT    = 0.5     # Δε floor ≈ 0.5 * vgF * dp  -> p_rad floor ≈ dp/2
EWIN_DP2_MULT   = 0.0     # optional dp^2 floor (keep 0 unless explicitly needed)
PRAD_FLOOR_DP   = 0.5     # ring thickness floor in units of dp (≈ one radial pixel)

# NEW: target number of active states (kept approximately constant with T)
N_ACTIVE_TARGET = 5000

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


def precompute(nx, ny, dp: float, Theta: float, mu_bar: float,
               shift_x: float = 0.0, shift_y: float = 0.0):
    """
    Physical momenta are P = dp*(n+shift) in UNITS OF kF0 (dimensionless),
    so FS stays at P≈1 by construction.

    eps is dimensionless in units of MU_PHYS (so mu_bar=mu/MU_PHYS).
    """
    px = dp * (nx.astype(np.float64) + shift_x)
    py = dp * (ny.astype(np.float64) + shift_y)
    P  = np.sqrt(px * px + py * py)

    eps = eps_bilayer_vec(P)
    f = fermi_vec(eps, mu_bar, float(Theta))

    return px, py, P, eps, f


def active_indices(f: np.ndarray, P: np.ndarray, eps: np.ndarray,
                   Theta: float, cutoff: float, dp: float, mu_bar: float) -> np.ndarray:
    """
    Select active states around eps ≈ mu_bar with an energy window,
    then *fix* the number of active states to ~N_ACTIVE_TARGET by
    keeping those closest to the Fermi ring P≈1.

    Steps:
      1) Build an energy-based shell: |eps-mu_bar| < ewin with a dp-based floor.
      2) Apply the usual f(1-f) > cutoff filter.
      3) Among all candidates, sort by |P-1| and keep the N_ACTIVE_TARGET
         nearest to the Fermi radius.

    This keeps N_active approximately constant in T, while preserving
    a thin, nearly isotropic ring around the Fermi surface.
    """
    # Pauli weight
    w = f * (1.0 - f)

    # group velocity at FS (P=1) in dimensionless units:
    vgF = vgroup_scalar(1.0)  # d eps / dP at FS

    # Energy window with dp floor:
    # |eps - mu| < max(A*Theta, B*vgF*dp, C*dp^2)
    ewin = max(
        EWIN_THETA_MULT * Theta,
        EWIN_DP_MULT * vgF * dp,
        EWIN_DP2_MULT * (dp * dp),
    )

    shell = np.abs(eps - mu_bar) < ewin
    base_mask = shell & (w > cutoff)

    cand = np.where(base_mask)[0].astype(np.int32)
    total_cand = cand.size

    # If we don't even have N_ACTIVE_TARGET eligible states, just take all of them.
    target = int(N_ACTIVE_TARGET)
    if total_cand <= target:
        return cand

    # Prefer the pixels closest to the Fermi circle P=1.
    dr = np.abs(P[cand] - 1.0)
    order = np.argsort(dr)
    chosen = cand[order[:target]]

    return chosen.astype(np.int32)


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
    
    # chemical potential in units of MU_PHYS
    mu_bar = solve_mu_bar_for_density(Theta) if FIX_DENSITY else 1.0
    
    px, py, P, eps, f = precompute(nx, ny, dp_T, Theta, mu_bar, float(SHIFT_X), float(SHIFT_Y))
    Nstates = nx.size

    # Compute lam_eff (energy broadening for δ_ε)
    vgF = vgroup_scalar(1.0)
    lam_T   = LAMBDA_REL * Theta
    lam_dp  = 5.0 * vgF * dp_T                 # Dirac-like spacing ~ v_g * dp
    lam_dp2 = LAMBDA_DP2_REL * (dp_T * dp_T)   # Parabolic-like mismatch ~ dp^2

    # Blend dp and dp^2 floors based on U/μ ratio:
    #  - U_BAR << 1  -> Dirac-like, prefer lam_dp
    #  - U_BAR >> 1  -> parabolic near FS, prefer lam_dp2
    U_ratio = U_BAR
    w_par = U_ratio / (1.0 + U_ratio)          # -> 1 when U>>mu, ->0 when U<<mu
    lam_floor = w_par * lam_dp2 + (1.0 - w_par) * lam_dp

    lam_eff = max(lam_T, lam_floor, LAMBDA_MIN)

    # Active shell: explicit energy window + controlled ring thickness
    active = active_indices(f, P, eps, Theta, ACTIVE_CUTOFF, dp_T, mu_bar)
    Nactive = int(active.size)
    
    # Diagnostic: actual pmax reached by the grid (account for half-step)
    pmax = (float(half) - 0.5) * float(dp_T)
    
    print(f"Theta={Theta:.6g}  Nmax={Nmax_T}  dp={dp_T:.6g}  "
          f"Nstates={Nstates}  Nactive={Nactive}  USE_NUMBA={USE_NUMBA}  ACTIVE_ONLY={BUILD_ACTIVE_ONLY}  pmax={pmax:.4f}")
    
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

    if BUILD_ACTIVE_ONLY:
        # FAST PATH: Use numba kernel with dense matrix (parallel & fast)
        # Use float32 to halve memory bandwidth and file size
        from scipy.sparse import csr_matrix
        
        Ma = np.zeros((Nactive, Nactive), dtype=np.float32)
        
        if USE_NUMBA:
            assemble_rows_numba_active(
                Ma, nx, ny, idx_map, half, eps, f,
                active, pos,
                float(dp_T), float(lam_eff), float(pref), float(dimless_scale)
            )
        else:
            raise RuntimeError("No numba: this will be extremely slow; install numba.")
        
        # Optional: threshold tiny entries before converting to sparse (only helps if matrix is truly sparse)
        # Uncomment if you want smaller files and most entries are near-zero:
        # thr = 1e-18
        # Ma[np.abs(Ma) < thr] = 0.0
        
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
        "band": "bilayer_sqrt_vk_U",
        "v_band": float(V_BAND),
        "U_band": float(U_BAND),
        "mu_phys": float(MU_PHYS),
        "mu_bar": float(mu_bar),
        "kF0": float(K_F0),
        "U_bar": float(U_BAR),
        "V_bar": float(V_BAR),
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