# Integrals_BLG_nonlin_mesh.py
# Extension of Kryhin-Levitov collision integral to BLG non-parabolic dispersion
# Key changes:
# - BLG dispersion: ε(k) = (γ₁/2)(√(1+4(ℏvk/γ₁)²)-1)
# - Numerical chemical potential from density constraint
# - Numerical solution of energy conservation (replaces Eq. 18)
# - Proper δ(ΔE) Jacobian (replaces Eq. 28)

import os
import time
import math
import pickle
import multiprocessing as mp
import numpy as np

# ----------------------- Parameters -----------------------
N_p   = 40
N_th  = 100
N0_th = 101#201

# BLG parameter: ζ = ℏv k_F / γ₁
# Small ζ → parabolic limit; ζ~1 → BLG crossover
zetas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # sweep density via ζ

# Temperature range (dimensionless: Θ = k_B T / ε_F)
Thetas = np.geomspace(0.0025, 1.28, 15).tolist()#30

# Integration grids
NP_LOSS   = 160
NTHP_LOSS = 32#64
NTHV_LOSS = 32#64
NPHI_GAIN = 512#1024  # reduced from 4096 for faster solving

DIM1MESH_P = 2000

BASE_DIR = os.path.join(os.getcwd(), "Matrixes_BLG")
# ----------------------------------------------------------

# Optional numba
USE_NUMBA = False
try:
    import numba
    from numba import njit, prange
    USE_NUMBA = True
except Exception:
    USE_NUMBA = False


def _theta_str(theta: float) -> str:
    return f"{theta:.10g}"


def _zeta_str(zeta: float) -> str:
    return f"{zeta:.10g}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.empty_like(x)
    out[0] = 0.0
    out[1:] = np.cumsum(area)
    return out


def dim1Mesh(rho, N=100, xmin=0, xmax=1, p=200):
    points = np.linspace(xmin, xmax, num=p * N + 1, endpoint=True)
    Int = cumulative_trapezoid(rho(points), points)
    Norm = Int[-1]
    if Norm <= 0:
        raise ValueError("Bad rho: integral is non-positive.")

    Int *= (N + 1e-3) / Norm

    result = []
    volume = []
    iprev = 0
    k = 1
    for i in range(p * N + 1):
        if Int[i] >= k:
            if Int[i] >= k + 1:
                raise ValueError("Given distribution is too sharp for chosen p.")
            result.append(0.5 * (points[i] + points[iprev]))
            volume.append(points[i] - points[iprev])
            iprev = i
            k += 1

    if not volume:
        raise ValueError("dim1Mesh failed (check bounds).")

    volume[-1] += xmax - points[iprev]
    return np.array(result, dtype=np.float64), np.array(volume, dtype=np.float64)


def AngleMesh(N_th, N0_th, Theta):
    th_i = np.linspace(0, 2 * np.pi, num=N_th, endpoint=False)
    th_i_front = np.linspace(-4 * Theta, 4 * Theta, num=N0_th, endpoint=True)
    th_i_front = np.where(th_i_front < 0, th_i_front + 2 * np.pi, th_i_front)

    x = Theta * 10
    if x > np.pi / 2:
        x = np.pi / 2
    if Theta < 0.015:
        x = np.sqrt(Theta)

    th_i_back = np.linspace(np.pi - x, np.pi + x, num=N0_th, endpoint=True)

    ang = np.unique(np.concatenate([th_i, th_i_front, th_i_back]))
    ang.sort()

    ang_right = np.append(ang, ang[0] + 2 * np.pi)
    ang_left = np.append(ang[-1] - 2 * np.pi, ang)

    d_right = ang_right[1:] - ang_right[:-1]
    d_left = ang_left[1:] - ang_left[:-1]
    dV = 0.5 * (d_right + d_left)
    return ang.astype(np.float64), dV.astype(np.float64)


def midpoint_grid_0_2pi(n: int):
    h = 2.0 * np.pi / n
    x = (np.arange(n, dtype=np.float64) + 0.5) * h
    return x, h


def midpoint_grid_0_a(n: int, a: float):
    h = a / n
    x = (np.arange(n, dtype=np.float64) + 0.5) * h
    return x, h


# ============ BLG Dispersion ============

def eps_tilde_numpy(P: np.ndarray, zeta: float) -> np.ndarray:
    """Normalized BLG dispersion: ε(k_F*P)/ε(k_F)
    Reduces to P² when zeta→0 (parabolic limit)
    """
    num = np.sqrt(1.0 + 4.0*(zeta*zeta)*(P*P)) - 1.0
    den = np.sqrt(1.0 + 4.0*(zeta*zeta)) - 1.0
    return num / den


def deps_dP_tilde_numpy(P: np.ndarray, zeta: float) -> np.ndarray:
    """Derivative of normalized dispersion"""
    den = np.sqrt(1.0 + 4.0*(zeta*zeta)) - 1.0
    num = (4.0*zeta*zeta*P) / np.sqrt(1.0 + 4.0*zeta*zeta*P*P)
    return num / den


def P_from_eps_tilde(eps_tilde_val: float, zeta: float) -> float:
    """
    Invert normalized BLG dispersion: P from ε̃.
    ε̃ = (√(1+4ζ²P²)-1) / (√(1+4ζ²)-1) = (√(1+4ζ²P²)-1) / D
    => √(1+4ζ²P²) = 1 + ε̃*D
    => P² = ((1+ε̃*D)² - 1) / (4ζ²)
    """
    if zeta < 1e-10:
        # Parabolic limit: ε̃ = P²
        return math.sqrt(max(0.0, eps_tilde_val))
    
    D = math.sqrt(1.0 + 4.0*zeta*zeta) - 1.0
    if D < 1e-12:
        return math.sqrt(max(0.0, eps_tilde_val))
    
    inner = 1.0 + eps_tilde_val * D
    if inner < 1.0:
        return 0.0
    
    P_sq = (inner*inner - 1.0) / (4.0*zeta*zeta)
    return math.sqrt(max(0.0, P_sq))


def compute_p_bounds_from_mu(Theta: float, zeta: float, mu_tilde: float, 
                             cutoff: float = 1e-3) -> tuple:
    """
    Compute p_min, p_max from μ̃ and Θ using energy-based cutoffs.
    Ensures grid contains the same equilibrium used to solve μ̃.
    
    Uses: f = a when (ε-μ)/Θ = ln(1/a - 1) = Xcut
    """
    Xcut = math.log(1.0/cutoff - 1.0)  # ~6.9 for cutoff=1e-3
    
    eps_max = mu_tilde + Theta * Xcut
    eps_min = mu_tilde - Theta * Xcut
    
    p_max = P_from_eps_tilde(eps_max, zeta)
    p_min = max(0.01, P_from_eps_tilde(eps_min, zeta))  # Clamp at small positive
    
    return p_min, p_max


def solve_mu_tilde(Theta: float, zeta: float) -> float:
    """Solve for dimensionless chemical potential from density constraint
    ∫ P dP f(P; μ, Θ, ζ) = 1/2
    
    Uses adaptive grid with high resolution near P=1 (the Fermi step)
    """
    # Adaptive grid: dense near P=1 to resolve Fermi step
    w = max(0.05, 50.0 * Theta)  # window width around P=1
    Pmax = max(4.0, 1.0 + 10*w)  # don't need 2/Theta
    
    # Build grid: coarse away from P=1, dense near P=1
    Ps1 = np.linspace(0.0, max(1.0-w, 0.0), 800, endpoint=False)
    Ps2 = np.linspace(max(1.0-w, 0.0), 1.0+w, 6000, endpoint=False)  # dense
    Ps3 = np.linspace(1.0+w, Pmax, 1200)
    
    Ps = np.unique(np.concatenate([Ps1, Ps2, Ps3]))
    
    def I(mu):
        eps = eps_tilde_numpy(Ps, zeta)
        x = (eps - mu) / Theta
        # avoid overflow
        x = np.clip(x, -50, 50)
        f = 1.0 / (np.exp(x) + 1.0)
        return np.trapz(Ps * f, Ps) - 0.5
    
    # Bracket search
    lo, hi = -20.0, 20.0
    flo, fhi = I(lo), I(hi)
    
    if flo * fhi > 0:
        # Try wider bracket
        lo, hi = -50.0, 50.0
        flo, fhi = I(lo), I(hi)
    
    # Bisection
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fmid = I(mid)
        if abs(fmid) < 1e-12:
            break
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    
    return 0.5 * (lo + hi)


if USE_NUMBA:
    @njit(cache=True, fastmath=True)
    def eps_tilde(P: float, zeta: float) -> float:
        """Normalized BLG dispersion"""
        num = math.sqrt(1.0 + 4.0*zeta*zeta*P*P) - 1.0
        den = math.sqrt(1.0 + 4.0*zeta*zeta) - 1.0
        return num / den
    
    @njit(cache=True, fastmath=True)
    def deps_dP_tilde(P: float, zeta: float) -> float:
        """Derivative of normalized dispersion"""
        den = math.sqrt(1.0 + 4.0*zeta*zeta) - 1.0
        if P == 0:
            return 0.0
        num = (4.0*zeta*zeta*P) / math.sqrt(1.0 + 4.0*zeta*zeta*P*P)
        return num / den
    
    @njit(cache=True, fastmath=True)
    def f_scalar(P: float, Theta: float, zeta: float, mu_tilde: float) -> float:
        """Fermi-Dirac occupation for BLG"""
        eps = eps_tilde(P, zeta)
        x = (eps - mu_tilde) / Theta
        if x > 50:
            return 0.0
        if x < -50:
            return 1.0
        return 1.0 / (math.exp(x) + 1.0)
    
    @njit(cache=True, fastmath=True)
    def solve_q_final_state(Kx: float, Ky: float,
                            P1: float, P2: float,
                            nx: float, ny: float,
                            Theta: float, zeta: float, mu_tilde: float):
        """
        Multi-root solver: finds ALL roots of energy conservation and sums weights.
        Returns (p1_out, p2_out, weight_total, ok) where:
        - p1_out, p2_out: representative solution (last root found)
        - weight_total: sum of 2*q/|dE/dq| over all roots
        - ok: 1 if roots found, 0 otherwise
        """
        Ein = eps_tilde(P1, zeta) + eps_tilde(P2, zeta)
        
        # Determine search range
        K_mag = math.sqrt(Kx*Kx + Ky*Ky)
        q_init = 0.5 * math.sqrt((Kx - 2.0*P1)*(Kx - 2.0*P1) + Ky*Ky)
        q_min = 1e-6
        q_max = max(20.0, 2.0 * (P1 + P2) + K_mag + 2.0)
        
        # Coarse scan for sign changes (find brackets)
        n_scan = 32
        q_scan_lo = [0.0] * 16  # Max 16 brackets
        q_scan_hi = [0.0] * 16
        F_scan_lo = [0.0] * 16
        F_scan_hi = [0.0] * 16
        n_brackets = 0
        
        F_prev = 0.0
        q_prev = q_min
        
        # Initial eval at q_min
        ax = 0.5*Kx + q_min*nx
        ay = 0.5*Ky + q_min*ny
        bx = 0.5*Kx - q_min*nx
        by = 0.5*Ky - q_min*ny
        a = math.sqrt(ax*ax + ay*ay)
        b = math.sqrt(bx*bx + by*by)
        F_prev = eps_tilde(a, zeta) + eps_tilde(b, zeta) - Ein
        
        # Scan for sign changes
        for i in range(1, n_scan + 1):
            q = q_min + (q_max - q_min) * i / n_scan
            
            ax = 0.5*Kx + q*nx
            ay = 0.5*Ky + q*ny
            bx = 0.5*Kx - q*nx
            by = 0.5*Ky - q*ny
            a = math.sqrt(ax*ax + ay*ay)
            b = math.sqrt(bx*bx + by*by)
            F = eps_tilde(a, zeta) + eps_tilde(b, zeta) - Ein
            
            if F_prev * F <= 0.0 and n_brackets < 16:
                q_scan_lo[n_brackets] = q_prev
                q_scan_hi[n_brackets] = q
                F_scan_lo[n_brackets] = F_prev
                F_scan_hi[n_brackets] = F
                n_brackets += 1
            
            q_prev = q
            F_prev = F
        
        if n_brackets == 0:
            return -1.0, -1.0, 0.0, 0
        
        # Find all roots via bisection and sum weights
        weight_total = 0.0
        p1_rep = -1.0
        p2_rep = -1.0
        
        for br_idx in range(n_brackets):
            q_lo = q_scan_lo[br_idx]
            q_hi = q_scan_hi[br_idx]
            F_lo = F_scan_lo[br_idx]
            F_hi = F_scan_hi[br_idx]
            
            # Bisection to find root
            for _ in range(40):
                q_mid = 0.5 * (q_lo + q_hi)
                
                ax = 0.5*Kx + q_mid*nx
                ay = 0.5*Ky + q_mid*ny
                bx = 0.5*Kx - q_mid*nx
                by = 0.5*Ky - q_mid*ny
                a = math.sqrt(ax*ax + ay*ay)
                b = math.sqrt(bx*bx + by*by)
                
                F_mid = eps_tilde(a, zeta) + eps_tilde(b, zeta) - Ein
                
                if abs(F_mid) < 1e-11:
                    q = q_mid
                    break
                
                if F_mid > 0.0:
                    q_hi = q_mid
                    F_hi = F_mid
                else:
                    q_lo = q_mid
                    F_lo = F_mid
            else:
                q = 0.5 * (q_lo + q_hi)
            
            # Compute Jacobian weight for this root
            ax = 0.5*Kx + q*nx
            ay = 0.5*Ky + q*ny
            bx = 0.5*Kx - q*nx
            by = 0.5*Ky - q*ny
            a = math.sqrt(ax*ax + ay*ay)
            b = math.sqrt(bx*bx + by*by)
            
            da_dq = 0.0
            db_dq = 0.0
            if a > 1e-14:
                da_dq = (ax*nx + ay*ny) / a
            if b > 1e-14:
                db_dq = -(bx*nx + by*ny) / b
            
            dE = deps_dP_tilde(a, zeta)*da_dq + deps_dP_tilde(b, zeta)*db_dq
            if abs(dE) > 1e-14:
                weight_i = 2.0 * q / abs(dE)
                weight_total += weight_i
                # Keep last root as representative
                p1_rep = a
                p2_rep = b
        
        if weight_total <= 0.0:
            return -1.0, -1.0, 0.0, 0
        
        return p1_rep, p2_rep, weight_total, 1
    
    @njit(cache=True, fastmath=True)
    def int_I1_loss_for_p(P_i: float,
                          Theta: float, zeta: float, mu_tilde: float,
                          pj: np.ndarray, hp: float,
                          thp_cos: np.ndarray, thp_sin: np.ndarray, hthp: float,
                          thv_cos: np.ndarray, thv_sin: np.ndarray, hthv: float) -> float:
        """Loss term I1 with BLG dispersion - sums contributions from ALL roots"""
        om_f0 = 1.0 - f_scalar(0.0, Theta, zeta, mu_tilde)
        twopi = 2.0 * math.pi
        pref = f_scalar(P_i, Theta, zeta, mu_tilde) / (Theta * (twopi ** 4))
        
        res = 0.0
        
        # k_1 = (P_i, 0)
        k1x = P_i
        k1y = 0.0
        
        for ip in range(pj.shape[0]):
            P_j = pj[ip]
            fj = f_scalar(P_j, Theta, zeta, mu_tilde)
            
            for a in range(thp_cos.shape[0]):
                cthp = thp_cos[a]
                sthp = thp_sin[a]
                
                # k_2 = (P_j cos(θ_p), P_j sin(θ_p))
                k2x = P_j * cthp
                k2y = P_j * sthp
                
                Kx = k1x + k2x
                Ky = k1y + k2y
                
                for b in range(thv_cos.shape[0]):
                    nx = thv_cos[b]
                    ny = thv_sin[b]
                    
                    nroots, a1, b1, w1, a2, b2, w2 = solve_q_roots2(
                        Kx, Ky, P_i, P_j, nx, ny, Theta, zeta, mu_tilde
                    )
                    
                    if nroots == 0:
                        continue
                    
                    # Root 1 contribution
                    om1 = 1.0 - f_scalar(a1, Theta, zeta, mu_tilde)
                    om2 = 1.0 - f_scalar(b1, Theta, zeta, mu_tilde)
                    res += (P_j * fj * om1 * om2 * w1)
                    
                    # Root 2 contribution (if exists)
                    if nroots == 2:
                        om1 = 1.0 - f_scalar(a2, Theta, zeta, mu_tilde)
                        om2 = 1.0 - f_scalar(b2, Theta, zeta, mu_tilde)
                        res += (P_j * fj * om1 * om2 * w2)
        
        dV = hp * hthp * hthv
        return pref * res * dV

    @njit(cache=True, fastmath=True)
    def eval_F_only(Kx: float, Ky: float, nx: float, ny: float, Ein: float, zeta: float, q: float) -> float:
        """Evaluate F(q) = E_out(q) - E_in"""
        ax = 0.5*Kx + q*nx
        ay = 0.5*Ky + q*ny
        bx = 0.5*Kx - q*nx
        by = 0.5*Ky - q*ny
        a = math.sqrt(ax*ax + ay*ay)
        b = math.sqrt(bx*bx + by*by)
        if a < 1e-14 or b < 1e-14:
            return 1e30  # Reject pathological case
        return eps_tilde(a, zeta) + eps_tilde(b, zeta) - Ein
    
    @njit(cache=True, fastmath=True)
    def bisect_root_q(Kx: float, Ky: float, nx: float, ny: float, Ein: float, zeta: float, q_lo: float, q_hi: float) -> float:
        """Robust bisection using endpoint signs (NOT F_mid>0 shortcut)
        
        CRITICAL: Validates endpoints are not sentinel values (1e30) before trusting sign.
        """
        F_lo = eval_F_only(Kx, Ky, nx, ny, Ein, zeta, q_lo)
        F_hi = eval_F_only(Kx, Ky, nx, ny, Ein, zeta, q_hi)
        
        # Guard: reject if either endpoint is invalid (sentinel from eval_F_only)
        sentinel_thresh = 1e20
        if abs(F_lo) > sentinel_thresh or abs(F_hi) > sentinel_thresh:
            # Try to fix invalid endpoints by nudging toward the other
            if abs(F_lo) > sentinel_thresh:
                # Move q_lo slightly toward q_hi
                q_lo = q_lo + 0.01 * (q_hi - q_lo)
                if q_lo >= q_hi:
                    return -1.0
                F_lo = eval_F_only(Kx, Ky, nx, ny, Ein, zeta, q_lo)
                if abs(F_lo) > sentinel_thresh:
                    return -1.0
            
            if abs(F_hi) > sentinel_thresh:
                # Move q_hi slightly toward q_lo
                q_hi = q_hi - 0.01 * (q_hi - q_lo)
                if q_hi <= q_lo:
                    return -1.0
                F_hi = eval_F_only(Kx, Ky, nx, ny, Ein, zeta, q_hi)
                if abs(F_hi) > sentinel_thresh:
                    return -1.0
        
        # Must bracket (both endpoints now valid)
        if F_lo * F_hi > 0.0:
            return -1.0
        
        for _ in range(48):
            q_mid = 0.5 * (q_lo + q_hi)
            F_mid = eval_F_only(Kx, Ky, nx, ny, Ein, zeta, q_mid)
            
            # Reject if midpoint is invalid
            if abs(F_mid) > sentinel_thresh:
                return -1.0
            
            if abs(F_mid) < 1e-12:
                return q_mid
            if F_lo * F_mid <= 0.0:
                q_hi = q_mid
                F_hi = F_mid
            else:
                q_lo = q_mid
                F_lo = F_mid
        
        return 0.5 * (q_lo + q_hi)
    
    @njit(cache=True, fastmath=True)
    def add_root_if_new(Kx: float, Ky: float, nx: float, ny: float, zeta: float,
                       q_root: float, q_dedupe_base: float,
                       nroots: int, max_roots: int,
                       a1: float, b1: float, w1: float, q1: float,
                       a2: float, b2: float, w2: float, q2: float):
        """Compute (a,b,w) at q_root and insert if not duplicate
        
        Uses relative deduplication tolerance: q_dedupe = q_dedupe_base * (1 + q_root)
        """
        # Relative deduplication tolerance
        q_dedupe = q_dedupe_base * (1.0 + abs(q_root))
        
        # De-dupe
        if nroots >= 1 and abs(q_root - q1) < q_dedupe:
            return nroots, a1, b1, w1, q1, a2, b2, w2, q2
        if nroots >= 2 and abs(q_root - q2) < q_dedupe:
            return nroots, a1, b1, w1, q1, a2, b2, w2, q2
        if nroots >= max_roots:
            return nroots, a1, b1, w1, q1, a2, b2, w2, q2
        
        # Compute a, b and weight
        ax = 0.5*Kx + q_root*nx
        ay = 0.5*Ky + q_root*ny
        bx = 0.5*Kx - q_root*nx
        by = 0.5*Ky - q_root*ny
        a = math.sqrt(ax*ax + ay*ay)
        b = math.sqrt(bx*bx + by*by)
        if a < 1e-14 or b < 1e-14:
            return nroots, a1, b1, w1, q1, a2, b2, w2, q2
        
        da_dq = (ax*nx + ay*ny) / a
        db_dq = -(bx*nx + by*ny) / b
        dE = deps_dP_tilde(a, zeta)*da_dq + deps_dP_tilde(b, zeta)*db_dq
        if abs(dE) < 1e-14 or q_root < 1e-14:
            return nroots, a1, b1, w1, q1, a2, b2, w2, q2
        
        w = 2.0 * q_root / abs(dE)
        
        if nroots == 0:
            a1, b1, w1, q1 = a, b, w, q_root
            nroots = 1
        else:
            a2, b2, w2, q2 = a, b, w, q_root
            nroots = 2
        
        return nroots, a1, b1, w1, q1, a2, b2, w2, q2
    
    @njit(cache=True, fastmath=True)
    def solve_q_roots2(Kx: float, Ky: float,
                       P1: float, P2: float,
                       nx: float, ny: float,
                       Theta: float, zeta: float, mu_tilde: float):
        """
        Find up to TWO distinct roots of energy conservation for fixed direction n=(nx,ny):
            eps(|K/2 + q n|) + eps(|K/2 - q n|) = eps(P1)+eps(P2)
        
        Returns:
            nroots, a1, b1, w1, a2, b2, w2
        where wi = 2*q_i/|dE/dq| evaluated at root i.
        """
        Ein = eps_tilde(P1, zeta) + eps_tilde(P2, zeta)
        
        # Search range
        Kmag = math.sqrt(Kx*Kx + Ky*Ky)
        q_min = 1e-8
        q_max = max(20.0, 2.0*(P1 + P2) + Kmag + 2.0)
        
        # Scan parameters (scaled with energy for robustness)
        n_scan = 32
        Ftol_scan = 1e-10 * (1.0 + abs(Ein))  # Energy-scaled near-zero tolerance
        q_dedupe_base = 1e-6  # Base for relative deduplication tolerance
        max_roots = 2
        
        # Outputs
        nroots = 0
        a1 = b1 = w1 = 0.0
        a2 = b2 = w2 = 0.0
        q1 = -1.0
        q2 = -1.0
        
        # Initial point
        q_prev = q_min
        F_prev = eval_F_only(Kx, Ky, nx, ny, Ein, zeta, q_prev)
        
        # Scan for brackets / near-zero hits
        for i in range(1, n_scan + 1):
            q = q_min + (q_max - q_min) * (i / n_scan)
            F = eval_F_only(Kx, Ky, nx, ny, Ein, zeta, q)
            
            # 1) Near-zero hit -> create a tiny bracket around q
            # Skip if F is a sentinel value (invalid)
            if abs(F) < Ftol_scan and abs(F) < 1e20 and nroots < max_roots:
                q_lo = max(q_min, q - (q_max - q_min) * 0.5 / n_scan)
                q_hi = min(q_max, q + (q_max - q_min) * 0.5 / n_scan)
                q_root = bisect_root_q(Kx, Ky, nx, ny, Ein, zeta, q_lo, q_hi)
                if q_root > 0.0:
                    nroots, a1, b1, w1, q1, a2, b2, w2, q2 = add_root_if_new(
                        Kx, Ky, nx, ny, zeta,
                        q_root, q_dedupe_base,
                        nroots, max_roots,
                        a1, b1, w1, q1, a2, b2, w2, q2
                    )
                    if nroots >= max_roots:
                        break
            
            # 2) Strict sign change -> bracket
            # Skip if either endpoint is a sentinel value (invalid)
            if (F_prev * F < 0.0 and abs(F_prev) < 1e20 and abs(F) < 1e20 
                and nroots < max_roots):
                q_root = bisect_root_q(Kx, Ky, nx, ny, Ein, zeta, q_prev, q)
                if q_root > 0.0:
                    nroots, a1, b1, w1, q1, a2, b2, w2, q2 = add_root_if_new(
                        Kx, Ky, nx, ny, zeta,
                        q_root, q_dedupe_base,
                        nroots, max_roots,
                        a1, b1, w1, q1, a2, b2, w2, q2
                    )
                    if nroots >= max_roots:
                        break
            
            q_prev = q
            F_prev = F
        
        return nroots, a1, b1, w1, a2, b2, w2

    
    @njit(cache=True, fastmath=True, parallel=True)
    def compute_gain_slices(out: np.ndarray,
                            p_i: np.ndarray,
                            dV: np.ndarray,
                            th: np.ndarray,
                            dV_th: np.ndarray,
                            Theta: float, zeta: float, mu_tilde: float,
                            phi_cos: np.ndarray,
                            phi_sin: np.ndarray,
                            hphi: float,
                            k0: int,
                            k1: int):
        """Gain terms (I2 - I3/I4) with BLG dispersion
        
        CRITICAL: I3/I4 integrand is f(p2') * (1-f(p1')), NOT f(p1') * (1-f(p2'))
        This ensures detailed balance is satisfied.
        
        NOTE: To diagnose solver acceptance rates, temporarily modify this function
        to return acceptance counts (requires changing return type to tuple).
        Or add a Python wrapper that calls this and counts p1_out >= 0.0.
        """
        twopi = 2.0 * math.pi
        scale = 1.0 / (Theta * (twopi ** 4))
        
        dVth0 = dV_th[0]
        Np = p_i.shape[0]
        Nphi = phi_cos.shape[0]
        
        for kk in prange(k0, k1):
            ang = th[kk]
            dv_ang = dV_th[kk]
            cpk = math.cos(ang)
            spk = math.sin(ang)
            
            for i in range(Np):
                p1 = p_i[i]
                f1 = f_scalar(p1, Theta, zeta, mu_tilde)
                
                # k_1 = (p1, 0)
                k1x = p1
                k1y = 0.0
                
                for j in range(i, Np):
                    p2 = p_i[j]
                    f2 = f_scalar(p2, Theta, zeta, mu_tilde)
                    om_f2 = 1.0 - f2
                    
                    # k_2 = (p2 cos(ang), p2 sin(ang))
                    k2x = p2 * cpk
                    k2y = p2 * spk
                    
                    Kx = k1x + k2x
                    Ky = k1y + k2y
                    
                    avg_dV = math.sqrt((dV[i] * dVth0) * (dV[j] * dv_ang))
                    
                    I2_int = 0.0
                    I34_int = 0.0
                    
                    for n in range(Nphi):
                        nx = phi_cos[n]
                        ny = phi_sin[n]
                        
                        nroots, a1, b1, w1, a2, b2, w2 = solve_q_roots2(
                            Kx, Ky, p1, p2, nx, ny, Theta, zeta, mu_tilde
                        )
                        
                        if nroots == 0:
                            continue
                        
                        # Root 1 contributions
                        f1o = f_scalar(a1, Theta, zeta, mu_tilde)
                        f2o = f_scalar(b1, Theta, zeta, mu_tilde)
                        om1 = 1.0 - f1o
                        om2 = 1.0 - f2o
                        
                        # I2: (1-f1') * (1-f2') - symmetric
                        I2_int += (om1 * om2 * w1)
                        
                        # I3+I4: f2' * (1-f1') - CORRECT KL structure
                        # Note: φ integration over 0..2π already includes n→-n, so no double-counting
                        I34_int += (f2o * om1 * w1)
                        
                        # Root 2 contributions (if exists)
                        if nroots == 2:
                            f1o = f_scalar(a2, Theta, zeta, mu_tilde)
                            f2o = f_scalar(b2, Theta, zeta, mu_tilde)
                            om1 = 1.0 - f1o
                            om2 = 1.0 - f2o
                            
                            I2_int += (om1 * om2 * w2)
                            I34_int += (f2o * om1 * w2)
                    
                    I2_int *= hphi
                    I34_int *= hphi
                    
                    I2  = I2_int  * (f1 * f2)   * scale
                    I34 = I34_int * (f1 * om_f2) * scale
                    
                    val = (I2 - I34) * avg_dV
                    out[i, j, kk] += val
                    if i != j:
                        out[j, i, kk] += val


if USE_NUMBA:
    @njit(cache=True, fastmath=True)
    def diagnostic_solver_acceptance_sample(p_i: np.ndarray,
                                           th_i: np.ndarray,
                                           phi_cos: np.ndarray,
                                           phi_sin: np.ndarray,
                                           Theta: float, zeta: float, mu_tilde: float,
                                           sample_size: int):
        """
        Diagnostic: sample solver calls to estimate acceptance rate and root count distribution.
        Returns (n_zero, n_one, n_two, n_total, w1_avg, w2_avg, w2_w1_avg)
        """
        Np = p_i.shape[0]
        K = th_i.shape[0]
        Nphi = phi_cos.shape[0]
        
        n_zero = 0
        n_one = 0
        n_two = 0
        n_total = 0
        w1_sum = 0.0
        w2_sum = 0.0
        w2_w1_sum = 0.0
        n_two_counted = 0
        
        # Sample across different (i, j, k, n) combinations
        step_i = max(1, Np // 4)
        step_j = max(1, Np // 4)
        step_k = max(1, K // 4)
        step_n = max(1, Nphi // 8)
        
        for i in range(0, Np, step_i):
            p1 = p_i[i]
            k1x = p1
            k1y = 0.0
            
            for j in range(i, Np, step_j):
                p2 = p_i[j]
                ang_idx = min(K-1, (i + j) % K)
                ang = th_i[ang_idx]
                cpk = math.cos(ang)
                spk = math.sin(ang)
                
                k2x = p2 * cpk
                k2y = p2 * spk
                Kx = k1x + k2x
                Ky = k1y + k2y
                
                for n in range(0, Nphi, step_n):
                    if n_total >= sample_size:
                        w1_avg = w1_sum / n_one if n_one > 0 else 0.0
                        w2_avg = w2_sum / n_two if n_two > 0 else 0.0
                        w2_w1_avg = w2_w1_sum / n_two_counted if n_two_counted > 0 else 0.0
                        return n_zero, n_one, n_two, n_total, w1_avg, w2_avg, w2_w1_avg
                    
                    nx = phi_cos[n]
                    ny = phi_sin[n]
                    
                    nroots, a1, b1, w1, a2, b2, w2 = solve_q_roots2(
                        Kx, Ky, p1, p2, nx, ny, Theta, zeta, mu_tilde
                    )
                    
                    n_total += 1
                    if nroots == 0:
                        n_zero += 1
                    elif nroots == 1:
                        n_one += 1
                        w1_sum += w1
                    elif nroots == 2:
                        n_two += 1
                        w1_sum += w1
                        w2_sum += w2
                        if w1 > 1e-14:
                            w2_w1_sum += w2 / w1
                            n_two_counted += 1
        
        w1_avg = w1_sum / n_one if n_one > 0 else 0.0
        w2_avg = w2_sum / n_two if n_two > 0 else 0.0
        w2_w1_avg = w2_w1_sum / n_two_counted if n_two_counted > 0 else 0.0
        return n_zero, n_one, n_two, n_total, w1_avg, w2_avg, w2_w1_avg


def _compute_one_theta_zeta(Theta: float, zeta: float, idx: int, total: int):
    t0 = time.time()
    
    # Solve for chemical potential
    mu_tilde = solve_mu_tilde(Theta, zeta)
    
    # Momentum cutoffs from energy-based criterion (consistent with μ̃ solver)
    # Ensures grid contains same equilibrium used to solve μ̃
    p_min, p_max = compute_p_bounds_from_mu(Theta, zeta, mu_tilde, cutoff=1e-3)
    
    def rho(p):
        return np.ones_like(p)
    
    p_i, dV_p = dim1Mesh(rho, N=N_p, xmin=p_min, xmax=p_max, p=DIM1MESH_P)
    th_i, dV_th = AngleMesh(N_th, N0_th, Theta)
    K = th_i.shape[0]
    dV = p_i * dV_p
    
    computed_ints = np.zeros((N_p, N_p, K), dtype=np.float64)
    computed_I1 = np.zeros(N_p, dtype=np.float64)
    
    name = f"matrix_p-{N_p}_th-{N_th}_th0-{N0_th}_BLG"
    out_dir = os.path.join(BASE_DIR, name)
    _ensure_dir(out_dir)
    file_name = os.path.join(out_dir, 
                            f"{name}_T-{_theta_str(Theta)}_z-{_zeta_str(zeta)}-0.p")
    
    # I1 grids - use same energy-based criterion (slightly larger cutoff for phase space)
    _, pj_max = compute_p_bounds_from_mu(Theta, zeta, mu_tilde, cutoff=1e-4)
    pj, hp = midpoint_grid_0_a(NP_LOSS, pj_max)
    thp, hthp = midpoint_grid_0_2pi(NTHP_LOSS)
    thv, hthv = midpoint_grid_0_2pi(NTHV_LOSS)
    thp_cos = np.cos(thp); thp_sin = np.sin(thp)
    thv_cos = np.cos(thv); thv_sin = np.sin(thv)
    
    # Gain quadrature
    phi, hphi = midpoint_grid_0_2pi(NPHI_GAIN)
    phi_cos = np.cos(phi); phi_sin = np.sin(phi)
    
    print(f"[{idx+1}/{total}] Theta={Theta:.6f}  zeta={zeta:.4f}  mu={mu_tilde:.6f}  K={K}", flush=True)
    
    if not USE_NUMBA:
        raise RuntimeError("Install numba: pip install numba")
    
    # ---- Diagnostic: Check solver acceptance rate and root distribution ----
    if USE_NUMBA:
        n_zero, n_one, n_two, n_tot, w1_avg, w2_avg, w2_w1_avg = diagnostic_solver_acceptance_sample(
            p_i, th_i, phi_cos, phi_sin,
            float(Theta), float(zeta), float(mu_tilde),
            sample_size=5000
        )
        if n_tot > 0:
            acc_rate = (n_one + n_two) / n_tot
            print(f"    Solver: {n_zero} zero, {n_one} one-root, {n_two} two-root (accept: {acc_rate:.1%})", flush=True)
            if n_two > 0:
                print(f"    Weights: <w1>={w1_avg:.4f}, <w2>={w2_avg:.4f}, <w2/w1>={w2_w1_avg:.4f}", flush=True)
        else:
            print(f"    Solver diagnostic: WARNING - no samples tested", flush=True)
    
    # ---- I1 (loss) ----
    for i in range(N_p):
        computed_I1[i] = int_I1_loss_for_p(
            float(p_i[i]), float(Theta), float(zeta), float(mu_tilde),
            pj, float(hp),
            thp_cos, thp_sin, float(hthp),
            thv_cos, thv_sin, float(hthv)
        )
        if (i + 1) % max(1, N_p // 5) == 0 or (i + 1) == N_p:
            print(f"    I1: {i+1}/{N_p}", flush=True)
    
    # Add I1 to diagonal
    for i in range(N_p):
        computed_ints[i, i, 0] += computed_I1[i]
    
    # ---- Gain terms ----
    chunk = max(8, K // 25)
    done = 0
    while done < K:
        k0 = done
        k1 = min(K, done + chunk)
        compute_gain_slices(computed_ints, p_i, dV, th_i, dV_th, 
                           float(Theta), float(zeta), float(mu_tilde),
                           phi_cos, phi_sin, float(hphi), int(k0), int(k1))
        done = k1
        print(f"    Gain: {done}/{K}", flush=True)
    
    save_data = (float(Theta), float(zeta), float(mu_tilde), 
                computed_ints, computed_I1, p_i, th_i, dV_p, dV_th)
    with open(file_name, "wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    dt = time.time() - t0
    print(f"    Saved: {file_name}")
    print(f"    Done in {dt:.2f}s\n", flush=True)
    
    # DIAGNOSTIC: To check solver acceptance rates, temporarily modify compute_gain_slices
    # to accumulate counts. Example (non-parallel version for debugging):
    #   n_accepted = 0
    #   n_total = 0
    #   for n in range(Nphi):
    #       n_total += 1
    #       p1_out, p2_out, weight = solve_q_final_state(...)
    #       if p1_out >= 0.0:
    #           n_accepted += 1
    #   print(f"    Solver acceptance: {n_accepted}/{n_total} = {n_accepted/n_total:.1%}")


def main():
    _ensure_dir(BASE_DIR)
    
    print("=== BLG Matrix Generation ===")
    print(f"USE_NUMBA={USE_NUMBA}")
    print(f"CPU count: {mp.cpu_count()}")
    print(f"N_p={N_p}, N_th={N_th}, N0_th={N0_th}")
    print(f"Thetas: {len(Thetas)}")
    print(f"Zetas: {len(zetas)}")
    print(f"I1 grid: {NP_LOSS}x{NTHP_LOSS}x{NTHV_LOSS}")
    print(f"Gain: NPHI={NPHI_GAIN}\n", flush=True)
    
    if not USE_NUMBA:
        raise RuntimeError("Install numba: pip install numba")
    
    try:
        numba.set_num_threads(mp.cpu_count())
    except Exception:
        pass
    
    t_all = time.time()
    
    # Sweep over (Theta, zeta) pairs
    idx = 0
    total = len(Thetas) * len(zetas)
    
    for zeta in zetas:
        for Theta in Thetas:
            _compute_one_theta_zeta(float(Theta), float(zeta), idx, total)
            idx += 1
    
    print(f"=== All done in {time.time() - t_all:.2f}s ===", flush=True)


if __name__ == "__main__":
    if os.name == "nt":
        mp.freeze_support()
    main()

