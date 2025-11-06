#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Eq. (29) for the maximally unstable wave number k*.

What this script does:
1) Computes and plots the growth rate ζ(k) = Im(ω_+(k)) for a few drift velocities u,
   and overlays the Eq. (29) prediction for k* as vertical reference lines.
2) Sweeps γ by varying gamma0 and compares the numeric k* (near threshold) against
   the straight-line prediction k* = ( √(m / (4 U n)) ) * γ from Eq. (29).

Notes:
- We take γ(n) = γ0 * exp(-n / w), so γ'(n) = -γ / w < 0.
- Instability requires p * γ'(n) > 0, i.e., u < 0 for the parameters below.
- For clean comparison to Eq. (29) set diffusion symmetric: Dp = Dn (typically zero).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
})

# -----------------------
# Global parameters
# -----------------------
U0 = 1.0        # interaction/compressibility scale U
m  = 1.0        # effective mass
n  = 0.20       # density
w  = 0.25       # γ(n) decay scale
gamma0 = 2.5    # prefactor for γ(n)
eta_p = 0.0     # Dp (set equal to Dn to match derivation of Eq.29)
eta_n = 0.0     # Dn

# k-range used for plotting and searches
KMIN, KMAX, NK = -6.0, 6.0, 16001

# Optional "box size" for reporting a discrete mode index near k*
L = 10.0

# -----------------------
# Model helpers
# -----------------------
def gamma_of_n():
    """γ(n) = γ0 * exp(-n/w)."""
    return gamma0 * np.exp(-n / w)

def zeta_of_k(u, k):
    """
    Growth rate ζ(k) for the + branch, using current globals.
    Convention exp(i k x - i ω t), so ζ = Im(ω).
    """
    gamma = gamma_of_n()
    Dp, Dn = eta_p, eta_n

    p  = n * u
    Pn = n * U0 - p**2 / (n**2 * m)     # Π_n
    Pp = 2 * p / (n * m)                # Π_p

    Gamma_n = -gamma / w                # ∂γ/∂n
    Lambda  = (Gamma_n - gamma / n) * p # (∂γ/∂n - γ/n) p

    G_tilde = gamma + (Dp - Dn) * k**2  # γ̃
    Delta   = (G_tilde + 1j * k * Pp)**2 + 4j * k * Lambda / m - 4 * k**2 * Pn / m
    omega_plus = (-1j * G_tilde + k * Pp + 1j * np.sqrt(Delta)) / 2 - 1j * Dn * k**2
    return np.imag(omega_plus)

def has_positive_growth(u, eps=1e-12):
    k = np.linspace(KMIN, KMAX, NK)
    z = zeta_of_k(u, k)
    return np.max(z) > eps

def find_udrift(u0, u1, tol=1e-8, max_iter=80):
    """
    Bisection on g(u)=max_k ζ(k;u) crossing 0. Accepts either bracket ordering:
    (stable,unstable) or (unstable,stable). We internally swap if needed.
    """
    a, b = u0, u1
    fa = has_positive_growth(a)
    fb = has_positive_growth(b)
    # Ensure left is stable, right is unstable
    if fa and not fb:
        a, b = b, a
        fa, fb = fb, fa
    if fa:
        raise ValueError("Left bracket already unstable; pick a lower-magnitude u0.")
    if not fb:
        raise ValueError("Right bracket still stable; pick a higher-magnitude u1.")

    for _ in range(max_iter):
        if abs(b - a) <= tol:
            break
        mid = 0.5 * (a + b)
        if has_positive_growth(mid):
            b = mid  # mid is unstable → move right edge left
        else:
            a = mid  # mid is stable   → move left edge right
    return 0.5 * (a + b)

def u_star_for(scan=(-1.5, 0.0, 0.002)):
    """
    Scan u and detect the first *change* in stability (either False→True or True→False),
    then refine with bisection on that local bracket.
    """
    lo, hi, st = scan
    prev = has_positive_growth(lo)
    u = lo + st
    while u <= hi + 1e-15:
        cur = has_positive_growth(u)
        if prev != cur:
            # bracket is [u-st, u] regardless of transition direction
            return find_udrift(u - st, u)
        prev = cur
        u += st
    return None

def kstar_eq29():
    """Analytic k* from Eq. 29 (near threshold, Dp=Dn)."""
    gamma = gamma_of_n()
    return gamma * np.sqrt(m / (4 * U0 * n))

def small_k_curvature_central(u, kappa=1e-3):
    """
    Robust small-k curvature a via 5-point central difference:
    z''(0) ≈ (-z(2κ)+16 z(κ) - 30 z(0) + 16 z(-κ) - z(-2κ)) / (12 κ^2)
    and z(k) ≈ a k^2 + O(k^4) near 0 ⇒ a = 0.5 z''(0).
    Use very small κ to avoid contamination from k^3, but not so small that
    floating-point noise dominates; 1e-3–1e-2 is a good range here.
    """
    z0   = zeta_of_k(u, 0.0)
    z1p  = zeta_of_k(u, kappa)
    z1m  = zeta_of_k(u, -kappa)
    z2p  = zeta_of_k(u, 2.0*kappa)
    z2m  = zeta_of_k(u, -2.0*kappa)
    zpp0 = (-z2p + 16.0*z1p - 30.0*z0 + 16.0*z1m - z2m) / (12.0 * kappa**2)
    return 0.5 * zpp0  # a

def kstar_numeric_by_matching(u_factor=1.003, kappa=2e-3):
    """
    Near-threshold k* via matching using unbiased small-k curvature and a true
    high-k plateau. Returns (u_star, u_sample, k_match, z_inf, k29).
    """
    # 1) threshold (negative branch)
    u_star_val = u_star_for(scan=(-1.5, 0.0, 0.002))
    if u_star_val is None:
        u_star_val = find_udrift(-1.5, 0.0)

    u_sample = u_factor * u_star_val

    # 2) unbiased small-k curvature a
    a = small_k_curvature_central(u_sample, kappa=kappa)
    # guard: a must be positive near threshold
    if not np.isfinite(a) or a <= 0:
        return u_star_val, u_sample, np.nan, np.nan, kstar_eq29()

    # 3) high-k plateau z_inf (take it *far* out; with Dp=Dn=0 it's truly constant)
    k29   = kstar_eq29()
    K_hi  = max(100.0, 40.0 * k29)         # go very high to kill transients
    ksamp = np.linspace(0.8*K_hi, K_hi, 2001)
    ztail = zeta_of_k(u_sample, ksamp)
    z_inf = float(np.mean(ztail))          # constant plateau

    # 4) match
    num = max(z_inf - 0.0, 0.0)            # z(0)=0 for the + branch
    k_match = np.sqrt(num / a) if num > 0 else 0.0
    return u_star_val, u_sample, float(k_match), z_inf, float(k29)

# -----------------------
# Main: Figure 1 — ζ(k) with Eq.29 overlay
# -----------------------
def figure_zeta_with_eq29():
    print("Searching for critical drift u* on the unstable (negative) side...")
    u_star = u_star_for()
    if u_star is None:
        u_star = find_udrift(-1.5, 0.0)

    uc_theory = w * np.sqrt(U0 * m / n)
    gamma_here = gamma_of_n()
    k29 = kstar_eq29()

    print(f"  γ(n) = {gamma_here:.6f}")
    print(f"  Eq. (26): u_c (magnitude) = {uc_theory:.6f}  -> u* ≈ {-uc_theory:.6f}")
    print(f"  Numerical u* found = {u_star:.6f}")
    print(f"  Eq. (29): k* = {k29:.6f},  λ* = {2*np.pi/k29:.6f}\n")

    # Also compute the matching-based numeric k*
    _, u_slight, k_match, z_inf, _ = kstar_numeric_by_matching(u_factor=1.003, kappa=2e-3)
    print(f"  Matching method at u≈1.003·u*: k_match = {k_match:.6f}  (Eq29 k* = {k29:.6f})")

    # Modern color palette - perceptually uniform and distinguishable
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    line_styles = ['-', '-', '-']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    k = np.linspace(KMIN, KMAX, NK)

    u_vals = np.array([0.98, 1.00, 1.02]) * u_star
    for idx, u in enumerate(u_vals):
        z = zeta_of_k(u, k)
        label = (r'$u\simeq u^\star$' if np.isclose(u, u_star)
                 else (r'$u=%.3f$' % u))
        ax.plot(k, z, linewidth=2.5, label=label, color=colors[idx], 
                linestyle=line_styles[idx], alpha=0.9, zorder=3)

    # Overlay Eq.29 and matching k* with distinct colors and styles
    eq29_color = '#D32F2F'  # Deep red
    match_color = '#388E3C'  # Green
    ax.axvline(+k29, linestyle='--', linewidth=2.5, color=eq29_color, 
               label=fr'Eq. (29): $k^*={k29:.3f}$', alpha=0.85, zorder=4)
    ax.axvline(-k29, linestyle='--', linewidth=2.5, color=eq29_color, alpha=0.85, zorder=4)
    if np.isfinite(k_match):
        ax.axvline(+k_match, linestyle=':', linewidth=2.8, color=match_color, 
                   label=fr'Matching: $k^*={k_match:.3f}$', alpha=0.9, zorder=5)
        ax.axvline(-k_match, linestyle=':', linewidth=2.8, color=match_color, alpha=0.9, zorder=5)

    ax.axhline(0.0, color='black', linewidth=1.2, alpha=0.4, linestyle='-', zorder=1)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, zorder=0)
    ax.set_xlabel(r'$k$', fontweight='medium')
    ax.set_ylabel(r'$\zeta(k)$', fontweight='medium')
    ax.set_title(r'Linear Increment: Eq. (29) vs Matching Extraction of $k^*$', 
                 fontweight='medium', pad=15)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray', 
              facecolor='white', fancybox=True, shadow=False, borderpad=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    plt.tight_layout()
    plt.savefig("linear_instability_increment_with_matching.png", dpi=300, bbox_inches='tight')
    plt.savefig("linear_instability_increment_with_matching.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------
# Main: Figure 2 — k*_num vs γ with Eq.29 slope line
# -----------------------
def figure_kstar_vs_gamma():
    print("Sweeping gamma0: k* by matching vs Eq.29 (central-diff curvature)...")
    # Ensure exact symmetry for this check
    global eta_p, eta_n
    eta_p_saved, eta_n_saved = eta_p, eta_n
    eta_p = eta_n = 0.0

    gamma0_values = np.linspace(0.8, 4000.0, 90)
    gammas, k_num_list, k29_list = [], [], []
    slope = np.sqrt(m / (4 * U0 * n))

    for g0 in gamma0_values:
        global gamma0
        gamma0 = g0
        u_star_val, u_slight, k_match, z_inf, k29 = kstar_numeric_by_matching(
            u_factor=1.003, kappa=2e-3
        )
        gam = gamma_of_n()
        if k_match is None or not np.isfinite(k_match):
            continue
        gammas.append(gam)
        k_num_list.append(k_match)
        k29_list.append(k29)
        print(f"  gamma0={g0:.3f} -> γ(n)={gam:.4f} | u*={u_star_val:.4f} | "
              f"k_match={k_match:.4f} | Eq29={k29:.4f} | ratio={k_match/k29:.3f}")

    # Restore diffusion
    eta_p, eta_n = eta_p_saved, eta_n_saved

    # Plot numeric vs Eq.29 line with modern styling
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    if gammas:
        # Numeric points - vibrant, distinguishable color
        ax.plot(gammas, k_num_list, 'o', label=r'Numeric $k^*$', 
                color='#E53E3E', markersize=6, markeredgewidth=0.8, 
                markeredgecolor='white', alpha=0.85, zorder=3, linewidth=0)
        
        # Theory line - distinct color and style
        ax.plot(gammas, [slope*g for g in gammas], '--', 
                label=fr'Eq. (29): $k^* = \gamma\sqrt{{\frac{{m}}{{4Un}}}}$', 
                color='#1A237E', linewidth=2.8, alpha=0.9, zorder=2)
    
    ax.set_xlabel(r'$\gamma(n)$', fontweight='medium')
    ax.set_ylabel(r'$k^*$', fontweight='medium')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, zorder=0)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray',
              facecolor='white', fancybox=True, shadow=False, borderpad=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    
    # Set tick parameters for cleaner look
    ax.tick_params(which='both', width=1.2, length=5, direction='in', 
                   top=False, right=False, labelsize=13)
    
    plt.tight_layout()
    plt.savefig("verify_eq29_matching.png", dpi=300, bbox_inches='tight')
    plt.savefig("verify_eq29_matching.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------
# Script entry point
# -----------------------
if __name__ == "__main__":
    # Figure 1: ζ(k) with Eq. (29) overlay
    figure_zeta_with_eq29()

    # Figure 2: k*_num vs γ with Eq. (29) slope line
    figure_kstar_vs_gamma()
