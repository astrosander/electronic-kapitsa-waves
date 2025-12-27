# Integrals_BLG_analysis.py
# Eigenvalue and current relaxation rate analysis for BLG
# Computes:
# - Momentum relaxation eigenvalues (standard angular modes)
# - Current relaxation rate (velocity-weighted, physical observable)

import os
import numpy as np
import pickle
from matplotlib import pyplot as plt

# Must match matrix generator
N_p   = 40
N_th  = 100
N0_th = 101
N = 1

# Same as generator
zetas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
Thetas = np.geomspace(0.0025, 1.28, 15).tolist()

ms = [0, 1, 2, 3, 4, 5, 6]
k = 0

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
# Publication-ready font sizes
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 18   
plt.rcParams['figure.titlesize'] = 28


import matplotlib as mpl

# --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 24,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})

def _theta_str(theta: float) -> str:
    return f"{theta:.10g}"


def _zeta_str(zeta: float) -> str:
    return f"{zeta:.10g}"


def eps_tilde_numpy(P: np.ndarray, zeta: float) -> np.ndarray:
    """BLG dispersion (normalized)"""
    num = np.sqrt(1.0 + 4.0*(zeta*zeta)*(P*P)) - 1.0
    den = np.sqrt(1.0 + 4.0*(zeta*zeta)) - 1.0
    return num / den


def deps_dP_tilde_numpy(P: np.ndarray, zeta: float) -> np.ndarray:
    """Velocity (derivative of dispersion)"""
    den = np.sqrt(1.0 + 4.0*(zeta*zeta)) - 1.0
    num = (4.0*zeta*zeta*P) / np.sqrt(1.0 + 4.0*zeta*zeta*P*P)
    return num / den


def Fourier_transform2(func, th_i, dV_th, m):
    exp = np.cos(th_i * m)
    integ = func * exp
    return np.sum(integ)


# Storage
Orig_Matrixes = {}
p_is = {}
th_is = {}
dVp_s = {}
dVth_s = {}
I1s = {}
mu_tildes = {}
eigs_momentum = {}  # standard angular modes
lambda_current = {}  # current relaxation (velocity-weighted m=1) - eigenvalue (≤0)
Gamma_current = {}  # physical current relaxation rate = -λ_J (≥0)

print("=== BLG Eigenvalue & Current Relaxation Analysis ===\n", flush=True)

# Load matrices
print("Loading matrices...", flush=True)
for zeta in zetas:
    for Theta in Thetas:
        for n in range(N):
            name = f"matrix_p-{N_p}_th-{N_th}_th0-{N0_th}_BLG"
            file_name = f'./Matrixes_BLG/{name}/{name}_T-{_theta_str(Theta)}_z-{_zeta_str(zeta)}-{n}.p'
            
            try:
                (Theta_loaded, zeta_loaded, mu_tilde_loaded, 
                 matrn, I1, p_i, th_in, dV_p, dV_thn) = pickle.load(open(file_name, 'rb'))
            except FileNotFoundError:
                print(f"  WARNING: File not found: {file_name}", flush=True)
                continue
            
            if n == 0:
                matr = matrn
                th_i = th_in
                dV_th = dV_thn
            else:
                matr = np.append(matr, matrn, axis=2)
                th_i = np.append(th_i, th_in)
                dV_th = np.append(dV_th, dV_thn)
        
        key = (Theta, zeta)
        Orig_Matrixes[key] = matr
        p_is[key] = p_i
        th_is[key] = th_i
        dVp_s[key] = dV_p
        dVth_s[key] = dV_th
        I1s[key] = np.diag(I1)
        mu_tildes[key] = mu_tilde_loaded

print(f"Loaded {len(Orig_Matrixes)} matrices.\n", flush=True)

# Compute eigenvalues
print("Computing eigenvalues and current relaxation rates...\n", flush=True)

for key in Orig_Matrixes.keys():
    Theta, zeta = key
    print(f"  [compute] Theta={Theta:.6f}, zeta={zeta:.4f}", flush=True)
    
    p_i = p_is[key]
    th_i = th_is[key]
    dV_p = dVp_s[key]
    dV_th = dVth_s[key]
    I = Orig_Matrixes[key]
    
    # Standard weight (for number conservation check)
    eta = np.sqrt(p_i * dV_p)
    
    # Momentum weight: etaP = eta * p_i (proportional to momentum)
    eta_P = eta * p_i
    
    # Energy weight (for energy conservation check)
    eps = eps_tilde_numpy(p_i, zeta)
    eta_E = eta * eps
    
    # **Current weight: velocity with momentum projected out**
    # Project out Galilean zero-mode (conserved momentum)
    v = deps_dP_tilde_numpy(p_i, zeta)
    w_p = p_i * dV_p  # Radial weight (consistent with discretization)
    
    # Best-fit coefficient: v = α*p + v_perp, where v_perp ⟂ p
    alpha = np.sum(w_p * p_i * v) / np.sum(w_p * p_i * p_i)
    v_perp = v - alpha * p_i
    
    eta_J = eta * v_perp
    
    # Number distribution (m=0 should be conserved)
    core = np.einsum("p,pqi,q->i", eta, I, eta, optimize=True)
    dist = core * np.sqrt(dV_th[0]) * np.sqrt(dV_th)
    
    # Momentum distribution (m=1 should be conserved)
    coreP = np.einsum("p,pqi,q->i", eta_P, I, eta_P, optimize=True)
    dist_P = coreP * np.sqrt(dV_th[0]) * np.sqrt(dV_th)
    
    # Energy distribution (m=0 should be conserved)
    coreE = np.einsum("p,pqi,q->i", eta_E, I, eta_E, optimize=True)
    dist_E = coreE * np.sqrt(dV_th[0]) * np.sqrt(dV_th)
    
    # **Current distribution**
    coreJ = np.einsum("p,pqi,q->i", eta_J, I, eta_J, optimize=True)
    dist_J = coreJ * np.sqrt(dV_th[0]) * np.sqrt(dV_th)
    
    eigs_momentum[key] = {}
    
    # Momentum mode eigenvalues (angular harmonics)
    dist_normed = np.array(dist[1:]) / dV_th[1:] / dV_th[0]
    for m in ms:
        eigs_momentum[key][m] = Fourier_transform2(dist_normed * dV_th[1:], th_i[1:], dV_th, m)
    
    # **Current relaxation rate: m=1 mode of velocity-weighted distribution**
    # Like momentum modes, subtract m=0 offset for consistent sign and magnitude
    distJ_normed = np.array(dist_J[1:]) / dV_th[1:] / dV_th[0]
    lamJ1 = Fourier_transform2(distJ_normed * dV_th[1:], th_i[1:], dV_th, m=1)
    lamJ0 = Fourier_transform2(distJ_normed * dV_th[1:], th_i[1:], dV_th, m=0)
    lam = lamJ1 - lamJ0
    lambda_current[key] = lam
    # Physical rate: Γ_J = -λ_J (strictly positive for log-scale plotting)
    Gamma_current[key] = max(1e-300, -lam)

print("\nDone computing.\n", flush=True)

# ====== Conservation checks: Number, Momentum, Energy ======
def radial_matrix_for_m(I, th_i, dV_th, m):
    """Build m-th Fourier-projected radial matrix: M_m = Σ_k I[:,:,k] * cos(m*θ_k) * dV_th[k] / (2π)"""
    c = np.cos(m * th_i)
    # Weighted sum over angle index k
    return np.einsum("pqk,k,k->pq", I, c, dV_th, optimize=True) / (2.0 * np.pi)

print("="*80)
print("CONSERVATION CHECKS (Rate-Style: λ = m1 - m0)")
print("="*80)
print("Number: λ_N = N_m1 - N_m0 (should be small)")
print("Momentum: λ_P = P_m1 - P_m0 (should be ~0 if conserved)")
print("Energy: λ_E = E_m1 - E_m0 (should be small)")
print("Direct test: ||(M1-M0) @ p|| / ||p|| (should be << 1)")
print("="*80)
print(f"{'ζ':>6} {'Θ':>8} {'λ_N':>12} {'λ_P':>12} {'λ_E':>12} {'||(M1-M0)p||/||p||':>18}")
print("-"*80)

for zeta in zetas[:3]:  # Check first few zetas
    for Theta in Thetas[::5]:  # Subsample
        key = (Theta, zeta)
        if key not in eigs_momentum:
            continue
        
        p_i = p_is[key]
        th_i = th_is[key]
        dV_p = dVp_s[key]
        dV_th = dVth_s[key]
        I = Orig_Matrixes[key]
        
        # Number conservation: rate-style eigenvalue
        N_m0 = eigs_momentum[key][0]
        N_m1 = eigs_momentum[key][1]
        lamN = N_m1 - N_m0
        
        # Momentum conservation: compute with etaP
        eta = np.sqrt(p_i * dV_p)
        eta_P = eta * p_i
        coreP = np.einsum("p,pqi,q->i", eta_P, I, eta_P, optimize=True)
        dist_P = coreP * np.sqrt(dV_th[0]) * np.sqrt(dV_th)
        distP_normed = np.array(dist_P[1:]) / dV_th[1:] / dV_th[0]
        P_m0 = Fourier_transform2(distP_normed * dV_th[1:], th_i[1:], dV_th, m=0)
        P_m1 = Fourier_transform2(distP_normed * dV_th[1:], th_i[1:], dV_th, m=1)
        lamP = P_m1 - P_m0
        
        # Energy conservation: compute with etaE
        eps = eps_tilde_numpy(p_i, zeta)
        eta_E = eta * eps
        coreE = np.einsum("p,pqi,q->i", eta_E, I, eta_E, optimize=True)
        dist_E = coreE * np.sqrt(dV_th[0]) * np.sqrt(dV_th)
        distE_normed = np.array(dist_E[1:]) / dV_th[1:] / dV_th[0]
        E_m0 = Fourier_transform2(distE_normed * dV_th[1:], th_i[1:], dV_th, m=0)
        E_m1 = Fourier_transform2(distE_normed * dV_th[1:], th_i[1:], dV_th, m=1)
        lamE = E_m1 - E_m0
        
        # Direct operator test: apply (M1 - M0) to momentum vector
        M0 = radial_matrix_for_m(I, th_i, dV_th, m=0)
        M1 = radial_matrix_for_m(I, th_i, dV_th, m=1)
        C1 = M1 - M0  # Rate operator (consistent with lambda_current definition)
        
        gP = p_i.copy()  # Momentum radial profile
        res = C1 @ gP
        r_direct = np.linalg.norm(res) / max(1e-300, np.linalg.norm(gP))
        
        print(f"{zeta:6.4f} {Theta:8.5f} {lamN:12.6e} {lamP:12.6e} {lamE:12.6e} {r_direct:18.6e}")

print("="*80)
print("Interpretation:")
print("  - λ_P should be ~0 (momentum conserved) or at least << |P_m0|")
print("  - ||(M1-M0)p||/||p|| should be << 1 (operator kills momentum mode)")
print("  - If λ_P is NOT small, check I1 loss term handling in matrix generation")
print("="*80)
print("", flush=True)

# ====== Plot 1: Momentum eigenvalues vs Theta (for various zetas) ======

fig1, ax1 = plt.subplots(figsize=(10, 7))

# Pick a few representative zetas for clarity
plot_zetas = [0.01, 0.1, 0.3, 0.7]
# Modern color palette: deep teal, vibrant blue, rich purple, warm orange
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Collect all x and y values for setting limits
all_x_vals_1 = []
all_y_vals_1 = []

for iz, zeta in enumerate(plot_zetas):
    for m in [2, 3, 4, 5, 6]:  # skip m=0,1 for clarity
        y = []
        y0 = []
        valid_thetas = []
        for Theta in Thetas:
            key = (Theta, zeta)
            if key not in eigs_momentum:
                continue
            y.append(eigs_momentum[key][m])
            y0.append(eigs_momentum[key][0])
            valid_thetas.append(Theta)
        
        if len(y) < 5:
            continue
        
        power = 2
        y_plot = 3*np.log(2*np.pi) + np.log((np.array(y) - np.array(y0))/(np.array(valid_thetas)**power))
        x_plot = np.log(valid_thetas)[6:]
        y_plot_trimmed = y_plot[6:]
        
        label = f"ζ={zeta}, m={m}" if m == 2 else None
        ax1.plot(x_plot, y_plot_trimmed, 
                color=colors[iz], alpha=0.7 if m > 2 else 1.0,
                linewidth=2.5 if m == 2 else 2.0,
                label=label)
        
        all_x_vals_1.extend(x_plot.tolist())
        all_y_vals_1.extend(y_plot_trimmed.tolist())

# Set limits based on data range (no margin)
if len(all_x_vals_1) > 0 and len(all_y_vals_1) > 0:
    x_min_1 = min(all_x_vals_1)
    x_max_1 = max(all_x_vals_1)
    y_min_1 = min(all_y_vals_1)
    y_max_1 = max(all_y_vals_1)
    ax1.set_xlim(x_min_1, x_max_1)
    ax1.set_ylim(y_min_1, y_max_1)

ax1.set_xlabel(r'Temperature, $\ln(T/T_F)$', fontsize=26)
ax1.set_ylabel(r'Eigenvalues, $\ln(\lambda_m T_F^2 / T^2)$', fontsize=26)
ax1.tick_params(axis='both', which='major', labelsize=20)
legend1 = ax1.legend(title='Momentum modes', fontsize=20, title_fontsize=22)
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig('./BLG_eigenvals_momentum.png', dpi=300)
print("Saved: ./BLG_eigenvals_momentum.png", flush=True)

# ====== Plot 2: Current relaxation rate vs zeta ======

fig2, ax2 = plt.subplots(figsize=(10, 7))

# Collect all x and y values for setting limits
all_x_vals_2 = []
all_y_vals_2 = []

# Fix a few temperatures and plot Γ_J(ζ)
plot_thetas_idx = [5//2, 10//2, 15//2, 20//2, 25//2]  # sample across temperature range

for ith in plot_thetas_idx:
    if ith >= len(Thetas):
        continue
    Theta = Thetas[ith]
    
    zeta_vals = []
    gamma_J_vals = []
    
    for zeta in zetas:
        key = (Theta, zeta)
        if key not in Gamma_current:
            continue
        zeta_vals.append(zeta)
        gamma_J_vals.append(Gamma_current[key])
    
    if len(zeta_vals) < 2:
        continue
    
    ax2.plot(zeta_vals, gamma_J_vals, '-', label=f'Θ={Theta:.4f}', linewidth=2.5)
    all_x_vals_2.extend(zeta_vals)
    all_y_vals_2.extend(gamma_J_vals)

# Add k² reference line (normalized to temperature at index 25//2)
ref_theta_idx = 25 // 2
if ref_theta_idx < len(Thetas):
    Theta_ref = Thetas[ref_theta_idx]
    # Get first valid data point for normalization
    zeta_ref = None
    gamma_ref = None
    for zeta in zetas:
        key = (Theta_ref, zeta)
        if key in Gamma_current and Gamma_current[key] > 0:
            zeta_ref = zeta
            gamma_ref = Gamma_current[key]
            break
    
    if zeta_ref is not None and gamma_ref is not None:
        z_all = np.array([z for z in zetas if z >= zeta_ref])
        if len(z_all) > 0:
            g_ref_k2 = gamma_ref * (z_all / zeta_ref)**4
            ax2.loglog(z_all, g_ref_k2, '--', color='blue', alpha=1, linewidth=3.0, label=r'$\propto k^2 \propto \zeta^4$')
            all_x_vals_2.extend(z_all.tolist())
            all_y_vals_2.extend(g_ref_k2.tolist())
            # k⁻¹ reference: Γ ∝ k⁻¹ ∝ ζ⁻¹ (normalized to last data point)
            zeta_ref_km1 = None
            gamma_ref_km1 = None
            for zeta in reversed(zetas):
                key = (Theta_ref, zeta)
                if key in Gamma_current and Gamma_current[key] > 0:
                    zeta_ref_km1 = zeta
                    gamma_ref_km1 = Gamma_current[key]
                    break
            if zeta_ref_km1 is not None and gamma_ref_km1 is not None:
                # Only plot for the last 25% of zetas (by count)
                zetas_sorted = sorted([z for z in zetas if z <= zeta_ref_km1])
                if len(zetas_sorted) > 0:
                    n_25pct = max(1, int(len(zetas_sorted) * 0.7    ))  # Last 25% of points
                    z_all_km1 = np.array(zetas_sorted[-n_25pct:])  # Take last 25% of zetas
                    if len(z_all_km1) > 0:
                        g_ref_km1 = gamma_ref_km1 * (z_all_km1 / zeta_ref_km1)**(-1)
                        ax2.loglog(z_all_km1, g_ref_km1, '-.', color="black", alpha=1, linewidth=3.0, label=r'$\propto k^{-1} \propto \zeta^{-2}$')
                        all_x_vals_2.extend(z_all_km1.tolist())
                        all_y_vals_2.extend(g_ref_km1.tolist())

# Set limits based on data range (no margin)
if len(all_x_vals_2) > 0 and len(all_y_vals_2) > 0:
    x_min_2 = min(all_x_vals_2)
    x_max_2 = max(all_x_vals_2)
    y_min_2 = min(all_y_vals_2)
    y_max_2 = max(all_y_vals_2)
    ax2.set_xlim(x_min_2, x_max_2)
    ax2.set_ylim(y_min_2, y_max_2)

ax2.set_xlabel(r'$\zeta = \hbar v k_F / \gamma_1$', fontsize=26)
ax2.set_ylabel(r'Current relaxation rate, $\Gamma_J = -\lambda_J$', fontsize=26)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.tick_params(axis='both', which='major', labelsize=20)
legend2 = ax2.legend(fontsize=20)
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig('./BLG_current_relaxation_vs_density.png', dpi=300)
fig2.savefig('./BLG_current_relaxation_vs_density.pdf', dpi=300)
print("Saved: ./BLG_current_relaxation_vs_density.png", flush=True)

# ====== Plot 3: Current relaxation scaling at fixed Theta ======

fig3, ax3 = plt.subplots(figsize=(10, 7))

# Pick one intermediate temperature
Theta_fixed = Thetas[25//2]  # middle of range

zeta_vals = []
gamma_J_vals = []

for zeta in zetas:
    key = (Theta_fixed, zeta)
    if key not in Gamma_current:
        continue
    zeta_vals.append(zeta)
    gamma_J_vals.append(Gamma_current[key])

# Collect all x and y values for setting limits
all_x_vals_3 = []
all_y_vals_3 = []

if len(zeta_vals) >= 3:
    ax3.loglog(zeta_vals, gamma_J_vals, '-', linewidth=2.5, markersize=8, label='BLG current')
    all_x_vals_3.extend(zeta_vals)
    all_y_vals_3.extend(gamma_J_vals)
    
    # Reference scalings (using positive Gamma values)
    # k² reference: Γ ∝ k² ∝ ζ²
    z_all = np.array(zeta_vals)
    if len(z_all) > 0 and gamma_J_vals[0] > 0:
        # Normalize to first data point
        g_ref_k2 = gamma_J_vals[0] * (z_all / zeta_vals[0])**4
        ax3.loglog(z_all, g_ref_k2, '-.', color='#4A90E2', alpha=0.8, linewidth=3.0, label=r'$\propto k^2 \propto \zeta^4$')
        all_x_vals_3.extend(z_all.tolist())
        all_y_vals_3.extend(g_ref_k2.tolist())
    
    # Small ζ (intraband): Γ ∝ n², and since n ∝ k_F² ∝ ζ², we have Γ ∝ ζ⁴
    z_small = np.array([z for z in zeta_vals if z < 0.2])
    if len(z_small) > 0 and gamma_J_vals[0] > 0:
        g_ref_small = gamma_J_vals[0] * (z_small / zeta_vals[0])**4
        ax3.loglog(z_small, g_ref_small, '--', color='#6C757D', alpha=0.8, linewidth=3.0, label=r'$\propto n^2 \propto \zeta^4$ (intraband)')
        all_x_vals_3.extend(z_small.tolist())
        all_y_vals_3.extend(g_ref_small.tolist())
    
    # Large ζ (collinear suppression): Γ ∝ n⁻¹, and since n ∝ ζ², we have Γ ∝ ζ⁻²
    z_large = np.array([z for z in zeta_vals if z > 0.3])
    if len(z_large) > 0 and gamma_J_vals[-1] > 0:
        g_ref_large = gamma_J_vals[-1] * (z_large / zeta_vals[-1])**(-2)
        ax3.loglog(z_large, g_ref_large, '-.', color='#E63946', alpha=0.8, linewidth=3.0, label=r'$\propto n^{-1} \propto \zeta^{-2}$ (collinear)')
        all_x_vals_3.extend(z_large.tolist())
        all_y_vals_3.extend(g_ref_large.tolist())

# Set limits based on data range
if len(all_x_vals_3) > 0 and len(all_y_vals_3) > 0:
    x_min_3 = min(all_x_vals_3)
    x_max_3 = max(all_x_vals_3)
    y_min_3 = min(all_y_vals_3)
    y_max_3 = max(all_y_vals_3)
    # Add small margins for log scale
    x_margin_3 = (x_max_3 / x_min_3) ** 0.1
    y_margin_3 = (y_max_3 / y_min_3) ** 0.1
    ax3.set_xlim(x_min_3 / x_margin_3, x_max_3 * x_margin_3)
    ax3.set_ylim(y_min_3 / y_margin_3, y_max_3 * y_margin_3)

ax3.set_xlabel(r'$\zeta = \hbar v k_F / \gamma_1$', fontsize=26)
ax3.set_ylabel(r'Current relaxation rate, $\Gamma_J = -\lambda_J$', fontsize=26)
ax3.set_title(f'Current relaxation scaling at Θ={Theta_fixed:.4f}', fontsize=28)
ax3.tick_params(axis='both', which='major', labelsize=20)
legend3 = ax3.legend(fontsize=20)
ax3.grid(alpha=0.3)
fig3.tight_layout()
fig3.savefig('./BLG_current_scaling.png', dpi=300)
print("Saved: ./BLG_current_scaling.png", flush=True)

# ====== Save numerical results ======

results = {
    'zetas': zetas,
    'Thetas': Thetas,
    'eigs_momentum': eigs_momentum,
    'lambda_current': lambda_current,
    'Gamma_current': Gamma_current,  # Physical rate (positive)
    'mu_tildes': mu_tildes,
    'p_is': p_is,
    'th_is': th_is
}

with open('BLG_relaxation_results.pkl', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\nSaved: BLG_relaxation_results.pkl")

# ====== Print summary table ======

print("\n" + "="*80)
print("CURRENT RELAXATION RATE SUMMARY")
print("="*80)
print("NOTE: λ_J is the eigenvalue (typically ≤ 0). Physical rate is Γ_J = -λ_J ≥ 0.")
print("="*80)
print(f"{'ζ':>8} {'Θ':>8} {'μ̃':>8} {'λ_J':>12} {'Γ_J':>12} {'Γ_J/Θ²':>12}")
print("-"*80)

for zeta in zetas:
    for Theta in Thetas[::5]:  # subsample for readability
        key = (Theta, zeta)
        if key not in lambda_current:
            continue
        mu = mu_tildes[key]
        lam = lambda_current[key]
        Gamma = max(0.0, -lam)  # Physical rate (non-negative)
        Gamma_scaled = Gamma / (Theta**2)
        print(f"{zeta:8.4f} {Theta:8.5f} {mu:8.4f} {lam:12.6e} {Gamma:12.6e} {Gamma_scaled:12.6e}")

print("="*80)
print("\nAnalysis complete!")

