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
print("="*80)
print("CONSERVATION CHECKS")
print("="*80)
print("Number (m=0 with eta): should be conserved (m=0 dominant)")
print("Momentum (m=1 with etaP): should be conserved (m=1 ≈ 0)")
print("Energy (m=0 with etaE): should be conserved (m=0 dominant)")
print("="*80)
print(f"{'ζ':>6} {'Θ':>8} {'N_m0':>10} {'N_m1':>10} {'|N_m1/N_m0|':>12} {'P_m1':>10} {'|P_m1/P_m0|':>12} {'E_m0':>10} {'E_m1':>10} {'|E_m1/E_m0|':>12}")
print("-"*80)

for zeta in zetas[:3]:  # Check first few zetas
    for Theta in Thetas[::5]:  # Subsample
        key = (Theta, zeta)
        if key not in eigs_momentum:
            continue
        
        # Number conservation: m=0 should dominate
        N_m0 = eigs_momentum[key][0]
        N_m1 = eigs_momentum[key][1]
        N_ratio = abs(N_m1 / N_m0) if abs(N_m0) > 1e-15 else float('inf')
        
        # Momentum conservation: compute with etaP
        p_i = p_is[key]
        th_i = th_is[key]
        dV_p = dVp_s[key]
        dV_th = dVth_s[key]
        I = Orig_Matrixes[key]
        
        eta = np.sqrt(p_i * dV_p)
        eta_P = eta * p_i
        coreP = np.einsum("p,pqi,q->i", eta_P, I, eta_P, optimize=True)
        dist_P = coreP * np.sqrt(dV_th[0]) * np.sqrt(dV_th)
        distP_normed = np.array(dist_P[1:]) / dV_th[1:] / dV_th[0]
        P_m0 = Fourier_transform2(distP_normed * dV_th[1:], th_i[1:], dV_th, m=0)
        P_m1 = Fourier_transform2(distP_normed * dV_th[1:], th_i[1:], dV_th, m=1)
        P_ratio = abs(P_m1 / P_m0) if abs(P_m0) > 1e-15 else float('inf')
        
        # Energy conservation: compute with etaE
        eps = eps_tilde_numpy(p_i, zeta)
        eta_E = eta * eps
        coreE = np.einsum("p,pqi,q->i", eta_E, I, eta_E, optimize=True)
        dist_E = coreE * np.sqrt(dV_th[0]) * np.sqrt(dV_th)
        distE_normed = np.array(dist_E[1:]) / dV_th[1:] / dV_th[0]
        E_m0 = Fourier_transform2(distE_normed * dV_th[1:], th_i[1:], dV_th, m=0)
        E_m1 = Fourier_transform2(distE_normed * dV_th[1:], th_i[1:], dV_th, m=1)
        E_ratio = abs(E_m1 / E_m0) if abs(E_m0) > 1e-15 else float('inf')
        
        print(f"{zeta:6.4f} {Theta:8.5f} {N_m0:10.4e} {N_m1:10.4e} {N_ratio:12.4e} {P_m1:10.4e} {P_ratio:12.4e} {E_m0:10.4e} {E_m1:10.4e} {E_ratio:12.4e}")

print("="*80)
print("Interpretation:")
print("  - Number: |N_m1|/|N_m0| should be small (m=0 is isotropic)")
print("  - Momentum: |P_m1|/|P_m0| should be << 1 (momentum conserved)")
print("  - Energy: |E_m1|/|E_m0| should be small (m=0 is isotropic)")
print("="*80)
print("", flush=True)

# ====== Plot 1: Momentum eigenvalues vs Theta (for various zetas) ======

fig1, ax1 = plt.subplots(figsize=(8, 6))

# Pick a few representative zetas for clarity
plot_zetas = [0.01, 0.1, 0.3, 0.7]
colors = ['C0', 'C1', 'C2', 'C3']

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
        
        label = f"ζ={zeta}, m={m}" if m == 2 else None
        ax1.plot(np.log(valid_thetas)[6:], y_plot[6:], 
                color=colors[iz], alpha=0.7 if m > 2 else 1.0,
                linewidth=1.5 if m == 2 else 1.0,
                label=label)

# ax1.set_xlim(-4.7, 0.5)
# ax1.set_ylim(-3.0, 2.2)
ax1.set_xlabel(r'Temperature, $\ln(T/T_F)$')
ax1.set_ylabel(r'Eigenvalues, $\ln(\lambda_m T_F^2 / T^2)$')
ax1.legend(title='Momentum modes')
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig('./BLG_eigenvals_momentum.png', dpi=300)
print("Saved: ./BLG_eigenvals_momentum.png", flush=True)

# ====== Plot 2: Current relaxation rate vs zeta (density) ======

fig2, ax2 = plt.subplots(figsize=(8, 6))

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
    
    ax2.plot(zeta_vals, gamma_J_vals, 'o-', label=f'Θ={Theta:.4f}', linewidth=1.5)

ax2.set_xlabel(r'$\zeta = \hbar v k_F / \gamma_1$ (density)')
ax2.set_ylabel(r'Current relaxation rate, $\Gamma_J = -\lambda_J$ (dimensionless)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig('./BLG_current_relaxation_vs_density.png', dpi=300)
print("Saved: ./BLG_current_relaxation_vs_density.png", flush=True)

# ====== Plot 3: Current relaxation scaling at fixed Theta ======

fig3, ax3 = plt.subplots(figsize=(8, 6))

# Pick one intermediate temperature
Theta_fixed = Thetas[20//2]  # middle of range

zeta_vals = []
gamma_J_vals = []

for zeta in zetas:
    key = (Theta_fixed, zeta)
    if key not in Gamma_current:
        continue
    zeta_vals.append(zeta)
    gamma_J_vals.append(Gamma_current[key])

if len(zeta_vals) >= 3:
    ax3.loglog(zeta_vals, gamma_J_vals, 'o-', linewidth=2, markersize=8, label='BLG current')
    
    # Reference scalings (using positive Gamma values)
    # Small ζ (intraband): Γ ∝ n², and since n ∝ k_F² ∝ ζ², we have Γ ∝ ζ⁴
    z_small = np.array([z for z in zeta_vals if z < 0.2])
    if len(z_small) > 0 and gamma_J_vals[0] > 0:
        g_ref_small = gamma_J_vals[0] * (z_small / zeta_vals[0])**4
        ax3.loglog(z_small, g_ref_small, '--', color='gray', alpha=0.7, label=r'$\propto n^2 \propto \zeta^4$ (intraband)')
    
    # Large ζ (collinear suppression): Γ ∝ n⁻¹, and since n ∝ ζ², we have Γ ∝ ζ⁻²
    z_large = np.array([z for z in zeta_vals if z > 0.3])
    if len(z_large) > 0 and gamma_J_vals[-1] > 0:
        g_ref_large = gamma_J_vals[-1] * (z_large / zeta_vals[-1])**(-2)
        ax3.loglog(z_large, g_ref_large, '-.', color='red', alpha=0.7, label=r'$\propto n^{-1} \propto \zeta^{-2}$ (collinear)')

ax3.set_xlabel(r'$\zeta = \hbar v k_F / \gamma_1$ (density)')
ax3.set_ylabel(r'Current relaxation rate, $\Gamma_J = -\lambda_J$')
ax3.set_title(f'Current relaxation scaling at Θ={Theta_fixed:.4f}')
ax3.legend()
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

