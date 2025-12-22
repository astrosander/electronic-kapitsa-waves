import numpy as np
import matplotlib.pyplot as plt

def find_udrift(u0,u1,kmin=-8,kmax=8,N=20000,eps=1e-12,tol=1e-8,max_iter=80):
    k=np.linspace(kmin,kmax,N)
    def has_positive_growth(u):
        gamma = gamma0 * np.exp(-n / w)
        Dp, Dn = eta_p, eta_n
        p = n * u
        Pn = n * U0 - p**2 / (n**2 * m)
        Pp = 2 * p / (n * m)
        Gamma_n = -gamma / w
        Lambda = (Gamma_n - gamma / n) * p
        G_tilde = gamma + (Dp - Dn) * k**2
        Delta = (G_tilde + 1j * k * Pp)**2 + 4j * k * Lambda / m - 4 * k**2 * Pn / m
        omega_plus = (-1j * G_tilde + k * Pp + 1j * np.sqrt(Delta)) / 2 - 1j * Dn * k**2
        return np.max(np.imag(omega_plus)) > eps
    a,b=u0,u1
    for _ in range(max_iter):
        if b-a<=tol: break
        u_mid = (a + b) / 2
        if has_positive_growth(u_mid):
            b = u_mid
        else:
            a = u_mid
    return (a+b)/2

def u_star_for(scan=(0,5.0,0.01)):
    lo,hi,st=scan
    k=np.linspace(-8,8,20000)
    def has_positive_growth(u):
        gamma = gamma0 * np.exp(-n / w)
        Dp, Dn = eta_p, eta_n
        p = n * u
        Pn = n * U0 - p**2 / (n**2 * m)
        Pp = 2 * p / (n * m)
        Gamma_n = -gamma / w
        Lambda = (Gamma_n - gamma / n) * p
        G_tilde = gamma + (Dp - Dn) * k**2
        Delta = (G_tilde + 1j * k * Pp)**2 + 4j * k * Lambda / m - 4 * k**2 * Pn / m
        omega_plus = (-1j * G_tilde + k * Pp + 1j * np.sqrt(Delta)) / 2 - 1j * Dn * k**2
        return np.max(np.imag(omega_plus)) > 1e-12
    prev=False
    u=lo
    while u<=hi:
        cur=has_positive_growth(u)
        if not prev and cur:
            return find_udrift(u-st,u)
        prev=cur
        u+=st
    return None

U0 = 0.9
# n = 0.05#0.2
# w = 0.1
n = 0.94
w = 0.28#0.4
gamma0 = 16.5
m = 1.3
kmin = -7
kmax = -kmin
N = 20000

# Create figure with better proportions for publication-quality plots
fig, ax = plt.subplots(figsize=(11, 7))
# Set background color for better contrast
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

L = 10.0  # Physical box size

# Fixed velocity
u_c = 2.0
print(f"Using velocity u_c = {u_c:.4f}")

# Define two diffusion coefficient sets with distinct, contrasting colors
# Using cool colors for no diffusion, warm colors for with diffusion
diffusion_sets = [
    {'eta_p': 0.0, 'eta_n': 0.0, 'color_plus': '#1976D2', 'color_minus': '#00897B', 
     'label': 'No diffusion', 'lw': 2.0},  # Deep blue and teal
    {'eta_p': 0.1, 'eta_n': 0.1, 'color_plus': '#D32F2F', 'color_minus': '#F57C00', 
     'label': 'With diffusion', 'lw': 2.0}   # Deep red and deep orange
]

for diff_idx, diff_set in enumerate(diffusion_sets):
    eta_p = diff_set['eta_p']
    eta_n = diff_set['eta_n']
    color_plus = diff_set['color_plus']
    color_minus = diff_set['color_minus']
    lw = diff_set['lw']
    
    print(f"\nDiffusion set {diff_idx + 1}: η_p = {eta_p}, η_n = {eta_n}")
    k_out = np.linspace(kmin, kmax, N)
    
    # Physical parameters (vectorized)
    gamma = gamma0 * np.exp(-n / w)           # Γ(n)
    Dp, Dn = eta_p, eta_n
    p = n * u_c
    Pn = n * U0 - p**2 / (n**2 * m)           # Π_n
    Pp = 2 * p / (n * m)                      # Π_p
    Gamma_n = -gamma / w                      # ∂Γ/∂n
    Lambda = (Gamma_n - gamma / n) * p        # (∂Γ/∂n - Γ/n)p
    
    # Corrected formulas with proper diffusion terms
    G_tilde = gamma + (Dp - Dn) * k_out**2    # Γ̃ = Γ + (Dp - Dn)k²
    Delta = (G_tilde + 1j * k_out * Pp)**2 + 4j * k_out * Lambda / m - 4 * k_out**2 * Pn / m
    
    # The two branches (ω± according to convention e^{ikx - iωt})
    omega_plus  = (-1j * G_tilde + k_out * Pp + 1j * np.sqrt(Delta)) / 2 - 1j * Dn * k_out**2
    omega_minus = (-1j * G_tilde + k_out * Pp - 1j * np.sqrt(Delta)) / 2 - 1j * Dn * k_out**2
    
    # Growth rates ζ(k) = Im(ω±)
    zeta_plus = np.imag(omega_plus)
    zeta_minus = np.imag(omega_minus)
    
    # Plot omega_plus with thicker, more visible line
    ax.plot(k_out, zeta_plus, linewidth=lw, color=color_plus, linestyle='-', 
            alpha=0.9, zorder=3+diff_idx*2)
    
    # Plot omega_minus with thicker, more visible line
    ax.plot(k_out, zeta_minus, linewidth=lw, color=color_minus, linestyle='-', 
            alpha=0.9, zorder=2+diff_idx*2)
    
    # Print summary
    print(f"  ω_+: max ζ = {np.max(zeta_plus):.6f}, min ζ = {np.min(zeta_plus):.6f}")
    print(f"  ω_-: max ζ = {np.max(zeta_minus):.6f}, min ζ = {np.min(zeta_minus):.6f}")

# Show a few discrete mode lines for reference
mode_numbers = []
for i in mode_numbers:
    k_mode = i * 2 * np.pi / L
    if kmin <= k_mode <= kmax:
        ax.axvline(k_mode, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        if i in [1, 20, 40]:  # Label only a few
            ax.text(k_mode, ax.get_ylim()[1] * 0.95, f'n={i}', 
                   fontsize=8, ha='center', color='red', alpha=0.7)

# Enhanced styling for better visualization
ax.axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.5, zorder=1)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
ax.set_xlabel('Wave number $k$', fontsize=13, fontweight='medium')
ax.set_ylabel('Growth rate $\\zeta(k) = \\Im(\\omega_{\\pm})$', fontsize=13, fontweight='medium')
ax.set_ylim(-4, 2)
ax.set_xlim(kmin, kmax)

# Improve tick appearance
ax.tick_params(axis='both', which='major', labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# ax.set_title('Linear Instability Increment (corrected dispersion with diffusion)', fontsize=13)

plt.tight_layout()
plt.savefig("linear_instability_increment.pdf", dpi=300, bbox_inches='tight')
plt.savefig("linear_instability_increment.png", dpi=300, bbox_inches='tight')
plt.show()