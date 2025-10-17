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

U0 = 1.5
eta_p = 0.01
eta_n = 10#*0.01
# n = 0.05#0.2
# w = 0.1
n = 0.2
w = 0.22#0.4
gamma0 = 3.0
m = 1.0
kmin = -2
kmax = 2
N = 20000

fig, ax = plt.subplots(figsize=(10, 6))

L = 10.0  # Physical box size

print("Calculating u_star...")
u_star = u_star_for()

u_d_min = u_star-0.005#2.5
u_d_max = u_star+0.005#2.999
u_d_step = 0.001
u_values = np.arange(u_d_min, u_d_max + u_d_step, u_d_step)

if u_star is not None:
    u_values = np.append(u_values, u_star)
    print(f"Critical drift velocity u_star = {u_star:.6f}")
    
    # Verify u_star has exactly one intersection with zero
    k_test = np.linspace(-8, 8, 1000)
    gamma = gamma0 * np.exp(-n / w)
    Dp, Dn = eta_p, eta_n
    p = n * u_star
    Pn = n * U0 - p**2 / (n**2 * m)
    Pp = 2 * p / (n * m)
    Gamma_n = -gamma / w
    Lambda = (Gamma_n - gamma / n) * p
    G_tilde = gamma + (Dp - Dn) * k_test**2
    Delta = (G_tilde + 1j * k_test * Pp)**2 + 4j * k_test * Lambda / m - 4 * k_test**2 * Pn / m
    omega_plus = (-1j * G_tilde + k_test * Pp + 1j * np.sqrt(Delta)) / 2 - 1j * Dn * k_test**2
    zeta_test = np.imag(omega_plus)
    
    # Count zero crossings
    zero_crossings = np.sum(np.diff(np.sign(zeta_test)) != 0)
    print(f"Zero crossings in u_star curve: {zero_crossings}")
    print(f"Max growth rate at u_star: {np.max(zeta_test):.6f}")
    print(f"Min growth rate at u_star: {np.min(zeta_test):.6f}")
else:
    print("Warning: Could not find critical drift velocity u_star")

print(f"Scanning u values: {u_values}")
print(f"Physical box size L = {L}")
print(f"Fundamental mode spacing: 2π/L = {2*np.pi/L:.4f}\n")

# Store k_max values for each u
k_max_values = []
zeta_max_values = []

for u in u_values:
    k_out = np.linspace(kmin, kmax, N)
    
    # Physical parameters (vectorized)
    gamma = gamma0 * np.exp(-n / w)           # Γ(n)
    Dp, Dn = eta_p, eta_n
    p = n * u
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
    
    # Growth rate ζ(k) = Im(ω_+) is the potentially unstable branch
    zeta = np.imag(omega_plus)
    
    # Find maximum growth rate on both k < 0 and k > 0 sides
    mask_neg = k_out < 0
    mask_pos = k_out > 0
    
    if np.any(mask_neg):
        k_neg = k_out[mask_neg]
        zeta_neg = zeta[mask_neg]
        k_max_neg_idx = np.argmax(zeta_neg)
        k_max_neg = k_neg[k_max_neg_idx]
        zeta_max_neg = zeta_neg[k_max_neg_idx]
    else:
        k_max_neg = 0
        zeta_max_neg = 0
        
    if np.any(mask_pos):
        k_pos = k_out[mask_pos]
        zeta_pos = zeta[mask_pos]
        k_max_pos_idx = np.argmax(zeta_pos)
        k_max_pos = k_pos[k_max_pos_idx]
        zeta_max_pos = zeta_pos[k_max_pos_idx]
    else:
        k_max_pos = 0
        zeta_max_pos = 0
    
    # Store both maxima
    k_max_values.append((k_max_neg, k_max_pos))
    zeta_max_values.append((zeta_max_neg, zeta_max_pos))
    
    # Most unstable mode numbers for both sides
    n_star_neg = int(np.round(np.abs(k_max_neg) / (2 * np.pi / L))) if k_max_neg != 0 else 0
    n_star_pos = int(np.round(np.abs(k_max_pos) / (2 * np.pi / L))) if k_max_pos != 0 else 0
    lambda_max_neg = 2 * np.pi / np.abs(k_max_neg) if k_max_neg != 0 else np.inf
    lambda_max_pos = 2 * np.pi / np.abs(k_max_pos) if k_max_pos != 0 else np.inf
    
    print(f"u = {u:.1f}:")
    print(f"  k < 0: k_max = {k_max_neg:7.3f}, ζ_max = {zeta_max_neg:6.2f}, "
          f"λ_max = {lambda_max_neg:.3f}, n* ≈ {n_star_neg}")
    print(f"  k > 0: k_max = {k_max_pos:7.3f}, ζ_max = {zeta_max_pos:6.2f}, "
          f"λ_max = {lambda_max_pos:.3f}, n* ≈ {n_star_pos}")
    
    if u_star is not None and np.isclose(u, u_star):
        lw = 2.5
        label = f'$u^{{\\bigstar}} = {u:.2f}$'
        color = 'blue'
    else:
        lw = 1.5
        label = f'u = {u:.2f}'
        color = plt.cm.viridis((u - u_d_min) / (u_d_max - u_d_min))
    
    ax.plot(k_out, zeta, linewidth=lw, color=color, label=label)
    
    if u_star is not None and np.isclose(u, u_star):
        mid_idx = len(k_out) *3 // 4
        ax.annotate(
            f'$u^{{\\bigstar}} = {u_star:.2f}$',
            xy=(k_out[mid_idx], zeta[mid_idx]),
            xytext=(k_out[mid_idx]*0.7, zeta[mid_idx] ),
            arrowprops=dict(arrowstyle="->", color='blue', lw=1.5),
            fontsize=10,
            color='blue'
        )
    
    if u_star is None or not np.isclose(u, u_star):
        if k_max_neg != 0:
            ax.plot(k_max_neg, zeta_max_neg, 's', color=color, markersize=6, 
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        if k_max_pos != 0:
            ax.plot(k_max_pos, zeta_max_pos, 's', color=color, markersize=6, 
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)

# Show a few discrete mode lines for reference
mode_numbers = []
for i in mode_numbers:
    k_mode = i * 2 * np.pi / L
    if kmin <= k_mode <= kmax:
        ax.axvline(k_mode, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
        if i in [1, 20, 40]:  # Label only a few
            ax.text(k_mode, ax.get_ylim()[1] * 0.95, f'n={i}', 
                   fontsize=8, ha='center', color='red', alpha=0.7)

ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.3)
ax.grid(True, alpha=0.3)
ax.set_xlabel('$k$', fontsize=12)
ax.set_ylabel('$\\zeta(k) = \\Im\\omega_+$', fontsize=12)
# ax.set_title('Linear Instability Increment (corrected dispersion with diffusion)', fontsize=13)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=u_d_min, vmax=u_d_max))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Drift velocity u', fontsize=11)

# Legend with better tolerance for float comparison
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("linear_instability_increment.pdf", dpi=160)
plt.savefig("linear_instability_increment.png", dpi=160)
plt.show()