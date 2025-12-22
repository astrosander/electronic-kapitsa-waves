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
eta_p = 0#0.1
eta_n = 0#0.1#*0.01
# n = 0.05#0.2
# w = 0.1
n = 0.2
w = 0.2#0.4
gamma0 = 3.0
m = 1.0
kmin = -100
kmax = 100
N = 20000

fig, ax = plt.subplots(figsize=(10, 6))

L = 10.0  # Physical box size

print("Calculating u_star...")
u_star = u_star_for()

if u_star is not None:
    u_d_min = u_star - 0.005
    u_d_max = u_star + 0.005
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
    # Default range if u_star is not found
    u_d_min = 2.5
    u_d_max = 3.0

print(f"Physical box size L = {L}")
print(f"Fundamental mode spacing: 2π/L = {2*np.pi/L:.4f}\n")

# Select 4 representative u values for plotting
if u_star is not None:
    # Select 4 u values: one below u_star, u_star, and two above
    # u_plot_values = [
    #     u_star - 0.003,
    #     u_star - 0.001,
    #     u_star,
    #     u_star + 0.002
    # ]
    u_plot_values = [1,2,3,4]
else:
    # If u_star not found, use evenly spaced values in the range
    u_plot_values = np.linspace(u_d_min, u_d_max, 4)

u_plot_values = sorted(u_plot_values)
print(f"Plotting for 4 u values: {[f'{u:.4f}' for u in u_plot_values]}")

# Define colors for the 4 u values
colors = ['#440154', '#31688e', '#35b779', '#fde725']  # viridis colors
linestyles_plus = ['-', '-', '-', '-']  # solid for omega_plus
linestyles_minus = ['--', '--', '--', '--']  # dashed for omega_minus

for idx, u in enumerate(u_plot_values):
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
    
    # Growth rates ζ(k) = Im(ω±)
    zeta_plus = np.imag(omega_plus)
    zeta_minus = np.imag(omega_minus)
    
    # Get color for this u value
    color = colors[idx % len(colors)]
    
    # Determine label and line width
    if u_star is not None and np.isclose(u, u_star, atol=1e-5):
        lw = 2.5
        u_label = f'$u^{{\\bigstar}} = {u:.4f}$'
    else:
        lw = 1.5
        u_label = f'$u = {u:.4f}$'
    
    # Plot omega_plus (solid line)
    ax.plot(k_out, zeta_plus, linewidth=lw, color=color, 
            linestyle=linestyles_plus[idx])
    
    # Plot omega_minus (dashed line)
    ax.plot(k_out, zeta_minus, linewidth=lw, color=color, 
            linestyle=linestyles_minus[idx])
    
    # Print summary for this u value
    print(f"u = {u:.4f}:")
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

ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.3)
ax.grid(True, alpha=0.3)
ax.set_xlabel('$k$', fontsize=12)
ax.set_ylabel('$\\zeta(k) = \\Im(\\omega_{\\pm})$', fontsize=12)
# ax.set_title('Linear Instability Increment (corrected dispersion with diffusion)', fontsize=13)

# Legend with better tolerance for float comparison
ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)

plt.tight_layout()
plt.savefig("linear_instability_increment.pdf", dpi=160)
plt.savefig("linear_instability_increment.png", dpi=160)
plt.show()