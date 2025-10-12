import numpy as np
import matplotlib.pyplot as plt

# --- constants (SI) ---
hbar = 1.054571817e-34   # J·s
m_e  = 9.1093837e-31     # kg

# --- model / material params ---
g = 4                    # degeneracy
m = 0.04*m_e            # effective mass (change per your system)
cm2_to_m2 = 1e4

# densities entered in cm^-2 then converted to m^-2
n_min_cm2, n_max_cm2 = 0.5e10, 5.0e10
w_cm2 = 3.0e10
n_vals = np.linspace(n_min_cm2, n_max_cm2, 800)*cm2_to_m2
w = w_cm2*cm2_to_m2

# U ≈ 2/nu for TF-screened, k≈0 constant interaction
nu = g*m/(2*np.pi*hbar**2)   # 1/(J·m^2)
U  = 2.0/nu                  # J·m^2

# Eq. (32) with Gamma ∝ e^{-n/w}  -> u_c = w*sqrt(U/(m*n))
u_c = w*np.sqrt(U/(m*n_vals))  # m/s

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(u_c, n_vals/cm2_to_m2, 'k-', linewidth=2)
ax.fill_betweenx(n_vals/cm2_to_m2, 0, u_c, color='green', alpha=0.2, label='Stable')
ax.fill_betweenx(n_vals/cm2_to_m2, u_c, u_c.max()*1.1, color='red', alpha=0.2, label='Unstable')
ax.set_xlabel(r'$u_d$ (m/s)', fontsize=13)
ax.set_ylabel(r'$\bar{n}$ (cm$^{-2}$)', fontsize=13)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=0.6)
plt.tight_layout()
plt.savefig('phase_diagram_analytical.pdf', dpi=300, bbox_inches='tight')
plt.savefig('phase_diagram_analytical.png', dpi=300, bbox_inches='tight')
plt.show()
