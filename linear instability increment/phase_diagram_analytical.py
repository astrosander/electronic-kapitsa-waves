import numpy as np
import matplotlib.pyplot as plt

hbar = 1.054571817e-34
m_e  = 9.1093837e-31
pi   = np.pi
cm2_to_m2 = 1e4

g = 4
m = 0.04 * m_e

n_min_cm2, n_max_cm2 = 0.5e10, 5.0e10
n_vals_cm2 = np.linspace(n_min_cm2, n_max_cm2, 800)
n_vals = n_vals_cm2 * cm2_to_m2

w_cm2 = 3.0e10
w = w_cm2 * cm2_to_m2

nu = g * m / (2 * pi * hbar**2)
U_factor = 1.0
U  = U_factor * (2.0 / nu)

u_c = w * np.sqrt(U / (m * n_vals))

kF  = np.sqrt(4 * pi * n_vals / g)
v_F = (hbar * kF) / m

idx_star = np.argmin(np.abs(u_c - v_F))
n_star_cm2 = n_vals_cm2[idx_star]
u_star = 0.5 * (u_c[idx_star] + v_F[idx_star])

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(u_c, n_vals_cm2, 'k-',  lw=2, label='$u_c(n)$  (instability boundary)')
ax.plot(v_F, n_vals_cm2, 'b--', lw=2, label='$v_F(n)$  (kinetic ceiling)')

ax.fill_betweenx(n_vals_cm2, 0, u_c, color='green', alpha=0.18, label='stable: $u < u_c$')
ax.fill_betweenx(n_vals_cm2, u_c, max(u_c.max(), v_F.max())*1.05, color='red', alpha=0.10, label='$u > u_c$')
u_right = np.minimum(v_F, max(u_c.max(), v_F.max())*1.05)
ax.fill_betweenx(n_vals_cm2, u_c, u_right, where=(v_F > u_c), color='orange', alpha=0.25, label='reachable instability: $u_c < u < v_F$')

ax.plot([u_star], [n_star_cm2], 'ko', ms=5)
n_star_display = n_star_cm2 / 1e10
ax.annotate(f'$u_c = v_F$ at $n = {n_star_display:.1f} \\times 10^{{10}}$ cm$^{{-2}}$',
            xy=(u_star, n_star_cm2), xytext=(0.55, 0.8), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', lw=1), fontsize=10)

ax.set_xlim(0, max(u_c.max(), v_F.max()) * 1.05)
ax.set_ylim(n_min_cm2, n_max_cm2)
ax.set_xlabel('$u$ (m/s)', fontsize=13)
ax.set_ylabel('$n$ (cm$^{-2}$)', fontsize=13)
ax.grid(True, alpha=0.3, linewidth=0.6)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('blg_phase_diagram.pdf', dpi=300, bbox_inches='tight')
plt.savefig('blg_phase_diagram.png', dpi=300, bbox_inches='tight')
plt.show()

n_spec_cm2 = 2.0e10
n_spec = n_spec_cm2 * cm2_to_m2
u_spec = w * np.sqrt(U / (m * n_spec))
vF_spec = (hbar/m) * np.sqrt(4*pi*n_spec/g)
print(f"w = {w_cm2:.2e} cm^-2 | u_c(2e10 cm^-2) = {u_spec:.3e} m/s | v_F(2e10 cm^-2) = {vF_spec:.3e} m/s")
