import numpy as np
import matplotlib.pyplot as plt

U0 = 0.5
eta_p = 0.2*2000
eta_n = 0.2*2000
n = 1.0
w = 0.5
gamma0 = 2.5
kmin = -0.02
kmax = 0.02
N = 20000
Lambda = -(1 / w + 1 / n) * np.exp(-n / w)
fig, ax = plt.subplots(figsize=(10, 8))

u_star = 0.37671861

u_values = np.arange(2.0, 50, 2.0)#np.append(np.arange(0.4, 0.46, 0.01), u_star)

L = 200*3.1415926
k_line = 2 * np.pi / L

for u in u_values:
    k_out = np.linspace(kmin, kmax, N)
    omega1 = np.zeros(N, dtype=complex)
    omega2 = np.zeros(N, dtype=complex)
    
    for i1, k in enumerate(k_out):
        gamma = gamma0 * np.exp(-n / w)
        p = n * u
        Pn = n * U0 - p**2 / n**2
        Pp = 2 * p / n
        Gamma_n = -(gamma / w)
        Lambda = (Gamma_n - gamma / n) * p

        Delta = (gamma + 1j * k * Pp) ** 2 + 4j * k * Lambda - 4 * k**2 * Pn

        omega1[i1] = (-1j * gamma + k * Pp + 1j * np.sqrt(Delta)) / 2 - 1j * eta_p * k**2
        omega2[i1] = (-1j * gamma + k * Pp - 1j * np.sqrt(Delta)) / 2 - 1j * eta_p * k**2

    # Highlight u_star
    lw = 2.5 if np.isclose(u, u_star) else 1.0
    label = f'$u_\\ast = {u:.2f}$' if np.isclose(u, u_star) else f'u = {u:.2f}'
    color = 'k' if np.isclose(u, u_star) else None
    ax.plot(k_out, np.imag(omega1), label=label, linewidth=lw, color=color)
    
    k_intersect = k_line#0.23561944901923448
    omega1_intersect = np.interp(k_intersect, k_out, np.imag(omega1))
    ax.plot(k_intersect, omega1_intersect, 'ro', markersize=3, alpha=0.7)
    
    # ax.plot(k_out, np.imag(omega2), linewidth=lw, color=color)
    
    # Find and plot maximum for k > 0
    mask = k_out > 0
    if np.any(mask):
        k_right = k_out[mask]
        im1_right = np.imag(omega1)[mask]
        max_idx = np.argmax(im1_right)
        k_max_right = k_right[max_idx]
        im1_max_right = im1_right[max_idx]
        
        # Plot dark small dot at maximum
        ax.plot(k_max_right, im1_max_right, 'ko', markersize=2, alpha=0.8)

    if np.isclose(u, u_star):
        mid_idx = len(k_out) // 3  # pick a midpoint for annotation
        ax.annotate(
            f'$u_\\ast = {u_star:.2f}$',
            xy=(k_out[mid_idx], np.imag(omega1[mid_idx])),   # point on the curve
            xytext=(k_out[mid_idx]*1.2, np.imag(omega1[mid_idx]) - 0.02),  # shifted down
            arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
            fontsize=10,
            color='black'
        )

ax.axvline(k_line, color="red", linestyle="--", linewidth=1.2, label=f"$k = 6\\pi/{L:.0f}$")
ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
print(k_line)

# k_line = 20 * np.pi / L
# ax.axvline(k_line, color="purple", linestyle="--", linewidth=1.2, label=f"$k = 6\\pi/{L:.0f}$")

# k_line = 12 * np.pi / L
# ax.axvline(k_line, color="green", linestyle="--", linewidth=1.2, label=f"$k = 6\\pi/{L:.0f}$")

ax.grid(True)
ax.set_xlabel('Wavenumber k')
ax.set_ylabel('Instability increment')
# ax.set_ylim([-5, 5])
# ax.set_xlim([-20, 20])
ax.legend(title='Drift velocity (u)')
ax.set_title('Instability increments vs Wavenumber')

plt.tight_layout(); plt.savefig(f"linear_instability_increment.pdf", dpi=160); plt.savefig(f"linear_instability_increment.png", dpi=160); plt.show()#plt.close()