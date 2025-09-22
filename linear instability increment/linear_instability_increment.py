import numpy as np
import matplotlib.pyplot as plt

U0 = 1.0
eta_p = 0.1
eta_n = 0.5
n = 0.2
w = 5.0
gamma0 = 2.5
kmin = -3.5
kmax = 3.5
N = 20000
Lambda = -(1 / w + 1 / n) * np.exp(-n / w)
fig, ax = plt.subplots()

L = 10.0#314.15936/1.5

u_star = 0.37671861

u_values = [20]#np.arange(0.40, 0.41, 0.1)#np.append(np.arange(0.4, 0.46, 0.01), u_star)

print(u_values)
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
    
    for i in range(1, 11):
        k_intersect = i * 2 * np.pi / L
        if kmin <= k_intersect <= kmax:
            omega1_intersect = np.interp(k_intersect, k_out, np.imag(omega1))
            ax.plot(k_intersect, omega1_intersect, 'ro', markersize=2, alpha=0.8)
    
    # ax.plot(k_out, np.imag(omega2), linewidth=lw, color=color)

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
mask = k_out >= 0
k_right = k_out[mask]
im1_right = np.imag(omega1)[mask]
k_max_right = k_right[np.argmax(im1_right)]
ax.axvline(k_max_right, color="blue", linestyle="--", linewidth=1.2,
           label=f"max at k={k_max_right:.3f}")

print(L)

for i in range(1, 6):
    k_line = i * 2 * np.pi / L
    ax.axvline(k_line, color="red", linestyle="--", linewidth=0.5, alpha=0.8, label=f"$k = {i}\\pi/{L:.0f}$" if i == 1 else "")
    print(k_line)

k_line = 2 * np.pi / L
# ax.axvline(k_line, color="red", linestyle="--", linewidth=1.2, label=f"$k = 2\\pi/{L:.0f}$")
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