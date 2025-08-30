import numpy as np
import matplotlib.pyplot as plt

U0 = 1.0
eta_p = 5e-3
eta_n = 5e-3
n = 0.2
w = 5.0
gamma0 = 2.50
N = 512
kmin = -20
kmax = 20
Lambda = -(1 / w + 1 / n) * np.exp(-n / w)
fig, ax = plt.subplots()

u_values = [20.0]

for u in u_values:
    k_out = np.linspace(kmin, kmax, N)
    omega1 = np.zeros(N, dtype=complex)
    omega2 = np.zeros(N, dtype=complex)
    
    for i1 in range(N):
        k = k_out[i1]
        
        gamma = gamma0 * np.exp(-n / w)
        p = n * u
        Pn = n * U0 - p**2 / n**2
        Pp = 2 * p / n
        Gamma_n = -(gamma / w)
        Lambda = (Gamma_n - gamma / n) * p
        
        Delta = (gamma + 1j * k * Pp) ** 2 + 4 * 1j * k * Lambda - 4 * k**2 * Pn
        
        omega1[i1] = (-1j * gamma + k * Pp + 1j * np.sqrt(Delta)) / 2 - 1j * eta_p * k**2
        omega2[i1] = (-1j * gamma + k * Pp - 1j * np.sqrt(Delta)) / 2 - 1j * eta_p * k**2

    ax.plot(k_out, np.imag(omega1), label=f'u = {u}', linewidth=1.0)
    ax.plot(k_out, np.imag(omega2), linewidth=1.0)


mask = k_out >= 0
k_right = k_out[mask]
im1_right = np.imag(omega1)[mask]
k_max_right = k_right[np.argmax(im1_right)]
ax.axvline(k_max_right, color="blue", linestyle="--", linewidth=1.2,
           label=f"max at k={k_max_right:.2f}")

L = 10.0
k_line = 6 * np.pi / L
ax.axvline(k_line, color="red", linestyle="--", linewidth=1.2, label=f"$k = 6\\pi/{L:.0f}$")

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

plt.tight_layout(); plt.savefig(f"linear_instability_increment.png", dpi=160); plt.show()#plt.close()