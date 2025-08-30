import numpy as np
import matplotlib.pyplot as plt

U0 = 0.5
eta_p = 0.01
eta_n = 0.01
n = 1
w = 0.5
gamma0 = 10
N = 200
kmin = -8
kmax = 8
Lambda = -(1 / w + 1 / n) * np.exp(-n / w)
fig, ax = plt.subplots()

u_values = [0.6, 0.5, 0.4, 0.3]

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

ax.grid(True)
ax.set_xlabel('Wavenumber k')
ax.set_ylabel('Instability increment')
ax.set_ylim([-3, 1])
ax.set_xlim([-10, 10])
ax.legend(title='Drift velocity (u)')
ax.set_title('Instability increments vs Wavenumber')

plt.tight_layout(); plt.savefig(f"linear_instability_increment.png", dpi=160); plt.close()