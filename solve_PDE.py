import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass

@dataclass
class Params:
    m: float = 1.0
    U: float = 1.0
    n0: float = 1.0
    u0: float = 0.20
    Gamma0: float = 0.10
    w: float = 1.0
    e_charge: float = 1.0
    E0: float = 0.0
    epsilon: float = 5.0
    include_poisson: bool = True
    L: float = 100.0
    Nx: int = 128*8          # reduced grid for speed
    t_final: float = 50.0  # shorter horizon for demo
    n_save: int = 160
    rtol: float = 1e-6
    atol: float = 1e-8
    n_floor: float = 1e-6
    amp_n: float = 5e-3
    amp_u: float = 0.0
    mode: int = 3
    isPreview: bool = True

P = Params()

def spectral_ops(Nx, L):
    x = np.linspace(0, L, Nx, endpoint=False)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ik = 1j * k
    inv_k2 = np.zeros_like(k, dtype=np.complex128)
    nonzero = k != 0.0
    inv_k2[nonzero] = 1.0 / (k[nonzero] ** 2)
    return x, dx, k, ik, inv_k2

x, dx, k, ik, inv_k2 = spectral_ops(P.Nx, P.L)

def Dx(f):
    return np.fft.ifft(ik * np.fft.fft(f)).real

def mu_of_n(n, U):
    return 0.5 * U * n**2

def Pi_of_np(n, p, m, U, n_floor):
    n_eff = np.maximum(n, n_floor)
    return mu_of_n(n_eff, U) + (p**2) / (m * n_eff)

def Gamma_of_n(n, Gamma0, w):
    return Gamma0 * np.exp(-n / w)

def phi_from_n(n, n0, epsilon, inv_k2):
    rhs = (n - n0) / epsilon
    rhs_hat = np.fft.fft(rhs)
    phi_hat = inv_k2 * rhs_hat
    phi_hat[0] = 0.0
    return np.fft.ifft(phi_hat).real

def rhs_full(t, y):
    N = P.Nx
    n = y[:N]
    p = y[N:]
    n_eff = np.maximum(n, P.n_floor)
    v = p / (P.m * n_eff)
    dn_dt = -Dx(n * v)
    Pi = Pi_of_np(n, p, P.m, P.U, P.n_floor)
    if P.include_poisson:
        phi = phi_from_n(n, P.n0, P.epsilon, inv_k2)
        Ex = P.E0 - Dx(phi)
    else:
        Ex = P.E0 * np.ones_like(n)
    dp_dt = -Gamma_of_n(n, P.Gamma0, P.w) * p - Dx(Pi) + P.e_charge * n * Ex
    return np.concatenate([dn_dt, dp_dt])

n = P.n0 * np.ones(P.Nx)
u = P.u0 * np.ones(P.Nx)
if P.mode is not None and P.amp_n != 0.0:
    kx = 2*np.pi*P.mode / P.L
    n += P.amp_n * np.cos(kx * x)
if P.mode is not None and P.amp_u != 0.0:
    kx = 2*np.pi*P.mode / P.L
    u += P.amp_u * np.cos(kx * x)
p = P.m * n * u
y0 = np.concatenate([n, p])
t_eval = np.linspace(0.0, P.t_final, P.n_save)

sol = solve_ivp(rhs_full, (0.0, P.t_final), y0, t_eval=t_eval,
                method="RK45", rtol=P.rtol, atol=P.atol)

N = P.Nx
n_t = sol.y[:N, :]
p_t = sol.y[N:, :]

plt.figure(figsize=(8, 4.5))
extent = [x.min(), x.max(), sol.t.min(), sol.t.max()]
plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent)
plt.xlabel("x"); plt.ylabel("t"); plt.title("n(x,t)")
plt.colorbar(label="n"); plt.tight_layout(); plt.savefig("img/density_profile.png", bbox_inches='tight'); plt.savefig("img/density_profile.pdf", bbox_inches='tight')
if not P.isPreview:
    plt.show()

plt.figure(figsize=(8, 4.5))
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    idx = int(frac * (len(sol.t) - 1))
    plt.plot(x, n_t[:, idx], label=f"t={sol.t[idx]:.1f}")
plt.xlabel("x"); plt.ylabel("n(x,t)"); plt.title("Density at different times")
plt.legend(); plt.tight_layout(); plt.savefig("img/density_snapshots.png", bbox_inches='tight'); plt.savefig("img/density_snapshots.pdf", bbox_inches='tight')
if not P.isPreview:
    plt.show()

total_mass = n_t.sum(axis=0) * dx
plt.figure(figsize=(8, 4.5))
plt.plot(sol.t, total_mass)
plt.xlabel("t"); plt.ylabel("∫ n dx"); plt.title("Mass conservation")
plt.tight_layout(); plt.savefig("img/mass_conservation.png", bbox_inches='tight'); plt.savefig("img/mass_conservation.pdf", bbox_inches='tight')
if not P.isPreview:
    plt.show()

mean_u = (p_t / (P.m * np.maximum(n_t, P.n_floor))).mean(axis=0)
plt.figure(figsize=(8, 4.5))
plt.plot(sol.t, mean_u)
plt.xlabel("t"); plt.ylabel("⟨u⟩"); plt.title("Mean velocity vs time")
plt.tight_layout(); plt.savefig("img/mean_velocity.png", bbox_inches='tight'); plt.savefig("img/mean_velocity.pdf", bbox_inches='tight')
if not P.isPreview:
    plt.show()
