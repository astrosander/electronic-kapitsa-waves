import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass

@dataclass
class Params:
    m: float = 1.0
    U: float = 0.04          # c_s = sqrt(U*n0/m) ~ 0.2 при n0=1
    n0: float = 1.0
    u_target: float = 0.20   # desired mean velocity
    Gamma0: float = 0.10
    w: float = 1.0
    e_charge: float = 1.0
    E0: float = 0.0
    epsilon: float = 20.0    # larger -> weaker Φ contribution
    include_poisson: bool = True
    
    # Flow control feedback (for stabilizing mean flow)
    feedback_Kp: float = 0.3  # proportional feedback gain
    
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

# Base field for maintaining mean flow
Gamma_bg = Gamma_of_n(P.n0, P.Gamma0, P.w)
E_base = Gamma_bg * P.m * P.u_target / P.e_charge

def rhs_full(t, y):
    N = P.Nx
    n = y[:N]
    p = y[N:]
    n_eff = np.maximum(n, P.n_floor)
    v = p / (P.m * n_eff)
    
    # Mean velocity for feedback control
    mean_u = v.mean()
    E = E_base + P.feedback_Kp * (P.u_target - mean_u)  # uniform field with feedback
    
    dn_dt = -Dx(n * v)
    Pi = Pi_of_np(n, p, P.m, P.U, P.n_floor)
    if P.include_poisson:
        phi = phi_from_n(n, P.n0, P.epsilon, inv_k2)
        Ex = E - Dx(phi)
    else:
        Ex = E * np.ones_like(n)
    dp_dt = -Gamma_of_n(n, P.Gamma0, P.w) * p - Dx(Pi) + P.e_charge * n * Ex
    return np.concatenate([dn_dt, dp_dt])

n = P.n0 * np.ones(P.Nx)
u = P.u_target * np.ones(P.Nx)
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
v_t = p_t / (P.m * np.maximum(n_t, P.n_floor))
mean_u = v_t.mean(axis=0)

# Characteristic lines for wave propagation analysis
c_s = np.sqrt(P.U * P.n0 / P.m)
u_plus, u_minus = P.u_target + c_s, P.u_target - c_s

plt.figure(figsize=(8, 4.5))
extent = [x.min(), x.max(), sol.t.min(), sol.t.max()]
plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent)
# Characteristic lines x = x0 + (u±c_s)t
x0 = P.L / 4
for uu in [P.u_target, u_plus, u_minus]:
    t_line = np.array([sol.t.min(), sol.t.max()])
    x_line = (x0 + uu * t_line) % P.L
    plt.plot(x_line, t_line, "--", alpha=0.7)
plt.xlabel("x"); plt.ylabel("t"); plt.title("n(x,t) with characteristic lines")
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

plt.figure(figsize=(8, 4.5))
plt.plot(sol.t, mean_u); plt.axhline(P.u_target, ls=":", alpha=0.7)
plt.xlabel("t"); plt.ylabel("⟨u⟩"); plt.title("Mean velocity vs time (should stay ~ constant)")
plt.tight_layout(); plt.savefig("img/mean_velocity.png", bbox_inches='tight'); plt.savefig("img/mean_velocity.pdf", bbox_inches='tight')
if not P.isPreview:
    plt.show()
