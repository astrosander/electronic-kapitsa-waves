import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp
import os

# -------------------------- Parameters --------------------------
@dataclass
class Par:
    m: float = 1.0
    e: float = 1.0
    n0: float = 1.0
    U: float = 1.0                   # Π0'(n)=U n  -> c_s = sqrt(U n0 / m)
    Gamma0: float = 0.08
    w: float = 1.0
    epsilon: float = 15.0
    # background (fixed ionic profile used in Poisson)
    n1: float = 0.02
    lam: float = 0.005 / 4
    x0: float = 0.55                 # as fraction of L
    # numerics
    L: float = 100.0
    Nx: int = 256
    t_final: float = 160.0
    n_save: int = 220
    rtol: float = 1e-7
    atol: float = 1e-9
    n_floor: float = 1e-12
    # output
    save_frames: bool = True
    frames_dir: str = "frames_nbar"
    cmap: str = "inferno"

# >>> choose your target drift here
u_target = 0.0   # try 0.0, 0.1, 0.2, ...

par = Par()

# ----------------------- Grid & spectral ops ---------------------
x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k  = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j*k
inv_k2 = np.zeros_like(k, dtype=np.complex128)
nz = (k != 0); inv_k2[nz] = 1.0/(k[nz]**2)

def Dx(f): return (ifft(ik*fft(f))).real

# -------------------- Material and helpers ----------------------
def Gamma(n): return par.Gamma0 * np.exp(-n/par.w)
def Pi0(n):   return 0.5 * par.U * n**2

def periodic_delta(x, x0, L):
    return (x - x0 + 0.5*L) % L - 0.5*L

def nbar_profile():
    xc = par.x0 * par.L
    d = periodic_delta(x, xc, par.L)
    return par.n0 + par.n1 * np.exp(-0.5 * par.lam * d**2)

# Poisson with background: φ'' = -(e/ε) (n - nbar)
def phi_from_n_with_bg(n, nbar):
    rhs_hat = fft((par.e/par.epsilon) * (n - nbar))
    phi_hat = inv_k2 * rhs_hat
    phi_hat[0] = 0.0
    return ifft(phi_hat).real

# -------------------- Equilibrium E(x) for given u ---------------------
def E_equilibrium_profile(nbar, u):
    # From steady momentum balance with phi_x = 0 and n = nbar:
    # Γ(nbar) m nbar u + ∂x[ 0.5 U nbar^2 + m nbar u^2 ] = e nbar E(x)
    # => E(x) = (m/e) Γ(nbar) u + ((U + m u^2)/(e nbar)) * ∂x nbar
    dnb = Dx(nbar)
    return (par.m/par.e) * Gamma(nbar) * u + ((par.U + par.m*u*u)/(par.e*nbar)) * dnb

# ----------------------- Initial fields --------------------------
def initial_fields(u_star):
    nbar = nbar_profile()
    # Start exactly on the static branch: n = nbar, p = m nbar u_star
    n0 = np.maximum(nbar.copy(), par.n_floor)
    p0 = par.m * n0 * u_star
    # Spatially varying E to keep (nbar, u_star) steady when phi_x=0
    E_x = E_equilibrium_profile(nbar, u_star)
    return n0, p0, nbar, E_x

# ----------------------- PDE RHS (full) --------------------------
# (∂t + Γ(n)) p + ∂x Π = e n E(x) - e n ∂x φ ; Π = Π0(n) + p^2/(m n)
def rhs_pde(t, y, E_x, nbar):
    N = par.Nx
    n = y[:N]; p = y[N:]
    n_eff = np.maximum(n, par.n_floor)
    v = p/(par.m*n_eff)

    # continuity
    dn_dt = -Dx(n_eff * v)

    # momentum flux
    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)

    # fields
    phi  = phi_from_n_with_bg(n, nbar)
    Ex   = E_x - Dx(phi)     # E(x) profile minus induced field

    # momentum
    dp_dt = -Gamma(n_eff)*p - Dx(Pi) + par.e*n_eff*Ex

    return np.concatenate([dn_dt, dp_dt])

# ----------------------- Run one experiment ---------------------
def run_once(u_star):
    n0, p0, nbar, E_x = initial_fields(u_star)
    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)

    # sanity: residual of steady momentum balance at t=0 (should be ~0)
    Pi0_now = Pi0(n0) + (p0**2)/(par.m*np.maximum(n0,par.n_floor))
    R0 = Gamma(n0)*p0 + Dx(Pi0_now) - par.e*n0*E_x
    print(f"[check t=0] max|residual| ≈ {np.max(np.abs(R0)):.2e}")

    sol = solve_ivp(lambda t,y: rhs_pde(t,y,E_x,nbar),
                    (0.0, par.t_final), y0, t_eval=t_eval,
                    method="BDF", rtol=par.rtol, atol=par.atol)

    N = par.Nx
    n_t = sol.y[:N, :]
    p_t = sol.y[N:, :]
    u_t = p_t/(par.m*np.maximum(n_t, par.n_floor))
    return sol.t, n_t, p_t, u_t, nbar, E_x

# ----------------------- Diagnostics & plots --------------------
def spacetime_dn(t, n_t, nbar, title=r"$\delta n(x,t)=n-\bar n$"):
    dn = n_t - nbar[:, None]            # <-- true δn
    extent = [x.min(), x.max(), t.min(), t.max()]
    plt.figure(figsize=(10,4.6))
    plt.imshow(dn.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    x0 = par.x0 * par.L
    plt.plot([x0, x0], [t.min(), t.max()], 'w--', lw=1.3, alpha=0.8, label="defect center")
    plt.colorbar(label=r"$\delta n$")
    plt.xlabel("x"); plt.ylabel("t"); plt.title(title)
    plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig("spacetime_dn.png", dpi=160); plt.close()

def save_animation_frames(t, n_t, p_t, nbar, u_star):
    os.makedirs(par.frames_dir, exist_ok=True)
    for j in range(len(t)):
        n = n_t[:, j]; p = p_t[:, j]
        n_eff = np.maximum(n, par.n_floor)
        Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)
        # residual w.r.t. *equilibrium* field only (phi excluded)
        R = Gamma(n_eff)*p + Dx(Pi)  # - e n E_eq omitted on purpose here

        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axs[0].plot(x, n - nbar, lw=2, label=r"$\delta n$")
        axs[0].axhline(0, ls='--', lw=1)
        axs[0].set_ylabel(r"$\delta n$")
        axs[0].set_title(f"u_target={u_star:.3f}  |  t={t[j]:.2f}")

        axs[1].plot(x, p/(par.m*np.maximum(n, par.n_floor)), lw=2, label="u(x,t)")
        axs[1].axhline(u_star, ls='--', lw=1.4, label="target u")
        axs[1].set_ylabel("u = p/(m n)")
        axs[1].legend(loc="best", fontsize=9)

        axs[2].plot(x, R, lw=1.8)
        axs[2].set_ylabel("residual (no φ)")
        axs[2].set_xlabel("x")

        plt.tight_layout()
        fname = f"{par.frames_dir}/frame_{j:04d}.png"
        plt.savefig(fname, dpi=140); plt.close()

def norms(a): return np.sqrt(np.mean(a*a)), np.max(np.abs(a))

# ---------------------------- Run -------------------------------
t, n_t, p_t, u_t, nbar, E_x = run_once(u_target)

# Space–time of δn (fixed)
spacetime_dn(t, n_t, nbar)

if par.save_frames:
    save_animation_frames(t, n_t, p_t, nbar, u_target)

# Report achieved drift
u_mean_0 = u_t[:, 0].mean()
u_mean_T = u_t[:, -1].mean()
print(f"<u>(t=0)  ≈ {u_mean_0:.4f}")
print(f"<u>(t_end)≈ {u_mean_T:.4f}  (target {u_target:.4f})")

# Extra diagnostics
dn0 = n_t[:, 0] - nbar
dnT = n_t[:, -1] - nbar
l2_0, li_0 = norms(dn0)
l2_T, li_T = norms(dnT)
print(f"||δn||_L2:  t=0 -> {l2_0:.3e},   t_end -> {l2_T:.3e}")
print(f"||δn||_L∞:  t=0 -> {li_0:.3e},  t_end -> {li_T:.3e}")
