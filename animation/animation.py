# PDE + Poisson traveling-wave model
# Parameter sweep -> one labeled PNG per case (0001.png, 0002.png, ...)
# Use ffmpeg later, e.g.:  ffmpeg -framerate 4 -i %04d.png -pix_fmt yuv420p out.mp4

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.fft import fft, ifft, fftfreq
from dataclasses import dataclass

# ---------------- Parameters ----------------
@dataclass
class P:
    # Physical
    m: float = 1.0
    e: float = 1.0
    U: float = 0.04         # barotropic EOS slope (mu' = U n)
    n0: float = 1.0
    Gamma0: float = 1.0
    w: float = 1.0
    epsilon: float = 15.0   # dielectric
    E: float = 0.035
    # Numerics (PDE)
    L: float = 10.0
    Nx: int = 384
    t_final: float = 10.0
    n_save: int = 240
    rtol: float = 1e-6
    atol: float = 1e-8
    n_floor: float = 1e-6
    # Initial perturbation: wave train
    amp_n: float = 0.004    # δ in δ n0 cos(kx)
    mode: int = 2           # k = 2π * mode / L
    # Localized background bump: n̄(x) = n0 + n1 * exp(-λ (x - x0)^2 / 2)
    n1: float = 0.02
    lam: float = 0.6
    x0: float = 6.0
    wave_phase: float = 0.0
    # Small feedback to keep <u> near c_pred
    Kp: float = 0.15

par = P()

# ---------------- Grid / spectral utilities ----------------
x = np.linspace(0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j * k
inv_k2 = np.zeros_like(k, dtype=np.complex128)
nz = k != 0
inv_k2[nz] = 1.0 / (k[nz]**2)

def Dx(f):
    return (ifft(ik * fft(f))).real

def phi_from_n(n):
    rhs_hat = fft((n - par.n0) / par.epsilon)
    phi_hat = inv_k2 * rhs_hat
    phi_hat[0] = 0.0
    return (ifft(phi_hat)).real

def Gamma(n):
    return par.Gamma0 * np.exp(-n / par.w)

def current_c_pred():
    # c_pred = e E / (m Γ(n0))
    return par.e * par.E / (par.m * Gamma(par.n0))

def set_U_and_u(U_value, u_target, use_feedback=False):
    """
    Choose U and E so that predicted speed c_pred equals u_target.
    Optionally disable feedback for an exact open-loop match during the run.
    """
    par.U = U_value
    par.Kp = (par.Kp if use_feedback else 0.0)  # 0 => no feedback
    Gamma_n0 = Gamma(par.n0)
    par.E = par.m * Gamma_n0 * u_target / par.e
    print(f"[set_U_and_u] U={par.U:.6f}, E={par.E:.6f}, c_pred≈{current_c_pred():.6f} (target={u_target:.6f})")

def periodic_delta(x, x0, L):
    # shortest signed distance on a periodic domain of length L
    return (x - x0 + 0.5*L) % L - 0.5*L

def init_fields_with_u(u_target):
    """
    Initial condition: wave train on top of a localized bump.
       n(x,0) = n̄(x) + δ n0 cos(k x + φ),   p = m n u_target
    """
    # localized background
    if par.n1 != 0.0:
        d = periodic_delta(x, par.x0, par.L)
        n_bar = par.n0 + par.n1 * np.exp(-0.5 * par.lam * d**2)
    else:
        n_bar = par.n0 * np.ones(par.Nx)

    # wave train
    n_wave = 0.0
    if par.amp_n != 0.0 and par.mode != 0:
        kx = 2.0 * np.pi * par.mode / par.L
        n_wave = (par.amp_n * par.n0) * np.cos(kx * x + par.wave_phase)

    n_init = np.maximum(n_bar + n_wave, par.n_floor)
    p_init = par.m * n_init * u_target
    return n_init, p_init

# ---------------- Short open-loop calibration ----------------
def measure_mean_speed(n_t, p_t):
    n_eff = np.maximum(n_t, par.n_floor)
    v_t = p_t / (par.m * n_eff)
    return v_t.mean(axis=0)[-1]

def rhs_pde_for_calibration(t, y):
    # identical physics as main rhs_pde, but no feedback (open-loop)
    N = par.Nx
    n = y[:N]; p = y[N:]
    n_eff = np.maximum(n, par.n_floor)
    v = p/(par.m*n_eff)
    dn_dt = -Dx(n_eff*v)
    Pi = 0.5 * par.U * n_eff**2 + (p**2) / (par.m * n_eff)
    phi = phi_from_n(n)
    Ex = par.E - Dx(phi)
    dp_dt = -Gamma(n_eff)*p - Dx(Pi) + par.e*n_eff*Ex
    return np.concatenate([dn_dt, dp_dt])

def calibrate_E_to_speed(u_target, t_short=6.0, iters=4, tol=1e-3):
    """
    Iteratively retune E so that mean speed ≈ u_target after a short warm-up run.
    Keeps U fixed. Uses open-loop during calibration.
    """
    Gamma_n0 = Gamma(par.n0)
    for k_iter in range(iters):
        n0, p0 = init_fields_with_u(u_target)
        y0 = np.concatenate([n0, p0])
        sol = solve_ivp(rhs_pde_for_calibration, (0.0, t_short), y0,
                        t_eval=np.linspace(0.0, t_short, 50),
                        method="BDF", rtol=par.rtol, atol=par.atol)
        N = par.Nx
        n_t = sol.y[:N, :]
        p_t = sol.y[N:, :]
        u_meas = measure_mean_speed(n_t, p_t)
        err = u_target - u_meas
        print(f"[cal] iter {k_iter}: E={par.E:.6f}, u_meas={u_meas:.6f}, err={err:.3e}")
        if abs(err) <= tol: break
        slope = par.e / (par.m * Gamma_n0)   # du/dE
        par.E += err / slope
    print(f"[cal] final E={par.E:.6f}, c_pred≈{current_c_pred():.6f}")

# ---------------- Full PDE (with gentle feedback) ----------------
def rhs_pde(t, y):
    N = par.Nx
    n = y[:N]
    p = y[N:]
    n_eff = np.maximum(n, par.n_floor)
    v = p / (par.m * n_eff)

    # Gentle feedback to keep <u> near c_pred
    mean_u = v.mean()
    c_pred = current_c_pred()
    E_eff = par.E + par.Kp * (c_pred - mean_u)

    # Continuity
    dn_dt = -Dx(n * v)

    # Momentum flux Π
    Pi = 0.5 * par.U * n_eff**2 + (p**2) / (par.m * n_eff)

    # Electric field
    phi = phi_from_n(n)
    Ex = E_eff - Dx(phi)

    # Momentum balance
    dp_dt = -Gamma(n_eff) * p - Dx(Pi) + par.e * n_eff * Ex

    return np.concatenate([dn_dt, dp_dt])

# ---------------- Single run -> save labeled frame ----------------
def run_once_and_save(frame_idx, u_desired, label):
    # Build ICs and solve
    n_init, p_init = init_fields_with_u(u_desired)
    y0 = np.concatenate([n_init, p_init])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)

    sol = solve_ivp(rhs_pde, (0.0, par.t_final), y0, t_eval=t_eval,
                    method="BDF", rtol=par.rtol, atol=par.atol)

    N = par.Nx
    n_t = sol.y[:N, :]

    # Space-time heatmap with label
    plt.figure(figsize=(8, 5.2))
    extent = [x.min(), x.max(), sol.t.min(), sol.t.max()]
    plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    plt.xlabel("x"); plt.ylabel("t"); plt.title("Eq. (3) PDE: n(x,t)")
    cb = plt.colorbar(); cb.set_label("n")

    plt.text(0.02, 0.98, label, transform=plt.gca().transAxes,
             va="top", ha="left", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.3", alpha=0.85))
    plt.tight_layout()
    fname = f"img1/{frame_idx:04d}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[frame] saved {fname}")

# ---------------- Parameter sweep -> frames ----------------
param_sweep = [
    dict(U_desired=0.04, u_desired=0.30, n1=0.010, lam=0.40, mode=1, amp_n=0.002, x0=6.0),
    dict(U_desired=0.04, u_desired=0.35, n1=0.020, lam=0.60, mode=2, amp_n=0.004, x0=6.0),
    dict(U_desired=0.04, u_desired=0.40, n1=0.030, lam=0.80, mode=2, amp_n=0.004, x0=5.0),
    dict(U_desired=0.04, u_desired=0.45, n1=0.040, lam=1.00, mode=3, amp_n=0.004, x0=4.0),
    dict(U_desired=0.04, u_desired=0.50, n1=0.050, lam=1.20, mode=3, amp_n=0.004, x0=3.0),
    dict(U_desired=0.04, u_desired=0.55, n1=0.060, lam=1.40, mode=3, amp_n=0.004, x0=2.0),
    dict(U_desired=0.04, u_desired=0.60, n1=0.070, lam=1.60, mode=3, amp_n=0.004, x0=1.0),
    dict(U_desired=0.04, u_desired=0.65, n1=0.080, lam=1.80, mode=3, amp_n=0.004, x0=0.0),
    dict(U_desired=0.04, u_desired=0.70, n1=0.090, lam=2.00, mode=3, amp_n=0.004, x0=-1.0),
    dict(U_desired=0.04, u_desired=0.75, n1=0.100, lam=2.20, mode=3, amp_n=0.004, x0=-2.0),
]

frame_idx = 1
for cfg in param_sweep:
    # apply configuration
    par.mode  = cfg.get("mode", par.mode)
    par.amp_n = cfg.get("amp_n", par.amp_n)
    par.n1    = cfg.get("n1", par.n1)
    par.lam   = cfg.get("lam", par.lam)
    par.x0    = cfg.get("x0", par.x0)

    U_desired = cfg.get("U_desired", par.U)
    u_desired = cfg.get("u_desired", 0.35)

    # set transport params and calibrate E to hit u_desired
    set_U_and_u(U_desired, u_desired, use_feedback=False)
    calibrate_E_to_speed(u_desired, t_short=6.0, iters=4, tol=1e-3)

    # compose label text for the frame
    label = (f"U={par.U:.3f}, u={u_desired:.3f}, c_pred={current_c_pred():.3f}\n"
             f"n1={par.n1:.3f}, λ={par.lam:.3f}, mode={par.mode}, δ={par.amp_n:.4f}, x0={par.x0:.1f}")

    # run and save PNG
    run_once_and_save(frame_idx, u_desired, label)
    frame_idx += 1

# After running:
# ffmpeg -framerate 4 -i %04d.png -pix_fmt yuv420p out.mp4
