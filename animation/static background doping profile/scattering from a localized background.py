# Wave-packet scattering from a localized background (defect) in Eq. (3) + Poisson(bg)
# Produces: (A) spacetime plot; (B) animation frames; (C) R/T vs defect amplitude sweep.

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp

# -------------------------- Parameters --------------------------
@dataclass
class Par:
    # physics
    m: float = 1.0
    e: float = 1.0
    n0: float = 1.0
    U: float = 1.0             # Π0'(n) = U n  -> c_s = sqrt(U n0/m)
    Gamma0: float = 0.05
    w: float = 1.0
    epsilon: float = 15.0
    # background defect: nbar(x) = n0 + n1 * exp(-lambda*(x-x0)^2/2)
    n1: float = 0.03
    lam: float = 0.005 / 4
    x0: float = 0.6             # in units of L (fraction along domain)
    # mean drift (set exact via E_ext)
    u_mean: float = 0.5
    # numerics
    L: float = 100.0
    Nx: int = 128
    t_final: float = 140.0
    n_save: int = 560
    rtol: float = 1e-7
    atol: float = 1e-9
    n_floor: float = 1e-10
    # wave packet (right-moving)
    A_pack: float = 0e-3        # amplitude of δn
    mode: int = 6               # carrier k0 = 2π*mode/L
    x_src: float = 0.15         # packet center (fraction of L)
    sigma: float = 8.0          # packet width in x
    # animation
    save_frames: bool = True
    frames_dir: str = "frames_scatter"
    cmap: str = "inferno"

par = Par()

# --------------------- Grid & spectral ops ----------------------
x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1]-x[0]
k = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j*k
inv_k2 = np.zeros_like(k, dtype=np.complex128)
nz = (k!=0); inv_k2[nz] = 1.0/(k[nz]**2)

def Dx(f): return (ifft(ik*fft(f))).real

# -------------------- Material and helpers ----------------------
def Gamma(n): return par.Gamma0 * np.exp(-n/par.w)
def Pi0(n):   return 0.5*par.U*n**2
def cs():     return np.sqrt(par.U*par.n0/par.m)

def periodic_delta(x, x0, L):
    # shortest signed distance on periodic domain
    return (x - x0 + 0.5*L) % L - 0.5*L

def nbar_profile():
    x0 = par.x0 * par.L
    d = periodic_delta(x, x0, par.L)
    return par.n0 + par.n1*np.exp(-0.5*par.lam*d**2) 

# Poisson with background: φ'' = -(e/ε) (n - nbar)
def phi_from_n_with_bg(n, nbar):
    rhs_hat = fft( (par.e/par.epsilon) * (n - nbar) )
    phi_hat = inv_k2 * rhs_hat
    phi_hat[0] = 0.0
    phi = ifft(phi_hat).real
    return phi

# exact mean drift via E
def external_field_for_u(u_target):
    G = Gamma(par.n0)
    return par.m * G * u_target / par.e

E_ext = external_field_for_u(par.u_mean)

# ----------------------- Wave packet ICs ------------------------
def right_moving_packet():
    """ δn(x) = A * exp(-(x-xc)^2/(2σ^2)) * cos(k0 (x-xc))
        δu chosen so that r_- = 0 (pure right-moving acoustic packet). """
    k0 = 2*np.pi*par.mode/par.L
    xc = par.x_src * par.L
    d = periodic_delta(x, xc, par.L)
    env = np.exp(-0.5*(d/par.sigma)**2)
    dn = par.A_pack * np.cos(k0*d) * env
    du = 0#(cs()/par.n0) * dn          # r_- = 0 → du = + (c_s/n0) dn
    return dn, du

def initial_fields():
    nbar = nbar_profile()
    dn, du = right_moving_packet()
    n0 = np.maximum(nbar + dn, par.n_floor)
    u0 = par.u_mean + du
    p0 = par.m * n0 * u0
    # print(p0)
    return n0, p0, nbar

# ----------------------- PDE RHS (full) ------------------------
def rhs_pde(t, y):
    N = par.Nx
    n = y[:N]; p = y[N:]
    n_eff = np.maximum(n, par.n_floor)
    v = p/(par.m*n_eff)

    # continuity
    dn_dt = -Dx(n_eff*v)

    # momentum flux
    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)

    # fields
    nbar = nbar_profile()          # static bg
    phi  = phi_from_n_with_bg(n, nbar)
    Ex   = E_ext - Dx(phi)

    # momentum
    dp_dt = -Gamma(n_eff)*p - Dx(Pi) + par.e*n_eff*Ex

    return np.concatenate([dn_dt, dp_dt])

# ----------------------- Run one experiment --------------------
def run_once():
    n0, p0, nbar = initial_fields()
    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)

    sol = solve_ivp(rhs_pde, (0.0, par.t_final), y0, t_eval=t_eval,
                    method="BDF", rtol=par.rtol, atol=par.atol)
    N = par.Nx
    n_t = sol.y[:N,:]
    p_t = sol.y[N:,:]
    u_t = p_t/(par.m*np.maximum(n_t, par.n_floor))
    return sol.t, n_t, u_t, nbar

# ------------------ Reflection / Transmission -------------------
# Demodulate δn(x,t) by e^{-i k0 x} in windows left/right of the defect.
def window_indices(x_center, half_width):
    d = periodic_delta(x, x_center, par.L)
    return np.where(np.abs(d) <= half_width)[0]

def complex_envelope(delta_n, k0, idx):
    w = np.hanning(len(idx))
    xx = x[idx]
    sig = delta_n[idx] * w
    ref = np.exp(-1j*k0*xx)
    # complex projection (normalized by window L1)
    a = np.trapz(sig*ref, xx) / np.trapz(w, xx)
    return a

def measure_RT(t, n_t, nbar, k0, xL_c, xR_c, halfW, t_win=(0.1, 0.9)):
    """ Return reflection & transmission based on envelopes before/after scattering.
        Left window center xL_c (left of defect), right window xR_c (right of defect). """
    iL = window_indices(xL_c, halfW)
    iR = window_indices(xR_c, halfW)

    # choose early and late time windows
    t0 = int(t_win[0]*len(t)); t1 = int(0.25*len(t))          # early
    t2 = int(0.75*len(t)); t3 = int(t_win[1]*len(t))          # late

    # incident amplitude (early, left window, +k0)
    a_in = []
    for j in range(t0, t1):
        dn = n_t[:,j] - nbar
        a_in.append( complex_envelope(dn, +k0, iL) )
    a_in = np.mean(a_in)

    # reflected amplitude (late, left window, -k0)
    a_ref = []
    for j in range(t2, t3):
        dn = n_t[:,j] - nbar
        # detect -k by flipping sign of k in demodulation
        a_ref.append( complex_envelope(dn, -k0, iL) )
    a_ref = np.mean(a_ref)

    # transmitted amplitude (late, right window, +k0)
    a_tr = []
    for j in range(t2, t3):
        dn = n_t[:,j] - nbar
        a_tr.append( complex_envelope(dn, +k0, iR) )
    a_tr = np.mean(a_tr)

    # define energy-like measures ~ |amplitude|^2
    Ain = np.abs(a_in)**2
    R   = np.abs(a_ref)**2 / (Ain + 1e-16)
    T   = np.abs(a_tr)**2 / (Ain + 1e-16)
    return R, T, a_in, a_ref, a_tr

# ----------------------------- Plots ----------------------------
def plot_spacetime(t, n_t, nbar, title="δn(x,t) scattering"):
    dn = n_t - nbar[:,None]
    extent=[x.min(), x.max(), t.min(), t.max()]
    plt.figure(figsize=(10,4.6))
    plt.imshow(dn.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    # mark defect location
    x0 = par.x0 * par.L
    plt.plot([x0, x0], [t.min(), t.max()], 'w--', lw=1.5, alpha=0.8, label="defect")
    plt.colorbar(label="δn")
    plt.xlabel("x"); plt.ylabel("t"); plt.title(title)
    plt.legend(loc='upper right'); plt.tight_layout()
    plt.savefig("spacetime_scatter.png", dpi=160)

def save_animation_frames(t, n_t, nbar):
    import os
    os.makedirs(par.frames_dir, exist_ok=True)
    dn = n_t - nbar[:,None]
    for j in range(len(t)):
        plt.figure(figsize=(9,3.2))
        plt.plot(x, dn[:,j], lw=1.8)
        plt.axvline(par.x0*par.L, color='k', ls='--', alpha=0.5, label="defect")
        plt.ylim(1.2*dn.min(), 1.2*dn.max())
        plt.xlim(0, par.L)
        plt.xlabel("x"); plt.ylabel("δn"); plt.title(f"t={t[j]:.2f}")
        if j==0: plt.legend()
        plt.tight_layout()
        fname = f"{par.frames_dir}/frame_{j:04d}.png"
        plt.savefig(fname, dpi=140); plt.close()

# -------------------------- Run once ---------------------------
t, n_t, u_t, nbar = run_once()

# spacetime
plot_spacetime(t, n_t, nbar, title="Wave-packet → localized defect (Poisson with background)")

# frames for animation (use: ffmpeg -framerate 12 -i frames_scatter/frame_%04d.png -pix_fmt yuv420p scatter.mp4)
if par.save_frames:
    save_animation_frames(t, n_t, nbar)

# ----------------- Measure R/T for this run --------------------
k0 = 2*np.pi*par.mode/par.L
x_def = par.x0*par.L
R, T, a_in, a_ref, a_tr = measure_RT(
    t, n_t, nbar, k0,
    xL_c = (x_def - 30.0) % par.L,  # left window center
    xR_c = (x_def + 30.0) % par.L,  # right window center
    halfW = 15.0,                   # window half-width
    t_win = (0.05, 0.95)
)
print(f"[single] R≈{R:.3f}, T≈{T:.3f},  R+T≈{R+T:.3f}  (damping may make R+T<1)")

# ----------------- Sweep over defect amplitude -----------------
def sweep_R_T_over_n1(n1_list):
    R_list, T_list = [], []
    n1_old = par.n1
    for n1 in n1_list:
        par.n1 = float(n1)
        t, n_t, u_t, nbar = run_once()
        R, T, *_ = measure_RT(
            t, n_t, nbar, k0,
            xL_c=(par.x0*par.L-30)%par.L, xR_c=(par.x0*par.L+30)%par.L,
            halfW=15.0, t_win=(0.05,0.95)
        )
        R_list.append(R); T_list.append(T)
        print(f"[sweep] n1={n1:.4f} -> R={R:.3f}, T={T:.3f}, R+T={R+T:.3f}")
    par.n1 = n1_old
    return np.array(R_list), np.array(T_list)

# Example sweep (quick)
n1_vals = np.linspace(0.0, 0.06, 8)
R_vals, T_vals = sweep_R_T_over_n1(n1_vals)

plt.figure(figsize=(7,4))
plt.plot(n1_vals, R_vals, 'o-', label='R (reflection)')
plt.plot(n1_vals, T_vals, 's-', label='T (transmission)')
plt.plot(n1_vals, 1 - (R_vals+T_vals), 'x--', label='loss (damping)')
plt.xlabel("defect amplitude n1"); plt.ylabel("coefficient")
plt.title("Scattering on localized background: R/T vs n1")
plt.legend(); plt.tight_layout(); plt.savefig("RT_vs_n1.png", dpi=160)
