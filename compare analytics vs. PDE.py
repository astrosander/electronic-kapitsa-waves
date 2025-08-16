# Validating Eq. (3): analytical speed/period derivation (Poisson case, small-amplitude)
# + numerical PDE simulation + traveling-wave ODE shooting.
#
# We provide:
# 1) Small-amplitude analytical predictions with Poisson coupling:
#       - Speed  c_pred = e E / (m Γ(n0))
#       - Linearized period in time at a point: T_lin = 2π / ω_lin,
#         where ω_lin is from the 2×2 linearization around (n0, Φ'=0).
#         Spatial wavelength Λ_lin = c_pred * T_lin.
# 2) Traveling-wave ODE in ξ = x - c t:
#       n' = [ e n (E - z) - Γ(n) (m c n + J) ] / ( U n - J^2/(m n^2) ),
#       z' = - (n - n0)/ε,   (z = Φ')
#    For J=0 this reduces to n' = [ e (E - z) - m c Γ(n) ] / U.
#    We integrate one closed orbit and measure its period.
# 3) Full PDE solve of Eq. (3) with Poisson, spectral derivatives, and constant E
#    (plus tiny proportional feedback to keep ⟨u⟩ near c_pred).
#    We extract: dominant k*, spatial wavelength, and speed c_sim from the
#    phase of the k*-mode over time.
#
# Notes:
# - Units are normalized (m=e=1 typical). Adjust parameters below as needed.
# - We keep parameters moderate so this cell runs quickly. Increase Nx/t_final for accuracy.

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
    epsilon: float = 15.0  # dielectric
    E: float = 0.035       # choose so that c_pred ~ 0.2
    # Numerics (PDE)
    L: float = 10.0
    Nx: int = 384
    t_final: float = 10.0
    n_save: int = 240
    rtol: float = 1e-6
    atol: float = 1e-8
    n_floor: float = 1e-6
    # Initial perturbation (PDE)
    amp_n: float = 8e-3
    mode: int = 3
    # Small feedback to keep <u> near c_pred
    Kp: float = 0.15

par = P()

# ---------------- Utilities ----------------
x = np.linspace(0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j * k
inv_k2 = np.zeros_like(k, dtype=np.complex128); nz = k!=0
inv_k2[nz] = 1.0 / (k[nz]**2)

def Dx(f): return (ifft(ik * fft(f))).real

def phi_from_n(n):
    rhs_hat = fft((n - par.n0) / par.epsilon)
    phi_hat = inv_k2 * rhs_hat
    phi_hat[0] = 0.0
    return (ifft(phi_hat)).real

def Gamma(n): return par.Gamma0 * np.exp(-n / par.w)

def set_U_and_u(U_value, u_target, use_feedback=False):
    """
    Set U and the electric field E so that the predicted speed equals u_target.
    Optionally disable feedback for an exact open-loop match.
    """
    par.U = U_value
    par.Kp = (par.Kp if use_feedback else 0.0)  # 0 => no feedback
    Gamma_n0 = Gamma(par.n0)
    par.E = par.m * Gamma_n0 * u_target / par.e
    # For diagnostics:
    c_pred = par.e * par.E / (par.m * Gamma_n0)
    print(f"[set_U_and_u] U={par.U:.6f}, E={par.E:.6f}, predicted u={c_pred:.6f} (target={u_target:.6f})")

def init_fields_with_u(u_target):
    """
    Build initial fields with the chosen target speed.
    """
    n_init = par.n0 * np.ones(par.Nx)
    if par.amp_n != 0.0:
        kx = 2*np.pi*par.mode / par.L
        # n_init += par.amp_n * np.cos(kx * x)
        n_init += 0.01 * np.cos(5 * x)
        # print("kx", kx)
        # print("amp_n", par.amp_n)
    p_init = par.m * n_init * u_target
    return n_init, p_init

def measure_mean_speed(n_t, p_t):
    n_eff = np.maximum(n_t, par.n_floor)
    v_t = p_t / (par.m * n_eff)
    return v_t.mean(axis=0)[-1]  # mean <u> at final time of the short run

def rhs_pde_for_calibration(t, y):
    # identical physics as your main rhs_pde, but *no feedback* to measure open-loop
    N = par.Nx
    n = y[:N]; p = y[N:]
    n_eff = np.maximum(n, par.n_floor)
    v = p/(par.m*n_eff)
    dn_dt = -Dx(n_eff*v)
    Pi = 0.5 * par.U * n_eff**2 + (p**2) / (par.m * n_eff)
    phi = phi_from_n(n)  # if you use Poisson; if not, set Ex = par.E
    Ex = par.E - Dx(phi)
    dp_dt = -Gamma(n_eff)*p - Dx(Pi) + par.e*n_eff*Ex
    return np.concatenate([dn_dt, dp_dt])

def calibrate_E_to_speed(u_target, t_short=10.0, iters=5, tol=1e-3):
    """
    Iteratively retune E so that mean speed ≈ u_target after a short warm-up run.
    Keeps U fixed. Uses open-loop (no feedback) during calibration.
    """
    Gamma_n0 = Gamma(par.n0)
    for k_iter in range(iters):
        # init fields with current E and intended u_target
        n0, p0 = init_fields_with_u(u_target)
        y0 = np.concatenate([n0, p0])
        t_eval_short = np.linspace(0.0, t_short, 50)
        sol_short = solve_ivp(rhs_pde_for_calibration, (0.0, t_short), y0,
                              t_eval=t_eval_short, method="BDF",
                              rtol=par.rtol, atol=par.atol)
        N = par.Nx
        n_t = sol_short.y[:N, :]
        p_t = sol_short.y[N:, :]
        u_meas = measure_mean_speed(n_t, p_t)
        err = u_target - u_meas
        print(f"[cal] iter {k_iter}: E={par.E:.6f}, u_meas={u_meas:.6f}, err={err:.6e}")
        if abs(err) <= tol:
            break
        # simple proportional retune of E using mean-balance slope du/dE = e/(m Γ(n0))
        slope = par.e / (par.m * Gamma_n0)
        par.E += err / slope
    print(f"[cal] final E={par.E:.6f} for target u={u_target:.6f}")

def mu_prime(n): return par.U * n
def Pi0(n): return 0.5 * par.U * n**2

# ---------------- Analytical predictions (small amplitude) ----------------
# Linearization around n0, z=Φ'=0 with J=0 and choose c so equilibrium is stationary:
# c_pred satisfies e E = m c Γ(n0) => c_pred = e E / (m Γ(n0))
Gamma0_at_n0 = Gamma(par.n0)
c_pred = par.e * par.E / (par.m * Gamma0_at_n0)

# Linearized 2×2 system in variables (δn, δz) with z=Φ':
# δn' = -(e/U) δz - (m c Γ'(n0)/U) δn  (in ξ=x-ct)  but we estimate temporal period at fixed x via PDE below.
# For a quick analytical spatial wavelength in moving frame, linearize the (n,z) ODE in ξ:
Gn0 = - (par.Gamma0 / par.w) * np.exp(-par.n0/par.w)
a = - par.m * c_pred * Gn0 / par.U
b = - par.e / par.U
ccoef = - 1.0 / par.epsilon
# Eigenvalues λ solve λ^2 - a λ + b c = 0. For weak |a| we take ω_lin ≈ sqrt(-b c).
disc = a*a - 4*b*ccoef
if disc < 0:
    omega_lin = np.sqrt(-b*ccoef)  # primary oscillation freq in ξ variable
else:
    # if damping is significant, take imaginary part of roots as ω
    lam1 = 0.5*(a + np.sqrt(disc))
    lam2 = 0.5*(a - np.sqrt(disc))
    omega_lin = abs(np.imag(lam1))

Lambda_lin = 2*np.pi / max(omega_lin, 1e-12)  # spatial period in ξ
T_lin = Lambda_lin / max(c_pred, 1e-12)       # temporal period at a fixed x

print(f"Analytical (small-amplitude): c_pred ≈ {c_pred:.4f}, Λ_lin ≈ {Lambda_lin:.2f}, T_lin ≈ {T_lin:.2f}")

# ---------------- Traveling-wave ODE (n,z) in ξ = x - c t ----------------
def nz_system_xi(xi, y, c, J=0.0):
    n, z = y  # z = Φ' = dΦ/dξ
    n_eff = max(n, par.n_floor)
    p = par.m * c * n_eff + J
    Gprime = par.U * n_eff - (J**2) / (par.m * n_eff**2)
    RHS = par.e * n_eff * (par.E - z) - Gamma(n_eff) * p
    dn_dxi = RHS / max(Gprime, 1e-12)
    dz_dxi = - (n_eff - par.n0) / par.epsilon
    return [dn_dxi, dz_dxi]

# Integrate one orbit near equilibrium (J=0)
def integrate_orbit(c, amp=0.02):
    # initial perturbation along n, z(0)=0
    y0 = [par.n0*(1+amp), 0.0]
    # Integrate until we return to z=0 with n decreasing (one period)
    def event_cross(xi, y): return y[1]  # z=0
    event_cross.terminal = False
    event_cross.direction = -1  # crossing from + to -
    sol = solve_ivp(lambda xi,y: nz_system_xi(xi,y,c,0.0),
                    (0.0, 2000.0), y0, rtol=1e-9, atol=1e-11,
                    events=event_cross, max_step=0.5)
    # find the second crossing (first is at xi=0)
    if len(sol.t_events[0]) >= 2:
        Xi_period = sol.t_events[0][1] - sol.t_events[0][0]
        return Xi_period, sol
    else:
        return np.nan, sol

Xi_period, orbit_sol = integrate_orbit(c_pred, amp=0.02)
print(f"Traveling-wave ODE: Xi_period ≈ {Xi_period:.2f} → Λ_ode ≈ {Xi_period:.2f} (since ξ is spatial), T_ode ≈ {Xi_period/c_pred if np.isfinite(Xi_period) else np.nan:.2f}")



# ---------------- Full PDE solve of Eq. (3) with Poisson ----------------
# State y = [n, p]; p = m n v
def rhs_pde(t, y):
    N = par.Nx
    n = y[:N]
    p = y[N:]
    n_eff = np.maximum(n, par.n_floor)
    v = p / (par.m * n_eff)

    # Gentle feedback to keep <u> near c_pred
    mean_u = v.mean()
    E_eff = par.E + par.Kp*(c_pred - mean_u)

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

# Initial fields: uniform flow ~ c_pred with small density perturbation
# n_init = par.n0 * np.ones(par.Nx)
# u_init = c_pred * np.ones(par.Nx)
# kx = 2*np.pi*par.mode / par.L
# n_init += par.amp_n * np.cos(kx * x)
# p_init = par.m * n_init * u_init
# y0 = np.concatenate([n_init, p_init])
# t_eval = np.linspace(0.0, par.t_final, par.n_save)
# sol = solve_ivp(rhs_pde, (0.0, par.t_final), y0, t_eval=t_eval,
#                 method="BDF", rtol=par.rtol, atol=par.atol)

U_desired = 0.04
u_desired = 0.6

# 2) Set them (open-loop, no feedback to keep exactness):
set_U_and_u(U_desired, u_desired, use_feedback=False)

# 3) (Optional but recommended) quick auto-calibration of E:
calibrate_E_to_speed(u_desired, t_short=10.0, iters=5, tol=5e-4)


n_init, p_init = init_fields_with_u(u_desired)
y0 = np.concatenate([n_init, p_init])
t_eval = np.linspace(0.0, par.t_final, par.n_save)
sol = solve_ivp(rhs_pde, (0.0, par.t_final), y0, t_eval=t_eval,
                method="BDF", rtol=par.rtol, atol=par.atol)


N = par.Nx
n_t = sol.y[:N, :]
p_t = sol.y[N:, :]
v_t = p_t / (par.m * np.maximum(n_t, par.n_floor))

# ---------------- Measure dominant wavelength and speed from PDE ----------------
# Dominant k* from average power spectrum
spec = np.mean(np.abs(fft(n_t, axis=0))**2, axis=1)
# exclude k=0
spec[0] = 0.0
m_star = np.argmax(spec[:par.Nx//2])  # positive k side
k_star = 2*np.pi * m_star / par.L
Lambda_sim = 2*np.pi / max(k_star, 1e-12)

# Speed from phase of the k* Fourier mode over time
nk_t = fft(n_t, axis=0)[m_star, :]  # complex amplitude of k*
phase = np.unwrap(np.angle(nk_t))
# linear fit of phase vs. t: phase ≈ k* c t + const  => slope/k* = c
coeffs = np.polyfit(sol.t, phase, 1)
omega_eff = coeffs[0]
c_sim = omega_eff / max(k_star, 1e-12)

# ---------- (1) EOS / analytic and finite-difference ----------
def cs_from_EOS(par):
    return np.sqrt(par.U * par.n0 / par.m)

def cs_from_pressure_FD(par, delta=1e-4):
    # Pi0(n) = 0.5 * U * n**2 in your model; keep general form if you later change Pi0
    def Pi0(n): return 0.5 * par.U * n**2
    num = Pi0(par.n0 + delta) - Pi0(par.n0 - delta)
    dPidn_at_n0 = num / (2*delta)
    return np.sqrt( (par.n0/par.m) * dPidn_at_n0 )

# ---------- (2A) Phase-fit method ----------
def measure_cs_phase_fit(x, t, n_t, par, m_pick=None, time_window_frac=0.5):
    """
    Returns (cs_phase, u_mean, k_pick). Uses the dominant mode if m_pick is None.
    """
    Nx = par.Nx
    # choose time window to avoid initial transient
    t0 = int(len(t) * (1.0 - time_window_frac))
    n_slice = n_t[:, t0:]
    # pick mode
    if m_pick is None:
        spec = np.mean(np.abs(np.fft.fft(n_slice, axis=0))**2, axis=1)
        spec[0] = 0.0
        m_pick = np.argmax(spec[:Nx//2])
    k_pick = 2*np.pi * m_pick / par.L

    # complex amplitude vs time for that mode
    nk_t = np.fft.fft(n_t, axis=0)[m_pick, :]
    phase = np.unwrap(np.angle(nk_t))
    slope = np.polyfit(t, phase, 1)[0]          # d(phase)/dt = omega
    omega = slope

    # mean flow (use late-time average)
    N = par.Nx
    v_t = (p_t / (par.m * np.maximum(n_t, par.n_floor)))
    u_mean = v_t.mean(axis=0)[t0:].mean()

    cs_phase = omega / k_pick - u_mean
    return cs_phase, u_mean, k_pick

# ---------- (2B) Cross-correlation method ----------
def bandlimit_mode(signal, m_pick):
    # project signal at one x onto mode m_pick (optional helper)
    return signal

def measure_cs_xcorr(x, t, n_t, par, x1=None, x2=None, detrend=True):
    """
    Returns (cs_corr, u_mean, v_front) using cross-correlation between two probes.
    """
    if x1 is None: x1 = par.L * 0.25
    if x2 is None: x2 = par.L * 0.75
    i1 = int((x1 / par.L) * par.Nx) % par.Nx
    i2 = int((x2 / par.L) * par.Nx) % par.Nx
    s1 = n_t[i1, :].copy()
    s2 = n_t[i2, :].copy()
    if detrend:
        s1 -= s1.mean(); s2 -= s2.mean()
    # cross-correlation (full), find lag of max
    corr = np.correlate(s2, s1, mode='full')
    lags = np.arange(-len(s1)+1, len(s2))
    kmax = np.argmax(corr)
    lag = lags[kmax]
    dt = t[1] - t[0]
    tau = lag * dt
    dx = (x2 - x1) % par.L
    v_front = dx / tau if tau != 0 else np.nan

    # mean flow (global avg)
    v_t = (p_t / (par.m * np.maximum(n_t, par.n_floor)))
    u_mean = v_t.mean(axis=0).mean()

    cs_corr = v_front - u_mean
    return cs_corr, u_mean, v_front

# ---------- run the measurements ----------
cs_eos  = cs_from_EOS(par)
cs_fd   = cs_from_pressure_FD(par, delta=1e-4)
cs_phase, u_mean_phase, k_used = measure_cs_phase_fit(x, sol.t, n_t, par)
cs_corr,  u_mean_corr,  v_front = measure_cs_xcorr(x, sol.t, n_t, par)

print(f"c_s (EOS)           = {cs_eos:.6f}")
print(f"c_s (finite diff.)  = {cs_fd:.6f}")
print(f"c_s (phase-fit)     = {cs_phase:.6f}  [k_used={k_used:.4f},  u_mean≈{u_mean_phase:.6f}]")
print(f"c_s (cross-corr)    = {cs_corr:.6f}  [v_front≈{v_front:.6f}, u_mean≈{u_mean_corr:.6f}]")



print(f"PDE (measured): k*={k_star:.4f}, Λ_sim ≈ {Lambda_sim:.2f}, c_sim ≈ {c_sim:.4f}")

plt.figure(figsize=(8,6))
extent=[x.min(),x.max(),sol.t.min(),sol.t.max()]
plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap="inferno")
plt.xlim(3,9)
plt.xlabel("x"); plt.ylabel("t"); plt.title("Eq. (3) PDE: n(x,t)")
plt.colorbar(label="n"); plt.tight_layout(); #plt.show()

plt.tight_layout()

plt.savefig("img.pdf")

plt.figure(figsize=(8,4.5))
for frac in [0.0,0.25,0.5,0.75,1.0]:
    i = int(frac*(len(sol.t)-1))
    plt.plot(x, n_t[:,i], label=f"t={sol.t[i]:.1f}")
plt.legend(); plt.xlabel("x"); plt.ylabel("n"); plt.title("Density snapshots"); plt.tight_layout(); #plt.show()

plt.figure(figsize=(8,4.5))
plt.plot(sol.t, v_t.mean(axis=0), label="⟨u⟩")
plt.axhline(c_pred, ls=":", label="c_pred")
plt.legend(); plt.xlabel("t"); plt.ylabel("⟨u⟩"); plt.title("Mean velocity vs time"); plt.tight_layout(); #plt.show()

labels = ["c_pred (analytical)", "c_sim (PDE)", "Λ_lin (analytical)", "Λ_sim (PDE)", "Λ_ode (orbit)"]
vals = [c_pred, c_sim, Lambda_lin, Lambda_sim, Xi_period]
plt.figure(figsize=(7,4))
plt.bar(labels, vals)
plt.xticks(rotation=20, ha='right'); plt.title("Predicted vs simulated"); plt.tight_layout(); #plt.show()