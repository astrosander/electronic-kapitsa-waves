import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.fft import fft, ifft, fftfreq
from dataclasses import dataclass

@dataclass
class P:
    m: float = 1.0
    e: float = 1.0
    U: float = 0.04
    n0: float = 1.0
    Gamma0: float = 2.0
    w: float = 1.0
    epsilon: float = 15.0
    E: float = 0.035
    L: float = 10.0
    Nx: int = 384
    t_final: float = 10.0
    n_save: int = 240
    rtol: float = 1e-6
    atol: float = 1e-8
    n_floor: float = 1e-6
    amp_n: float = 8e-3
    mode: int = 3
    Kp: float = 0.15

par = P()

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
    par.U = U_value
    par.Kp = (par.Kp if use_feedback else 0.0)
    Gamma_n0 = Gamma(par.n0)
    par.E = par.m * Gamma_n0 * u_target / par.e
    c_pred = par.e * par.E / (par.m * Gamma_n0)
    print(f"[set_U_and_u] U={par.U:.6f}, E={par.E:.6f}, predicted u={c_pred:.6f} (target={u_target:.6f})")

def init_fields_with_u(u_target):
    n_init = par.n0 * np.ones(par.Nx)
    if par.amp_n != 0.0:
        kx = 2*np.pi*par.mode / par.L
        n_init += 0.01 * np.cos(5 * x)
    p_init = par.m * n_init * u_target
    return n_init, p_init

def measure_mean_speed(n_t, p_t):
    n_eff = np.maximum(n_t, par.n_floor)
    v_t = p_t / (par.m * n_eff)
    return v_t.mean(axis=0)[-1]

def rhs_pde_for_calibration(t, y):
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

def calibrate_E_to_speed(u_target, t_short=10.0, iters=5, tol=1e-3):
    Gamma_n0 = Gamma(par.n0)
    for k_iter in range(iters):
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
        slope = par.e / (par.m * Gamma_n0)
        par.E += err / slope
    print(f"[cal] final E={par.E:.6f} for target u={u_target:.6f}")

def mu_prime(n): return par.U * n
def Pi0(n): return 0.5 * par.U * n**2

Gamma0_at_n0 = Gamma(par.n0)
c_pred = par.e * par.E / (par.m * Gamma0_at_n0)

Gn0 = - (par.Gamma0 / par.w) * np.exp(-par.n0/par.w)
a = - par.m * c_pred * Gn0 / par.U
b = - par.e / par.U
ccoef = - 1.0 / par.epsilon
disc = a*a - 4*b*ccoef
if disc < 0:
    omega_lin = np.sqrt(-b*ccoef)
else:
    lam1 = 0.5*(a + np.sqrt(disc))
    lam2 = 0.5*(a - np.sqrt(disc))
    omega_lin = abs(np.imag(lam1))

Lambda_lin = 2*np.pi / max(omega_lin, 1e-12)
T_lin = Lambda_lin / max(c_pred, 1e-12)

print(f"Analytical (small-amplitude): c_pred ≈ {c_pred:.4f}, Λ_lin ≈ {Lambda_lin:.2f}, T_lin ≈ {T_lin:.2f}")

def nz_system_xi(xi, y, c, J=0.0):
    n, z = y
    n_eff = max(n, par.n_floor)
    p = par.m * c * n_eff + J
    Gprime = par.U * n_eff - (J**2) / (par.m * n_eff**2)
    RHS = par.e * n_eff * (par.E - z) - Gamma(n_eff) * p
    dn_dxi = RHS / max(Gprime, 1e-12)
    dz_dxi = - (n_eff - par.n0) / par.epsilon
    return [dn_dxi, dz_dxi]

def integrate_orbit(c, amp=0.02):
    y0 = [par.n0*(1+amp), 0.0]
    def event_cross(xi, y): return y[1]
    event_cross.terminal = False
    event_cross.direction = -1
    sol = solve_ivp(lambda xi,y: nz_system_xi(xi,y,c,0.0),
                    (0.0, 2000.0), y0, rtol=1e-9, atol=1e-11,
                    events=event_cross, max_step=0.5)
    if len(sol.t_events[0]) >= 2:
        Xi_period = sol.t_events[0][1] - sol.t_events[0][0]
        return Xi_period, sol
    else:
        return np.nan, sol

Xi_period, orbit_sol = integrate_orbit(c_pred, amp=0.02)
print(f"Traveling-wave ODE: Xi_period ≈ {Xi_period:.2f} → Λ_ode ≈ {Xi_period:.2f} (since ξ is spatial), T_ode ≈ {Xi_period/c_pred if np.isfinite(Xi_period) else np.nan:.2f}")

def rhs_pde(t, y):
    N = par.Nx
    n = y[:N]
    p = y[N:]
    n_eff = np.maximum(n, par.n_floor)
    v = p / (par.m * n_eff)

    mean_u = v.mean()
    E_eff = par.E + par.Kp*(c_pred - mean_u)

    dn_dt = -Dx(n * v)

    Pi = 0.5 * par.U * n_eff**2 + (p**2) / (par.m * n_eff)

    phi = phi_from_n(n)
    Ex = E_eff - Dx(phi)

    dp_dt = -Gamma(n_eff) * p - Dx(Pi) + par.e * n_eff * Ex

    return np.concatenate([dn_dt, dp_dt])

U_desired = 0.04
u_desired = 0.6

set_U_and_u(U_desired, u_desired, use_feedback=False)

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

spec = np.mean(np.abs(fft(n_t, axis=0))**2, axis=1)
spec[0] = 0.0
m_star = np.argmax(spec[:par.Nx//2])
k_star = 2*np.pi * m_star / par.L
Lambda_sim = 2*np.pi / max(k_star, 1e-12)

nk_t = fft(n_t, axis=0)[m_star, :]
phase = np.unwrap(np.angle(nk_t))
coeffs = np.polyfit(sol.t, phase, 1)
omega_eff = coeffs[0]
c_sim = omega_eff / max(k_star, 1e-12)

def cs_from_EOS(par):
    return np.sqrt(par.U * par.n0 / par.m)

def cs_from_pressure_FD(par, delta=1e-4):
    def Pi0(n): return 0.5 * par.U * n**2
    num = Pi0(par.n0 + delta) - Pi0(par.n0 - delta)
    dPidn_at_n0 = num / (2*delta)
    return np.sqrt( (par.n0/par.m) * dPidn_at_n0 )

def measure_cs_phase_fit(x, t, n_t, par, m_pick=None, time_window_frac=0.5):
    Nx = par.Nx
    t0 = int(len(t) * (1.0 - time_window_frac))
    n_slice = n_t[:, t0:]
    if m_pick is None:
        spec = np.mean(np.abs(np.fft.fft(n_slice, axis=0))**2, axis=1)
        spec[0] = 0.0
        m_pick = np.argmax(spec[:Nx//2])
    k_pick = 2*np.pi * m_pick / par.L

    nk_t = np.fft.fft(n_t, axis=0)[m_pick, :]
    phase = np.unwrap(np.angle(nk_t))
    slope = np.polyfit(t, phase, 1)[0]
    omega = slope

    N = par.Nx
    v_t = (p_t / (par.m * np.maximum(n_t, par.n_floor)))
    u_mean = v_t.mean(axis=0)[t0:].mean()

    cs_phase = omega / k_pick - u_mean
    return cs_phase, u_mean, k_pick

def bandlimit_mode(signal, m_pick):
    return signal

def measure_cs_xcorr(x, t, n_t, par, x1=None, x2=None, detrend=True):
    if x1 is None: x1 = par.L * 0.25
    if x2 is None: x2 = par.L * 0.75
    i1 = int((x1 / par.L) * par.Nx) % par.Nx
    i2 = int((x2 / par.L) * par.Nx) % par.Nx
    s1 = n_t[i1, :].copy()
    s2 = n_t[i2, :].copy()
    if detrend:
        s1 -= s1.mean(); s2 -= s2.mean()
    corr = np.correlate(s2, s1, mode='full')
    lags = np.arange(-len(s1)+1, len(s2))
    kmax = np.argmax(corr)
    lag = lags[kmax]
    dt = t[1] - t[0]
    tau = lag * dt
    dx = (x2 - x1) % par.L
    v_front = dx / tau if tau != 0 else np.nan

    v_t = (p_t / (par.m * np.maximum(n_t, par.n_floor)))
    u_mean = v_t.mean(axis=0).mean()

    cs_corr = v_front - u_mean
    return cs_corr, u_mean, v_front

cs_eos  = cs_from_EOS(par)
cs_fd   = cs_from_pressure_FD(par, delta=1e-4)
cs_phase, u_mean_phase, k_used = measure_cs_phase_fit(x, sol.t, n_t, par)
cs_corr,  u_mean_corr,  v_front = measure_cs_xcorr(x, sol.t, n_t, par)

print(f"c_s (EOS)           = {cs_eos:.6f}")
print(f"c_s (finite diff.)  = {cs_fd:.6f}")
print(f"c_s (phase-fit)     = {cs_phase:.6f}  [k_used={k_used:.4f},  u_mean≈{u_mean_phase:.6f}]")
print(f"c_s (cross-corr)    = {cs_corr:.6f}  [v_front≈{v_front:.6f}, u_mean≈{u_mean_corr:.6f}]")

print(f"PDE (measured): k*={k_star:.4f}, Λ_sim ≈ {Lambda_sim:.2f}, c_sim ≈ {c_sim:.4f}")

# Store results for current Gamma0
gamma0_current = par.Gamma0
n_t_current = n_t.copy()
sol_t_current = sol.t.copy()

# Run simulations for three different Gamma0 values
#gamma0_values = [0.5, 1.0, 2.0]
gamma0_values = [0.9, 1.0, 1.1]
results = {}

for gamma0 in gamma0_values:
    print(f"\n--- Running simulation with Gamma0 = {gamma0} ---")
    
    # Set new Gamma0 value
    par.Gamma0 = gamma0
    
    # Recalibrate for this Gamma0
    set_U_and_u(U_desired, u_desired, use_feedback=False)
    calibrate_E_to_speed(u_desired, t_short=10.0, iters=5, tol=5e-4)
    
    # Run simulation
    n_init, p_init = init_fields_with_u(u_desired)
    y0 = np.concatenate([n_init, p_init])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)
    sol = solve_ivp(rhs_pde, (0.0, par.t_final), y0, t_eval=t_eval,
                    method="BDF", rtol=par.rtol, atol=par.atol)
    
    N = par.Nx
    n_t = sol.y[:N, :]
    
    # Store results
    results[gamma0] = {
        'n_t': n_t,
        't': sol.t,
        'x': x
    }

# Create three parallel figures with shared logarithmic colorbar
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Calculate global min/max for logarithmic colorbar (relative to n0=1)
all_log_ratios = []
for gamma0 in gamma0_values:
    log_ratio = np.log10(results[gamma0]['n_t'] / par.n0)
    all_log_ratios.append(log_ratio)

vmin = np.min([np.min(lr) for lr in all_log_ratios])
vmax = np.max([np.max(lr) for lr in all_log_ratios])

# Plot each simulation
for i, gamma0 in enumerate(gamma0_values):
    n_t = results[gamma0]['n_t']
    t = results[gamma0]['t']
    x = results[gamma0]['x']
    
    # Calculate log ratio relative to n0
    log_ratio = np.log10(n_t / par.n0)
    
    extent = [x.min(), x.max(), t.min(), t.max()]
    im = axes[i].imshow(log_ratio.T, origin="lower", aspect="auto", 
                       extent=extent, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[i].set_xlim(3, 9)
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("t")
    axes[i].set_title(f"Γ₀ = {gamma0}")

# Add shared colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30)
cbar.set_label("log₁₀(n/n₀)")

plt.tight_layout()
plt.savefig("img.png", dpi=150, bbox_inches='tight')

plt.figure(figsize=(8,4.5))
for frac in [0.0,0.25,0.5,0.75,1.0]:
    i = int(frac*(len(sol.t)-1))
    plt.plot(x, n_t[:,i], label=f"t={sol.t[i]:.1f}")
plt.legend(); plt.xlabel("x"); plt.ylabel("n"); plt.title("Density snapshots"); plt.tight_layout()

plt.figure(figsize=(8,4.5))
plt.plot(sol.t, v_t.mean(axis=0), label="⟨u⟩")
plt.axhline(c_pred, ls=":", label="c_pred")
plt.legend(); plt.xlabel("t"); plt.ylabel("⟨u⟩"); plt.title("Mean velocity vs time"); plt.tight_layout()

labels = ["c_pred (analytical)", "c_sim (PDE)", "Λ_lin (analytical)", "Λ_sim (PDE)", "Λ_ode (orbit)"]
vals = [c_pred, c_sim, Lambda_lin, Lambda_sim, Xi_period]
plt.figure(figsize=(7,4))
plt.bar(labels, vals)
plt.xticks(rotation=20, ha='right'); plt.title("Predicted vs simulated"); plt.tight_layout()