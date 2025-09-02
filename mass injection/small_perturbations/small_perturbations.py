import os, multiprocessing
n_threads = max(1, multiprocessing.cpu_count() - 0)
os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_threads))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.fft import fft, ifft, fftfreq
try:
    from threadpoolctl import threadpool_limits
    _THREADPOOL_OK = True
except Exception:
    _THREADPOOL_OK = False

@dataclass
class P:
    m: float = 1.0
    e: float = 1.0
    U: float = 1.0
    nbar0: float = 0.2
    Gamma0: float = 2.50
    w: float = 5.0

    include_poisson: bool = False
    eps: float = 20.0

    u_d: float = 20.0
    maintain_drift: str = "field"
    Kp: float = 0.15

    Dn: float = 0.8
    Dp: float = 0.2

    L: float = 10.0
    Nx: int = 512
    t_final: float = 4.0
    n_save: int = 360
    n_floor: float = 1e-7
    n_cap: float = 50.0
    p_cap: float = 1e3
    dealias_23: bool = True

    sv_nu: float = 0.15
    sv_pow: int = 8
    sv_on: bool = True

    seed_amp_n: float = 20e-2
    seed_amp_p: float = 20e-2
    seed_mode: int = 1

    CFL: float = 0.25
    dt_max: float = 2e-2
    dt_min: float = 1e-5

    outdir: str = "out_imex_mt"
    cmap: str = "inferno"

par = P()

x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j * k
k2 = k**2
kmax = np.max(np.abs(k))

def dealias(f):
    if not par.dealias_23: 
        return f
    fh = fft(np.asarray(f, dtype=np.float64))
    kc = par.Nx//3
    fh[kc:-kc] = 0.0
    return (ifft(fh)).real

def Dx(f):
    fh = fft(np.asarray(f, dtype=np.float64))
    return (ifft(ik * fh)).real

def Dxx(f):
    fh = fft(np.asarray(f, dtype=np.float64))
    return (ifft((-k2) * fh)).real

def spectral_viscosity(f, dt):
    if not par.sv_on or par.sv_nu <= 0.0:
        return f
    fh = fft(np.asarray(f, dtype=np.float64))
    damp = np.exp(-par.sv_nu * (np.abs(k)/max(kmax,1e-12))**par.sv_pow * dt)
    fh *= damp
    return (ifft(fh)).real

def Gamma(n):
    n_eff = np.maximum(n, par.n_floor)
    return par.Gamma0 * np.exp(-n_eff/par.w)

def dGamma_dn(n):
    return -Gamma(n)/par.w

def Pi0(n):
    n_eff = np.maximum(n, par.n_floor)
    return 0.5 * par.U * n_eff**2

def phi_from_n(n, nbar):
    rhs_hat = fft((par.e/par.eps) * (n - nbar))
    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    nz = (k2 != 0)
    phi_hat[nz] = rhs_hat[nz] / (-k2[nz])
    return (ifft(phi_hat)).real

def nbar_profile():
    return np.full_like(x, par.nbar0)

def pbar_profile(nbar):
    return par.m * nbar * par.u_d

def E_base_from_drift(nbar):
    return par.m * par.u_d * np.mean(Gamma(nbar)) / par.e

n0 = par.nbar0
u0 = par.u_d
p0 = par.m * n0 * u0
Gamma0_const = Gamma(n0)
Gp0_const = dGamma_dn(n0)
A = par.U*n0 - u0**2
B = 2*u0

def linear_E_mats(dt):
    a11 = -par.Dn*k2
    a12 = -1j*k
    a21 = -1j*k*A
    a22 = -(Gamma0_const + par.Dp*k2) - 1j*k*B

    Lk = np.empty((par.Nx,2,2), dtype=np.complex128)
    Lk[:,0,0] = a11; Lk[:,0,1] = a12
    Lk[:,1,0] = a21; Lk[:,1,1] = a22

    tr = a11 + a22
    det = a11*a22 - a12*a21
    disc = np.sqrt(np.maximum(0.0, 0.0) + tr*0)
    tr2 = 0.5*tr
    M = Lk - tr2[:,None,None]*np.eye(2, dtype=np.complex128)
    s2 = tr2**2 - det
    s = np.sqrt(s2)
    sh_over_s = np.where(np.abs(s)>1e-14, np.sinh(s*dt)/s, dt + (s*dt)**2/6.0)
    common = np.exp(tr2*dt)[:,None,None]
    I = np.eye(2, dtype=np.complex128)

    E_full = common * (np.cosh(s*dt)[:,None,None]*I + sh_over_s[:,None,None]*M)
    E_half = common * (np.cosh(s*(0.5*dt))[:,None,None]*I + (np.sinh(s*(0.5*dt))/np.where(np.abs(s)>1e-14,s,1.0))[:,None,None]*M)
    return E_full, E_half

def apply_linear_step_fft(dn, dp, E):
    nh = fft(np.asarray(dn, dtype=np.float64))
    ph = fft(np.asarray(dp, dtype=np.float64))
    V = np.stack([nh, ph], axis=-1)
    W = np.einsum('kij,kj->ki', E, V, optimize=True)
    out_n = (ifft(W[:,0])).real
    out_p = (ifft(W[:,1])).real
    return out_n, out_p

def nonlinear_residual(dn, dp, E_eff):
    n_full = np.clip(n0 + dn, par.n_floor, par.n_cap)
    p_full = np.clip(p0 + dp, -par.p_cap, par.p_cap)
    n_eff = np.maximum(n_full, par.n_floor)

    dn_full = -Dx(p_full) + par.Dn * Dxx(n_full)

    Pi = Pi0(n_eff) + (p_full**2)/(par.m*n_eff)
    Pi = dealias(Pi)
    dp_full = -Gamma(n_eff)*p_full - Dx(Pi) + par.e*n_eff*E_eff + par.Dp * Dxx(p_full)
    if par.include_poisson:
        phi = phi_from_n(n_eff, nbar_profile())
        dp_full -= n_eff * Dx(phi)

    dn_lin = -Dx(dp) + par.Dn * Dxx(dn)
    grad_term = Dx(A*dn + B*dp)
    dp_lin = -Gamma0_const*dp - grad_term + par.Dp * Dxx(dp)

    Nn = dn_full - dn_lin
    Np = dp_full - dp_lin
    Nn = spectral_viscosity(Nn, dt_current)
    Np = spectral_viscosity(Np, dt_current)
    Nn = dealias(Nn); Np = dealias(Np)
    return Nn, Np

def estimate_dt(n, p):
    cs = np.sqrt(par.U * n0 / par.m)
    u = p/(par.m*np.maximum(n, par.n_floor))
    umax = float(np.max(np.abs(u))) + cs
    dt_adv = par.CFL * dx / max(umax, 1e-12)
    Dmax = max(par.Dn, par.Dp)
    dt_diff = 0.45 * dx*dx / max(Dmax, 1e-12)
    dt = min(par.dt_max, dt_adv, dt_diff)
    return max(par.dt_min, dt)

def march_imex():
    os.makedirs(par.outdir, exist_ok=True)

    n = nbar_profile()
    p = pbar_profile(n)
    if par.seed_amp_n and par.seed_mode:
        kx = 2*np.pi*par.seed_mode/par.L
        n += par.seed_amp_n*np.cos(kx*x)
    if par.seed_amp_p and par.seed_mode:
        kx = 2*np.pi*par.seed_mode/par.L
        p += par.seed_amp_p*np.cos(kx*x)

    ts = np.linspace(0.0, par.t_final, par.n_save)
    n_t = np.empty((par.Nx, par.n_save))
    p_t = np.empty((par.Nx, par.n_save))
    n_t[:,0] = n; p_t[:,0] = p

    E_base = E_base_from_drift(n) if par.maintain_drift in ("field","feedback") else 0.0

    global dt_current
    dt_current = estimate_dt(n, p)
    print(f"[imex] Using dt={dt_current:.4e} with {n_threads} threads")

    E_full, E_half = linear_E_mats(dt_current)

    t = 0.0
    save_idx = 1
    next_tsave = ts[save_idx]

    n_prev = n.copy(); p_prev = p.copy()

    while t < par.t_final - 1e-14:
        n_eff = np.maximum(n, par.n_floor)
        u_mean = float(np.mean(p/(par.m*n_eff)))
        if par.maintain_drift == "feedback":
            E_eff = E_base + par.Kp*(par.u_d - u_mean)
        elif par.maintain_drift == "field":
            E_eff = E_base
        else:
            E_eff = 0.0

        dn = n - n0
        dp = p - p0

        dn_half, dp_half = apply_linear_step_fft(dn, dp, E_half)

        Nn1, Np1 = nonlinear_residual(dn_half, dp_half, E_eff)
        dn_e = dn_half + dt_current*Nn1
        dp_e = dp_half + dt_current*Np1
        Nn2, Np2 = nonlinear_residual(dn_e, dp_e, E_eff)
        dn_new = dn_half + 0.5*dt_current*(Nn1 + Nn2)
        dp_new = dp_half + 0.5*dt_current*(Np1 + Np2)

        dn_out, dp_out = apply_linear_step_fft(dn_new, dp_new, E_half)

        n = np.clip(n0 + dn_out, par.n_floor, par.n_cap)
        p = np.clip(p0 + dp_out, -par.p_cap, par.p_cap)
        n = spectral_viscosity(n, dt_current)
        p = spectral_viscosity(p, dt_current)

        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(p))):
            n[:] = n_prev; p[:] = p_prev
            dt_current *= 0.5
            if dt_current < par.dt_min:
                raise RuntimeError("dt collapsed below dt_min")
            E_full, E_half = linear_E_mats(dt_current)
            continue

        dt_target = estimate_dt(n, p)
        dt_current = max(par.dt_min, min(par.dt_max, 0.9*dt_current + 0.1*dt_target))
        E_full, E_half = linear_E_mats(dt_current)

        t_next = t + dt_current
        while save_idx < par.n_save and next_tsave <= t_next + 1e-14:
            alpha = (next_tsave - t)/dt_current
            n_t[:,save_idx] = (1-alpha)*n_t[:,save_idx-1] + alpha*n
            p_t[:,save_idx] = (1-alpha)*p_t[:,save_idx-1] + alpha*p
            save_idx += 1
            if save_idx < par.n_save:
                next_tsave = ts[save_idx]

        n_prev[:] = n; p_prev[:] = p
        t = t_next

    for s in range(save_idx, par.n_save):
        n_t[:,s] = n; p_t[:,s] = p
    return ts, n_t, p_t

def main():
    if _THREADPOOL_OK:
        with threadpool_limits(limits=n_threads):
            t, n_t, p_t = march_imex()
    else:
        t, n_t, p_t = march_imex()

    dn_t = n_t - par.nbar0
    extent=[x.min(), x.max(), t.min(), t.max()]

    os.makedirs(par.outdir, exist_ok=True)

    plt.figure(figsize=(11,4.2))
    plt.imshow(dn_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.xlabel("x"); plt.ylabel("t"); plt.title(r"$\delta n(x,t)=n-n_0$")
    plt.colorbar(label=r"$\delta n$")
    plt.tight_layout(); plt.savefig(f"{par.outdir}/spacetime_deltan.png", dpi=180); plt.close()

    plt.figure(figsize=(10,3.6))
    for frac in [0.0, 0.5, 1.0]:
        j = int(frac*(len(t)-1))
        plt.plot(x, dn_t[:,j], lw=1.6, label=f"t={t[j]:.2f}")
    plt.legend(); plt.xlabel("x"); plt.ylabel(r"$\delta n$")
    plt.title("Snapshots of δn")
    plt.tight_layout(); plt.savefig(f"{par.outdir}/snapshots_deltan.png", dpi=180); plt.close()

    print(f"[saved] {par.outdir}/spacetime_deltan.png, snapshots_deltan.png")

if __name__ == "__main__":
    main()