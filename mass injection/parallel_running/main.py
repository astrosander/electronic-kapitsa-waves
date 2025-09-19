#!/usr/bin/env python3

import os
NTHREADS = int(os.environ.get("NTHREADS", os.cpu_count() or 1))
os.environ["OMP_NUM_THREADS"] = str(NTHREADS)
os.environ["MKL_NUM_THREADS"] = str(NTHREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NTHREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NTHREADS)

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
from scipy.fft import rfft, irfft, rfftfreq, set_workers
from scipy.integrate import solve_ivp

try:
    from threadpoolctl import threadpool_limits, threadpool_info
except Exception:
    class _NoOpCtx:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def threadpool_limits(*a, **k): return _NoOpCtx()
    def threadpool_info(): return []

@dataclass
class P:
    m: float = 1.0
    e: float = 1.0
    U: float = 0.5
    nbar0: float = 1.0
    Gamma0: float = 2.50
    w: float = 0.5
    include_poisson: bool = False
    eps: float = 20.0

    u_d: float = 0.0
    maintain_drift: str = "field"
    Kp: float = 0.15

    Dn: float = 1.0
    Dp: float = 1.0

    J0: float = 1.0
    sigma_J: float = 2.0**1/2
    x0: float = 5.0
    source_model: str = "as_given"

    L: float = 10.0 * np.pi
    Nx: int = 512

    t_final: float = 10.0
    n_save: int = 900
    rtol: float = 1e-3
    atol: float = 1e-7

    n_floor: float = 1e-7
    dealias_23: bool = True

    seed_amp_n: float = 2e-2
    seed_mode: int = 3
    seed_amp_p: float = 2e-2

    outdir: str = "out_fast"
    cmap: str = "inferno"

    frame_mode: str = "co_fixed"
    Uc: float = 50.0

par = P()

x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
kx = 2.0 * np.pi * rfftfreq(par.Nx, d=dx)
ik = 1j * kx
k2 = kx**2

if par.dealias_23:
    kc = par.Nx // 3
    mask_dealias = np.ones_like(kx, dtype=bool)
    mask_dealias[kc:] = False
else:
    mask_dealias = None

def periodic_delta(xv, x0, L): return (xv - x0 + 0.5*L) % L - 0.5*L

def nbar_profile():
    return np.full_like(x, par.nbar0)

def J_profile():
    d = periodic_delta(x, par.x0, par.L)
    return par.J0 * np.exp(-0.5*(d/par.sigma_J)**2)

def gamma_from_J(Jx):
    return np.trapz(Jx, x) / par.L

def S_injection(n, nbar, Jx, gamma):
    if par.source_model == "as_given":
        return Jx * nbar - gamma * (n - nbar)
    elif par.source_model == "balanced":
        return Jx * nbar - gamma * n
    else:
        raise ValueError("source_model must be 'as_given' or 'balanced'")

def Gamma(n):
    return par.Gamma0 * np.exp(-np.maximum(n, par.n_floor)/par.w)

def E_base_from_drift(nbar):
    return par.m * par.u_d * np.mean(Gamma(nbar)) / par.e

_ws = dict(
    Fn=np.zeros_like(kx, dtype=np.complex128),
    Fp=np.zeros_like(kx, dtype=np.complex128),
    Ftmp=np.zeros_like(kx, dtype=np.complex128),
    dx_n=np.empty(par.Nx, float),
    dx_p=np.empty(par.Nx, float),
    dxx_n=np.empty(par.Nx, float),
    dxx_p=np.empty(par.Nx, float),
    Pi=np.empty(par.Nx, float),
    gradPi=np.empty(par.Nx, float),
    Dx_phi=np.empty(par.Nx, float),
)

def Dx_from_F(F, out_real):
    _ws["Ftmp"][:] = ik * F
    if mask_dealias is not None:
        _ws["Ftmp"][~mask_dealias] = 0.0
    out_real[:] = irfft(_ws["Ftmp"], n=par.Nx)

def Dxx_from_F(F, out_real):
    _ws["Ftmp"][:] = -k2 * F
    if mask_dealias is not None:
        _ws["Ftmp"][~mask_dealias] = 0.0
    out_real[:] = irfft(_ws["Ftmp"], n=par.Nx)

def rhs(t, y, E_base):
    N = par.Nx
    n = y[:N]
    p = y[N:]

    nbar = nbar_profile()
    n_eff = np.maximum(n, par.n_floor)

    v = p/(par.m*n_eff)
    u_mean = float(np.mean(v))
    if par.maintain_drift == "feedback":
        E_eff = E_base + par.Kp * (par.u_d - u_mean)
    elif par.maintain_drift == "field":
        E_eff = E_base
    else:
        E_eff = 0.0

    _ws["Fn"][:] = rfft(n)
    _ws["Fp"][:] = rfft(p)
    if mask_dealias is not None:
        _ws["Fn"][~mask_dealias] = 0.0
        _ws["Fp"][~mask_dealias] = 0.0

    Dx_from_F(_ws["Fp"], _ws["dx_p"])
    Dx_from_F(_ws["Fn"], _ws["dx_n"])
    Dxx_from_F(_ws["Fn"], _ws["dxx_n"])
    Dxx_from_F(_ws["Fp"], _ws["dxx_p"])

    SJ = 0.0

    _ws["Pi"][:] = 0.5*par.U*(n_eff**2) + (p**2)/(par.m*n_eff)
    if par.frame_mode == "co_fixed":
        Pi_corr = _ws["Pi"] - par.Uc * p
    else:
        Pi_corr = _ws["Pi"]

    FPi = rfft(Pi_corr)
    if mask_dealias is not None:
        FPi[~mask_dealias] = 0.0
    Dx_from_F(FPi, _ws["gradPi"])

    force_Phi = 0.0
    if par.include_poisson:
        rhs_hat = rfft((par.e/par.eps) * (n_eff - nbar))
        phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
        nz = k2 != 0
        phi_hat[nz] = rhs_hat[nz] / (-k2[nz])
        _ws["Ftmp"][:] = ik * phi_hat
        if mask_dealias is not None:
            _ws["Ftmp"][~mask_dealias] = 0.0
        _ws["Dx_phi"][:] = irfft(_ws["Ftmp"], n=N)
        force_Phi = n_eff * _ws["Dx_phi"]

    if par.frame_mode == "co_fixed":
        adv_n = -(_ws["dx_p"] - par.Uc * _ws["dx_n"])
    else:
        adv_n = -_ws["dx_p"]

    dn_dt = adv_n + par.Dn * _ws["dxx_n"] + (SJ if np.ndim(SJ) else 0.0)
    dp_dt = -Gamma(n_eff)*p - _ws["gradPi"] + par.e*n_eff*E_eff - force_Phi + par.Dp * _ws["dxx_p"]

    return np.concatenate([dn_dt, dp_dt])

def initial_fields():
    n0 = nbar_profile().copy()
    p0 = (par.m * n0 * par.u_d).copy()
    if par.seed_amp_n != 0.0 and par.seed_mode != 0:
        kx0 = 2*np.pi*par.seed_mode / par.L
        n0 += par.seed_amp_n * np.cos(kx0 * x)
    if par.seed_amp_p != 0.0 and par.seed_mode != 0:
        kx0 = 2*np.pi*par.seed_mode / par.L
        p0 += par.seed_amp_p * np.cos(kx0 * x)
    return n0, p0

def run_once(tag="fast"):
    os.makedirs(par.outdir, exist_ok=True)
    n0, p0 = initial_fields()
    E_base = 15.0

    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)

    with threadpool_limits(limits=NTHREADS, user_api=("blas", "lapack", "openmp")):
        with set_workers(NTHREADS):
            sol = solve_ivp(lambda t,y: rhs(t,y,E_base),
                            (0.0, par.t_final), y0, t_eval=t_eval,
                            method="BDF", rtol=par.rtol, atol=par.atol,
                            dense_output=False)

    if not sol.success:
        print("[warn] solver did not converge:", sol.message)

    N = par.Nx
    n_t = sol.y[:N,:]
    p_t = sol.y[N:,:]

    n_eff_t = np.maximum(n_t, par.n_floor)
    v_t = p_t/(par.m*n_eff_t)
    print(f"[run]  <u>(t=0)={np.mean(v_t[:,0]):.4f},  <u>(t_end)={np.mean(v_t[:,-1]):.4f},  target u_d={par.u_d:.4f}")

    extent=[x.min(), x.max(), sol.t.min(), sol.t.max()]
    plt.figure(figsize=(10.0,4.2))
    plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.xlabel("x"); plt.ylabel("t"); plt.title(f"n(x,t)  [{tag}]")
    plt.colorbar(label="n"); plt.tight_layout()
    plt.savefig(f"{par.outdir}/spacetime_n_{tag}.png", dpi=160); plt.close()

    plt.figure(figsize=(11.0,3.6))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        j = int(frac*(len(sol.t)-1))
        plt.plot(x, n_t[:,j], lw=1.2, label=f"t={sol.t[j]:.2f}")
    plt.legend(); plt.xlabel("x"); plt.ylabel("n"); plt.title(f"Density snapshots  [{tag}]")
    plt.tight_layout(); plt.savefig(f"{par.outdir}/snapshots_n_{tag}.png", dpi=160); plt.close()

    return sol.t, n_t, p_t

def save_simulation_data(t, n_t, p_t, u_d_val, tag=""):
    os.makedirs(par.outdir, exist_ok=True)
    param_str = f"ud{u_d_val:.1f}_U{par.U:.2f}_G{par.Gamma0:.2f}_w{par.w:.2f}_Dn{par.Dn:.2f}_Dp{par.Dp:.2f}_L{par.L:.1f}_Nx{par.Nx}_tf{par.t_final:.1f}"
    filename_base = f"sim_data_{param_str}_{tag}" if tag else f"sim_data_{param_str}"
    data_file = f"{par.outdir}/{filename_base}.npz"
    np.savez_compressed(data_file, t=t, n_t=n_t, p_t=p_t, x=x, u_d=u_d_val)

    keys = ['U','Gamma0','w','nbar0','Dn','Dp','L','Nx','t_final','n_save','rtol','atol',
            'maintain_drift','include_poisson','source_model','seed_amp_n','seed_amp_p',
            'seed_mode','eps','e','m','J0','sigma_J','x0','dealias_23','n_floor','Kp',
            'frame_mode','Uc']
    metadata = {k: getattr(par,k) for k in keys}
    metadata['u_d'] = float(u_d_val)
    with open(f"{par.outdir}/{filename_base}_params.json",'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[save] Saved: {data_file}")
    return data_file

def run_all_ud_snapshots(tag="ud_panels"):
    os.makedirs(par.outdir, exist_ok=True)
    u_d_values = [0.9]
    results = []
    old_ud = par.u_d
    try:
        for ud in u_d_values:
            print(f"[run] u_d = {ud}")
            par.u_d = ud
            t, n_t, p_t = run_once(tag=f"ud{ud:.1f}")
            results.append((ud, t, n_t))
            save_simulation_data(t, n_t, p_t, ud, tag=f"ud{ud:.1f}")

        fig, axes = plt.subplots(len(u_d_values), 1, sharex=True,
                                 figsize=(10, 2.5*len(u_d_values)), constrained_layout=True)
        if not isinstance(axes, (list, np.ndarray)): axes = [axes]
        for ax, (ud, t, n_t) in zip(axes, results):
            ax.plot(x, n_t[:, -1], lw=1.2, label=f"t={t[-1]:.2f}")
            ax.legend(fontsize=8, loc="upper right")
            ax.set_ylabel(f"$u_d={ud:.1f}$")
        axes[-1].set_xlabel("x")
        plt.suptitle(f"Density snapshots  [{tag}]")
        outpath = f"{par.outdir}/snapshots_panels_{tag}.png"
        plt.savefig(outpath, dpi=160); plt.close()
        print(f"[plot] saved {outpath}")
    finally:
        par.u_d = old_ud

if __name__ == "__main__":
    os.makedirs(par.outdir, exist_ok=True)

    par.maintain_drift = "feedback"
    par.include_poisson = False
    par.source_model = "as_given"

    par.frame_mode = "co_fixed"
    par.Uc = 37.0

    run_all_ud_snapshots(tag="ud_comparison_fast")
