import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp
import os

@dataclass
class P:
    m: float = 1.0
    e: float = 1.0
    U: float = 0.03
    nbar0: float = 1.0
    Gamma0: float = 0.08
    w: float = 1.0
    include_poisson: bool = False
    eps: float = 20.0
    u_d: float = 0.10
    maintain_drift: str = "field"
    Kp: float = 0.15
    Dn: float = 0.00
    Dp: float = 0.06
    J0: float = 0.004
    sigma_J: float = 6.0
    x0: float = 60.0
    source_model: str = "as_given"
    use_nbar_gaussian: bool = False
    nbar_amp: float = 0.0
    nbar_sigma: float = 120.0
    L: float = 200.0
    Nx: int = 512
    t_final: float = 2000.0
    n_save: int = 360
    rtol: float = 5e-7
    atol: float = 5e-9
    n_floor: float = 1e-9
    dealias_23: bool = True
    seed_amp_n: float = 0.0
    seed_mode: int = 3
    seed_amp_p: float = 0.0
    outdir: str = "out_drift"
    cmap: str = "inferno"

par = P()

x = np.linspace(0.0, par.L, par.Nx, endpoint=False)
dx = x[1] - x[0]
k = 2*np.pi*fftfreq(par.Nx, d=dx)
ik = 1j*k
k2 = k**2

def Dx(f):  return (ifft(ik * fft(f))).real
def Dxx(f): return (ifft((-k2) * fft(f))).real

def filter_23(f):
    if not par.dealias_23: return f
    fh = fft(f)
    kc = par.Nx//3
    fh[kc:-kc] = 0.0
    return (ifft(fh)).real

def Gamma(n):
    return par.Gamma0 * np.exp(-np.maximum(n, par.n_floor)/par.w)

def Pi0(n):
    return 0.5 * par.U * n**2

def phi_from_n(n, nbar):
    rhs_hat = fft((par.e/par.eps) * (n - nbar))
    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)
    nz = (k2 != 0)
    phi_hat[nz] = rhs_hat[nz] / (-k2[nz])
    return (ifft(phi_hat)).real

def periodic_delta(xv, x0, L): return (xv - x0 + 0.5*L) % L - 0.5*L

def nbar_profile():
    if par.use_nbar_gaussian and par.nbar_amp != 0.0:
        d = periodic_delta(x, par.x0, par.L)
        return par.nbar0 + par.nbar_amp * np.exp(-0.5*(d/par.nbar_sigma)**2)
    else:
        return np.full_like(x, par.nbar0)

def pbar_profile(nbar):
    return par.m * nbar * par.u_d

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

def E_base_from_drift(nbar):
    return par.m * par.u_d * np.mean(Gamma(nbar)) / par.e

def rhs(t, y, E_base):
    N = par.Nx
    n = y[:N]
    p = y[N:]

    nbar = nbar_profile()
    n_eff = np.maximum(n, par.n_floor)

    Jx = J_profile()
    gamma = gamma_from_J(Jx)
    SJ = S_injection(n_eff, nbar, Jx, gamma)

    v = p/(par.m*n_eff)
    u_mean = float(np.mean(v))
    if par.maintain_drift == "feedback":
        E_eff = E_base + par.Kp * (par.u_d - u_mean)
    elif par.maintain_drift == "field":
        E_eff = E_base
    else:
        E_eff = 0.0

    dn_dt = -Dx(p) + (par.Dn * Dxx(n) + SJ)
    dn_dt = filter_23(dn_dt)

    Pi = Pi0(n_eff) + (p**2)/(par.m*n_eff)
    grad_Pi = Dx(Pi)
    force_Phi = 0.0
    if par.include_poisson:
        phi = phi_from_n(n_eff, nbar)
        force_Phi = n_eff * Dx(phi)

    dp_dt = -Gamma(n_eff)*p - grad_Pi + par.e*n_eff*E_eff - force_Phi + par.Dp * Dxx(p)
    dp_dt = filter_23(dp_dt)

    return np.concatenate([dn_dt, dp_dt])

def initial_fields():
    nbar = nbar_profile()
    pbar = pbar_profile(nbar)
    n0 = nbar.copy()
    p0 = pbar.copy()
    if par.seed_amp_n != 0.0 and par.seed_mode != 0:
        kx = 2*np.pi*par.seed_mode / par.L
        n0 += par.seed_amp_n * np.cos(kx * x)
    if par.seed_amp_p != 0.0 and par.seed_mode != 0:
        kx = 2*np.pi*par.seed_mode / par.L
        p0 += par.seed_amp_p * np.cos(kx * x)
    return n0, p0

def run_once(tag="drift"):
    os.makedirs(par.outdir, exist_ok=True)
    n0, p0 = initial_fields()
    E_base = E_base_from_drift(nbar_profile()) if par.maintain_drift in ("field","feedback") else 0.0

    y0 = np.concatenate([n0, p0])
    t_eval = np.linspace(0.0, par.t_final, par.n_save)

    sol = solve_ivp(lambda t,y: rhs(t,y,E_base),
                    (0.0, par.t_final), y0, t_eval=t_eval,
                    method="BDF", rtol=par.rtol, atol=par.atol)

    N = par.Nx
    n_t = sol.y[:N,:]
    p_t = sol.y[N:,:]

    n_eff_t = np.maximum(n_t, par.n_floor)
    v_t = p_t/(par.m*n_eff_t)
    print(f"[run]  <u>(t=0)={np.mean(v_t[:,0]):.4f},  <u>(t_end)={np.mean(v_t[:,-1]):.4f},  target u_d={par.u_d:.4f}")

    extent=[x.min(), x.max(), sol.t.min(), sol.t.max()]
    plt.figure(figsize=(9.6,4.3))
    plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.xlabel("x"); plt.ylabel("t"); plt.title(f"n(x,t)  [lab]  {tag}")
    plt.colorbar(label="n")
    plt.plot([par.x0, par.x0], [sol.t.min(), sol.t.max()], 'w--', lw=1, alpha=0.7)
    plt.tight_layout(); plt.savefig(f"{par.outdir}/spacetime_n_lab_{tag}.png", dpi=160); plt.close()

    return sol.t, n_t, p_t

def crop_xt(n_t, t, x, x_min=60.0, x_max=200.0, t_max=750.0):
    mask_t = (t < t_max)
    mask_x = (x >= x_min) & (x <= x_max)
    return x[mask_x], t[mask_t], n_t[mask_x][:, mask_t]

def find_crossings_x_linear(profile_x, level, x_grid):
    y = profile_x - level
    xs = []
    for i in range(len(x_grid) - 1):
        y1, y2 = y[i], y[i+1]
        if y1 == 0.0:
            xs.append(x_grid[i])
        if y1 * y2 < 0.0 or (y1 == 0.0 and y2 != 0.0):
            tloc = y1 / (y1 - y2)
            xs.append(x_grid[i] + tloc * (x_grid[i+1] - x_grid[i]))
    return np.array(xs) if len(xs) > 0 else np.array([])

def nonperiodic_nearest(target_x, candidates):
    if len(candidates) == 0:
        return None
    return int(np.argmin(np.abs(candidates - target_x)))

def track_iso_density(n_t, t, levels, anchors=None, x_win=(80.0,200.0), t_max=750.0):
    x_sub, t_sub, n_sub = crop_xt(n_t, t, x, x_min=x_win[0], x_max=x_win[1], t_max=t_max)
    Nt = n_sub.shape[1]

    if anchors is None:
        anchors = np.linspace(x_sub[0] + 0.1*(x_sub[-1]-x_sub[0]),
                              x_sub[-1] - 0.1*(x_sub[-1]-x_sub[0]),
                              len(levels))
    else:
        anchors = np.clip(np.asarray(anchors, dtype=float), x_sub[0], x_sub[-1])

    crossings_per_t = []
    for j in range(Nt):
        prof = n_sub[:, j]
        cs_j = {}
        for lev in levels:
            cs_j[lev] = find_crossings_x_linear(prof, lev, x_sub)
        crossings_per_t.append(cs_j)

    tracks = {}
    for lev, anchor in zip(levels, anchors):
        x_path = np.full(Nt, np.nan)

        c0 = crossings_per_t[0][lev]
        if len(c0) > 0:
            i0 = nonperiodic_nearest(anchor, c0)
            x_path[0] = c0[i0]

        for j in range(1, Nt):
            prev = x_path[j-1]
            if np.isnan(prev):
                prev = anchor
            cand = crossings_per_t[j][lev]
            idx = nonperiodic_nearest(prev, cand)
            if idx is not None:
                x_path[j] = cand[idx]
            else:
                x_path[j] = np.nan

        x_unw = x_path.copy()
        if np.any(np.isnan(x_unw)):
            good = ~np.isnan(x_unw)
            if good.sum() >= 2:
                x_unw = np.interp(np.arange(len(x_unw)), np.where(good)[0], x_unw[good])

        v_inst = np.gradient(x_unw, t_sub)

        tracks[lev] = dict(
            x_path=x_path,
            x_unwrap=x_unw,
            v_inst=v_inst,
            x_grid=x_sub,
            t_grid=t_sub
        )

    return tracks, (x_sub, t_sub, n_sub)

def plot_spacetime_with_tracks(n_sub, t_sub, x_sub, levels, tracks, tag="traj"):
    extent=[x_sub.min(), x_sub.max(), t_sub.min(), t_sub.max()]
    plt.figure(figsize=(10.0, 4.6))
    plt.imshow(n_sub.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.xlabel("x"); plt.ylabel("t"); plt.title("n(x,t) (cropped) with iso-density trajectories")
    plt.colorbar(label="n")

    colors = plt.cm.tab20(np.linspace(0, 1, len(levels)))
    for i, lev in enumerate(levels):
        t_curve = tracks[lev]["t_grid"]
        xp = tracks[lev]["x_path"]
        good = ~np.isnan(xp)
        plt.plot(xp[good], t_curve[good],
                 lw=1.2, color=colors[i],
                 label=f"n = {lev:.3f}", alpha=1.0)

    plt.legend(loc="upper right", fontsize=9, frameon=True)
    os.makedirs(par.outdir, exist_ok=True)
    fname = f"{par.outdir}/spacetime_n_with_tracks_{tag}.png"
    plt.tight_layout(); plt.savefig(fname, dpi=170); plt.close()
    print(f"[saved] {fname}")

def plot_velocities(levels, tracks, tag="traj"):
    vmin = 1e-6
    acc_thresh = 5e-3
    min_pts = 8

    plt.figure(figsize=(10.0, 4.4))
    colors = plt.cm.tab20(np.linspace(0, 1, len(levels)))

    for i, lev in enumerate(levels):
        t = tracks[lev]["t_grid"]
        x_unw = tracks[lev]["x_unwrap"].astype(float)

        good0 = ~np.isnan(x_unw)
        if good0.sum() < min_pts:
            continue

        v = np.gradient(x_unw, t)
        a = np.gradient(v, t)

        good = good0 & (v > vmin) & (np.abs(a) < acc_thresh)

        plt.plot(t[good0], v[good0], color=colors[i], lw=1.0, alpha=0.25)

        idx = np.flatnonzero(np.diff(good.astype(int)))
        seg_starts = np.r_[0, idx + 1]
        seg_ends   = np.r_[idx, len(good)-1]

        for s, e in zip(seg_starts, seg_ends):
            if not good[s]:
                continue
            if (e - s + 1) < min_pts:
                continue

            t_seg = t[s:e+1]
            x_seg = x_unw[s:e+1]

            coeff = np.polyfit(t_seg, x_seg, 1)
            v_ls = coeff[0]

            plt.hlines(v_ls, t_seg[0], t_seg[-1],
                       color=colors[i], lw=2.4, alpha=0.95)

        vls_list = []
        for s, e in zip(seg_starts, seg_ends):
            if not good[s]: continue
            if (e - s + 1) < min_pts: continue
            t_seg = t[s:e+1]; x_seg = x_unw[s:e+1]
            vls_list.append(np.polyfit(t_seg, x_seg, 1)[0])
        if len(vls_list) > 0:
            v_med = float(np.median(vls_list))
            plt.plot([], [], color=colors[i], lw=2.4,
                     label=rf"$n={lev:.3f},\, v_{{\rm LS}}\!\approx\!{v_med:.3f}$")
        else:
            plt.plot([], [], color=colors[i], lw=2.4,
                     label=rf"$n={lev:.3f},\, v_{{\rm LS}}\ {{\rm n/a}}$")

    plt.axhline(par.u_d, color="k", ls="--", alpha=0.6, label="drift $u_d$")
    plt.xlabel("t"); plt.ylabel("trajectory speed  $dx/dt$")
    plt.title("Least-squares segment speeds of iso-density trajectories")
    plt.legend(ncol=2, fontsize=9, frameon=False)
    os.makedirs(par.outdir, exist_ok=True)
    fname = f"{par.outdir}/velocities_iso_density_LS_{tag}.png"
    plt.tight_layout(); plt.savefig(fname, dpi=170); plt.close()
    print(f"[saved] {fname}")

if __name__ == "__main__":
    os.makedirs(par.outdir, exist_ok=True)

    par.maintain_drift = "feedback"
    par.include_poisson = False
    par.source_model = "as_given"

    t, n_t, p_t = run_once(tag="drift_run")

    x_sub_preview, t_sub_preview, n_sub_preview = crop_xt(n_t, t, x, x_min=60.0, x_max=200.0, t_max=750.0)

    qs = np.quantile(n_sub_preview, np.arange(0.4, 1.0, 0.05))
    levels = [float(q) for q in qs]
    print("[levels] chosen n levels:", ", ".join(f"{lv:.6f}" for lv in levels))

    tracks, (x_sub, t_sub, n_sub) = track_iso_density(n_t, t, levels, anchors=None,
                                                      x_win=(80.0,200.0), t_max=750.0)

    plot_spacetime_with_tracks(n_sub, t_sub, x_sub, levels, tracks, tag="drift_run")

    plot_velocities(levels, tracks, tag="drift_run")
