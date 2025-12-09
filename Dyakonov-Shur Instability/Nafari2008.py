import sys
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

VERSION = "DS_NONLINEAR_TVD_2025-12-09_v1"

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

def load_cfg():
    cfg = {
        "case_name": "nonlinear_ideal",
        "boundary_mode": "ideal",
        "Cs_fF": 5.0,
        "Cd_fF": 1.0,
        "dt_bc_s": 1e-17,
        "L_nm": 110.0,
        "dx_nm": 2.0,
        "t_end_ps": 30.0,
        "store_every_ps": 0.02,
        "cfl": 0.35,
        "dt_s": 0.0,
        "dt_max_s": 8e-16,
        "dt_min_s": 1e-18,
        "W_um": 100.0,
        "d_nm": 20.0,
        "eps_r": 13.0,
        "m_eff_m0": 0.04,
        "tau_ps": 2.0,
        "n0_m2": 2.2e15,
        "v0_mps": 5.0e5,
        "excite_nm": 10.0,
        "excite_factor": 1.02,
        "noise_level": 0.0,
        "seed": 0,
        "limiter": "minmod",
        "flux": "rusanov",
        "rk2": True,
        "plot_time": True,
        "plot_spacetime": True,
        "plot_spacetime_norm": True,
        "plot_spectrum": False,
        "save_npz": True,
        "outdir": "ds_out_nl",
        "debug_dump": False
    }
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) and sys.argv[1].lower().endswith(".json"):
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            user = json.load(f)
        if isinstance(user, dict):
            deep_update(cfg, user)
    return cfg

def consts():
    return (
        1.602176634e-19,
        8.8541878128e-12,
        1.054571817e-34,
        9.1093837015e-31
    )

def compute_alpha_and_CGC(cfg, e, eps0, hbar, m0):
    m_eff = float(cfg["m_eff_m0"]) * m0
    d = float(cfg["d_nm"]) * 1e-9
    eps_r = float(cfg["eps_r"])
    CGC = eps_r * eps0 / d
    alpha = e * e / (2.0 * m_eff * CGC) + math.pi * hbar * hbar / (2.0 * m_eff * m_eff)
    return alpha, CGC, m_eff

def minmod(a, b):
    out = np.zeros_like(a)
    same = (a * b) > 0.0
    out[same] = np.where(np.abs(a[same]) < np.abs(b[same]), a[same], b[same])
    return out

def vanleer(a, b):
    out = np.zeros_like(a)
    same = (a * b) > 0.0
    out[same] = (2.0 * a[same] * b[same]) / (a[same] + b[same])
    return out

def limiter_slopes(q, kind):
    N = q.size
    s = np.zeros_like(q)
    if N < 3:
        return s
    dm = q[1:-1] - q[:-2]
    dp = q[2:] - q[1:-1]
    if kind == "vanleer":
        s[1:-1] = vanleer(dm, dp)
    else:
        s[1:-1] = minmod(dm, dp)
    return s

def sound_speed(n, alpha):
    return np.sqrt(np.maximum(2.0 * alpha * n, 0.0))

def flux_components(n, j, alpha):
    u = j / n
    f1 = j
    f2 = j * u + alpha * n * n
    return f1, f2, u

def rusanov_flux(nL, jL, nR, jR, alpha):
    f1L, f2L, uL = flux_components(nL, jL, alpha)
    f1R, f2R, uR = flux_components(nR, jR, alpha)
    cL = sound_speed(nL, alpha)
    cR = sound_speed(nR, alpha)
    a = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)
    dn = nR - nL
    dj = jR - jL
    f1 = 0.5 * (f1L + f1R) - 0.5 * a * dn
    f2 = 0.5 * (f2L + f2R) - 0.5 * a * dj
    return f1, f2

def apply_bc_inplace(n, j, n_prev, cfg, CGC, dt, n0, j0):
    mode = str(cfg.get("boundary_mode", "ideal")).lower()
    if mode == "ideal":
        n[0] = n0
        j[-1] = j0
        return
    Cs = float(cfg.get("Cs_fF", 0.0)) * 1e-15
    Cd = float(cfg.get("Cd_fF", 0.0)) * 1e-15
    W = float(cfg.get("W_um", 1.0)) * 1e-6
    dt_bc = float(cfg.get("dt_bc_s", dt))
    coef = 1.0 / (W * CGC * dt_bc)
    j[0] = j0 + Cs * coef * (n[0] - n_prev[0])
    j[-1] = j0 - Cd * coef * (n[-1] - n_prev[-1])

def compute_dt(n, j, alpha, dx, cfg):
    if float(cfg.get("dt_s", 0.0)) > 0.0:
        return float(cfg["dt_s"])
    u = j / n
    c = sound_speed(n, alpha)
    s = np.abs(u) + c
    smax = float(np.max(s)) if s.size else 1.0
    if not np.isfinite(smax) or smax <= 0.0:
        smax = 1.0
    dt = float(cfg["cfl"]) * dx / smax
    dt = min(dt, float(cfg.get("dt_max_s", dt)))
    dt = max(dt, float(cfg.get("dt_min_s", 0.0)))
    return dt

def rhs(n, j, alpha, dx, tau_s, limiter_kind, flux_kind):
    N = n.size
    dn_dt = np.zeros_like(n)
    dj_dt = np.zeros_like(j)
    if N < 3:
        return dn_dt, dj_dt

    sn = limiter_slopes(n, limiter_kind)
    sj = limiter_slopes(j, limiter_kind)

    nL = n[:-1] + 0.5 * sn[:-1]
    jL = j[:-1] + 0.5 * sj[:-1]
    nR = n[1:] - 0.5 * sn[1:]
    jR = j[1:] - 0.5 * sj[1:]

    n_floor = 1e-12 * float(np.max([1.0, np.median(n)]))
    nL = np.maximum(nL, n_floor)
    nR = np.maximum(nR, n_floor)

    if flux_kind == "rusanov":
        f1, f2 = rusanov_flux(nL, jL, nR, jR, alpha)
    else:
        f1, f2 = rusanov_flux(nL, jL, nR, jR, alpha)

    dn_dt[1:-1] = -(f1[1:] - f1[:-1]) / dx
    dj_dt[1:-1] = -(f2[1:] - f2[:-1]) / dx - j[1:-1] / tau_s

    return dn_dt, dj_dt

def run(cfg):
    e, eps0, hbar, m0 = consts()

    L = float(cfg["L_nm"]) * 1e-9
    dx = float(cfg["dx_nm"]) * 1e-9
    N = int(round(L / dx))
    if N < 20:
        N = 20
    L = N * dx
    dx = L / N

    n0 = float(cfg["n0_m2"])
    v0 = float(cfg["v0_mps"])
    tau_s = float(cfg["tau_ps"]) * 1e-12
    j0 = n0 * v0

    alpha, CGC, m_eff = compute_alpha_and_CGC(cfg, e, eps0, hbar, m0)

    x = (np.arange(N) + 0.5) * dx

    np.random.seed(int(cfg.get("seed", 0)))
    n = np.full(N, n0, dtype=np.float64)
    j = np.full(N, j0, dtype=np.float64)

    L_exc = float(cfg.get("excite_nm", 0.0)) * 1e-9
    if L_exc > 0.0:
        mask = x >= (L - L_exc)
        fac = float(cfg.get("excite_factor", 1.0))
        n[mask] = fac * n[mask]

    noise = float(cfg.get("noise_level", 0.0))
    if noise > 0.0:
        n *= (1.0 + noise * (np.random.rand(N) - 0.5))
        j *= (1.0 + noise * (np.random.rand(N) - 0.5))

    n_prev = n.copy()

    t_end = float(cfg["t_end_ps"]) * 1e-12
    store_every = float(cfg["store_every_ps"]) * 1e-12

    t = 0.0
    next_store_t = 0.0

    n_store = []
    t_store = []
    J_trace = []
    t_trace = []

    limiter_kind = str(cfg.get("limiter", "minmod")).lower()
    flux_kind = str(cfg.get("flux", "rusanov")).lower()
    use_rk2 = bool(cfg.get("rk2", True))

    max_steps = int(cfg.get("max_steps", 5000000))
    steps = 0

    while t < t_end and steps < max_steps:
        steps += 1

        dt = compute_dt(n, j, alpha, dx, cfg)
        if t + dt > t_end:
            dt = t_end - t
        if dt <= 0.0:
            break

        apply_bc_inplace(n, j, n_prev, cfg, CGC, dt, n0, j0)

        dn0_dt, dj0_dt = rhs(n, j, alpha, dx, tau_s, limiter_kind, flux_kind)

        if use_rk2:
            n1 = n + dt * dn0_dt
            j1 = j + dt * dj0_dt
            apply_bc_inplace(n1, j1, n_prev, cfg, CGC, dt, n0, j0)

            dn1_dt, dj1_dt = rhs(n1, j1, alpha, dx, tau_s, limiter_kind, flux_kind)

            n_new = 0.5 * n + 0.5 * (n1 + dt * dn1_dt)
            j_new = 0.5 * j + 0.5 * (j1 + dt * dj1_dt)
        else:
            n_new = n + dt * dn0_dt
            j_new = j + dt * dj0_dt

        apply_bc_inplace(n_new, j_new, n_prev, cfg, CGC, dt, n0, j0)

        if not np.isfinite(n_new).all() or not np.isfinite(j_new).all():
            break

        n_prev = n
        n = n_new
        j = j_new

        t += dt

        J_ac = -e * (j[-1] - j0)
        J_trace.append(J_ac)
        t_trace.append(t)

        if t >= next_store_t:
            n_store.append(n.copy())
            t_store.append(t)
            next_store_t += store_every

    n_store = np.array(n_store, dtype=np.float64) if len(n_store) else np.empty((0, N))
    t_store = np.array(t_store, dtype=np.float64) if len(t_store) else np.empty((0,))
    t_trace = np.array(t_trace, dtype=np.float64) if len(t_trace) else np.empty((0,))
    J_trace = np.array(J_trace, dtype=np.float64) if len(J_trace) else np.empty((0,))

    return {
        "x": x, "L": L, "dx": dx, "alpha": alpha, "CGC": CGC,
        "n0": n0, "v0": v0, "j0": j0, "tau_s": tau_s,
        "t_trace": t_trace, "J_trace": J_trace,
        "t_store": t_store, "n_store": n_store,
        "steps": int(t_trace.size), "frames": int(t_store.size)
    }

def debug_print_10x10(data):
    n_store = data["n_store"]
    t_store = data["t_store"]
    if n_store.size == 0 or t_store.size == 0:
        return
    Nt, Nx = n_store.shape
    ti = np.linspace(0, Nt - 1, min(10, Nt)).astype(int)
    start = int(0.7 * (Nx - 1))
    xi = np.linspace(start, Nx - 1, min(10, Nx)).astype(int)
    print("n_store (showing 10 x values for 10 time steps):")
    for it in ti:
        vals = n_store[it, xi]
        print(f"  t={t_store[it]*1e12:.3f} ps: {vals}")

def plot_and_save(cfg, data):
    outdir = str(cfg.get("outdir", "")).strip()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "config_used.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    label = str(cfg.get("case_name", "case"))
    x = data["x"]
    L = data["L"]
    n0 = data["n0"]
    t_trace = data["t_trace"]
    J_trace = data["J_trace"]
    t_store = data["t_store"]
    n_store = data["n_store"]

    if bool(cfg.get("plot_time", True)) and t_trace.size:
        plt.figure()
        plt.plot(t_trace * 1e12, J_trace)
        plt.xlabel("Time (ps)")
        plt.ylabel("J_ac (A/m)")
        plt.title(label)
        if outdir:
            plt.savefig(os.path.join(outdir, f"J_time_{label}.png"), dpi=200, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    if bool(cfg.get("plot_spacetime", True)) and n_store.size and t_store.size >= 2:
        plt.figure()
        extent = [0.0, L * 1e9, t_store[0] * 1e12, t_store[-1] * 1e12]
        if bool(cfg.get("plot_spacetime_norm", True)):
            plt.imshow(n_store / n0, aspect="auto", origin="lower", extent=extent)
            plt.title(f"n(x,t)/n0 {label}")
        else:
            plt.imshow(n_store, aspect="auto", origin="lower", extent=extent)
            plt.title(f"n(x,t) {label}")
        plt.xlabel("x (nm)")
        plt.ylabel("t (ps)")
        if outdir:
            plt.savefig(os.path.join(outdir, f"n_spacetime_{label}.png"), dpi=200, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    if bool(cfg.get("plot_spectrum", False)) and J_trace.size > 256:
        y = J_trace - np.mean(J_trace)
        if t_trace.size > 1:
            dt_eff = float(np.mean(np.diff(t_trace)))
        else:
            dt_eff = 1.0
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(y.size, d=dt_eff)
        plt.figure()
        plt.plot(f * 1e-12, np.abs(Y))
        plt.xlabel("Frequency (THz)")
        plt.ylabel("|FFT(J_ac)|")
        plt.title(label)
        if outdir:
            plt.savefig(os.path.join(outdir, f"J_fft_{label}.png"), dpi=200, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    if bool(cfg.get("save_npz", True)) and outdir:
        np.savez(
            os.path.join(outdir, f"data_{label}.npz"),
            x_nm=x * 1e9,
            t_ps=t_trace * 1e12,
            J_ac=J_trace,
            t_store_ps=t_store * 1e12,
            n_xt=n_store,
            params=cfg,
            derived={
                "n0": float(data["n0"]),
                "v0": float(data["v0"]),
                "j0": float(data["j0"]),
                "tau_s": float(data["tau_s"]),
                "alpha": float(data["alpha"]),
                "CGC": float(data["CGC"])
            }
        )

    if outdir:
        with open(os.path.join(outdir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump([{
                "label": label,
                "version": VERSION,
                "steps": int(data["steps"]),
                "frames": int(data["frames"]),
                "dx_nm": float(cfg["dx_nm"]),
                "t_end_ps": float(cfg["t_end_ps"]),
                "boundary_mode": str(cfg.get("boundary_mode", "ideal")),
                "limiter": str(cfg.get("limiter", "minmod")),
                "flux": str(cfg.get("flux", "rusanov")),
                "rk2": bool(cfg.get("rk2", True))
            }], f, indent=2)

def main():
    cfg = load_cfg()
    print(VERSION)
    print(os.path.abspath(__file__))

    data = run(cfg)

    if bool(cfg.get("debug_dump", False)):
        debug_print_10x10(data)

    plot_and_save(cfg, data)

if __name__ == "__main__":
    main()
