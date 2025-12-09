import sys
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

VERSION = "DS_AC_CLEAN_2025-12-09_v1"

def load_cfg():
    base = {
        "case_name": "ac_nonideal_Cs5_Cd1_clean",
        "boundary_mode": "nonideal",
        "Cs_fF": 5.0,
        "Cd_fF": 1.0,
        "L_nm": 110.0,
        "dx_nm": 2.0,
        "t_end_ps": 60.0,
        "store_every_ps": 0.02,
        "cfl": 0.35,
        "dt_s": 0.0,
        "W_um": 100.0,
        "d_nm": 20.0,
        "eps_r": 13.0,
        "m_eff_m0": 0.04,
        "tau_ps": 2.0,
        "n0_m2": 2.2e15,
        "v0_mps": 5.0e5,
        "excite_nm": 10.0,
        "excite_factor": 1.005,
        "noise_level": 5e-4,
        "seed": 0,
        "plot_time": True,
        "plot_spacetime": True,
        "plot_spacetime_norm": True,
        "plot_spectrum": False,
        "save_npz": True,
        "outdir": "ds_out",
        "debug_dump": False
    }
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) and sys.argv[1].lower().endswith(".json"):
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            user = json.load(f)
        if isinstance(user, dict):
            for k in base.keys():
                if k in user:
                    base[k] = user[k]
    return base

def consts():
    return (
        1.602176634e-19,
        8.8541878128e-12,
        1.054571817e-34,
        9.1093837015e-31
    )

def alpha_and_c2(e, eps0, hbar, m0, m_eff_m0, eps_r, d_nm, n0):
    m_eff = m_eff_m0 * m0
    d = d_nm * 1e-9
    CGC = eps_r * eps0 / d
    alpha = e * e / (2.0 * m_eff * CGC) + math.pi * hbar * hbar / (2.0 * m_eff * m_eff)
    c2 = 2.0 * alpha * n0
    return alpha, c2, CGC, m_eff

def upwind_grad(a, dx, v):
    g = np.empty_like(a)
    if v >= 0.0:
        g[0] = 0.0
        g[1:] = (a[1:] - a[:-1]) / dx
    else:
        g[-1] = 0.0
        g[:-1] = (a[1:] - a[:-1]) / dx
    return g

def apply_bc(dn, du, dn_prev, cfg, CGC, dt, n0, v0):
    mode = str(cfg["boundary_mode"]).lower()
    if mode == "ideal":
        dn[0] = 0.0
        du[-1] = (-v0 * dn[-1]) / n0
        return
    Cs = float(cfg["Cs_fF"]) * 1e-15
    Cd = float(cfg["Cd_fF"]) * 1e-15
    W = float(cfg["W_um"]) * 1e-6
    coef = 1.0 / (W * CGC * dt)
    djL = Cs * coef * (dn[0] - dn_prev[0])
    djR = -Cd * coef * (dn[-1] - dn_prev[-1])
    du[0] = (djL - v0 * dn[0]) / n0
    du[-1] = (djR - v0 * dn[-1]) / n0

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
    tau = float(cfg["tau_ps"]) * 1e-12
    eps_r = float(cfg["eps_r"])
    d_nm = float(cfg["d_nm"])
    m_eff_m0 = float(cfg["m_eff_m0"])

    alpha, c2, CGC, m_eff = alpha_and_c2(e, eps0, hbar, m0, m_eff_m0, eps_r, d_nm, n0)
    s = math.sqrt(max(c2, 0.0))

    if float(cfg.get("dt_s", 0.0)) > 0.0:
        dt = float(cfg["dt_s"])
    else:
        denom = abs(v0) + s
        if denom <= 0.0:
            denom = 1.0
        dt = float(cfg["cfl"]) * dx / denom

    t_end = float(cfg["t_end_ps"]) * 1e-12
    store_every = float(cfg["store_every_ps"]) * 1e-12

    steps = int(math.ceil(t_end / dt))
    if steps < 1:
        steps = 1

    store_stride = max(1, int(round(store_every / dt)))
    frames_est = steps // store_stride + 2

    x = (np.arange(N) + 0.5) * dx

    np.random.seed(int(cfg.get("seed", 0)))

    dn = np.zeros(N, dtype=np.float64)
    du = np.zeros(N, dtype=np.float64)

    L_exc = float(cfg.get("excite_nm", 0.0)) * 1e-9
    if L_exc > 0.0:
        mask = x >= (L - L_exc)
        fac = float(cfg.get("excite_factor", 1.0))
        dn[mask] += (fac - 1.0) * n0

    noise = float(cfg.get("noise_level", 0.0))
    if noise > 0.0:
        dn += noise * n0 * (np.random.rand(N) - 0.5)

    dn_prev = dn.copy()

    want_xt = bool(cfg.get("plot_spacetime", True) or cfg.get("save_npz", True))
    n_store = np.empty((frames_est, N), dtype=np.float64) if want_xt else None
    t_store = np.empty((frames_est,), dtype=np.float64) if want_xt else None

    t_trace = np.empty(steps, dtype=np.float64)
    J_trace = np.empty(steps, dtype=np.float64)

    frame_i = 0
    last_k = 0

    for k in range(steps):
        t = (k + 1) * dt

        apply_bc(dn, du, dn_prev, cfg, CGC, dt, n0, v0)

        dn_dx = upwind_grad(dn, dx, v0)
        du_dx = upwind_grad(du, dx, v0)

        dn_new = dn - dt * (v0 * dn_dx + n0 * du_dx)
        du_new = du - dt * (v0 * du_dx + (c2 / n0) * dn_dx + du / tau)

        dn_prev = dn
        dn = dn_new
        du = du_new

        apply_bc(dn, du, dn_prev, cfg, CGC, dt, n0, v0)

        djR = n0 * du[-1] + v0 * dn[-1]

        t_trace[k] = t
        J_trace[k] = -e * djR

        if want_xt and (k % store_stride == 0):
            if frame_i < frames_est:
                n_store[frame_i, :] = n0 + dn
                t_store[frame_i] = t
                frame_i += 1

        if not np.isfinite(dn).all() or not np.isfinite(du).all():
            last_k = k + 1
            break

        last_k = k + 1

    t_trace = t_trace[:last_k]
    J_trace = J_trace[:last_k]

    if want_xt:
        n_store = n_store[:frame_i, :]
        t_store = t_store[:frame_i]
    else:
        n_store = np.empty((0, N), dtype=np.float64)
        t_store = np.empty((0,), dtype=np.float64)

    return {
        "x": x, "L": L, "dx": dx, "dt": dt,
        "n0": n0, "v0": v0, "tau": tau,
        "alpha": alpha, "c2": c2, "s": s, "CGC": CGC,
        "t_trace": t_trace, "J_trace": J_trace,
        "t_store": t_store, "n_store": n_store,
        "frames": int(frame_i), "steps": int(last_k)
    }

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
    dt = float(data["dt"])
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
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(y.size, d=dt)
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
            dt_s=dt,
            t_ps=t_trace * 1e12,
            J_ac=J_trace,
            t_store_ps=t_store * 1e12,
            n_xt=n_store,
            params=cfg,
            derived={
                "n0": float(data["n0"]),
                "v0": float(data["v0"]),
                "tau_s": float(data["tau"]),
                "alpha": float(data["alpha"]),
                "c2": float(data["c2"]),
                "s": float(data["s"]),
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
                "dt": float(data["dt"]),
                "dx_nm": float(cfg["dx_nm"]),
                "t_end_ps": float(cfg["t_end_ps"])
            }], f, indent=2)

    if bool(cfg.get("debug_dump", False)) and n_store.size:
        idx_t = np.linspace(0, n_store.shape[0] - 1, min(10, n_store.shape[0])).astype(int)
        idx_x = np.linspace(0, n_store.shape[1] - 1, min(10, n_store.shape[1])).astype(int)
        print("n_store (showing 10 x values for 10 time steps):")
        for it in idx_t:
            vals = n_store[it, idx_x]
            print(f"  t={t_store[it]*1e12:.3f} ps: {vals}")

def main():
    cfg = load_cfg()
    print(VERSION)
    print(os.path.abspath(__file__))

    data = run(cfg)
    plot_and_save(cfg, data)

if __name__ == "__main__":
    main()
