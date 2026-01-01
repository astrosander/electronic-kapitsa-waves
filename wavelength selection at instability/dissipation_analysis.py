import numpy as np
from scipy.fft import fft


def Gamma(n, Gamma0, w, n_floor):
    return Gamma0 * np.exp(-np.maximum(n, n_floor) / w)


def Gamma_spatial(n, Gamma0, w, n_floor, L, lambda_diss, sigma_diss, x0):
    Gamma_base = Gamma(n, Gamma0, w, n_floor)
    if lambda_diss != 0.0:
        x_local = np.linspace(0.0, L, len(n), endpoint=False)
        d = periodic_delta(x_local, x0, L)
        perturbation = lambda_diss * np.exp(-0.5 * (d / sigma_diss)**2)
        return Gamma_base + perturbation
    else:
        return Gamma_base


def periodic_delta(x, x0, L):
    return (x - x0 + 0.5*L) % L - 0.5*L


def E_base_from_drift(nbar, m, u_d, e, Gamma0, w, n_floor):
    return m * u_d * np.mean(Gamma(nbar, Gamma0, w, n_floor)) / e


def E_eff_from_snapshot(n_snap, p_snap, E_base, m, n_floor, maintain_drift, u_d, Kp):
    n_eff = np.maximum(n_snap, n_floor)
    u_mean = float(np.mean(p_snap / (m * n_eff)))
    if maintain_drift == "feedback":
        return E_base + Kp * (u_d - u_mean)
    else:
        return E_base


def sigma_profile(n_snap, e, m, n_floor, Gamma0, w, L, lambda_diss, sigma_diss, x0):
    n_eff = np.maximum(n_snap, n_floor)
    gamma = Gamma_spatial(n_eff, Gamma0, w, n_floor, L, lambda_diss, sigma_diss, x0)
    return (e**2 / m) * (n_eff / gamma)


def compute_W_series(t, n_t, p_t, E_base, meta):
    nt = n_t.shape[1]
    E_t = np.empty(nt)
    sigma_t = np.empty(nt)
    W_t = np.empty(nt)
    
    m = meta['m']
    e = meta['e']
    n_floor = meta['n_floor']
    maintain_drift = meta['maintain_drift']
    u_d = meta['u_d']
    Kp = meta.get('Kp', 0.15)
    Gamma0 = meta['Gamma0']
    w = meta['w']
    L = meta.get('L', 10.0)
    lambda_diss = meta.get('lambda_diss', 0.0)
    sigma_diss = meta.get('sigma_diss', 2.0)
    x0 = meta.get('x0', 10.0)
    
    for j in range(nt):
        n_snap = n_t[:, j]
        p_snap = p_t[:, j]
        E = E_eff_from_snapshot(n_snap, p_snap, E_base, m, n_floor, 
                                maintain_drift, u_d, Kp)
        sig_bar = float(np.mean(sigma_profile(n_snap, e, m, n_floor, Gamma0, w, 
                                               L, lambda_diss, sigma_diss, x0)))
        E_t[j] = E
        sigma_t[j] = sig_bar
        W_t[j] = (E**2) * sig_bar
    
    return E_t, sigma_t, W_t


def estimate_period_time_fft(t, signal, frac_tail=0.5):
    n0 = int(len(t) * (1.0 - frac_tail))
    tt = t[n0:]
    ss = signal[n0:] - np.mean(signal[n0:])
    dt = float(tt[1] - tt[0])
    
    freqs = np.fft.rfftfreq(len(tt), d=dt)
    S = np.abs(np.fft.rfft(ss))
    
    if len(S) < 3:
        return np.nan, np.nan
    k = 1 + np.argmax(S[1:])
    f0 = freqs[k]
    if f0 <= 0:
        return np.nan, np.nan
    return 1.0 / f0, f0


def _power_spectrum_1d(n_slice, L):
    N = n_slice.size
    dn = n_slice - np.mean(n_slice)
    nhat = fft(dn)
    P = (nhat * np.conj(nhat)).real / (N*N)
    m = np.arange(N//2 + 1)
    kpos = 2*np.pi*m / L
    return kpos[1:], P[1:N//2+1]


def estimate_velocity_fourier(n_t1, n_t2, t1, t2, L, power_floor=1e-3):
    N = n_t1.size
    
    f1 = n_t1 - n_t1.mean()
    f2 = n_t2 - n_t2.mean()
    
    k = 2*np.pi * np.fft.rfftfreq(N, d=L/N)
    F1 = np.fft.rfft(f1)
    F2 = np.fft.rfft(f2)
    C = np.conj(F1) * F2
    phi = np.angle(C)
    
    k = k[1:]
    phi = phi[1:]
    w = np.abs(C[1:])
    mask = w > (power_floor * w.max())
    k, phi, w = k[mask], phi[mask], w[mask]
    
    phi = np.unwrap(phi)
    
    num = np.sum(w * k * phi)
    den = np.sum(w * k**2)
    shift = num / den
    
    shift = -shift
    shift = (shift + 0.5*L) % L - 0.5*L
    
    dt = float(t2 - t1)
    u = shift / dt
    return u, shift


def time_average_over_last_period(t, W_t, T):
    t_end = float(t[-1])
    t0 = t_end - float(T)
    mask = t >= t0
    if np.count_nonzero(mask) < 2:
        return np.nan
    return np.trapz(W_t[mask], t[mask]) / (t[mask][-1] - t[mask][0])


def compute_dissipation_from_npz(npz_file, method='time_fft', frac_tail=0.5):
    data = np.load(npz_file, allow_pickle=True)
    files = list(data.files)
    
    print(f"[Debug] Available keys in {npz_file}: {files}")
    
    if 'n_t' in files and 'p_t' in files:
        t = data['t']
        n_t = data['n_t']
        p_t = data['p_t']
        L = float(data['L'])
        meta = data['meta'].item() if hasattr(data['meta'], 'item') else data['meta']
        has_p = True
        
    elif all(k in files for k in ('n', 'p', 'x', 't')):
        n = data['n']
        p = data['p']
        x = data['x']
        t = data['t']
        
        if n.shape[0] == len(t):
            n_t = n.T
            p_t = p.T
        else:
            n_t = n
            p_t = p
        
        L = float(data['L']) if 'L' in files else (x[-1] - x[0] + (x[1] - x[0]) if len(x) > 1 else 10.0)
        meta = data['meta'].item() if 'meta' in files and hasattr(data['meta'], 'item') else (data['meta'] if 'meta' in files else {})
        has_p = True
        
    elif all(k in files for k in ('n', 'x', 't')):
        n = data['n']
        x = data['x']
        t = data['t']
        
        if n.shape[0] == len(t):
            n_t = n.T
        else:
            n_t = n
        
        p_t = np.zeros_like(n_t)
        L = float(data['L']) if 'L' in files else (x[-1] - x[0] + (x[1] - x[0]) if len(x) > 1 else 10.0)
        meta = data['meta'].item() if 'meta' in files and hasattr(data['meta'], 'item') else (data['meta'] if 'meta' in files else {})
        has_p = False
        
    else:
        raise ValueError(f"Unsupported npz format. Available keys: {files}")
    
    if not meta or len(meta) == 0:
        meta = {}
        print(f"[Warning] No metadata found in npz file. Using default parameters.")
    
    defaults = {
        'm': 1.0, 'e': 1.0, 'u_d': 5.245, 'nbar0': 0.2,
        'Gamma0': 2.5, 'w': 0.04, 'n_floor': 1e-4,
        'maintain_drift': 'field', 'Kp': 0.15,
        'lambda_diss': 0.0, 'sigma_diss': 2.0, 'x0': 10.0,
        'U': 1.0
    }
    for key, default_val in defaults.items():
        if key not in meta:
            meta[key] = default_val
    
    if 'L' not in meta:
        meta['L'] = L
    
    nbar0 = meta['nbar0']
    Nx = n_t.shape[0]
    nbar = np.full(Nx, nbar0)
    
    E_base = E_base_from_drift(nbar, meta['m'], meta['u_d'], meta['e'],
                                meta['Gamma0'], meta['w'], meta['n_floor'])
    
    if has_p:
        E_t, sigma_t, W_t = compute_W_series(t, n_t, p_t, E_base, meta)
    else:
        E_t, sigma_t, W_t = compute_W_series(t, n_t, p_t, E_base, meta)
        E_t[:] = E_base
    
    results = {
        'E_t': E_t,
        'sigma_t': sigma_t,
        'W_t': W_t,
        't': t,
        'E_base': E_base,
    }
    
    if method == 'time_fft':
        i_probe = Nx // 2
        T, f = estimate_period_time_fft(t, n_t[i_probe, :], frac_tail=frac_tail)
        results['T'] = T
        results['f'] = f
        results['method'] = 'time_fft'
        
    elif method == 'washboard':
        kpos, Pk = _power_spectrum_1d(n_t[:, -1], L)
        k_peak = float(kpos[np.argmax(Pk)])
        lam = 2*np.pi / k_peak
        
        idx_t1 = -5 if len(t) >= 5 else 0
        idx_t2 = -1
        u_drift, shift_opt = estimate_velocity_fourier(
            n_t[:, idx_t1], n_t[:, idx_t2], t[idx_t1], t[idx_t2], L
        )
        
        T = lam / abs(u_drift) if abs(u_drift) > 1e-10 else np.nan
        f = abs(u_drift) / lam if lam > 1e-10 else np.nan
        
        results['T'] = T
        results['f'] = f
        results['lambda'] = lam
        results['u_drift'] = u_drift
        results['k_peak'] = k_peak
        results['method'] = 'washboard'
    
    if not np.isnan(results['T']):
        W_avg = time_average_over_last_period(t, W_t, results['T'])
        results['W_avg'] = W_avg
    else:
        results['W_avg'] = np.nan
    
    j_t = (meta['e'] / meta['m']) * np.mean(p_t, axis=0)
    P_Ej = E_t * j_t
    results['j_t'] = j_t
    results['P_Ej_mean'] = np.mean(P_Ej)
    results['W_t_mean'] = np.mean(W_t)
    
    return results


def print_dissipation_summary(results):
    print("\n" + "="*60)
    print("JOULE DISSIPATION ANALYSIS")
    print("="*60)
    
    if results['method'] == 'time_fft':
        print(f"Method: Time FFT (probe point)")
        print(f"  Dominant frequency: f ≈ {results['f']:.6g} Hz")
        print(f"  Period: T ≈ {results['T']:.6g}")
    elif results['method'] == 'washboard':
        print(f"Method: Washboard (T = λ/u)")
        print(f"  k_peak: {results['k_peak']:.6g}")
        print(f"  Wavelength: λ = {results['lambda']:.6g}")
        print(f"  Drift velocity: u = {results['u_drift']:.6g}")
        print(f"  Period: T = {results['T']:.6g}")
        print(f"  Frequency: f = {results['f']:.6g} Hz")
    
    print(f"\nBase electric field: E_base = {results['E_base']:.6g}")
    
    t = results['t']
    sigma_t = results['sigma_t']
    W_t = results['W_t']
    
    n_tail = int(0.2 * len(t))
    if n_tail > 0:
        sigma_tail = np.mean(sigma_t[-n_tail:])
        W_tail = np.mean(W_t[-n_tail:])
        print(f"\nTime averages over last 20% of time:")
        print(f"  <sigma>_x(t) = {sigma_tail:.6g}")
        print(f"  $\\langle\\sigma\\rangle_x(t)$ = {sigma_tail:.6g}")
        print(f"  W(t)=E(t)^2<sigma>_x = {W_tail:.6g}")
        print(f"  $W(t)=E(t)^2\\langle\\sigma\\rangle_x$ = {W_tail:.6g}")
    
    print(f"\nDissipation metrics:")
    print(f"  <W(t)>_time = {results['W_t_mean']:.6g}")
    print(f"  <E^2 * σ>_T = {results['W_avg']:.6g}")
    print(f"\nSanity check:")
    print(f"  <E*j>_time = {results['P_Ej_mean']:.6g}")
    print(f"  (should be similar to <W> for Drude model)")
    print("="*60 + "\n")


def plot_dissipation_diagnostics(results, n_t, meta, L, x0_label="", tail_window=None):
    import matplotlib.pyplot as plt
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["legend.frameon"] = False
    # Publication-ready font sizes
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['figure.titlesize'] = 16

    t = results["t"]
    E_t = results["E_t"]
    sigma_t = results["sigma_t"]
    W_t = results["W_t"]
    T = results.get("T", np.nan)

    if tail_window is not None:
        if np.isfinite(T) and T > 0:
            t0 = t[-1] - tail_window*T
        else:
            t0 = t[int((1.0 - tail_window)*len(t))]
        mask = t >= t0
    else:
        mask = np.ones(len(t), dtype=bool)
    
    tt = t[mask]

    fig, ax = plt.subplots(1, 1, figsize=(9, 4))

    ax.plot(tt, sigma_t[mask], label=r"$\langle\sigma\rangle_x(t)$")
    ax.plot(tt, W_t[mask], label=r"$W(t)=E(t)^2\langle\sigma\rangle_x$")
    ax.set_ylabel("amplitude")
    ax.set_xlabel("$t$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if x0_label:
        base_name = x0_label.replace('.npz', '').replace('\\', '_').replace('/', '_')
        fig.savefig(f"dissipation_diagnostics_{base_name}.svg", dpi=300, bbox_inches='tight')
    else:
        fig.savefig("dissipation_diagnostics.png", dpi=300, bbox_inches='tight')
    
    return fig


def plot_local_dissipation_heatmap(results, n_t, meta, L):
    import matplotlib.pyplot as plt
    
    t = results["t"]
    E_t = results["E_t"]
    Nx = n_t.shape[0]
    x = np.linspace(0, L, Nx, endpoint=False)

    w_xt = np.empty_like(n_t)
    for j in range(n_t.shape[1]):
        sig_x = sigma_profile(
            n_t[:, j],
            e=meta["e"], m=meta["m"], n_floor=meta["n_floor"],
            Gamma0=meta["Gamma0"], w=meta["w"],
            L=L, lambda_diss=meta.get("lambda_diss",0.0),
            sigma_diss=meta.get("sigma_diss",2.0), x0=meta.get("x0",10.0)
        )
        w_xt[:, j] = (E_t[j]**2) * sig_x

    fig = plt.figure(figsize=(9,4.8))
    plt.imshow(
        w_xt.T, origin="lower", aspect="auto",
        extent=[x.min(), x.max(), t.min(), t.max()]
    )
    plt.colorbar(label="$w(x,t)=E(t)^2 \\sigma(x,t)$")
    plt.xlabel("$x$")
    plt.ylabel("$t$")
    # plt.title("Local dissipation density")
    plt.tight_layout()
    return fig


def plot_final_density_profile(results, n_t, L, x0_label=""):
    import matplotlib.pyplot as plt
    
    t = results["t"]
    Nx = n_t.shape[0]
    x = np.linspace(0, L, Nx, endpoint=False)
    
    n_final = n_t[:, -1]
    t_final = t[-1]
    
    T = results.get("T", np.nan)
    
    fig = plt.figure(figsize=(9, 5))
    plt.plot(x, n_final, linewidth=2, label="$n(x)$")
    
    if np.isfinite(T) and T > 0:
        if 'lambda' in results and np.isfinite(results['lambda']):
            lam = results['lambda']
        elif 'k_peak' in results and np.isfinite(results['k_peak']):
            lam = 2*np.pi / results['k_peak']
        else:
            lam = None
        
        if lam is not None and np.isfinite(lam) and lam > 0:
            k = 2*np.pi / lam
            n_mean = np.mean(n_final)
            n_amp = np.std(n_final)
            sine_wave = n_mean + n_amp * np.sin(k * x)
            plt.plot(x, sine_wave, '--', linewidth=2, alpha=0.7, 
                    label=rf"$\sin(2\pi x/\lambda)$, $\lambda={lam:.4g}$, $T={T:.4g}$")
    
    plt.xlabel("$x$")
    plt.ylabel("$n(x)$")
    # plt.title(f"{x0_label}  Density profile at t = {t_final:.4g}")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if x0_label:
        base_name = x0_label.replace('.npz', '').replace('\\', '_').replace('/', '_')
        fig.savefig(f"final_density_profile_{base_name}.svg", bbox_inches='tight')
        fig.savefig(f"final_density_profile_{base_name}.png", dpi=300, bbox_inches='tight')
    else:
        fig.savefig("final_density_profile.svg", bbox_inches='tight')
        fig.savefig("final_density_profile.png", dpi=300, bbox_inches='tight')
    
    return fig


def predicted_uc(meta, n_ref=None):
    m = meta.get("m", 1.0)
    U = meta.get("U", 1.0)
    w = meta.get("w", 0.04)
    n = float(n_ref if n_ref is not None else meta.get("nbar0", 0.2))
    n = max(n, 1e-30)
    return w * np.sqrt(U / (m * n))


def predicted_W_drude(u_d, meta, n_ref=None):
    m = meta.get("m", 1.0)
    Gamma0 = meta.get("Gamma0", 2.5)
    w = meta.get("w", 0.04)
    n_floor = meta.get("n_floor", 1e-4)
    n = float(n_ref if n_ref is not None else meta.get("nbar0", 0.2))
    gam = Gamma(n, Gamma0, w, n_floor)
    return m * n * gam * (u_d ** 2)


def fit_onset_coeff(u_d, W_meas, W0, u_c, nfit=6):
    u_d = np.asarray(u_d, float)
    W_meas = np.asarray(W_meas, float)
    W0 = np.asarray(W0, float)

    mask = np.isfinite(u_d) & np.isfinite(W_meas) & np.isfinite(W0) & (u_d > u_c)
    if np.count_nonzero(mask) < 3:
        return np.nan

    uu = u_d[mask]
    eps = uu - u_c
    y = (W_meas[mask] - W0[mask]) / (uu**2 + 1e-30)

    idx = np.argsort(eps)
    eps = eps[idx][:nfit]
    y = y[idx][:nfit]

    den = float(np.dot(eps, eps))
    if den <= 0:
        return np.nan
    a = float(np.dot(eps, y) / den)
    return a


def plot_W_vs_u_d(u_d_values, W_values, meta_ref=None, n_ref=None, output_file="W_vs_u_d", also_plot_onset_test=True):
    import matplotlib.pyplot as plt
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["legend.frameon"] = False
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['figure.titlesize'] = 16
    
    u = np.asarray(u_d_values, float)
    W = np.asarray(W_values, float)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(u, W, linewidth=2,color="blue", label=r"measured $W$")

    if meta_ref is not None:
        u_c = predicted_uc(meta_ref, n_ref=n_ref)
        W0 = np.array([predicted_W_drude(ui, meta_ref, n_ref=n_ref) for ui in u])

        plt.plot(u, W0, linestyle="--", linewidth=2,
                 label=r"$W_0=m n\gamma(n)\,u_d^2$")

        plt.axvline(u_c, linestyle="--", color="red", linewidth=2,
                    label=rf"$u_c$")

        a = fit_onset_coeff(u, W, W0, u_c)
        if np.isfinite(a):
            W_onset = W0 + a * (u**2) * np.maximum(u - u_c, 0.0)
            plt.plot(u, W_onset, linestyle=":", linewidth=2,
                     label=r"$\Delta W/u^2\propto(u-u_c)$")

    plt.xlabel(r"$u_d$")
    plt.ylabel(r"$W = E(t)^2\langle\sigma\rangle_x$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"{output_file}.svg", bbox_inches='tight')
    fig.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')

    if also_plot_onset_test and meta_ref is not None:
        u_c = predicted_uc(meta_ref, n_ref=n_ref)
        W0 = np.array([predicted_W_drude(ui, meta_ref, n_ref=n_ref) for ui in u])
        dW_over_u2 = (W - W0) / (u**2 + 1e-30)
        eps = u - u_c

        fig2 = plt.figure(figsize=(8, 6))
        mask = (eps > 0) & np.isfinite(dW_over_u2)
        plt.loglog(eps[mask], dW_over_u2[mask], marker=".", color="black", linewidth=2)
        plt.xlabel(r"$u_d-u_c$")
        plt.ylabel(r"$(W-W_0)/u_d^2$")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(f"{output_file}_onset_test.svg", bbox_inches='tight')
        fig2.savefig(f"{output_file}_onset_test.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    return fig


def plot_un_vs_u_d(u_d_values, un_values, meta_ref=None, output_file="un_vs_u_d"):
    import matplotlib.pyplot as plt
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["legend.frameon"] = False
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['figure.titlesize'] = 16
    
    u = np.asarray(u_d_values, float)
    un = np.asarray(un_values, float)
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(u, un, linewidth=2, color="blue", label=r"$\langle u \cdot n \rangle_t$")
    
    if meta_ref is not None:
        u_c = predicted_uc(meta_ref)
        plt.axvline(u_c, linestyle="--", color="red", linewidth=2, label=rf"$u_c$")
    
    plt.xlabel(r"$u_d$")
    plt.ylabel(r"$\langle u \cdot n \rangle_t$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"{output_file}.svg", bbox_inches='tight')
    fig.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    import sys
    import glob
    import os
    
    base_dir = r"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\w=0.15_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.03, seed_amp_p=0.03)"#w=1_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.001, seed_amp_p=0.001)"
    #few sharp: w=0.3_dp=0.025_dn=0.2(seed_amp_n=0.001, seed_amp_p=0.001)
    # w=0.2_m=0.7_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.001, seed_amp_p=0.001) -- ideal match
    #w=0.4_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.001, seed_amp_p=0.001)


    npz_files = sorted(glob.glob(os.path.join(base_dir, "**", "*.npz"), recursive=True))
    
    if not npz_files:
        print(f"No .npz files found in {base_dir}")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} .npz files to process")
    
    u_d_list = []
    W_list = []
    un_list = []
    meta_ref = None
    n_ref = None
    
    for idx, npz_file in enumerate(npz_files, 1):
        print(f"\n[{idx}/{len(npz_files)}] Processing: {npz_file}")
        try:
            data = np.load(npz_file, allow_pickle=True)
            files = list(data.files)
            
            if 'n_t' in files and 'p_t' in files:
                n_t = data['n_t']
                p_t = data['p_t']
                L = float(data['L'])
                meta = data['meta'].item() if hasattr(data['meta'], 'item') else data['meta']
                has_p = True
            elif all(k in files for k in ('n', 'p', 'x', 't')):
                n = data['n']
                p = data['p']
                x = data['x']
                t = data['t']
                if n.shape[0] == len(t):
                    n_t = n.T
                    p_t = p.T
                else:
                    n_t = n
                    p_t = p
                L = float(data['L']) if 'L' in files else (x[-1] - x[0] + (x[1] - x[0]) if len(x) > 1 else 10.0)
                meta = data['meta'].item() if 'meta' in files and hasattr(data['meta'], 'item') else (data['meta'] if 'meta' in files else {})
                has_p = True
            elif all(k in files for k in ('n', 'x', 't')):
                n = data['n']
                x = data['x']
                t = data['t']
                if n.shape[0] == len(t):
                    n_t = n.T
                else:
                    n_t = n
                p_t = np.zeros_like(n_t)
                L = float(data['L']) if 'L' in files else (x[-1] - x[0] + (x[1] - x[0]) if len(x) > 1 else 10.0)
                meta = data['meta'].item() if 'meta' in files and hasattr(data['meta'], 'item') else (data['meta'] if 'meta' in files else {})
                has_p = False
            else:
                raise ValueError(f"Cannot load data from {npz_file}")
            print("meta=",meta)
            
            defaults = {
                'm': 1.0, 'e': 1.0, 'u_d': 5.245, 'nbar0': 0.2,
                'Gamma0': 2.5, 'w': 0.04, 'n_floor': 1e-4,
                'maintain_drift': 'field', 'Kp': 0.15,
                'lambda_diss': 0.0, 'sigma_diss': 2.0, 'x0': 10.0,
                'U': 1.0
            }
            for key, default_val in defaults.items():
                if key not in meta:
                    meta[key] = default_val
            if 'L' not in meta:
                meta['L'] = L
            
            if meta_ref is None:
                meta_ref = dict(meta)
                n_tail = max(5, int(0.2 * n_t.shape[1]))
                n_ref = float(np.mean(n_t[:, -n_tail:]))
            
            results_fft = compute_dissipation_from_npz(npz_file, method='time_fft')
            print_dissipation_summary(results_fft)
            
            results_wash = compute_dissipation_from_npz(npz_file, method='washboard')
            print_dissipation_summary(results_wash)
            
            u_d = meta.get('u_d', np.nan)
            t = results_wash['t']
            W_t = results_wash['W_t']
            n_tail = int(0.2 * len(t))
            if n_tail > 0:
                W_avg_tail = np.mean(W_t[-n_tail:])
                u_d_list.append(u_d)
                W_list.append(W_avg_tail)
                
                if has_p:
                    m = meta.get('m', 1.0)
                    p_mean_x = np.mean(p_t, axis=0)
                    un_avg_tail = np.mean(p_mean_x[-n_tail:]) / m
                    un_list.append(un_avg_tail)
                else:
                    un_list.append(np.nan)
            
            import matplotlib.pyplot as plt
            # fig1 = plot_dissipation_diagnostics(results_wash, n_t, meta, L, x0_label=os.path.basename(npz_file))
            # fig2 = plot_local_dissipation_heatmap(results_wash, n_t, meta, L)
            fig3 = plot_final_density_profile(results_wash, n_t, L, x0_label=os.path.basename(npz_file))
            # plt.show()
            
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if u_d_list and W_list:
        u_d_array = np.array(u_d_list)
        W_array = np.array(W_list)
        un_array = np.array(un_list)
        sort_idx = np.argsort(u_d_array)
        u_d_sorted = u_d_array[sort_idx]
        W_sorted = W_array[sort_idx]
        un_sorted = un_array[sort_idx]
        
        import matplotlib.pyplot as plt
        fig4 = plot_W_vs_u_d(u_d_sorted, W_sorted, meta_ref=meta_ref, n_ref=n_ref,
                             output_file="W_vs_u_d", also_plot_onset_test=True)
        print(f"\nSaved W vs u_d plot: W_vs_u_d.svg and W_vs_u_d.png")
        print(f"Saved onset test plot: W_vs_u_d_onset_test.svg and W_vs_u_d_onset_test.png")
        plt.close(fig4)
        
        if np.any(np.isfinite(un_sorted)):
            fig5 = plot_un_vs_u_d(u_d_sorted, un_sorted, meta_ref=meta_ref, output_file="un_vs_u_d")
            print(f"Saved <u*n>_t vs u_d plot: un_vs_u_d.svg and un_vs_u_d.png")
            plt.close(fig5)
