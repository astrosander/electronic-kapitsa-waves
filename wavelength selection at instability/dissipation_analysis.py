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
        'lambda_diss': 0.0, 'sigma_diss': 2.0, 'x0': 10.0
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
    print(f"\nDissipation metrics:")
    print(f"  <W(t)>_time = {results['W_t_mean']:.6g}")
    print(f"  <E^2 * σ>_T = {results['W_avg']:.6g}")
    print(f"\nSanity check:")
    print(f"  <E*j>_time = {results['P_Ej_mean']:.6g}")
    print(f"  (should be similar to <W> for Drude model)")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    import glob
    
    npz_files=["npz/complete_m01_m1.npz"]
    
    for npz_file in npz_files:
        print(f"\nProcessing: {npz_file}")
        try:
            results_fft = compute_dissipation_from_npz(npz_file, method='time_fft')
            print_dissipation_summary(results_fft)
            
            results_wash = compute_dissipation_from_npz(npz_file, method='washboard')
            print_dissipation_summary(results_wash)
            
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
            import traceback
            traceback.print_exc()
