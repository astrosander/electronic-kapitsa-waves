import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename):
    data = np.load(filename, allow_pickle=True)
    return {
        'm': int(data['m']),
        't': data['t'],
        'n_t': data['n_t'],
        'p_t': data['p_t'],
        'L': float(data['L']),
        'Nx': int(data['Nx']),
        'meta': data['meta'].item() if 'meta' in data else {}
    }

def _power_spectrum_1d(n_slice, L):
    N = n_slice.size
    dn = n_slice - np.mean(n_slice)
    nhat = np.fft.fft(dn)
    P = (nhat * np.conj(nhat)).real / (N*N)
    m = np.arange(N//2 + 1)
    kpos = 2*np.pi*m / L
    return kpos[1:], P[1:N//2+1]

def plot_spacetime_lab(data, tag="spacetime_lab"):
    n_t = data['n_t']
    t = data['t']
    L = data['L']
    m = data['m']
    
    x = np.linspace(0, L, n_t.shape[0], endpoint=False)
    extent = [x.min(), x.max(), t.min(), t.max()]
    
    plt.figure(figsize=(9.6, 4.3))
    plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"n(x,t) [lab] m={m}")
    plt.colorbar(label="n")
    
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{tag}_m{m}.png", dpi=160)
    plt.close()

def plot_spacetime_comoving(data, u_d, tag="spacetime_comoving"):
    n_t = data['n_t']
    t = data['t']
    L = data['L']
    m = data['m']
    
    x = np.linspace(0, L, n_t.shape[0], endpoint=False)
    dx = x[1] - x[0]
    
    n_co = np.empty_like(n_t)
    for j, tj in enumerate(t):
        shift = (u_d * tj) % L
        s_idx = int(np.round(shift/dx)) % n_t.shape[0]
        n_co[:, j] = np.roll(n_t[:, j], -s_idx)
    
    extent = [x.min(), x.max(), t.min(), t.max()]
    plt.figure(figsize=(9.6, 4.3))
    plt.imshow(n_co.T, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    plt.xlabel("ξ = x - u_d t")
    plt.ylabel("t")
    plt.title(f"n(ξ,t) [co-moving u_d={u_d}] m={m}")
    plt.colorbar(label="n")
    
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{tag}_m{m}.png", dpi=160)
    plt.close()

def plot_snapshots(data, tag="snapshots"):
    n_t = data['n_t']
    t = data['t']
    L = data['L']
    m = data['m']
    
    x = np.linspace(0, L, n_t.shape[0], endpoint=False)
    
    plt.figure(figsize=(9.6, 3.4))
    for frac in [0.0, 1.0]:
        j = int(frac*(len(t)-1))
        plt.plot(x, n_t[:,j], label=f"t={t[j]:.1f}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("n")
    plt.title(f"Density snapshots m={m}")
    
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{tag}_m{m}.png", dpi=160)
    plt.close()

def plot_fft_compare(data, tag="fft_compare"):
    n_t = data['n_t']
    t = data['t']
    L = data['L']
    m = data['m']
    
    k0, P0 = _power_spectrum_1d(n_t[:, 0], L)
    k1, P1 = _power_spectrum_1d(n_t[:, -1], L)
    
    i0 = np.argmax(P0)
    i1 = np.argmax(P1)
    k0_peak, k1_peak = k0[i0], k1[i1]
    
    plt.figure(figsize=(8.6, 4.2))
    plt.plot(k0, P0, label="t = 0")
    plt.plot(k1, P1, label=f"t = {t[-1]:.2f}")
    plt.plot([k0_peak], [P0[i0]], "o", ms=6, label=f"peak0 k={k0_peak:.3f}")
    plt.plot([k1_peak], [P1[i1]], "s", ms=6, label=f"peak1 k={k1_peak:.3f}")
    
    plt.xlabel("$k$")
    plt.ylabel("power $|\\hat{n}(k)|^2$")
    plt.title("Fourier spectrum of $n(x,t)$: initial vs final")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(frameon=False, ncol=2)
    
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{tag}_m{m}.png", dpi=160)
    plt.close()

def plot_velocity_detection(data, u_d, tag="velocity_detection"):
    n_t = data['n_t']
    t = data['t']
    L = data['L']
    m = data['m']
    
    idx_t1 = -2
    idx_t2 = -1
    
    n_t1 = n_t[:, idx_t1]
    n_t2 = n_t[:, idx_t2]
    t1 = t[idx_t1]
    t2 = t[idx_t2]
    
    x = np.linspace(0, L, len(n_t1), endpoint=False)
    dx = L / len(n_t1)
    
    dn_t1 = n_t1 - np.mean(n_t1)
    dn_t2 = n_t2 - np.mean(n_t2)
    
    n_shifts = len(n_t1)
    shifts = np.arange(-n_shifts//2, n_shifts//2) * dx
    correlations = np.zeros(len(shifts))
    
    for i, shift in enumerate(shifts):
        if shift >= 0:
            dn_t1_shifted = np.roll(dn_t1, -int(shift/dx))
        else:
            dn_t1_shifted = np.roll(dn_t1, int(-shift/dx))
        correlations[i] = np.corrcoef(dn_t1_shifted, dn_t2)[0, 1]
    
    max_idx = np.argmax(correlations)
    shift_opt = shifts[max_idx]
    corr_max = correlations[max_idx]
    u_drift = shift_opt / (t2 - t1)
    
    if shift_opt >= 0:
        n_t1_shifted = np.roll(n_t1, -int(shift_opt/dx))
    else:
        n_t1_shifted = np.roll(n_t1, int(-shift_opt/dx))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(x, n_t1, 'b-', label=f'Initial n(x,{t1:.2f})', alpha=0.6, linewidth=1.5)
    ax1.plot(x, n_t2, 'r-', label=f'Final n(x,{t2:.2f})', alpha=0.8, linewidth=2)
    ax1.plot(x, n_t1_shifted, 'g--', label=f'Initial shifted by {shift_opt:.3f}', alpha=0.8, linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('n(x)')
    ax1.set_title('Velocity Detection: Shifted Initial vs Final')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(shifts, correlations, 'b-', linewidth=2, label='Correlation')
    ax2.axvline(shift_opt, color='r', linestyle='--', linewidth=2, label=f'Optimal shift={shift_opt:.3f}')
    ax2.plot([shift_opt], [corr_max], 'ro', markersize=8, label=f'Max corr={corr_max:.3f}')
    ax2.set_xlabel('Spatial shift')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation vs Shift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    delta_t = t2 - t1
    fig.suptitle(f'Instantaneous velocity at t={t2:.3f}: u_drift={u_drift:.3f} (shift={shift_opt:.3f}, Δt={delta_t:.4f}) | u_target={u_d:.3f}', 
                 y=0.98, fontsize=10.5)
    
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{tag}_m{m}.png", dpi=160, bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_velocity_evolution(data, u_d, tag="velocity_evolution"):
    n_t = data['n_t']
    p_t = data['p_t']
    t = data['t']
    L = data['L']
    m = data['m']
    meta = data['meta']
    
    m_par = meta.get('m', 1.0)
    n_floor = meta.get('n_floor', 1e-7)
    
    x = np.linspace(0, L, n_t.shape[0], endpoint=False)
    dx = L / len(x)
    
    n_eff_t = np.maximum(n_t, n_floor)
    v_t = p_t / (m_par * n_eff_t)
    u_momentum_t = np.mean(v_t, axis=0)
    
    n_times = len(t)
    u_drift_t = np.zeros(n_times - 1)
    
    for i in range(n_times - 1):
        n_t1 = n_t[:, i]
        n_t2 = n_t[:, i + 1]
        t1 = t[i]
        t2 = t[i + 1]
        
        dn_t1 = n_t1 - np.mean(n_t1)
        dn_t2 = n_t2 - np.mean(n_t2)
        
        n_shifts = len(n_t1)
        shifts = np.arange(-n_shifts//2, n_shifts//2) * dx
        correlations = np.zeros(len(shifts))
        
        for j, shift in enumerate(shifts):
            if shift >= 0:
                dn_t1_shifted = np.roll(dn_t1, -int(shift/dx))
            else:
                dn_t1_shifted = np.roll(dn_t1, int(-shift/dx))
            correlations[j] = np.corrcoef(dn_t1_shifted, dn_t2)[0, 1]
        
        max_idx = np.argmax(correlations)
        shift_opt = shifts[max_idx]
        corr_max = correlations[max_idx]
        u_drift = shift_opt / (t2 - t1)
        
        u_drift_t[i] = u_drift
    
    t_mid = (t[:-1] + t[1:]) / 2
    plt.figure(figsize=(8, 5))
    plt.plot(t_mid, -u_drift_t, 'b-', linewidth=2, label='Measured $u_d$ (shift method)')
    plt.plot(t, u_momentum_t, 'r-', linewidth=2, label='$\\langle v \\rangle$ (momentum)')
    # plt.axhline(y=u_d, color='r', linestyle='--', linewidth=2, label=f'Target $u_d={u_d}$')
    plt.xlabel('$t$')
    plt.ylabel('$u_d$')
    plt.title('Velocity Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{tag}_m{m}.png", dpi=160, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":
    filename = "out_drift/data_m01_m1.npz"
    data = load_data(filename)
    
    u_d = data['meta'].get('u_d', 20.0)
    
    plot_spacetime_lab(data)
    plot_spacetime_comoving(data, u_d)
    plot_snapshots(data)
    plot_fft_compare(data)
    plot_velocity_detection(data, u_d)
    plot_velocity_evolution(data, u_d)
    
    print("All plots generated!")
