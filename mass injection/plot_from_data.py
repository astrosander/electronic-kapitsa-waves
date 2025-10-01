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
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
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
    dt_skip = 1

    valid_indices = []
    u_drift_values = []
    t_mid_values = []
    
    for i in range(0, n_times - dt_skip, dt_skip):
        n_t1 = n_t[:, i]
        n_t2 = n_t[:, i + dt_skip]
        t1 = t[i]
        t2 = t[i + dt_skip]
        
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
        
        u_drift_values.append(abs(u_drift))
        t_mid_values.append((t1 + t2) / 2)
    
    u_drift_t = np.array(u_drift_values)
    t_mid = np.array(t_mid_values)
    plt.figure(figsize=(8, 5))
    print(u_drift_t[:-1], u_momentum_t[:-1])
    plt.plot(t_mid, u_drift_t, 'b-', linewidth=2, label='Measured $u_d$ (shift method)')
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

def plot_velocity_field(data, u_d, tag="velocity_field"):
    """Plot v(x) = p(x)/n(x) alongside p(x) and n(x) at different time snapshots"""
    n_t = data['n_t']
    p_t = data['p_t']
    t = data['t']
    L = data['L']
    m = data['m']
    meta = data['meta']
    
    m_par = meta.get('m', 1.0)
    n_floor = meta.get('n_floor', 1e-7)
    
    x = np.linspace(0, L, n_t.shape[0], endpoint=False)
    
    # Calculate velocity field v(x,t) = p(x,t) / (m * n(x,t))
    n_eff_t = np.maximum(n_t, n_floor)
    v_t = p_t / (m_par * n_eff_t)
    
    # Select time snapshots
    time_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    fig, axes = plt.subplots(len(time_fractions), 1, figsize=(12, 2.5*len(time_fractions)))
    if len(time_fractions) == 1:
        axes = [axes]
    
    for i, frac in enumerate(time_fractions):
        j = int(frac * (len(t) - 1))
        t_val = t[j]
        
        n_slice = n_t[:, j]
        p_slice = p_t[:, j]
        v_slice = v_t[:, j]
        
        # Plot n(x), p(x), and v(x) on the same subplot
        ax = axes[i]
        
        # Use different y-axes for different quantities
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        # Plot n(x) on left axis
        line1 = ax.plot(x, n_slice, 'b-', linewidth=2, alpha=0.8)
        ax.set_ylabel('$n(x)$', color='b', fontsize=12)
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim(0, np.max(n_slice) * 1.1)
        
        # Plot p(x) on middle axis
        line2 = ax2.plot(x, p_slice, 'r-', linewidth=2, alpha=0.8)
        ax2.set_ylabel('$p(x)$', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Plot v(x) on right axis
        line3 = ax3.plot(x, v_slice, 'g-', linewidth=2, alpha=0.8)
        ax3.set_ylabel('$v(x)$', color='g', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='g')
        
        # Add horizontal line at v=0 for reference
        # ax3.axhline(y=0, color='g', linestyle='--', alpha=0.5)
        
        # Add target velocity line
        # ax3.axhline(y=u_d, color='orange', linestyle=':', alpha=0.7, label=f'Target $u_d={u_d}$')
        
        # Only add x-label for the bottom panel
        if i == len(time_fractions) - 1:
            ax.set_xlabel('$x$', fontsize=12)
        else:
            ax.set_xticklabels([])  # Remove x-axis labels for upper panels
        
        # ax.set_title(f'Velocity Field Analysis at $t={t_val:.2f}$ (m={m})', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=10)
        
        # Add statistics text
        v_mean = np.mean(v_slice)
        v_std = np.std(v_slice)
        v_min = np.min(v_slice)
        v_max = np.max(v_slice)
        
        # stats_text = f'$\\langle v \\rangle = {v_mean:.3f}$\n$\\sigma_v = {v_std:.3f}$\n$v_{{min}} = {v_min:.3f}$\n$v_{{max}} = {v_max:.3f}$'
        stats_text = f'$t={t_val:.1f}$'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.subplots_adjust(hspace=0.1)  # Reduce space between subplots
    plt.tight_layout()
    
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{tag}_m{m}.png", dpi=160, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print summary statistics
    print(f"\nVelocity Field Analysis Summary (m={m}):")
    print(f"Target u_d = {u_d}")
    for i, frac in enumerate(time_fractions):
        j = int(frac * (len(t) - 1))
        t_val = t[j]
        v_slice = v_t[:, j]
        v_mean = np.mean(v_slice)
        v_std = np.std(v_slice)
        v_min = np.min(v_slice)
        v_max = np.max(v_slice)
        print(f"  t={t_val:.2f}: <v>={v_mean:.3f}, σ_v={v_std:.3f}, v_range=[{v_min:.3f}, {v_max:.3f}]")

def plot_multiple_ud_panel():
    u_d_values = [1.5, 2, 3, 3.5, 3.6, 3.75, 4, 5, 6, 7]
    filenames = [f"multiple_u_d/10e-3/out_drift_ud{ud}/data_m01_ud{ud}.npz" for ud in u_d_values]
    
    fig, axes = plt.subplots(10, 1, figsize=(10, 16))
    
    for i, (filename, u_d) in enumerate(zip(filenames, u_d_values)):
        try:
            data = load_data(filename)
            n_t = data['n_t']
            t = data['t']
            L = data['L']
            
            x = np.linspace(0, L, n_t.shape[0], endpoint=False)
            n_final = n_t[:, -1]
            
            axes[i].plot(x, n_final, 'b-', linewidth=1.5)
            axes[i].text(0.02, 0.95, f'$u_d={u_d}$', transform=axes[i].transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, L)
            axes[i].tick_params(labelsize=8)
            
            if i == 9:
                axes[i].set_xlabel('$x$', fontsize=12)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            axes[i].text(0.5, 0.5, f'Error loading u_d={u_d}', 
                        transform=axes[i].transAxes, ha='center', va='center')
    
    # plt.suptitle('Final density profiles $n(x,t_{final})$ for different $u_d$', fontsize=14, y=0.98)
    plt.subplots_adjust(hspace=0.1, top=0.95)
    
    os.makedirs("multiple_u_d", exist_ok=True)
    plt.savefig("multiple_u_d/final_profiles_panel.png", dpi=160, bbox_inches='tight')
    plt.savefig("multiple_u_d/final_profiles_panel.svg", dpi=160, bbox_inches='tight')
    # plt.show()
    plt.close()

if __name__ == "__main__":
    # Single file analysis
    filename = "out_drift/data_m01_m1_t10.npz"#"out_drift/data_m01_m1.npz"
    data = load_data(filename)
    
    u_d = data['meta'].get('u_d', 20.0)
    print(u_d)
    # plot_spacetime_lab(data)
    # plot_spacetime_comoving(data, u_d)
    # plot_snapshots(data)
    # plot_fft_compare(data)
    # plot_velocity_detection(data, u_d)
    # plot_velocity_evolution(data, u_d)
    # plot_velocity_field(data, u_d)
    
    # Multiple u_d panel
    plot_multiple_ud_panel()
    
    print("All plots generated!")
