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
    dt_skip = 3

    valid_indices = []
    u_drift_values = []
    t_mid_values = []
    n_peaks_values = []
    amplitude_values = []
    print(n_times)
    for i in range(0, n_times - dt_skip, dt_skip):
        print(i)
        # if i<1000:
        #     break
        n_t1 = n_t[:, i]
        n_t2 = n_t[:, i + dt_skip]
        t1 = t[i]
        t2 = t[i + dt_skip]
        
        # Calculate number of peaks in n_t1
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(n_t1, height=np.mean(n_t1), distance=5)
        n_peaks = len(peaks)
        
        # Calculate amplitude (delta n = n_max - n_min)
        amplitude = np.max(n_t1) - np.min(n_t1)
        
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
        n_peaks_values.append(n_peaks)
        amplitude_values.append(amplitude)
    
    u_drift_t = np.array(u_drift_values)
    t_mid = np.array(t_mid_values)
    n_peaks_t = np.array(n_peaks_values)
    amplitude_t = np.array(amplitude_values)
    
    # Create subplots for all three quantities
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Velocity evolution
    ax1.plot(t_mid, u_drift_t, 'b-', linewidth=2, label='Measured $u_d$ (shift method)')
    ax1.plot(t, u_momentum_t, 'r-', linewidth=2, label='$\\langle v \\rangle$ (momentum)')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$u_d$')
    ax1.set_title('Velocity Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of peaks evolution
    ax2.plot(t_mid, n_peaks_t, 'g-', linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('Number of Peaks')
    ax2.set_title('Number of Peaks Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Amplitude evolution (delta n = n_max - n_min)
    ax3.plot(t_mid, amplitude_t, 'm-', linewidth=2, marker='s', markersize=3)
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('Amplitude $\\Delta n = n_{max} - n_{min}$')
    ax3.set_title('Amplitude Evolution')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{tag}_m{m}.png", dpi=160, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary for u_d = {u_d}:")
    print(f"  Number of peaks: {np.min(n_peaks_t)} to {np.max(n_peaks_t)} (mean: {np.mean(n_peaks_t):.1f})")
    print(f"  Amplitude range: {np.min(amplitude_t):.6f} to {np.max(amplitude_t):.6f}")
    print(f"  Velocity range: {np.min(u_drift_t):.4f} to {np.max(u_drift_t):.4f}")

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

def plot_velocity_vs_ud(data_files, tag="velocity_vs_ud"):
    """Plot measured velocity (spatial shifting) vs target u_d at t=t_final"""
    u_d_values = []
    u_true_values = []
    n_pulses_values = []
    frequency_values = []
    
    for filename, u_d in data_files:
        try:
            data = load_data(filename)
            n_t = data['n_t']
            t = data['t']
            L = data['L']
            
            # Use last two time points for velocity calculation
            idx_t1 = -2
            idx_t2 = -1
            
            n_t1 = n_t[:, idx_t1]
            n_t2 = n_t[:, idx_t2]
            t1 = t[idx_t1]
            t2 = t[idx_t2]
            
            x = np.linspace(0, L, len(n_t1), endpoint=False)
            dx = L / len(n_t1)
            
            # Detrend the data
            dn_t1 = n_t1 - np.mean(n_t1)
            dn_t2 = n_t2 - np.mean(n_t2)
            
            # Calculate correlation for different shifts
            n_shifts = len(n_t1)
            shifts = np.arange(-n_shifts//2, n_shifts//2) * dx
            correlations = np.zeros(len(shifts))
            
            for i, shift in enumerate(shifts):
                if shift >= 0:
                    dn_t1_shifted = np.roll(dn_t1, -int(shift/dx))
                else:
                    dn_t1_shifted = np.roll(dn_t1, int(-shift/dx))
                correlations[i] = np.corrcoef(dn_t1_shifted, dn_t2)[0, 1]
            
            # Find optimal shift
            max_idx = np.argmax(correlations)
            shift_opt = shifts[max_idx]
            u_true = shift_opt / (t2 - t1)
            
            # Calculate density of pulses (count peaks in final density profile)
            n_final = n_t[:, -1]
            n_mean = np.mean(n_final)
            n_std = np.std(n_final)
            
            # Find peaks (local maxima above threshold)
            from scipy.signal import find_peaks
            threshold = n_mean + 0.5 * n_std
            peaks, _ = find_peaks(n_final, height=threshold, distance=5)
            N_pulses = len(peaks)
            n_pulses = N_pulses / L
            
            # Calculate frequency
            frequency = u_true * n_pulses
            
            u_d_values.append(abs(u_d))
            u_true_values.append(u_true)
            n_pulses_values.append(n_pulses)
            frequency_values.append(frequency)
            
        except Exception as e:
            continue
    
    if not u_d_values:
        return
    
    # Convert to numpy arrays
    u_d_values = np.array(u_d_values)
    u_true_values = np.array(u_true_values)
    n_pulses_values = np.array(n_pulses_values)
    frequency_values = np.array(frequency_values)
    
    # Filter for u_d > 1.4
    mask = u_d_values > 0#1.41
    u_d_filtered = u_d_values[mask]
    u_true_filtered = u_true_values[mask]
    n_pulses_filtered = n_pulses_values[mask]
    frequency_filtered = frequency_values[mask]
    
    if len(u_d_filtered) == 0:
        print("No data points with u_d > 1.4 found!")
        return
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: u_true vs u_d
    ax1.plot(u_d_filtered, np.abs(u_true_filtered), 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('$u_d$')
    ax1.set_ylabel('$u_{\\text{true}}$')
    ax1.set_title('$u_{\\text{true}}$ vs $u_d$')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: n_pulses vs u_d
    ax2.plot(u_d_filtered, n_pulses_filtered, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('$u_d$')
    ax2.set_ylabel('$n_{\\text{pulses}} = N/L$')
    ax2.set_title('$n_{\\text{pulses}}$ vs $u_d$')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: frequency vs u_d
    ax3.plot(u_d_filtered, np.abs(frequency_filtered), 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('$u_d$')
    ax3.set_ylabel('$f = u_{\\text{true}} \\cdot n_{\\text{pulses}}$')
    ax3.set_title('$f$ vs $u_d$')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("multiple_u_d", exist_ok=True)
    plt.savefig(f"multiple_u_d/{tag}.png", dpi=160, bbox_inches='tight')
    plt.savefig(f"multiple_u_d/{tag}.pdf", dpi=160, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return u_d_values, u_true_values, n_pulses_values, frequency_values

def plot_multiple_ud_panel(data_files=None):
    """Plot density profiles for multiple u_d values"""
    if data_files is None:
        data_files = find_available_simulations()
    
    if not data_files:
        print("No data files found for panel plot!")
        return
    
    # Limit to first 10 files for readability
    data_files = data_files#[:20]
    n_files = len(data_files)
    
    fig, axes = plt.subplots(n_files, 1, figsize=(len(data_files), 2*n_files))
    if n_files == 1:
        axes = [axes]
    
    for i, (filename, u_d) in enumerate(data_files):
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
            
            if i == n_files - 1:
                axes[i].set_xlabel('$x$', fontsize=12)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            axes[i].text(0.5, 0.5, f'Error loading u_d={u_d}', 
                        transform=axes[i].transAxes, ha='center', va='center')
    
    plt.subplots_adjust(hspace=0.1, top=0.95)
    
    os.makedirs("multiple_u_d", exist_ok=True)
    plt.savefig("multiple_u_d/final_profiles_panel.png", dpi=160, bbox_inches='tight')
    plt.savefig("multiple_u_d/final_profiles_panel.svg", dpi=160, bbox_inches='tight')
    plt.show()
    plt.close()

def load_multi_dataset(base_dirs, labels=None):
    """Load data from multiple base directories for comparison.
    
    Args:
        base_dirs: List of base directory paths
        labels: Optional list of labels for each dataset
        
    Returns:
        list: [(label, u_d, data_dict), ...] structure
    """
    import re
    combined_data = []
    
    if labels is None:
        labels = [os.path.basename(d) for d in base_dirs]
    
    for base_dir, label in zip(base_dirs, labels):
        print(f"\nScanning {label}...")
        
        for item in os.listdir(base_dir):
            if item.startswith("out_drift_ud"):
                subdir = os.path.join(base_dir, item)
                if not os.path.isdir(subdir):
                    continue
                    
                # Extract u_d from directory name
                match = re.search(r'ud([\d.]+)', item)
                if not match:
                    continue
                u_d = float(match.group(1))
                
                # Find data file
                data_files = [f for f in os.listdir(subdir) if f.startswith("data_") and f.endswith(".npz")]
                if not data_files:
                    continue
                    
                filepath = os.path.join(subdir, data_files[0])
                try:
                    data = load_data(filepath)
                    combined_data.append((label, u_d, data))
                    print(f"  Loaded u_d={u_d:.4f}")
                except Exception as e:
                    print(f"  Error loading {filepath}: {e}")
    
    return combined_data

def plot_combined_velocity_analysis(base_dirs, labels=None, outdir="multiple_u_d"):
    """Plot u_true, n_pulses, and frequency for combined datasets.
    
    Args:
        base_dirs: List of base directory paths
        labels: Optional custom labels for each dataset
        outdir: Output directory for plots
    """
    from scipy.signal import find_peaks
    
    if labels is None:
        labels = [os.path.basename(d) for d in base_dirs]
    
    # Load all data
    all_data = load_multi_dataset(base_dirs, labels)
    
    if not all_data:
        print("No data found!")
        return
    
    # Organize data by label
    data_by_label = {label: {'u_d': [], 'u_true': [], 'n_pulses': [], 'frequency': []} for label in labels}
    
    # Collect all data for interpolation
    all_u_d = []
    all_u_true = []
    all_n_pulses = []
    all_frequency = []
    
    for data_label, u_d, data in all_data:
        n_t = data['n_t']
        t = data['t']
        L = data['L']
        
        # Calculate u_true using spatial correlation
        idx_t1 = -2
        idx_t2 = -1
        
        n_t1 = n_t[:, idx_t1]
        n_t2 = n_t[:, idx_t2]
        t1 = t[idx_t1]
        t2 = t[idx_t2]
        
        dx = L / len(n_t1)
        
        # Detrend the data
        dn_t1 = n_t1 - np.mean(n_t1)
        dn_t2 = n_t2 - np.mean(n_t2)
        
        # Calculate correlation for different shifts
        n_shifts = len(n_t1)
        shifts = np.arange(-n_shifts//2, n_shifts//2) * dx
        correlations = np.zeros(len(shifts))
        
        for i, shift in enumerate(shifts):
            if shift >= 0:
                dn_t1_shifted = np.roll(dn_t1, -int(shift/dx))
            else:
                dn_t1_shifted = np.roll(dn_t1, int(-shift/dx))
            correlations[i] = np.corrcoef(dn_t1_shifted, dn_t2)[0, 1]
        
        # Find optimal shift
        max_idx = np.argmax(correlations)
        shift_opt = shifts[max_idx]
        u_true = shift_opt / (t2 - t1)
        
        # Calculate density of pulses (count peaks in final density profile)
        n_final = n_t[:, -1]
        n_mean = np.mean(n_final)
        n_std = np.std(n_final)
        
        # Find peaks (local maxima above threshold)
        threshold = n_mean + 0.5 * n_std
        peaks, _ = find_peaks(n_final, height=threshold, distance=5)
        N_pulses = len(peaks)
        n_pulses = N_pulses / L
        
        # Calculate frequency
        frequency = abs(u_true) * n_pulses
        
        data_by_label[data_label]['u_d'].append(u_d)
        data_by_label[data_label]['u_true'].append(u_true)
        data_by_label[data_label]['n_pulses'].append(n_pulses)
        data_by_label[data_label]['frequency'].append(frequency)
        
        # Collect all data for interpolation
        all_u_d.append(u_d)
        all_u_true.append(u_true)
        all_n_pulses.append(n_pulses)
        all_frequency.append(frequency)
    
    # Create plots - more compact design similar to original velocity_vs_ud.png
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Same color for all datasets, different marker shapes
    color = 'black'  # Single color for all datasets
    markers = ['o', 's', '^', '*', 'v', '<', '>', 'p', '*', 'h', 
               'H', '+', 'x', 'X', '|']
    
    for idx, label in enumerate(labels):
        if not data_by_label[label]['u_d']:
            continue
        
        # Filter for u_d > 1.4
        u_d_arr = np.array(data_by_label[label]['u_d'])
        u_true_arr = np.array(data_by_label[label]['u_true'])
        n_pulses_arr = np.array(data_by_label[label]['n_pulses'])
        freq_arr = np.array(data_by_label[label]['frequency'])
        
        mask = u_d_arr > 0#1.41
        
        if not np.any(mask):
            continue
        
        u_d_filtered = u_d_arr[mask]
        u_true_filtered = u_true_arr[mask]
        n_pulses_filtered = n_pulses_arr[mask]
        freq_filtered = freq_arr[mask]
        
        marker = markers[idx % len(markers)]
        
        # Plot 1: u_true vs u_d (points only, no lines, larger size)
        ax1.scatter(u_d_filtered, np.abs(u_true_filtered), marker=marker, color=color,
                   label=label, s=24, alpha=0.8)
        
        # Plot 2: n_pulses vs u_d (points only, no lines, larger size)
        ax2.scatter(u_d_filtered, n_pulses_filtered, marker=marker, color=color,
                   label=label, s=24, alpha=0.8)
        
        # Plot 3: frequency vs u_d (points only, no lines, larger size)
        ax3.scatter(u_d_filtered, freq_filtered, marker=marker, color=color,
                   label=label, s=24, alpha=0.8)
    
    # Add smooth interpolation lines with specific ranges for each plot
    if all_u_d:
        from scipy.interpolate import UnivariateSpline
        from scipy.ndimage import uniform_filter1d
        
        # Sort data for smooth line
        sorted_indices = np.argsort(all_u_d)
        u_d_sorted = np.array(all_u_d)[sorted_indices]
        u_true_sorted = np.abs(np.array(all_u_true))[sorted_indices]  # Use absolute values
        n_pulses_sorted = np.array(all_n_pulses)[sorted_indices]
        freq_sorted = np.array(all_frequency)[sorted_indices]
        
        try:
            # Plot 1: u_true vs u_d - interpolation for 2 < u_d < 8
            mask1 = (u_d_sorted > 1.5) & (u_d_sorted < 8.0)
            if np.sum(mask1) > 3:  # Need at least 4 points for spline
                u_d_1 = u_d_sorted[mask1]
                u_true_1 = u_true_sorted[mask1]
                
                # Use moving average for binning points at same u_d
                u_d_unique = []
                u_true_avg = []
                tolerance = 0.01  # Group points within this tolerance
                
                i = 0
                while i < len(u_d_1):
                    # Find all points within tolerance
                    mask_same = np.abs(u_d_1 - u_d_1[i]) < tolerance
                    u_d_unique.append(np.mean(u_d_1[mask_same]))
                    u_true_avg.append(np.mean(u_true_1[mask_same]))
                    i = np.where(mask_same)[0][-1] + 1
                
                u_d_unique = np.array(u_d_unique)
                u_true_avg = np.array(u_true_avg)
                
                if len(u_d_unique) > 3:
                    # Much stronger smoothing to avoid fluctuations
                    spline_u_true = UnivariateSpline(u_d_unique, u_true_avg, s=len(u_d_unique)*2.0, k=5)
                    u_d_smooth_1 = np.linspace(u_d_unique.min(), u_d_unique.max(), 200)
                    u_true_smooth = spline_u_true(u_d_smooth_1)
                    ax1.plot(u_d_smooth_1, u_true_smooth, 'r-', linewidth=2.5, alpha=0.9, label='Mean trend', zorder=100)
            
            # Plot 2: n_pulses vs u_d - two separate interpolations
            # First segment: 1.0 <= u_d <= 2.5
            mask2a = (u_d_sorted >= 1.0) & (u_d_sorted <= 2.5)
            if np.sum(mask2a) > 3:
                u_d_2a = u_d_sorted[mask2a]
                n_pulses_2a = n_pulses_sorted[mask2a]
                
                # Bin and average
                u_d_unique_2a = []
                n_pulses_avg_2a = []
                i = 0
                tolerance = 0.01
                while i < len(u_d_2a):
                    mask_same = np.abs(u_d_2a - u_d_2a[i]) < tolerance
                    u_d_unique_2a.append(np.mean(u_d_2a[mask_same]))
                    n_pulses_avg_2a.append(np.mean(n_pulses_2a[mask_same]))
                    i = np.where(mask_same)[0][-1] + 1
                
                u_d_unique_2a = np.array(u_d_unique_2a)
                n_pulses_avg_2a = np.array(n_pulses_avg_2a)
                
                if len(u_d_unique_2a) > 3:
                    # Increased smoothing parameter to reduce fluctuations
                    spline_n_pulses_a = UnivariateSpline(u_d_unique_2a, n_pulses_avg_2a, s=len(u_d_unique_2a)*1.5, k=5)
                    u_d_smooth_2a = np.linspace(u_d_unique_2a.min(), u_d_unique_2a.max(), 200)
                    n_pulses_smooth_a = spline_n_pulses_a(u_d_smooth_2a)
                    ax2.plot(u_d_smooth_2a, n_pulses_smooth_a, 'r-', linewidth=2.5, alpha=0.9, label='Mean trend', zorder=100)
            
            # Second segment: 2.5 <= u_d <= 5.0
            mask2b = (u_d_sorted >= 2.5) & (u_d_sorted <= 8.0)
            if np.sum(mask2b) > 3:
                u_d_2b = u_d_sorted[mask2b]
                n_pulses_2b = n_pulses_sorted[mask2b]
                
                # Bin and average
                u_d_unique_2b = []
                n_pulses_avg_2b = []
                i = 0
                tolerance = 0.01
                while i < len(u_d_2b):
                    mask_same = np.abs(u_d_2b - u_d_2b[i]) < tolerance
                    u_d_unique_2b.append(np.mean(u_d_2b[mask_same]))
                    n_pulses_avg_2b.append(np.mean(n_pulses_2b[mask_same]))
                    i = np.where(mask_same)[0][-1] + 1
                
                u_d_unique_2b = np.array(u_d_unique_2b)
                n_pulses_avg_2b = np.array(n_pulses_avg_2b)
                
                if len(u_d_unique_2b) > 2:
                    # Use very strong smoothing and lower degree to avoid multiple extrema
                    # k=2 (quadratic) ensures at most one extremum
                    spline_n_pulses_b = UnivariateSpline(u_d_unique_2b, n_pulses_avg_2b, s=len(u_d_unique_2b)*8.0, k=5)
                    u_d_smooth_2b = np.linspace(u_d_unique_2b.min(), u_d_unique_2b.max(), 200)
                    n_pulses_smooth_b = spline_n_pulses_b(u_d_smooth_2b)
                    ax2.plot(u_d_smooth_2b, n_pulses_smooth_b, 'r-', linewidth=2.5, alpha=0.9, zorder=100)
            
            # Plot 3: frequency vs u_d - two separate interpolations
            # First segment: 1.0 <= u_d <= 2.5
            mask3a = (u_d_sorted >= 1.0) & (u_d_sorted <= 2.5)
            if np.sum(mask3a) > 3:
                u_d_3a = u_d_sorted[mask3a]
                freq_3a = freq_sorted[mask3a]
                
                # Bin and average
                u_d_unique_3a = []
                freq_avg_3a = []
                i = 0
                tolerance = 0.01
                while i < len(u_d_3a):
                    mask_same = np.abs(u_d_3a - u_d_3a[i]) < tolerance
                    u_d_unique_3a.append(np.mean(u_d_3a[mask_same]))
                    freq_avg_3a.append(np.mean(freq_3a[mask_same]))
                    i = np.where(mask_same)[0][-1] + 1
                
                u_d_unique_3a = np.array(u_d_unique_3a)
                freq_avg_3a = np.array(freq_avg_3a)
                
                if len(u_d_unique_3a) > 3:
                    spline_freq_a = UnivariateSpline(u_d_unique_3a, freq_avg_3a, s=len(u_d_unique_3a)*1.5, k=5)
                    u_d_smooth_3a = np.linspace(u_d_unique_3a.min(), u_d_unique_3a.max(), 200)
                    freq_smooth_a = spline_freq_a(u_d_smooth_3a)
                    ax3.plot(u_d_smooth_3a, freq_smooth_a, 'r-', linewidth=2.5, alpha=0.9, label='Mean trend', zorder=100)
            
            # Second segment: 2.5 <= u_d <= 5.0
            mask3b = (u_d_sorted >= 2.5) & (u_d_sorted <= 8.0)
            if np.sum(mask3b) > 3:
                u_d_3b = u_d_sorted[mask3b]
                freq_3b = freq_sorted[mask3b]
                
                # Bin and average
                u_d_unique_3b = []
                freq_avg_3b = []
                i = 0
                tolerance = 0.01
                while i < len(u_d_3b):
                    mask_same = np.abs(u_d_3b - u_d_3b[i]) < tolerance
                    u_d_unique_3b.append(np.mean(u_d_3b[mask_same]))
                    freq_avg_3b.append(np.mean(freq_3b[mask_same]))
                    i = np.where(mask_same)[0][-1] + 1
                
                u_d_unique_3b = np.array(u_d_unique_3b)
                freq_avg_3b = np.array(freq_avg_3b)
                
                if len(u_d_unique_3b) > 2:
                    # Use very strong smoothing and lower degree to avoid multiple extrema
                    # k=2 (quadratic) ensures at most one extremum
                    spline_freq_b = UnivariateSpline(u_d_unique_3b, freq_avg_3b, s=len(u_d_unique_3b)*8.0, k=5)
                    u_d_smooth_3b = np.linspace(u_d_unique_3b.min(), u_d_unique_3b.max(), 200)
                    freq_smooth_b = spline_freq_b(u_d_smooth_3b)
                    ax3.plot(u_d_smooth_3b, freq_smooth_b, 'r-', linewidth=2.5, alpha=0.9, zorder=100)
            
        except Exception as e:
            print(f"Warning: Could not create smooth lines: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot 1: u_true vs u_d (exact labels from original)
    ax1.set_xlabel('$u_d$')
    ax1.set_ylabel('$u_{\\text{true}}$')
    ax1.set_title('$u_{\\text{true}}$ vs $u_d$')
    ax1.legend(fontsize=8, ncol=2, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: n_pulses vs u_d (exact labels from original)
    ax2.set_xlabel('$u_d$')
    ax2.set_ylabel('$n_{\\text{pulses}} = N/L$')
    ax2.set_title('$n_{\\text{pulses}}$ vs $u_d$')
    ax2.legend(fontsize=8, ncol=2, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: frequency vs u_d (exact labels from original)
    ax3.set_xlabel('$u_d$')
    ax3.set_ylabel('$f = u_{\\text{true}} \\cdot n_{\\text{pulses}}$')
    ax3.set_title('$f$ vs $u_d$')
    ax3.legend(fontsize=8, ncol=2, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/velocity_vs_ud_combined.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{outdir}/velocity_vs_ud_combined.pdf", dpi=200, bbox_inches='tight')
    print(f"\nSaved velocity analysis to {outdir}/velocity_vs_ud_combined.png")
    plt.show()
    plt.close()
    
    return data_by_label

def plot_delta_n_vs_ud(base_dirs, labels=None, outdir="multiple_u_d"):
    """Plot delta n (n_max - n_min) vs u_d for combined datasets.
    
    Args:
        base_dirs: List of base directory paths
        labels: Optional custom labels for each dataset
        outdir: Output directory for plots
    """
    if labels is None:
        labels = [os.path.basename(d) for d in base_dirs]
    
    # Load all data
    all_data = load_multi_dataset(base_dirs, labels)
    
    if not all_data:
        print("No data found!")
        return
    
    # Organize data by label
    data_by_label = {label: {'u_d': [], 'delta_n': []} for label in labels}
    
    # Collect all data for interpolation
    all_u_d = []
    all_delta_n = []
    
    for data_label, u_d, data in all_data:
        n_t = data['n_t']
        t = data['t']
        
        # Calculate delta n = n_max - n_min for the final time step
        n_final = n_t[:, -1]  # Last time step
        delta_n = np.max(n_final) - np.min(n_final)
        
        data_by_label[data_label]['u_d'].append(u_d)
        data_by_label[data_label]['delta_n'].append(delta_n)
        
        # Collect all data for interpolation
        all_u_d.append(u_d)
        all_delta_n.append(delta_n)
    
    # Create plot - same style as plot_combined_velocity_analysis
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Same color for all datasets, different marker shapes
    color = 'black'  # Single color for all datasets
    markers = ['o', 's', '^', '*', 'v', '<', '>', 'p', '*', 'h', 
               'H', '+', 'x', 'X', '|']
    
    for idx, label in enumerate(labels):
        if not data_by_label[label]['u_d']:
            continue
        
        # Filter for u_d <= 10
        u_d_arr = np.array(data_by_label[label]['u_d'])
        delta_n_arr = np.array(data_by_label[label]['delta_n'])
        
        mask = u_d_arr <= 10.0
        
        if not np.any(mask):
            continue
        
        u_d_filtered = u_d_arr[mask]
        delta_n_filtered = delta_n_arr[mask]
        
        marker = markers[idx % len(markers)]
        
        # Plot points only, no lines, larger size
        ax.scatter(u_d_filtered, delta_n_filtered, marker=marker, color=color,
                  label=label, s=24, alpha=0.8)
    
    # Add smooth interpolation line
    if all_u_d:
        from scipy.interpolate import UnivariateSpline
        from scipy.ndimage import uniform_filter1d
        
        # Sort data for smooth line
        sorted_indices = np.argsort(all_u_d)
        u_d_sorted = np.array(all_u_d)[sorted_indices]
        delta_n_sorted = np.array(all_delta_n)[sorted_indices]
        
        # Filter for u_d <= 10
        mask = u_d_sorted <= 10.0
        u_d_sorted = u_d_sorted[mask]
        delta_n_sorted = delta_n_sorted[mask]
        
        try:
            # Interpolation for u_d > 2.74
            mask_interp = (u_d_sorted > 2.68) & (u_d_sorted <= 10.0)
            if np.sum(mask_interp) > 2.7:  # Need at least 4 points for spline
                u_d_interp = u_d_sorted[mask_interp]
                delta_n_interp = delta_n_sorted[mask_interp]
                
                # Create smooth interpolation
                spline = UnivariateSpline(u_d_interp, delta_n_interp, s=0.1)
                u_d_smooth = np.linspace(u_d_interp.min(), u_d_interp.max(), 100)
                delta_n_smooth = spline(u_d_smooth)
                
                # Apply additional smoothing
                delta_n_smooth = uniform_filter1d(delta_n_smooth, size=3)
                
                ax.plot(u_d_smooth, delta_n_smooth, 'r-', linewidth=2, alpha=0.7, label='Fit')
        except:
            pass  # Skip interpolation if it fails
    
    # Add vertical line at u* = 2.74
    ax.axvline(x=2.74, color='blue', linestyle='--', linewidth=2.0, alpha=0.8, label='$u^{\\bigstar} = 2.74$')
    
    ax.set_xlabel('$u_d$', fontsize=12)
    ax.set_ylabel('$n_{\\rm max} - n_{\\rm min}$', fontsize=12)
    ax.legend(fontsize=10, ncol=1, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    # ax.set_xlim(0.5, 8)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/delta_n_vs_ud.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{outdir}/delta_n_vs_ud.pdf", dpi=200, bbox_inches='tight')
    print(f"\nSaved delta n vs u_d plot to {outdir}/delta_n_vs_ud.png")
    plt.show()
    plt.close()
    
    return data_by_label

def plot_combined_comparison(base_dirs, labels=None, outdir="multiple_u_d"):
    """Compare results across multiple parameter sets.
    
    Args:
        base_dirs: List of base directory paths
        labels: Optional custom labels for each dataset
        outdir: Output directory for combined plots
    """
    if labels is None:
        labels = [os.path.basename(d) for d in base_dirs]
    
    # Load all data
    all_data = load_multi_dataset(base_dirs, labels)
    
    if not all_data:
        print("No data found!")
        return
    
    print(f"\nTotal data points loaded: {len(all_data)}")
    
    # Organize data by label
    data_by_label = {label: {'u_d': [], 'u_measured': [], 'amplitude': []} for label in labels}
    
    for data_label, u_d, data in all_data:
        n_t = data['n_t']
        p_t = data['p_t']
        
        # Calculate measured velocity (from momentum)
        n_eff = np.maximum(n_t, 1e-7)
        v_t = p_t / n_eff
        u_measured = np.mean(v_t[:, -1])
        
        # Calculate max amplitude
        nbar = np.mean(n_t[:, -1])
        max_amp = np.max(n_t[:, -1]) - nbar
        
        data_by_label[data_label]['u_d'].append(u_d)
        data_by_label[data_label]['u_measured'].append(u_measured)
        data_by_label[data_label]['amplitude'].append(max_amp)
    
    # Prepare plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
    markers = ['o', 's', '^', 'D', 'v']
    
    all_u_d = []
    
    for idx, label in enumerate(labels):
        if not data_by_label[label]['u_d']:
            print(f"Warning: No data for {label}")
            continue
            
        u_d_list = data_by_label[label]['u_d']
        u_measured_list = data_by_label[label]['u_measured']
        max_amplitude_list = data_by_label[label]['amplitude']
        
        all_u_d.extend(u_d_list)
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        print(f"\nPlotting {label}: {len(u_d_list)} points")
        print(f"  u_d range: [{min(u_d_list):.2f}, {max(u_d_list):.2f}]")
        
        # Plot velocity comparison with visible points
        axes[0].plot(u_d_list, u_measured_list, marker=marker, color=color, 
                     label=label, linewidth=1.5, markersize=8, markeredgewidth=1.5,
                     markeredgecolor='white', alpha=0.9, linestyle='-')
        
        # Plot amplitude with visible points
        axes[1].plot(u_d_list, max_amplitude_list, marker=marker, color=color,
                     label=label, linewidth=1.5, markersize=8, markeredgewidth=1.5,
                     markeredgecolor='white', alpha=0.9, linestyle='-')
    
    # Add diagonal line for reference in velocity plot
    if all_u_d:
        u_min, u_max = min(all_u_d), max(all_u_d)
        axes[0].plot([u_min, u_max], [u_min, u_max], 
                     'k--', alpha=0.4, linewidth=2, label='$u_{measured} = u_d$')
    
    axes[0].set_ylabel('Measured velocity $\\langle u \\rangle$', fontsize=12)
    axes[0].legend(fontsize=10, ncol=1, loc='best', framealpha=0.9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Velocity and Amplitude Comparison', fontsize=13, fontweight='bold')
    
    axes[1].set_xlabel('Target drift velocity $u_d$', fontsize=12)
    axes[1].set_ylabel('Max amplitude $\\Delta n_{max}$', fontsize=12)
    axes[1].legend(fontsize=10, ncol=1, loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/combined_comparison.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{outdir}/combined_comparison.pdf", dpi=200, bbox_inches='tight')
    print(f"\nSaved combined comparison to {outdir}/combined_comparison.png")
    plt.show()
    plt.close()
    
    return all_data

def find_available_simulations():
    """Automatically find all available simulation files in out_drift_ud* subdirectories"""
    data_files = []
    
    # Search in multiple_u_d/out_drift_ud* subdirectories
    if os.path.exists("multiple_u_d"):
        for item in os.listdir("multiple_u_d"):
            if item.startswith("out_drift_ud") and os.path.isdir(os.path.join("multiple_u_d", item)):
                # Extract u_d from directory name
                try:
                    u_d_str = item.replace("out_drift_ud", "")
                    u_d = float(u_d_str)
                    
                    # Look for data file in this subdirectory
                    subdir_path = os.path.join("multiple_u_d", item)
                    for file in os.listdir(subdir_path):
                        if file.startswith("data_m01_ud") and file.endswith(".npz"):
                            filepath = os.path.join(subdir_path, file)
                            data_files.append((filepath, u_d))
                            break  # Only take first matching file
                except ValueError:
                    continue
    
    # Search in out_drift directory (main level only)
    if os.path.exists("out_drift"):
        for file in os.listdir("out_drift"):
            if file.startswith("data_m01_ud") and file.endswith(".npz"):
                try:
                    u_d_str = file.replace("data_m01_ud", "").replace(".npz", "")
                    u_d = float(u_d_str)
                    filepath = os.path.join("out_drift", file)
                    data_files.append((filepath, u_d))
                except ValueError:
                    continue
    
    # Sort by u_d value
    data_files.sort(key=lambda x: x[1])
    
    print(f"Found {len(data_files)} simulation files:")
    for filepath, u_d in data_files:
        print(f"  u_d={u_d}: {filepath}")
    
    return data_files

if __name__ == "__main__":
    # Compare results from three different parameter sets
    base_dirs = [
        "multiple_u_d/delta n=delta p=0.03(cos3x+cos5x+cos8x+cos13x)",
        "multiple_u_d/delta n=delta p=0.05(cos3x+cos5x+cos8x+cos13x)",
        "multiple_u_d/quadratic;delta n=delta p=0.05(cos3x+cos5x+cos8x+cos13x)"
    ]
    
    custom_labels = [
        "δn,δp = 0.03",
        "δn,δp = 0.05 (uniform)",
        "δn,δp = 0.05 (quadratic)"
    ]
    
    print("=" * 60)
    print("VELOCITY ANALYSIS: Multiple Parameter Sets")
    print("=" * 60)
    
    # Plot velocity, pulse density, and frequency
    # print("Generating velocity analysis (u_true, n_pulses, frequency)...")
    # velocity_data = plot_combined_velocity_analysis(base_dirs, labels=custom_labels)
    
    print("\n" + "=" * 60)
    print(f"Analysis complete! Generated velocity_vs_ud_combined.png")
    print("=" * 60)
    
    # Load data and plot velocity evolution
    filename = "multiple_u_d/out_drift_ud18.0000/data_m01_ud18.0.npz"
    data = load_data(filename)
    plot_velocity_evolution(data, 18.0)



    # Single file analysis (moved to separate file)
    # Use plot_u_true_vs_time.py for u_true vs time analysis
    
    # Automatically find and analyze all available simulations (commented out)
    data_files = find_available_simulations()
    if data_files:
        # plot_velocity_vs_ud(data_files)
        # plot_multiple_ud_panel(data_files)
        pass
    else:
        print("No simulation files found!")
    
    print("\nAll plots generated!")