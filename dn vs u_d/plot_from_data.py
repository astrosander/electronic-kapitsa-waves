import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# --- unified TeX-style appearance (MathText, no system LaTeX needed) ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "STIXGeneral",  # match math fonts
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,   # proper minus sign
})

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

def calculate_N_from_fourier(n_profile, L):
    """
    Calculate n_pulses from the Fourier spectrum of n(x) at t=t_final.
    
    This function:
    1. Takes the FFT of the density profile
    2. Finds the dominant wavenumber k_dominant
    3. Calculates N = k_dominant * L / (2π)
    4. Converts to density: n_pulses = N / L
    
    Parameters:
    -----------
    n_profile : array
        Density profile n(x) at t=t_final
    L : float
        Domain length
        
    Returns:
    --------
    n_pulses_fourier : float
        Pulse density calculated from Fourier spectrum (N/L)
    k_dominant : float
        Dominant wavenumber
    k_spectrum : array
        Wavenumber array
    power_spectrum : array
        Power spectrum
    """
    # Remove mean to focus on fluctuations
    n_fluct = n_profile - np.mean(n_profile)
    
    # Take FFT
    n_hat = np.fft.fft(n_fluct)
    power_spectrum = np.abs(n_hat)**2
    
    # Create wavenumber array
    N = len(n_profile)
    k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    
    # Only consider positive wavenumbers
    k_pos = k[1:N//2+1]
    power_pos = power_spectrum[1:N//2+1]
    
    # Find dominant wavenumber (excluding k=0)
    if len(k_pos) > 0:
        dominant_idx = np.argmax(power_pos)
        k_dominant = k_pos[dominant_idx]
        N_fourier = k_dominant * L / (2 * np.pi)
        n_pulses_fourier = N_fourier / L  # Convert to density
    else:
        k_dominant = 0.0
        N_fourier = 0.0
        n_pulses_fourier = 0.0
    
    return n_pulses_fourier, k_dominant, k_pos, power_pos

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
            threshold = n_mean + 0.1 * n_std
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

def plot_multiple_ud_panel(data_files=None, base_dirs=None, labels=None, max_rows=None):
    """Plot density profiles panels.
    
    Modes:
      - Single column (default): pass data_files or none to auto-discover in one dir.
      - Multi-column (requested): pass base_dirs (list) and optional labels; columns per base_dir.
    """
    # Helper: scan a base_dir for u_d->filepath mapping
    def scan_dir(dir_path):
        import re
        mapping = {}
        if not os.path.isdir(dir_path):
            return mapping
        for item in os.listdir(dir_path):
            if item.startswith("out_drift_ud"):
                subdir = os.path.join(dir_path, item)
                if not os.path.isdir(subdir):
                    continue
                
                # Handle both old and new directory name formats
                u_d_str = item.replace("out_drift_ud", "")
                # Handle new format: out_drift_ud4p6000 -> 4p6000 -> 4.6000
                if "p" in u_d_str:
                    u_d_str = u_d_str.replace("p", ".")
                
                try:
                    u_d_val = float(u_d_str)
                except ValueError:
                    continue
                
                # pick first data_*.npz (support both old and new formats)
                files = [f for f in os.listdir(subdir) if f.startswith("data_") and f.endswith(".npz")]
                if files:
                    mapping[u_d_val] = os.path.join(subdir, files[0])
        return mapping

    if base_dirs is None:
        # Single column fallback
        if data_files is None:
            data_files = find_available_simulations()
        if not data_files:
            print("No data files found for panel plot!")
            return
        if max_rows is not None:
            data_files = data_files[:max_rows]
        n_files = len(data_files)
        fig, axes = plt.subplots(n_files, 1, figsize=(8.0, 2*n_files))
        if n_files == 1:
            axes = [axes]
        for i, (filename, u_d) in enumerate(data_files):
            try:
                data = load_data(filename)
                n_t = data['n_t']
                L = data['L']
                x = np.linspace(0, L, n_t.shape[0], endpoint=False)
                
                # Use final time step (no time averaging)
                n_final = n_t[:, -1]
                axes[i].plot(x, n_final, 'b-', linewidth=1.3)
                axes[i].text(0.02, 0.95, f'$u_d={u_d}$', transform=axes[i].transAxes,
                             fontsize=9, va='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlim(0, L)
                axes[i].tick_params(labelsize=8)
                if i == n_files - 1:
                    axes[i].set_xlabel('$x$', fontsize=11)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                axes[i].text(0.5, 0.5, f'Error loading u_d={u_d}', transform=axes[i].transAxes, ha='center', va='center')
        plt.subplots_adjust(hspace=0.1, top=0.95)
        os.makedirs("multiple_u_d", exist_ok=True)
        plt.savefig("multiple_u_d/final_profiles_panel.png", dpi=160, bbox_inches='tight')
        plt.savefig("multiple_u_d/final_profiles_panel.svg", dpi=160, bbox_inches='tight')
        plt.show()
        plt.close()
        return

    # Multi-column grid
    if labels is None:
        labels = [os.path.basename(d.rstrip('/\\')) or d for d in base_dirs]
    n_cols = len(base_dirs)
    dir_maps = [scan_dir(d) for d in base_dirs]
    # Union of all u_d values
    all_ud = sorted(set().union(*[set(m.keys()) for m in dir_maps]))
    if max_rows is not None:
        all_ud = all_ud[:max_rows]
    n_rows = len(all_ud)
    if n_rows == 0:
        print("No data files found across provided base_dirs for panel plot!")
        return
    # Wider figure per column, compact per row
    fig_width = max(4.8 * n_cols, 8.0)
    fig_height = max(1.6 * n_rows, 3.2)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False, sharex=False, sharey=True
    )
    for r, u_d in enumerate(all_ud):
        for c, (label, dmap) in enumerate(zip(labels, dir_maps)):
            ax = axes[r][c]
            filepath = dmap.get(u_d)
            if filepath is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.axis('off')
                continue
            try:
                data = load_data(filepath)
                n_t = data['n_t']
                L = data['L']
                x = np.linspace(0, L, n_t.shape[0], endpoint=False)
                # Use final time step (no time averaging)
                n_final = n_t[:, -1]
                ax.plot(x, n_final, lw=1.2)
                ax.grid(True, alpha=0.25)
                ax.set_xlim(0, L)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.axis('off')
                continue
            if c == 0:
                ax.set_ylabel(f"$u_d={u_d:g}$", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel('$x$', fontsize=10)
            if r == 0:
                ax.set_title(label, fontsize=10)
            ax.tick_params(labelsize=8)
    # Use available space with minimal gaps
    fig.subplots_adjust(left=0.06, right=0.995, top=0.97, bottom=0.06,
                        wspace=0.18, hspace=0.06)
    os.makedirs("multiple_u_d", exist_ok=True)
    plt.savefig("multiple_u_d/final_profiles_panel_grid_n.png", dpi=180, bbox_inches='tight')
    plt.savefig("multiple_u_d/final_profiles_panel_grid_n.pdf", dpi=180, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_multiple_ud_panel_p(data_files=None, base_dirs=None, labels=None, max_rows=None):
    """Plot momentum profiles panels (final time). Supports multiple columns for base_dirs."""
    def scan_dir(dir_path):
        import re
        mapping = {}
        if not os.path.isdir(dir_path):
            return mapping
        for item in os.listdir(dir_path):
            if item.startswith("out_drift_ud"):
                subdir = os.path.join(dir_path, item)
                if not os.path.isdir(subdir):
                    continue
                
                # Handle both old and new directory name formats
                u_d_str = item.replace("out_drift_ud", "")
                # Handle new format: out_drift_ud4p6000 -> 4p6000 -> 4.6000
                if "p" in u_d_str:
                    u_d_str = u_d_str.replace("p", ".")
                
                try:
                    u_d_val = float(u_d_str)
                except ValueError:
                    continue
                
                # pick first data_*.npz (support both old and new formats)
                files = [f for f in os.listdir(subdir) if f.startswith("data_") and f.endswith(".npz")]
                if files:
                    mapping[u_d_val] = os.path.join(subdir, files[0])
        return mapping

    if base_dirs is None:
        if data_files is None:
            data_files = find_available_simulations()
        if not data_files:
            print("No data files found for momentum panel plot!")
            return
        if max_rows is not None:
            data_files = data_files[:max_rows]
        n_files = len(data_files)
        fig, axes = plt.subplots(n_files, 1, figsize=(8.0, 2*n_files))
        if n_files == 1:
            axes = [axes]
        for i, (filename, u_d) in enumerate(data_files):
            try:
                data = load_data(filename)
                p_t = data['p_t']
                L = data['L']
                # x = np.linspace(0, L, p_t.shape[0], endpoint=False)
                # p_final = p_t[:, -1]
                # axes[i].plot(x, p_final, 'r-', linewidth=1.3)
                # Average over last 20% of time points
                p_avg = np.mean(p_t[:, -max(1, p_t.shape[1]//5):], axis=1)
                axes[i].plot(x, p_avg, 'r-', linewidth=1.3)
                axes[i].text(0.02, 0.95, f'$u_d={u_d}$', transform=axes[i].transAxes,
                             fontsize=9, va='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlim(0, L)
                axes[i].tick_params(labelsize=8)
                if i == n_files - 1:
                    axes[i].set_xlabel('$x$', fontsize=11)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                axes[i].text(0.5, 0.5, f'Error loading u_d={u_d}', transform=axes[i].transAxes, ha='center', va='center')
        plt.subplots_adjust(hspace=0.1, top=0.95)
        os.makedirs("multiple_u_d", exist_ok=True)
        plt.savefig("multiple_u_d/final_profiles_panel_p.png", dpi=160, bbox_inches='tight')
        plt.savefig("multiple_u_d/final_profiles_panel_p.svg", dpi=160, bbox_inches='tight')
        plt.show()
        plt.close()
        return

    if labels is None:
        labels = [os.path.basename(d.rstrip('/\\')) or d for d in base_dirs]
    n_cols = len(base_dirs)
    dir_maps = [scan_dir(d) for d in base_dirs]
    all_ud = sorted(set().union(*[set(m.keys()) for m in dir_maps]))
    if max_rows is not None:
        all_ud = all_ud[:max_rows]
    n_rows = len(all_ud)
    if n_rows == 0:
        print("No data files found across provided base_dirs for momentum panel plot!")
        return
    # Wider figure per column, compact per row
    fig_width = max(4.8 * n_cols, 8.0)
    fig_height = max(1.6 * n_rows, 3.2)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False, sharex=False, sharey=True
    )
    for r, u_d in enumerate(all_ud):
        for c, (label, dmap) in enumerate(zip(labels, dir_maps)):
            ax = axes[r][c]
            filepath = dmap.get(u_d)
            if filepath is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.axis('off')
                continue
            try:
                data = load_data(filepath)
                p_t = data['p_t']
                L = data['L']
                x = np.linspace(0, L, p_t.shape[0], endpoint=False)
                # p_final = p_t[:, -1]
                # ax.plot(x, p_final, lw=1.2, color='C1')


                # Average over last 20% of time points
                p_avg = np.mean(p_t[:, -max(1, p_t.shape[1]//5):], axis=1)
                ax.plot(x, p_avg, lw=1.2, color='C1')
                ax.grid(True, alpha=0.25)
                ax.set_xlim(0, L)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.axis('off')
                continue
            if c == 0:
                ax.set_ylabel(f"$u_d={u_d:g}$", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel('$x$', fontsize=10)
            if r == 0:
                ax.set_title(label, fontsize=10)
            ax.tick_params(labelsize=8)
    # Use available space with minimal gaps
    fig.subplots_adjust(left=0.06, right=0.995, top=0.97, bottom=0.06,
                        wspace=0.18, hspace=0.06)
    os.makedirs("multiple_u_d", exist_ok=True)
    plt.savefig("multiple_u_d/final_profiles_panel_grid_p.png", dpi=180, bbox_inches='tight')
    plt.savefig("multiple_u_d/final_profiles_panel_grid_p.pdf", dpi=180, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_overlay_final_profiles_n(data_files=None):
    """Overlay final n(x) for all u_d on a single axes."""
    if data_files is None:
        data_files = find_available_simulations()
    if not data_files:
        print("No data files found for overlay plot (n)!")
        return
    
    plt.figure(figsize=(9.0, 4.0))
    for filename, u_d in data_files:
        try:
            data = load_data(filename)
            n_t = data['n_t']
            L = data['L']
            x = np.linspace(0, L, n_t.shape[0], endpoint=False)
            n_final = n_t[:, -1]
            plt.plot(x, n_final, lw=1.2, label=f"u_d={u_d:g}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    plt.xlabel('$x$')
    plt.ylabel('$n(x, t_{\\rm final})$')
    plt.title('Final density profiles across $u_d$')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8, frameon=False)
    os.makedirs("multiple_u_d", exist_ok=True)
    plt.tight_layout()
    plt.savefig("multiple_u_d/final_profiles_overlay_n.png", dpi=180, bbox_inches='tight')
    plt.savefig("multiple_u_d/final_profiles_overlay_n.pdf", dpi=180, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_overlay_final_profiles_p(data_files=None):
    """Overlay final p(x) for all u_d on a single axes."""
    if data_files is None:
        data_files = find_available_simulations()
    if not data_files:
        print("No data files found for overlay plot (p)!")
        return
    
    plt.figure(figsize=(9.0, 4.0))
    for filename, u_d in data_files:
        try:
            data = load_data(filename)
            p_t = data['p_t']
            L = data['L']
            x = np.linspace(0, L, p_t.shape[0], endpoint=False)
            p_final = p_t[:, -1]
            plt.plot(x, p_final, lw=1.2, label=f"u_d={u_d:g}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    plt.xlabel('$x$')
    plt.ylabel('$p(x, t_{\\rm final})$')
    plt.title('Final momentum profiles across $u_d$')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8, frameon=False)
    os.makedirs("multiple_u_d", exist_ok=True)
    plt.tight_layout()
    plt.savefig("multiple_u_d/final_profiles_overlay_p.png", dpi=180, bbox_inches='tight')
    plt.savefig("multiple_u_d/final_profiles_overlay_p.pdf", dpi=180, bbox_inches='tight')
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
                    
                # Extract u_d from directory name (handle both old and new formats)
                u_d_str = item.replace("out_drift_ud", "")
                # Handle new format: out_drift_ud4p6000 -> 4p6000 -> 4.6000
                if "p" in u_d_str:
                    u_d_str = u_d_str.replace("p", ".")
                
                try:
                    u_d = float(u_d_str)
                except ValueError:
                    continue
                
                # Find data file (support both old and new formats)
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
    data_by_label = {label: {'u_d': [], 'u_true': [], 'n_pulses': [], 'frequency': [], 'delta_n': [], 'N_fourier': []} for label in labels}
    
    # Collect all data for interpolation
    all_u_d = []
    all_u_true = []
    all_n_pulses = []
    all_frequency = []
    all_delta_n = []
    all_N_fourier = []
    
    # Critical velocity u* = 1.2500
    u_star = 1.2500
    
    for data_label, u_d, data in all_data:
        # Filter: only process data where u_d > u*
        if u_d <= u_star:
            print(f"  Skipping u_d={u_d:.4f} (u_d <= u* = {u_star})")
            continue
            
        n_t = data['n_t']
        t = data['t']
        L = data['L']
        
        # Calculate u_true using spatial correlation
        idx_t1 = -2
        idx_t2 = -1
        
        print("len: ", len(n_t[1]))

        n_t1 = n_t[:, idx_t1]
        n_t2 = n_t[:, idx_t2]
        t1 = t[idx_t1]
        t2 = t[idx_t2]
        
        dx = L / len(n_t1)
        x = np.linspace(0, L, len(n_t1), endpoint=False)
        
        # Remove mean to focus on fluctuations (same as main.py)
        dn_t1 = n_t1 - np.mean(n_t1)
        dn_t2 = n_t2 - np.mean(n_t2)
        
        # Use robust Fourier-based velocity estimation (same method as main.py)
        from numpy.fft import rfft, irfft, rfftfreq
        
        def _fourier_shift_real(f, shift, L):
            """Circularly shift a real 1D signal by a non-integer amount using FFT."""
            N = f.size
            k = 2*np.pi * rfftfreq(N, d=L/N)      # wavenumbers (>=0)
            F = rfft(f)
            return irfft(F * np.exp(1j*k*shift), n=N)

        def estimate_velocity_fourier(n_t1, n_t2, t1, t2, L, power_floor=1e-3):
            """Estimate spatial shift and velocity using cross-spectrum phase."""
            N = n_t1.size
            f1 = n_t1 - n_t1.mean()
            f2 = n_t2 - n_t2.mean()

            k = 2*np.pi * rfftfreq(N, d=L/N)
            F1 = rfft(f1)
            F2 = rfft(f2)
            C  = np.conj(F1) * F2
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
            
            # Fix sign convention: positive shift = forward motion (rightward drift)
            shift = -shift  # Flip sign to match spatial convention
            
            shift = (shift + 0.5*L) % L - 0.5*L

            dt = float(t2 - t1)
            u = shift / dt
            return u, shift
        
        # Calculate velocity using Fourier method
        u_true, shift_opt = estimate_velocity_fourier(n_t1, n_t2, t1, t2, L)
        
        # Build correlation curve for visualization
        N = len(n_t1)
        f1 = n_t1 - n_t1.mean()
        f2 = n_t2 - n_t2.mean()
        F1 = rfft(f1)
        F2 = rfft(f2)
        xcorr = irfft(np.conj(F1) * F2, n=N)
        dx = L/N
        shifts = dx * (np.arange(N) - (N//2))
        xcorr = np.roll(xcorr, -N//2)
        correlations = (xcorr - xcorr.min())/(xcorr.max()-xcorr.min() + 1e-15)
        
        # Calculate density of pulses (count peaks in final density profile)
        n_final = n_t[:, -1]
        n_mean = np.mean(n_final)
        n_std = np.std(n_final)
        
        # Find peaks (local maxima above threshold)
        threshold = n_mean + 0.1 * n_std
        peaks, _ = find_peaks(n_final, height=threshold, distance=5)
        N_pulses = len(peaks)
        n_pulses = N_pulses / L
        
        # Calculate delta_n using peak/valley count method
        if len(peaks) > 0:
            # Find valleys (minima) between peaks
            valleys, _ = find_peaks(-n_final, distance=5)  # Find minima by inverting the signal
            
            # Filter valleys to be between peaks (in the valleys)
            if len(valleys) > 0:
                # Keep only valleys that are between peaks
                valid_valleys = []
                for valley in valleys:
                    # Check if this valley is between any two peaks
                    for i in range(len(peaks)-1):
                        if peaks[i] < valley < peaks[i+1]:
                            valid_valleys.append(valley)
                            break
                    # Also check if valley is before first peak or after last peak
                    if len(peaks) > 0:
                        if valley < peaks[0] or valley > peaks[-1]:
                            valid_valleys.append(valley)
                
                # Calculate delta_n as max(number_of_peaks, number_of_valleys)
                delta_n = max(len(peaks), len(valid_valleys))
            else:
                # If no valleys found, delta_n is just the number of peaks
                delta_n = len(peaks)
        else:
            # Fallback: if no peaks found, delta_n = 0
            delta_n = 0
        
        # Calculate frequency
        frequency = abs(u_true) * n_pulses
        
        # Calculate n_pulses from Fourier spectrum
        n_pulses_fourier, k_dominant, k_spectrum, power_spectrum = calculate_N_from_fourier(n_final, L)
        
        print(f"  Processing u_d={u_d:.4f} (u_d > u* = {u_star})")
        print(f"    Peaks: {len(peaks)}, Valleys: {len(valid_valleys) if 'valid_valleys' in locals() else 0}")
        print(f"    delta_n = max({len(peaks)}, {len(valid_valleys) if 'valid_valleys' in locals() else 0}) = {delta_n}")
        print(f"    n_pulses_fourier = {n_pulses_fourier:.4f}, k_dominant = {k_dominant:.3f}")
        
        data_by_label[data_label]['u_d'].append(u_d)
        data_by_label[data_label]['u_true'].append(u_true)
        data_by_label[data_label]['n_pulses'].append(n_pulses)
        data_by_label[data_label]['frequency'].append(frequency)
        data_by_label[data_label]['delta_n'].append(delta_n)
        data_by_label[data_label]['N_fourier'].append(n_pulses_fourier)
        
        # Collect all data for interpolation
        all_u_d.append(u_d)
        all_u_true.append(u_true)
        all_n_pulses.append(n_pulses)
        all_frequency.append(frequency)
        all_delta_n.append(delta_n)
        all_N_fourier.append(n_pulses_fourier)
    
    # Create plots - more compact design similar to original velocity_vs_ud.png
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Same color for all datasets, different marker shapes
    color = 'black'  # Single color for all datasets
    markers = ['o', 's', '^', '*', 'v', '<', '>', 'p', '*', 'h', 
               'H', '+', 'x', 'X', '|']
    
    # Add smooth interpolation lines with specific ranges for each plot
    # Plot this FIRST so it appears below the data points
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
            # Plot 1: u_true vs u_d - plot ALL data points (no upper limit on u_d)
            mask1 = (u_d_sorted > u_star)  # Only filter for u_d > u*, no upper limit
            if np.sum(mask1) > 0:  # Plot even with just 1 point
                u_d_1 = u_d_sorted[mask1]
                u_true_1 = u_true_sorted[mask1]
                
                # Plot ALL individual data points without binning or interpolation
                # This ensures every single u_d value is plotted
                ax1.plot(u_d_1, u_true_1, 'o', color='gray', alpha=0.3, markersize=2, zorder=1)
                
                # Optional: Add interpolation if we have enough points
                if len(u_d_1) > 3:
                    # Use moving average for binning points at same u_d (but keep all original points)
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
                        # ax1.plot(u_d_smooth_1, u_true_smooth, 'r-', linewidth=2.5, alpha=0.9, label='Mean trend', zorder=100)
            
            # Plot 2: n_pulses vs u_d - plot ALL data points (no upper limit on u_d)
            mask2 = (u_d_sorted > u_star)  # Only filter for u_d > u*, no upper limit
            if np.sum(mask2) > 0:  # Plot even with just 1 point
                u_d_2 = u_d_sorted[mask2]
                n_pulses_2 = n_pulses_sorted[mask2]
                
                # Plot ALL individual data points without binning or interpolation
                # This ensures every single u_d value is plotted
                ax2.plot(u_d_2, n_pulses_2, 'o', color='gray', alpha=0.3, markersize=2, zorder=1)
                
                # Optional: Add interpolation if we have enough points
                if len(u_d_2) > 3:
                    # First segment: u* < u_d <= 3.0 (if any data exists in this range)
                    mask2a = (u_d_sorted > u_star) & (u_d_sorted <= 3.0)
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
                            # ax2.plot(u_d_smooth_2a, n_pulses_smooth_a, 'r-', linewidth=2.5, alpha=0.9, label='Mean trend', zorder=100)
                    
                    # Second segment: 3.0 < u_d (no upper limit, plot ALL higher u_d values)
                    mask2b = (u_d_sorted > 3.0)
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
                            # ax2.plot(u_d_smooth_2b, n_pulses_smooth_b, 'r-', linewidth=2.5, alpha=0.9, zorder=100)
            
            # Add Fourier-based n_pulses values as additional curve on ax2
            if all_N_fourier:
                n_pulses_fourier_sorted = np.array(all_N_fourier)[sorted_indices]
                mask2_fourier = (u_d_sorted > u_star)  # Same filter as n_pulses
                if np.sum(mask2_fourier) > 0:
                    u_d_2_fourier = u_d_sorted[mask2_fourier]
                    n_pulses_fourier_2 = n_pulses_fourier_sorted[mask2_fourier]
                    
                    # Plot n_pulses_fourier as a line with different style
                    ax2.plot(u_d_2_fourier, n_pulses_fourier_2, 'g--', linewidth=2, alpha=0.8, 
                            label='n_pulses (Fourier)', zorder=10)
            
            # Plot 3: frequency vs u_d - plot ALL data points (no upper limit on u_d)
            mask3 = (u_d_sorted > u_star)  # Only filter for u_d > u*, no upper limit
            if np.sum(mask3) > 0:  # Plot even with just 1 point
                u_d_3 = u_d_sorted[mask3]
                freq_3 = freq_sorted[mask3]
                
                # Plot ALL individual data points without binning or interpolation
                # This ensures every single u_d value is plotted
                ax3.plot(u_d_3, freq_3, 'o', color='gray', alpha=0.3, markersize=2, zorder=1)
                
                # Optional: Add interpolation if we have enough points
                if len(u_d_3) > 3:
                    # First segment: u* < u_d <= 3.0 (if any data exists in this range)
                    mask3a = (u_d_sorted > u_star) & (u_d_sorted <= 3.0)
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
                            # ax3.plot(u_d_smooth_3a, freq_smooth_a, 'r-', linewidth=2.5, alpha=0.9, label='Mean trend', zorder=100)
                    
                    # Second segment: 3.0 < u_d (no upper limit, plot ALL higher u_d values)
                    mask3b = (u_d_sorted > 3.0)
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
                    # ax3.plot(u_d_smooth_3b, freq_smooth_b, 'r-', linewidth=2.5, alpha=0.9, zorder=100)
            
        except Exception as e:
            print(f"Warning: Could not create smooth lines: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot data points AFTER the interpolation so they appear on top
    for idx, label in enumerate(labels):
        if not data_by_label[label]['u_d']:
            continue
        
        # Filter for u_d > u* only (no upper limit - plot ALL data points)
        u_d_arr = np.array(data_by_label[label]['u_d'])
        u_true_arr = np.array(data_by_label[label]['u_true'])
        n_pulses_arr = np.array(data_by_label[label]['n_pulses'])
        freq_arr = np.array(data_by_label[label]['frequency'])
        delta_n_arr = np.array(data_by_label[label]['delta_n'])
        N_fourier_arr = np.array(data_by_label[label]['N_fourier'])
        
        mask = (u_d_arr > u_star)  # Only filter for u_d > u*, no upper limit
        
        if not np.any(mask):
            continue
        
        u_d_filtered = u_d_arr[mask]
        u_true_filtered = u_true_arr[mask]
        n_pulses_filtered = n_pulses_arr[mask]
        freq_filtered = freq_arr[mask]
        delta_n_filtered = delta_n_arr[mask]/L
        N_fourier_filtered = N_fourier_arr[mask]
        
        marker = markers[idx % len(markers)]
        
        # Plot 1: u_true vs u_d (points only, no lines, larger size)
        ax1.scatter(u_d_filtered, np.abs(u_true_filtered), marker=marker, #color=color,
                   label=label, s=24, alpha=0.8)
        
        # Plot 2: n_pulses vs u_d (points only, no lines, larger size)
        ax2.scatter(u_d_filtered, delta_n_filtered, marker=marker, #color=color,
                   label=label, s=24, alpha=0.8)
        
        # Plot 2: n_pulses_fourier vs u_d (line with different style for each dataset)
        linestyle = ['-', '--', '-.', ':'][idx % 4]
        ax2.plot(u_d_filtered, N_fourier_filtered, linestyle=linestyle, 
                linewidth=2, alpha=0.8, label=f'{label} (n_pulses_fourier)')
        
        # Plot 3: frequency vs u_d (points only, no lines, larger size)
        ax3.scatter(u_d_filtered, freq_filtered, marker=marker, #color=color,
                   label=label, s=24, alpha=0.8)
    
    # Plot 1: u_true vs u_d (exact labels from original)
    ax1.set_xlabel('$u_d$')
    ax1.set_ylabel('$u_{\\text{true}}$')
    ax1.set_title('$u_{\\text{true}}$ vs $u_d$')
    ax1.legend(fontsize=8, ncol=2, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=1.2500, color='blue', linestyle='--', linewidth=1, alpha=0.8, label='$u^{\\bigstar} = 1.2500$')
    
    # Plot 2: n_pulses vs u_d (exact labels from original)
    ax2.set_xlabel('$u_d$')
    ax2.set_ylabel('$n_{\\text{pulses}} = N/L$')
    ax2.set_title('$n_{\\text{pulses}}$ vs $u_d$')
    ax2.legend(fontsize=8, ncol=2, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=1.2500, color='blue', linestyle='--', linewidth=1, alpha=0.8, label='$u^{\\bigstar} = 1.2500$')
    
    # Plot 3: frequency vs u_d (exact labels from original)
    ax3.set_xlabel('$u_d$')
    ax3.set_ylabel('$f = u_{\\text{true}} \\cdot n_{\\text{pulses}}$')
    ax3.set_title('$f$ vs $u_d$')
    ax3.legend(fontsize=8, ncol=2, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=1.2500, color='blue', linestyle='--', linewidth=1, alpha=0.8, label='$u^{\\bigstar} = 1.2500$')
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/velocity_vs_ud_combined.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{outdir}/velocity_vs_ud_combined.svg", dpi=200, bbox_inches='tight')
    print(f"\nSaved velocity analysis to {outdir}/velocity_vs_ud_combined.png")
    plt.show()
    plt.close()
    
    return data_by_label

def plot_delta_n_vs_ud(base_dirs, labels=None, outdir="multiple_u_d", x0_fraction=0.5):
    """Plot delta n (n_max - n_min) vs u_d for combined datasets.
    
    Args:
        base_dirs: List of base directory paths
        labels: Optional custom labels for each dataset
        outdir: Output directory for plots
        x0_fraction: Fraction of domain length to use for current measurement (default: 0.5 = middle)
    """
    if labels is None:
        labels = [os.path.basename(d) for d in base_dirs]
    
    # Load all data
    all_data = load_multi_dataset(base_dirs, labels)
    
    if not all_data:
        print("No data found!")
        return
    
    # Print summary of loaded u_d values
    u_d_values = [u_d for _, u_d, _ in all_data]
    print(f"\nLoaded {len(u_d_values)} simulations with u_d values:")
    print(f"  Range: {min(u_d_values):.4f} to {max(u_d_values):.4f}")
    print(f"  Below u* = 1.2500: {sum(1 for u in u_d_values if u < 1.2500)} simulations")
    print(f"  Above u* = 1.2500: {sum(1 for u in u_d_values if u >= 1.2500)} simulations")
    
    # Organize data by label
    data_by_label = {label: {'u_d': [], 'delta_n': [], 'j_avg': [], 'sigma_p': []} for label in labels}
    
    # Collect all data for interpolation
    all_u_d = []
    all_delta_n = []
    all_j_avg = []
    
    # Critical velocity u* = 1.2500
    u_star = 1.2500
    
    for data_label, u_d, data in all_data:
        # Process ALL data (both subcritical and supercritical)
        print(f"  Processing u_d={u_d:.4f}")
            
        n_t = data['n_t']
        t = data['t']
        L = data['L']
        
        # Calculate delta n using peak-based n_max and valley-based n_min
        n_final = n_t[:, -1]  # Last time step
        
        # Calculate n_max as average of peak values and n_min as average of valley values
        from scipy.signal import find_peaks
        
        if len(n_final) > 0:
            # Find peaks (local maxima above threshold)
            n_mean = np.mean(n_final)
            n_std = np.std(n_final)
            threshold = n_mean + 0.1 * n_std
            peaks, _ = find_peaks(n_final, height=threshold, distance=5)
            
            if len(peaks) > 0:
                # Calculate n_max as average of peak values
                peak_values = n_final[peaks]
                n_max = np.mean(peak_values)  # Average of peak values
                
                # Find valleys (minima) between peaks
                valleys, _ = find_peaks(-n_final, distance=5)  # Find minima by inverting the signal
                
                # Filter valleys to be between peaks (in the valleys)
                if len(valleys) > 0:
                    # Keep only valleys that are between peaks
                    valid_valleys = []
                    for valley in valleys:
                        # Check if this valley is between any two peaks
                        for i in range(len(peaks)-1):
                            if peaks[i] < valley < peaks[i+1]:
                                valid_valleys.append(valley)
                                break
                        # Also check if valley is before first peak or after last peak
                        if len(peaks) > 0:
                            if valley < peaks[0] or valley > peaks[-1]:
                                valid_valleys.append(valley)
                    
                    if len(valid_valleys) > 0:
                        valley_values = n_final[valid_valleys]
                        n_min = np.mean(valley_values)  # Average of valley values
                    else:
                        # If no valid valleys found, use global minimum
                        n_min = np.min(n_final)
                else:
                    # If no valleys found, use global minimum
                    n_min = np.min(n_final)
                
                delta_n = n_max - n_min
            else:
                # Fallback to global min/max if no peaks found
                n_max = np.max(n_final)
                n_min = np.min(n_final)
                delta_n = n_max - n_min
        else:
            delta_n = 0.0
        
        # Calculate time-averaged current j = v*n = p at fixed location x0
        p_t = data['p_t']
        Nx = data['Nx']
        
        # Determine spatial index for measurement
        x0_idx = int(x0_fraction * Nx)
        
        # Extract momentum time series at x0
        p_at_x0 = p_t[x0_idx, :]
        
        # Calculate time-averaged current using Gaussian weighting
        # Gaussian centered at 85% of final time with width 10% of final time
        t_final = t[-1]
        t_center = 0.85 * t_final
        t_width = 0.1 * t_final
        
        # Create Gaussian weights
        gaussian_weights = np.exp(-0.5 * ((t - t_center) / t_width)**2)
        
        # Normalize weights so they sum to 1
        gaussian_weights = gaussian_weights / np.sum(gaussian_weights)
        
        # Calculate weighted average and weighted standard deviation
        j_avg = np.sum(p_at_x0 * gaussian_weights)
        
        # For weighted standard deviation, we need to be careful
        # Calculate weighted variance: sum(w * (x - mean)^2) / sum(w)
        weighted_mean = j_avg
        weighted_variance = np.sum(gaussian_weights * (p_at_x0 - weighted_mean)**2) / np.sum(gaussian_weights)
        # sigma_p = np.sqrt(weighted_variance)
        # effective_n = (np.sum(gaussian_weights))**2 / np.sum(gaussian_weights**2)
        # sigma_p = np.sqrt(weighted_variance) / np.sqrt(effective_n)
        
        effective_n = (np.sum(gaussian_weights))**2 / np.sum(gaussian_weights**2)
        sigma_p = np.sqrt(weighted_variance) / np.sqrt(effective_n)

        # Print detailed information about the calculation
        regime = "subcritical" if u_d < u_star else "supercritical"
        if len(n_final) > 0 and len(peaks) > 0:
            print(f"    [{regime}] Peaks: {len(peaks)}, Valleys: {len(valid_valleys) if 'valid_valleys' in locals() else 0}")
            print(f"    n_max = {n_max:.3f} (avg of {len(peaks)} peaks), n_min = {n_min:.3f} (avg of {len(valid_valleys) if 'valid_valleys' in locals() else 0} valleys)")
            print(f"    delta_n = {delta_n:.3f}")
            print(f"    j_avg (Gaussian t={t_center}±{t_width}, x={x0_fraction:.2f}L) = {j_avg:.4f}, σ_p = {sigma_p:.4f}")
        else:
            print(f"    [{regime}] Using global min/max")
            print(f"    j_avg (Gaussian t={t_center}±{t_width}, x={x0_fraction:.2f}L) = {j_avg:.4f}, σ_p = {sigma_p:.4f}")
        
        data_by_label[data_label]['u_d'].append(u_d)
        data_by_label[data_label]['delta_n'].append(delta_n)
        data_by_label[data_label]['j_avg'].append(j_avg)
        data_by_label[data_label]['sigma_p'].append(sigma_p)
        
        # Collect all data for interpolation
        all_u_d.append(u_d)
        all_delta_n.append(delta_n)
        all_j_avg.append(j_avg)
    
    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(1, 1, figsize=(12*0.8,9*0.8))
    ax2 = ax1.twinx()  # Create second y-axis for current
    
    # Same color for all datasets, different marker shapes
    color = 'black'  # Single color for all datasets
    markers = ['o', 's', '^', '*', 'v', '<', '>', 'p', '*', 'h', 
               'H', '+', 'x', 'X', '|']
    
    # Add square-root fit based on SH-type model: delta_n ~ a * (u_d - u_c)^(1/2)
    # Plot this FIRST so it appears below the data points
    if all_u_d:
        from scipy.optimize import curve_fit
        
        # Sort data for smooth line
        sorted_indices = np.argsort(all_u_d)
        u_d_sorted = np.array(all_u_d)[sorted_indices]
        delta_n_sorted = np.array(all_delta_n)[sorted_indices]
        
        # Filter for u_d <= 10
        mask = u_d_sorted <= 10.0
        u_d_sorted = u_d_sorted[mask]
        delta_n_sorted = delta_n_sorted[mask]
        
        u_c = 1.2500  # Critical u_d value (onset of instability)
        
        try:
            # Linear fit BELOW critical u_d: delta_n = b * u_d for u_d < u_c
            mask_linear = u_d_sorted < u_c
            if np.sum(mask_linear) > 2:  # Need at least 3 points for linear fit
                u_d_linear = u_d_sorted[mask_linear]
                delta_n_linear = delta_n_sorted[mask_linear]
                
                # Define linear model function
                def linear_model(u_d, b):
                    return b * u_d
                
                # Fit the model
                popt_lin, pcov_lin = curve_fit(linear_model, u_d_linear, delta_n_linear, p0=[0.01])
                b_fit = popt_lin[0]
                
                # Generate smooth curve for subcritical region
                if len(u_d_linear) > 0:
                    u_d_smooth_linear = np.linspace(u_d_linear.min(), u_c, 200)
                    delta_n_smooth_linear = linear_model(u_d_smooth_linear, b_fit)
                    
                    ax1.plot(u_d_smooth_linear, delta_n_smooth_linear, 'g--', linewidth=1.5, alpha=0.9, 
                            label=f'$u^{{\\bigstar}} = 1.2500$')
                    
                    print(f"\nLinear fit results (u_d < u*):")
                    print(f"  Fit parameter: b = {b_fit:.6f}")
                    print(f"  Model: Δn = {b_fit:.6f} * u_d")
                    print(f"  Standard error: {np.sqrt(pcov_lin[0,0]):.6f}")
                    print(f"  Fitted on {np.sum(mask_linear)} points with u_d < {u_c}")
        except Exception as e:
            print(f"Warning: Could not fit linear model: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Sqrt fit ABOVE critical u_d: delta_n = a * sqrt(u_d - u_c) for u_d > u_c
            # Fit only on points with u_d < 4.7
            mask_fit = (u_d_sorted > u_c) & (u_d_sorted < 4.7)
            if np.sum(mask_fit) > 3:  # Need at least 4 points for fitting
                u_d_fit = u_d_sorted[mask_fit]
                delta_n_fit = delta_n_sorted[mask_fit]
                
                # Define sqrt model function
                def sqrt_model(u_d, a):
                    return a * np.sqrt(u_d - u_c)
                
                # Fit the model
                popt, pcov = curve_fit(sqrt_model, u_d_fit, delta_n_fit, p0=[1.0])
                a_fit = popt[0]
                
                # Generate smooth curve over the whole x-axis range (not just fitted range)
                u_d_smooth = np.linspace(u_c, u_d_sorted.max(), 1000)
                delta_n_smooth = sqrt_model(u_d_smooth, a_fit)
                
                ax1.plot(u_d_smooth, delta_n_smooth, 'r-', linewidth=1.5, alpha=0.9, 
                        label=f'${a_fit:.3f} \\sqrt{{u_d - 1.2500}}$')
                
                print(f"\nSquare-root fit results (u_d > u*):")
                print(f"  Fit parameter: a = {a_fit:.4f}")
                print(f"  Model: Δn = {a_fit:.4f} * sqrt(u_d - 1.2500)")
                print(f"  Standard error: {np.sqrt(pcov[0,0]):.4f}")
                print(f"  Fitted on {np.sum(mask_fit)} points with {u_c} < u_d < 4.7")
        except Exception as e:
            print(f"Warning: Could not fit sqrt model: {e}")
            import traceback
            traceback.print_exc()
    
    # Collect all data for least squares fitting of ⟨j⟩ vs u_d
    all_u_d_j = []
    all_j_avg_j = []
    all_sigma_p_j = []
    for label in labels:
        if data_by_label[label]['u_d']:
            all_u_d_j.extend(data_by_label[label]['u_d'])
            all_j_avg_j.extend(data_by_label[label]['j_avg'])
            # Collect sigma_p values for error bars
            if 'sigma_p' in data_by_label[label]:
                all_sigma_p_j.extend(data_by_label[label]['sigma_p'])
            else:
                # If sigma_p not available, use zeros
                all_sigma_p_j.extend([0.0] * len(data_by_label[label]['u_d']))
    
    # Convert to numpy arrays for fitting
    all_u_d_j = np.array(all_u_d_j)
    all_j_avg_j = np.array(all_j_avg_j)
    
    # Fit least squares lines for j_avg vs u_d
    u_c = 1.2500  # Critical u_d value
    
    # Subcritical region (u_d < u_c)
    subcritical_mask = all_u_d_j < u_c
    if np.sum(subcritical_mask) > 1:
        u_d_sub = all_u_d_j[subcritical_mask]
        j_avg_sub = all_j_avg_j[subcritical_mask]
        
        # Linear fit: j_avg = a_sub * u_d + b_sub
        A_sub = np.vstack([u_d_sub, np.ones(len(u_d_sub))]).T
        a_sub, b_sub = np.linalg.lstsq(A_sub, j_avg_sub, rcond=None)[0]
        
        # Plot subcritical fit
        u_d_fit_sub = np.linspace(min(u_d_sub), u_c, 100)
        j_avg_fit_sub = a_sub * u_d_fit_sub + b_sub
        # ax2.plot(u_d_fit_sub, j_avg_fit_sub, 'g--', linewidth=2, alpha=0.8,
        #         label=f'$\\langle j \\rangle$ fit (subcritical): slope = {a_sub:.3f}')
        
        print(f"\nSubcritical ⟨j⟩ fit: ⟨j⟩ = {a_sub:.4f} * u_d + {b_sub:.4f}")
        # Calculate R²
        j_pred_sub = a_sub * u_d_sub + b_sub
        ss_res_sub = np.sum((j_avg_sub - j_pred_sub)**2)
        ss_tot_sub = np.sum((j_avg_sub - np.mean(j_avg_sub))**2)
        r2_sub = 1 - (ss_res_sub / ss_tot_sub) if ss_tot_sub > 0 else 0
        print(f"  R² = {r2_sub:.4f}")
    
    # Supercritical region (u_d > u_c)
    supercritical_mask = all_u_d_j > u_c
    if np.sum(supercritical_mask) > 1:
        u_d_sup = all_u_d_j[supercritical_mask]
        j_avg_sup = all_j_avg_j[supercritical_mask]
        
        # Linear fit: j_avg = a_sup * u_d + b_sup
        A_sup = np.vstack([u_d_sup, np.ones(len(u_d_sup))]).T
        a_sup, b_sup = np.linalg.lstsq(A_sup, j_avg_sup, rcond=None)[0]
        
        # Plot supercritical fit
        u_d_fit_sup = np.linspace(u_c, max(u_d_sup), 100)
        j_avg_fit_sup = a_sup * u_d_fit_sup + b_sup
        ax2.plot(u_d_fit_sup, j_avg_fit_sup, 'm--', linewidth=2, alpha=0.8,
                label=f'$\\langle j \\rangle$ fit ($u_d > u^{{\\bigstar}}$): slope = {a_sup:.3f}')
        
        print(f"\nSupercritical ⟨j⟩ fit: ⟨j⟩ = {a_sup:.4f} * u_d + {b_sup:.4f}")
        # Calculate R²
        j_pred_sup = a_sup * u_d_sup + b_sup
        ss_res_sup = np.sum((j_avg_sup - j_pred_sup)**2)
        ss_tot_sup = np.sum((j_avg_sup - np.mean(j_avg_sup))**2)
        r2_sup = 1 - (ss_res_sup / ss_tot_sup) if ss_tot_sup > 0 else 0
        print(f"  R² = {r2_sup:.4f}")
    
    # Overall fit (all data)
    if len(all_u_d_j) > 1:
        A_all = np.vstack([all_u_d_j, np.ones(len(all_u_d_j))]).T
        a_all, b_all = np.linalg.lstsq(A_all, all_j_avg_j, rcond=None)[0]
        
        # Plot overall fit
        u_d_fit_all = np.linspace(min(all_u_d_j), max(all_u_d_j), 100)
        j_avg_fit_all = a_all * u_d_fit_all + b_all
        ax2.plot(u_d_fit_all, j_avg_fit_all, 'k:', linewidth=2, alpha=0.6,
                label=f'$\\langle j \\rangle$ fit (overall): slope = {a_all:.3f}')
        
        print(f"\nOverall ⟨j⟩ fit: ⟨j⟩ = {a_all:.4f} * u_d + {b_all:.4f}")
        # Calculate R²
        j_pred_all = a_all * all_u_d_j + b_all
        ss_res_all = np.sum((all_j_avg_j - j_pred_all)**2)
        ss_tot_all = np.sum((all_j_avg_j - np.mean(all_j_avg_j))**2)
        r2_all = 1 - (ss_res_all / ss_tot_all) if ss_tot_all > 0 else 0
        print(f"  R² = {r2_all:.4f}")
        
        # Compare slopes if both subcritical and supercritical fits exist
        if np.sum(subcritical_mask) > 1 and np.sum(supercritical_mask) > 1:
            slope_change = a_sup - a_sub
            print(f"\nSlope change at u* = {u_c}: Δslope = {slope_change:.4f}")
            print(f"  Subcritical slope: {a_sub:.4f}")
            print(f"  Supercritical slope: {a_sup:.4f}")
            print(f"  Relative change: {slope_change/abs(a_sub)*100:.1f}%")

    # Plot data points AFTER the approximation so they appear on top
    for idx, label in enumerate(labels):
        if not data_by_label[label]['u_d']:
            continue
        
        # Plot ALL data points (both subcritical and supercritical)
        u_d_arr = np.array(data_by_label[label]['u_d'])
        delta_n_arr = np.array(data_by_label[label]['delta_n'])
        j_avg_arr = np.array(data_by_label[label]['j_avg'])
        
        if len(u_d_arr) == 0:
            continue
        
        marker = markers[idx % len(markers)]

        # Plot delta_n on left axis (filled markers)
        ax1.scatter(u_d_arr, delta_n_arr, marker=marker, color='black',
                   label=f'$\\Delta n=n_{{\\rm max}} - n_{{\\rm min}}$', s=50, alpha=0.8, zorder=10)
        
        # Plot j_avg on right axis (hollow markers)
        ax2.scatter(u_d_arr, j_avg_arr, marker=marker, color='blue',
                   label=f'$\\langle j \\rangle = \\langle u n \\rangle_t$', s=50, alpha=0.7, 
                   facecolors='none', edgecolors='blue', linewidths=1.5, zorder=10)

        # Plot points only, no lines, larger size
        # if "25" in label:
        #     ax.scatter(u_d_filtered, delta_n_filtered, marker=marker, color="magenta",
        #               label=label, s=24, alpha=0.8)
        # elif "100" in label:
        #     ax.scatter(u_d_filtered, delta_n_filtered, marker=marker, color="orange",
        #               label=label, s=24, alpha=0.8)
        # else:
        #     ax.scatter(u_d_filtered, delta_n_filtered, marker=marker, color="black",
        #               label=label, s=24, alpha=0.8)
    
    # Add vertical line at u* = 1.2500
    ax1.axvline(x=1.2500, color='green', linestyle='--', linewidth=2.0, alpha=0.8, label='$u^{\\bigstar} = 1.2500$')
    ax1.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)

    # Add zoomed inset around the critical region
    from matplotlib.patches import Rectangle
    
    # Define zoom region
    zoom_xlim = (2.4, 2.9)
    zoom_ylim = (-0.005, 0.030)
    
    # Add rectangle to indicate zoom region
    # rect = Rectangle((zoom_xlim[0], zoom_ylim[0]), 
    #                 zoom_xlim[1] - zoom_xlim[0], 
    #                 zoom_ylim[1] - zoom_ylim[0],
    #                 edgecolor="red", facecolor="none", linewidth=1.0, linestyle='--')
    # ax.add_patch(rect)
    
    # Create inset axes for zoom
    # ax_inset = fig.add_axes([0.15, 0.6, 0.25, 0.25])  # [left, bottom, width, height]
    
    # Add square-root fit to inset if available (plot FIRST so it appears below data points)
    if all_u_d:
        try:
            from scipy.optimize import curve_fit
            
            sorted_indices = np.argsort(all_u_d)
            u_d_sorted = np.array(all_u_d)[sorted_indices]
            delta_n_sorted = np.array(all_delta_n)[sorted_indices]
            
            u_c = 1.2500
            mask_fit = (u_d_sorted > u_c)  # No upper limit - use ALL data points
            if np.sum(mask_fit) > 3:
                u_d_fit = u_d_sorted[mask_fit]
                delta_n_fit = delta_n_sorted[mask_fit]
                
                def sqrt_model(u_d, a):
                    return a * np.sqrt(u_d - u_c)
                
                popt, pcov = curve_fit(sqrt_model, u_d_fit, delta_n_fit, p0=[1.0])
                a_fit = popt[0]
                
                # Plot fit in zoom region
                u_d_zoom_fit = np.linspace(max(u_c, zoom_xlim[0]), zoom_xlim[1], 100)
                delta_n_zoom_fit = sqrt_model(u_d_zoom_fit, a_fit)
                
                # Only plot if within zoom y-limits
                # mask_zoom_fit = (delta_n_zoom_fit >= zoom_ylim[0]) & (delta_n_zoom_fit <= zoom_ylim[1])
                # if np.any(mask_zoom_fit):
                #     ax_inset.plot(u_d_zoom_fit[mask_zoom_fit], delta_n_zoom_fit[mask_zoom_fit], 
                #                 'r-', linewidth=1.5, alpha=0.9)
        except Exception as e:
            pass  # Silent fail for inset
    
    # Plot the same data in the inset (show all data, let axis limits handle the zoom)
    # Plot data points AFTER the approximation so they appear on top
    # for idx, label in enumerate(labels):
    #     if not data_by_label[label]['u_d']:
    #         continue
        
    #     u_d_arr = np.array(data_by_label[label]['u_d'])
    #     delta_n_arr = np.array(data_by_label[label]['delta_n'])
        
    #     marker = markers[idx % len(markers)]
        
    #     ax_inset.scatter(u_d_arr, delta_n_arr, marker=marker, s=20, alpha=0.8)

        # if "25" in label:
        #     ax_inset.scatter(u_d_arr, delta_n_arr, marker=marker, color="magenta", s=20, alpha=0.8)
        # elif "100" in label:
        #     ax_inset.scatter(u_d_arr, delta_n_arr, marker=marker, color="orange", s=20, alpha=0.8)
        # else:
        #     ax_inset.scatter(u_d_arr, delta_n_arr, marker=marker, color="black", s=20, alpha=0.8)
    
    # Add reference lines in inset
    # ax_inset.axvline(x=1.2500, color='blue', linestyle='--', linewidth=1.0, alpha=0.8)
    # ax_inset.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    
    # Set inset properties
    # ax_inset.set_xlim(zoom_xlim)
    # ax_inset.set_ylim(zoom_ylim)
    # ax_inset.set_xticks([])
    # ax_inset.set_yticks([])
    # ax_inset.set_title("Critical region", fontsize=9)
    # ax_inset.grid(True, alpha=0.3)
    
    # Style inset spines
    # for spine in ax_inset.spines.values():
    #     spine.set_linewidth(0.8)
    
    # Set axis labels and properties
    ax1.set_xlabel('$u_d$', fontsize=14)
    ax1.set_ylabel('$\\Delta n = n_{\\rm max} - n_{\\rm min}$', fontsize=14, color='black')
    ax2.set_ylabel('$\\langle j \\rangle = \\langle u n \\rangle_t$', fontsize=14, color='blue')
    
    # Color the y-axis ticks to match the data
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Set y-axis limits
    ax2.set_ylim(bottom=-0.065)  # Current axis starts from 0
    # ax2.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, ncol=1, loc='upper left', framealpha=0.9)
    
    ax1.grid(True, alpha=0.3)
    # ax1.set_xlim(0.5, 8)
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/delta_n_vs_ud.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{outdir}/delta_n_vs_ud.pdf", dpi=200, bbox_inches='tight')
    plt.savefig(f"{outdir}/delta_n_vs_ud.svg", dpi=200, bbox_inches='tight')
    print(f"\nSaved delta n vs u_d plot (with time-averaged current) to {outdir}/delta_n_vs_ud.png")
    print(f"  Current measured at x = {x0_fraction:.2f}L")
    plt.show()
    plt.close()
    
    # Create separate plot for ⟨j⟩ residuals: ⟨j⟩ - 0.2*u_d
    print(f"\nCreating residual plot: ⟨j⟩ - 0.2*u_d")
    fig_res, ax_res = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot residuals for each dataset
    for idx, label in enumerate(labels):
        if not data_by_label[label]['u_d']:
            continue
        
        u_d_arr = np.array(data_by_label[label]['u_d'])
        j_avg_arr = np.array(data_by_label[label]['j_avg'])
        sigma_p_arr = np.array(data_by_label[label]['sigma_p'])
        
        if len(u_d_arr) == 0:
            continue
        
        # Calculate residuals: ⟨j⟩ - 0.2*u_d
        j_expected = 0.2 * u_d_arr
        j_residuals = j_avg_arr - j_expected
        
        marker = markers[idx % len(markers)]
        
        # Plot residuals with error bars (sigma_p represents uncertainty in j_avg)
        # ax_res.scatter(u_d_arr, j_residuals, marker=marker, color='red',
        #         label=f'$\\langle j \\rangle - 0.2 \\cdot u_d$', 
        #         s=60, alpha=0.8, zorder=10)
        ax_res.errorbar(u_d_arr, j_residuals, yerr=sigma_p_arr, 
                       marker=marker, color='red', linestyle='none',
                       label=f'$\\langle j \\rangle - 0.2 \\cdot u_d$', 
                       markersize=8, alpha=0.8, zorder=10, capsize=3, capthick=1)
        
        print(f"  {label}: {len(u_d_arr)} points, residual range: [{j_residuals.min():.4f}, {j_residuals.max():.4f}], σ_p range: [{sigma_p_arr.min():.4f}, {sigma_p_arr.max():.4f}]")
    
    # Add reference lines
    ax_res.axhline(y=0.0, color='black', linestyle='-', linewidth=1.0, alpha=0.5, label='$\\langle j \\rangle = 0.2 \\cdot u_d$')
    ax_res.axvline(x=1.2500, color='green', linestyle='--', linewidth=2.0, alpha=0.8, label='$u^{\\bigstar} = 1.2500$')
    
    # Fit lines to residuals in subcritical and supercritical regions
    if len(all_u_d_j) > 1:
        # Subcritical residuals
        if np.sum(subcritical_mask) > 1:
            u_d_sub = all_u_d_j[subcritical_mask]
            j_avg_sub = all_j_avg_j[subcritical_mask]
            j_expected_sub = 0.2 * u_d_sub
            j_residuals_sub = j_avg_sub - j_expected_sub
            
            # Linear fit to residuals
            A_sub_res = np.vstack([u_d_sub, np.ones(len(u_d_sub))]).T
            a_sub_res, b_sub_res = np.linalg.lstsq(A_sub_res, j_residuals_sub, rcond=None)[0]
            
            # Plot subcritical residual fit
            u_d_fit_sub = np.linspace(min(u_d_sub), u_c, 100)
            j_residuals_fit_sub = a_sub_res * u_d_fit_sub + b_sub_res
            # ax_res.plot(u_d_fit_sub, j_residuals_fit_sub, 'g--', linewidth=2, alpha=0.8,
            #            label=f'Residual fit ($u_d < u^{{\\bigstar}}$): slope = {a_sub_res:.3f}')
            
            print(f"\nSubcritical residual fit: residual = {a_sub_res:.4f} * u_d + {b_sub_res:.4f}")
        
        # Supercritical residuals
        if np.sum(supercritical_mask) > 1:
            u_d_sup = all_u_d_j[supercritical_mask]
            j_avg_sup = all_j_avg_j[supercritical_mask]
            j_expected_sup = 0.2 * u_d_sup
            j_residuals_sup = j_avg_sup - j_expected_sup
            
            # Linear fit to residuals
            A_sup_res = np.vstack([u_d_sup, np.ones(len(u_d_sup))]).T
            a_sup_res, b_sup_res = np.linalg.lstsq(A_sup_res, j_residuals_sup, rcond=None)[0]
            
            # Plot supercritical residual fit
            u_d_fit_sup = np.linspace(u_c, max(u_d_sup), 100)
            j_residuals_fit_sup = a_sup_res * u_d_fit_sup + b_sup_res
            ax_res.plot(u_d_fit_sup, j_residuals_fit_sup, 'm--', linewidth=2, alpha=0.8,
                       label=f'Residual fit ($u_d > u^{{\\bigstar}}$): slope = {a_sup_res:.3f}')
            
            print(f"\nSupercritical residual fit: residual = {a_sup_res:.4f} * u_d + {b_sup_res:.4f}")
    
    # Set axis labels and properties
    ax_res.set_xlabel('$u_d$', fontsize=14)
    ax_res.set_ylabel('$\\langle j \\rangle - 0.2 \\cdot u_d$', fontsize=14)
    # ax_res.set_title('Residuals from expected linear relationship $\\langle j \\rangle = 0.2 \\cdot u_d$', fontsize=14)
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(fontsize=10, loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/j_residuals_vs_ud.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{outdir}/j_residuals_vs_ud.pdf", dpi=200, bbox_inches='tight')
    plt.savefig(f"{outdir}/j_residuals_vs_ud.svg", dpi=200, bbox_inches='tight')
    print(f"\nSaved residual plot to {outdir}/j_residuals_vs_ud.png")
    plt.show()
    plt.close()
    
    return data_by_label

def plot_n_p_time_series(base_dirs, labels=None, outdir="multiple_u_d", x0_fraction=0.5, 
                        u_d_subcritical=None, u_d_supercritical=None):
    """Plot n(t) and p(t) at fixed location x₀ for subcritical and supercritical u_d values.
    
    Args:
        base_dirs: List of base directory paths
        labels: Optional custom labels for each dataset
        outdir: Output directory for plots
        x0_fraction: Fraction of domain length to use for measurement (default: 0.5 = middle)
        u_d_subcritical: u_d value below critical (default: 0.8 * 1.2500 = 2.192)
        u_d_supercritical: u_d value above critical (default: 1.2 * 1.2500 = 3.288)
    """
    if labels is None:
        labels = [os.path.basename(d) for d in base_dirs]
    
    # Set default u_d values if not provided
    u_star = 1.2500
    if u_d_subcritical is None:
        u_d_subcritical = 0.8 * u_star  # 2.192
    if u_d_supercritical is None:
        u_d_supercritical = 1.2 * u_star  # 3.288
    
    # Load all data
    all_data = load_multi_dataset(base_dirs, labels)
    
    if not all_data:
        print("No data found!")
        return
    
    # Find closest u_d values to target values
    u_d_values = [u_d for _, u_d, _ in all_data]
    
    def find_closest_u_d(target_u_d):
        if not u_d_values:
            return None, None
        closest_idx = min(range(len(u_d_values)), key=lambda i: abs(u_d_values[i] - target_u_d))
        closest_u_d = u_d_values[closest_idx]
        closest_data = all_data[closest_idx]
        return closest_u_d, closest_data
    
    # Find data for subcritical and supercritical cases
    u_d_sub, data_sub = find_closest_u_d(u_d_subcritical)
    u_d_sup, data_sup = find_closest_u_d(u_d_supercritical)
    
    print(f"\nTime series analysis:")
    print(f"  Subcritical: u_d = {u_d_sub:.4f} (target: {u_d_subcritical:.4f})")
    print(f"  Supercritical: u_d = {u_d_sup:.4f} (target: {u_d_supercritical:.4f})")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # First, collect all data to determine common scales
    all_n_data = []
    all_p_data = []
    
    # Collect subcritical data
    if data_sub is not None:
        n_t = data_sub[2]['n_t']
        p_t = data_sub[2]['p_t']
        Nx = data_sub[2]['Nx']
        x0_idx = int(x0_fraction * Nx)
        all_n_data.extend(n_t[x0_idx, :])
        all_p_data.extend(p_t[x0_idx, :])
    
    # Collect supercritical data
    if data_sup is not None:
        n_t = data_sup[2]['n_t']
        p_t = data_sup[2]['p_t']
        Nx = data_sup[2]['Nx']
        x0_idx = int(x0_fraction * Nx)
        all_n_data.extend(n_t[x0_idx, :])
        all_p_data.extend(p_t[x0_idx, :])
    
    # Determine common y-axis limits with some padding
    if all_n_data:
        n_min, n_max = min(all_n_data), max(all_n_data)
        n_padding = 0.05 * (n_max - n_min)
        n_ylim = (n_min - n_padding, n_max + n_padding)
    else:
        n_ylim = None
    
    if all_p_data:
        p_min, p_max = min(all_p_data), max(all_p_data)
        p_padding = 0.05 * (p_max - p_min)
        p_ylim = (p_min - p_padding, p_max + p_padding)
    else:
        p_ylim = None
    
    # Plot subcritical case
    if data_sub is not None:
        n_t = data_sub[2]['n_t']
        p_t = data_sub[2]['p_t']
        t = data_sub[2]['t']
        L = data_sub[2]['L']
        Nx = data_sub[2]['Nx']
        
        # Determine spatial index for measurement
        x0_idx = int(x0_fraction * Nx)
        x0 = x0_fraction * L
        
        # Extract time series at x₀
        n_at_x0 = n_t[x0_idx, :]
        p_at_x0 = p_t[x0_idx, :]
        
        # Plot n(t)
        axes[0, 0].plot(t, n_at_x0, 'b-', linewidth=0.5, alpha=0.8)
        axes[0, 0].set_ylabel('$n(x_0, t)$', fontsize=12)
        axes[0, 0].set_title(f'$u_d = {u_d_sub:.3f}$\n$n({x0:.2f}, t)$', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        if n_ylim is not None:
            axes[0, 0].set_ylim(n_ylim)
        
        # Plot p(t)
        axes[1, 0].plot(t, p_at_x0, 'r-', linewidth=0.5, alpha=0.8)
        axes[1, 0].set_xlabel('$t$', fontsize=12)
        axes[1, 0].set_ylabel('$p(x_0, t)$', fontsize=12)
        axes[1, 0].set_title(f'$p({x0:.2f}, t)$', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        if p_ylim is not None:
            axes[1, 0].set_ylim(p_ylim)
        
        # Add statistics
        n_mean = np.mean(n_at_x0)
        n_std = np.std(n_at_x0)
        p_mean = np.mean(p_at_x0)
        p_std = np.std(p_at_x0)
        
        axes[0, 0].text(0.02, 0.98, f'$\\langle n \\rangle = {n_mean:.3f}$\n$\\sigma_n = {n_std:.3f}$', 
                       transform=axes[0, 0].transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1, 0].text(0.02, 0.98, f'$\\langle p \\rangle = {p_mean:.3f}$\n$\\sigma_p = {p_std:.3f}$', 
                       transform=axes[1, 0].transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot supercritical case
    if data_sup is not None:
        n_t = data_sup[2]['n_t']
        p_t = data_sup[2]['p_t']
        t = data_sup[2]['t']
        L = data_sup[2]['L']
        Nx = data_sup[2]['Nx']
        
        # Determine spatial index for measurement
        x0_idx = int(x0_fraction * Nx)
        x0 = x0_fraction * L
        
        # Extract time series at x₀
        n_at_x0 = n_t[x0_idx, :]
        p_at_x0 = p_t[x0_idx, :]
        
        # Plot n(t)
        axes[0, 1].plot(t, n_at_x0, 'b-', linewidth=0.5, alpha=0.8)
        axes[0, 1].set_ylabel('$n(x_0, t)$', fontsize=12)
        axes[0, 1].set_title(f'$u_d = {u_d_sup:.3f}$\n$n({x0:.2f}, t)$', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        if n_ylim is not None:
            axes[0, 1].set_ylim(n_ylim)
        
        # Plot p(t)
        axes[1, 1].plot(t, p_at_x0, 'r-', linewidth=0.5, alpha=0.8)
        axes[1, 1].set_xlabel('$t$', fontsize=12)
        axes[1, 1].set_ylabel('$p(x_0, t)$', fontsize=12)
        axes[1, 1].set_title(f'$p({x0:.2f}, t)$', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        if p_ylim is not None:
            axes[1, 1].set_ylim(p_ylim)
        
        # Add statistics
        n_mean = np.mean(n_at_x0)
        n_std = np.std(n_at_x0)
        p_mean = np.mean(p_at_x0)
        p_std = np.std(p_at_x0)
        
        axes[0, 1].text(0.02, 0.98, f'$\\langle n \\rangle = {n_mean:.3f}$\n$\\sigma_n = {n_std:.3f}$', 
                       transform=axes[0, 1].transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[1, 1].text(0.02, 0.98, f'$\\langle p \\rangle = {p_mean:.3f}$\n$\\sigma_p = {p_std:.3f}$', 
                       transform=axes[1, 1].transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/n_p_time_series_comparison.png", dpi=200, bbox_inches='tight')
    plt.savefig(f"{outdir}/n_p_time_series_comparison.pdf", dpi=200, bbox_inches='tight')
    print(f"\nSaved n(t) and p(t) time series comparison to {outdir}/n_p_time_series_comparison.png")
    plt.show()
    plt.close()

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
    
    folder_name = "multiple_u_d/"
    # Search in multiple_u_d/out_drift_ud* subdirectories
    if os.path.exists(folder_name):
        for item in os.listdir(folder_name):
            if item.startswith("out_drift_ud") and os.path.isdir(os.path.join(folder_name, item)):
                # Extract u_d from directory name (handle both old and new formats)
                try:
                    u_d_str = item.replace("out_drift_ud", "")
                    # Handle new format: out_drift_ud4p6000 -> 4p6000 -> 4.6000
                    if "p" in u_d_str:
                        u_d_str = u_d_str.replace("p", ".")
                    u_d = float(u_d_str)
                    
                    # Look for data file in this subdirectory
                    subdir_path = os.path.join(folder_name, item)
                    for file in os.listdir(subdir_path):
                        # Support both old and new filename formats
                        if (file.startswith("data_m01_ud") and file.endswith(".npz")) or \
                           (file.startswith("data_m") and "_ud" in file and file.endswith(".npz")):
                            filepath = os.path.join(subdir_path, file)
                            data_files.append((filepath, u_d))
                            break  # Only take first matching file
                except ValueError:
                    continue
    
    # Search in out_drift directory (main level only)
    if os.path.exists("out_drift"):
        for file in os.listdir("out_drift"):
            # Support both old and new filename formats
            if (file.startswith("data_m01_ud") and file.endswith(".npz")) or \
               (file.startswith("data_m") and "_ud" in file and file.endswith(".npz")):
                try:
                    # Try new format first (data_mXX_udXpXXXX_tag.npz)
                    if "_ud" in file and "p" in file:
                        # Extract u_d from new format: data_mXX_udXpXXXX_tag.npz
                        parts = file.split("_ud")[1].split("_")[0]  # Get "XpXXXX" part
                        u_d_str = parts.replace("p", ".")  # Convert "XpXXXX" to "X.XXXX"
                        u_d = float(u_d_str)
                    else:
                        # Fallback to old format: data_m01_udX.XXXX.npz
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
        "multiple_u_d/2.5L(lambda=0.0, sigma=-1.0, seed_amp_n=0.05, seed_amp_p=0.05)",
    ]
    
    custom_labels = [
        # "δn,δp = 0.03",
        # "δn,δp = 0.05 (uniform)",
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
    # filename = "multiple_u_d/out_drift_ud18.0000/data_m01_ud18.0.npz"
    # data = load_data(filename)
    # plot_velocity_evolution(data, 18.0)



    # Single file analysis (moved to separate file)
    # Use plot_u_true_vs_time.py for u_true vs time analysis
    
    # Automatically find and analyze all available simulations (commented out)
    data_files = find_available_simulations()
    if data_files:
        # plot_velocity_vs_ud(data_files)
        plot_multiple_ud_panel(data_files)
        pass
    else:
        print("No simulation files found!")
    
    print("\nAll plots generated!")