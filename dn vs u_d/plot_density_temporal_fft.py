import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq

data_file = r"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\dn vs u_d\multiple_u_d\last\data_m07_ud0p0000_ud0.npz"
data = np.load(data_file)

t = data['t']
n_t = data['n_t']
p_t = data['p_t']
L = float(data['L'])
Nx = int(data['Nx'])

# Get metadata
try:
    meta = data['meta'].item()
    t_final = meta.get('t_final', t.max())
    x_source = meta.get('x_source', 2.5)
    x_drain = meta.get('x_drain', 7.5)
except:
    t_final = t.max()
    x_source = 2.5
    x_drain = 7.5

t_min_fft = 30.0
t_max_fft = 300.0
mask_time = (t >= t_min_fft) & (t <= t_max_fft)

if not np.any(mask_time):
    print(f"WARNING: No data points found in time range [{t_min_fft}, {t_max_fft}]")
    print(f"Available time range: [{t.min():.2f}, {t.max():.2f}]")
    mask_time = np.ones(len(t), dtype=bool)
    t_min_fft = t.min()
    t_max_fft = t.max()
    print(f"Using all available data: [{t_min_fft:.2f}, {t_max_fft:.2f}]")
else:
    print(f"Filtering data to time range: [{t_min_fft}, {t_max_fft}]")
    print(f"  Number of time points: {np.sum(mask_time)}/{len(t)}")

t_fft = t[mask_time]
n_t_fft = n_t[:, mask_time]
p_t_fft = p_t[:, mask_time]

x = np.linspace(0.0, L, Nx, endpoint=False)
dt = t_fft[1] - t_fft[0] if len(t_fft) > 1 else 1.0

probe_points = [
    (0.5 * (x_source + x_drain), "Midpoint (between source/drain)"),
    (x_source + 0.5, "Near source"),
    (x_drain - 0.5, "Near drain"),
    (L * 0.25, "Quarter point"),
    (L * 0.75, "Three-quarter point")
]

fig, axes = plt.subplots(len(probe_points), 2, figsize=(16, 4*len(probe_points)))
if len(probe_points) == 1:
    axes = axes.reshape(1, -1)

print("Computing temporal FFT for probe points...")
for i, (x_probe, label) in enumerate(probe_points):
    idx_probe = np.argmin(np.abs(x - x_probe))
    x_actual = x[idx_probe]
    
    n_series = n_t_fft[idx_probe, :]
    p_series = p_t_fft[idx_probe, :]
    
    n_series_detrended = n_series - np.mean(n_series)
    p_series_detrended = p_series - np.mean(p_series)
    
    n_fft = rfft(n_series_detrended)
    p_fft = rfft(p_series_detrended)
    
    freqs = rfftfreq(len(t_fft), d=dt)
    freqs = freqs[1:]
    n_fft = n_fft[1:]
    p_fft = p_fft[1:]
    
    n_power = np.abs(n_fft)**2
    p_power = np.abs(p_fft)**2
    
    n_peak_idx = np.argmax(n_power)
    p_peak_idx = np.argmax(p_power)
    f_n_peak = freqs[n_peak_idx]
    f_p_peak = freqs[p_peak_idx]
    period_n = 1.0 / f_n_peak if f_n_peak > 0 else np.inf
    period_p = 1.0 / f_p_peak if f_p_peak > 0 else np.inf
    
    ax_ts = axes[i, 0]
    if len(t) > len(t_fft):
        ax_ts.plot(t, n_t[idx_probe, :], 'b-', linewidth=0.3, alpha=0.2, label='_nolegend_')
        ax_ts_twin = ax_ts.twinx()
        ax_ts_twin.plot(t, p_t[idx_probe, :], 'r-', linewidth=0.3, alpha=0.2, label='_nolegend_')
    else:
        ax_ts_twin = ax_ts.twinx()
    ax_ts.plot(t_fft, n_series, 'b-', linewidth=0.7, label=f'$n(x={x_actual:.2f}, t)$', alpha=0.8)
    ax_ts_twin.plot(t_fft, p_series, 'r-', linewidth=0.7, label=f'$p(x={x_actual:.2f}, t)$', alpha=0.8)
    ax_ts.axvspan(t_min_fft, t_max_fft, alpha=0.1, color='gray', label='FFT window')
    ax_ts.set_xlabel("$t$", fontsize=12)
    ax_ts.set_ylabel("$n$", fontsize=12, color='b')
    ax_ts_twin.set_ylabel("$p$", fontsize=12, color='r')
    ax_ts.tick_params(axis='y', labelcolor='b')
    ax_ts_twin.tick_params(axis='y', labelcolor='r')
    ax_ts.set_title(f"{label}\n$x = {x_actual:.3f}$ (FFT: t=[{t_min_fft}, {t_max_fft}])", fontsize=12)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc='upper left', fontsize=10)
    ax_ts_twin.legend(loc='upper right', fontsize=10)
    
    ax_fft = axes[i, 1]
    ax_fft.semilogy(freqs, n_power, 'b-', linewidth=0.7, label='$|\\hat{n}(f)|^2$', alpha=0.8)
    ax_fft_twin = ax_fft.twinx()
    ax_fft_twin.semilogy(freqs, p_power, 'r-', linewidth=0.7, label='$|\\hat{p}(f)|^2$', alpha=0.8)
    
    ax_fft.axvline(f_n_peak, color='b', linestyle=':', linewidth=2, alpha=0.7, 
                   label=f'$f_{{n,peak}}={f_n_peak:.4f}$')
    ax_fft_twin.axvline(f_p_peak, color='r', linestyle=':', linewidth=2, alpha=0.7, 
                        label=f'$f_{{p,peak}}={f_p_peak:.4f}$')
    
    ax_fft.set_xlabel("Frequency $f$", fontsize=12)
    ax_fft.set_ylabel("$|\\hat{n}(f)|^2$", fontsize=12, color='b')
    ax_fft_twin.set_ylabel("$|\\hat{p}(f)|^2$", fontsize=12, color='r')
    ax_fft.tick_params(axis='y', labelcolor='b')
    ax_fft_twin.tick_params(axis='y', labelcolor='r')
    ax_fft.set_title(f"Frequency Spectrum\n$T_n={period_n:.3f}$, $T_p={period_p:.3f}$", fontsize=12)
    ax_fft.grid(True, alpha=0.3)
    ax_fft.legend(loc='upper left', fontsize=10)
    ax_fft_twin.legend(loc='upper right', fontsize=10)
    
    if len(freqs) > 0:
        f_max = min(10.0 * f_n_peak if f_n_peak > 0 else freqs[-1], freqs[-1])
        ax_fft.set_xlim(0, f_max)
        ax_fft_twin.set_xlim(0, f_max)
    
    print(f"  Probe {i+1}/{len(probe_points)}: x={x_actual:.3f}")
    print(f"    Peak frequency (n): {f_n_peak:.6f} Hz, Period: {period_n:.6f}")
    print(f"    Peak frequency (p): {f_p_peak:.6f} Hz, Period: {period_p:.6f}")

plt.tight_layout(pad=1.5)

output_file = "density_temporal_fft.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nSaved temporal FFT plot → {os.path.abspath(output_file)}")

output_file_pdf = "density_temporal_fft.svg"
plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight')
print(f"Saved temporal FFT plot (PDF) → {os.path.abspath(output_file_pdf)}")

print("\nComputing bulk-averaged temporal FFT...")
mask_bulk = (x > x_source + 0.5) & (x < x_drain - 0.5)
if np.any(mask_bulk):
    n_bulk_avg = np.mean(n_t_fft[mask_bulk, :], axis=0)
    p_bulk_avg = np.mean(p_t_fft[mask_bulk, :], axis=0)
    
    n_bulk_detrended = n_bulk_avg - np.mean(n_bulk_avg)
    p_bulk_detrended = p_bulk_avg - np.mean(p_bulk_avg)
    
    n_bulk_fft = rfft(n_bulk_detrended)[1:]
    p_bulk_fft = rfft(p_bulk_detrended)[1:]
    freqs_bulk = rfftfreq(len(t_fft), d=dt)[1:]
    
    n_bulk_power = np.abs(n_bulk_fft)**2
    p_bulk_power = np.abs(p_bulk_fft)**2
    
    n_bulk_peak_idx = np.argmax(n_bulk_power)
    p_bulk_peak_idx = np.argmax(p_bulk_power)
    f_n_bulk_peak = freqs_bulk[n_bulk_peak_idx]
    f_p_bulk_peak = freqs_bulk[p_bulk_peak_idx]
    
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
    
    ax2_ts = axes2[0]
    if len(t) > len(t_fft):
        n_bulk_full = np.mean(n_t[mask_bulk, :], axis=0)
        p_bulk_full = np.mean(p_t[mask_bulk, :], axis=0)
        ax2_ts.plot(t, n_bulk_full, 'b-', linewidth=0.3, alpha=0.2, label='_nolegend_')
        ax2_ts_twin = ax2_ts.twinx()
        ax2_ts_twin.plot(t, p_bulk_full, 'r-', linewidth=0.3, alpha=0.2, label='_nolegend_')
    else:
        ax2_ts_twin = ax2_ts.twinx()
    ax2_ts.plot(t_fft, n_bulk_avg, 'b-', linewidth=2, label='$\\langle n \\rangle_{bulk}(t)$', alpha=0.8)
    ax2_ts_twin.plot(t_fft, p_bulk_avg, 'r-', linewidth=2, label='$\\langle p \\rangle_{bulk}(t)$', alpha=0.8)
    ax2_ts.axvspan(t_min_fft, t_max_fft, alpha=0.1, color='gray', label='FFT window')
    ax2_ts.set_xlabel("$t$", fontsize=14)
    ax2_ts.set_ylabel("$\\langle n \\rangle_{bulk}$", fontsize=14, color='b')
    ax2_ts_twin.set_ylabel("$\\langle p \\rangle_{bulk}$", fontsize=14, color='r')
    ax2_ts.tick_params(axis='y', labelcolor='b')
    ax2_ts_twin.tick_params(axis='y', labelcolor='r')
    ax2_ts.set_title(f"Bulk-Averaged Time Series (FFT: t=[{t_min_fft}, {t_max_fft}])", fontsize=14)
    ax2_ts.grid(True, alpha=0.3)
    ax2_ts.legend(loc='upper left', fontsize=12)
    ax2_ts_twin.legend(loc='upper right', fontsize=12)
    
    ax2_fft = axes2[1]
    ax2_fft.semilogy(freqs_bulk, n_bulk_power, 'b-', linewidth=2, label='$|\\hat{\\langle n \\rangle}(f)|^2$', alpha=0.8)
    ax2_fft_twin = ax2_fft.twinx()
    ax2_fft_twin.semilogy(freqs_bulk, p_bulk_power, 'r-', linewidth=2, label='$|\\hat{\\langle p \\rangle}(f)|^2$', alpha=0.8)
    
    ax2_fft.axvline(f_n_bulk_peak, color='b', linestyle=':', linewidth=2, alpha=0.7,
                    label=f'$f_{{n,peak}}={f_n_bulk_peak:.4f}$')
    ax2_fft_twin.axvline(f_p_bulk_peak, color='r', linestyle=':', linewidth=2, alpha=0.7,
                         label=f'$f_{{p,peak}}={f_p_bulk_peak:.4f}$')
    
    ax2_fft.set_xlabel("Frequency $f$", fontsize=14)
    ax2_fft.set_ylabel("$|\\hat{\\langle n \\rangle}(f)|^2$", fontsize=14, color='b')
    ax2_fft_twin.set_ylabel("$|\\hat{\\langle p \\rangle}(f)|^2$", fontsize=14, color='r')
    ax2_fft.tick_params(axis='y', labelcolor='b')
    ax2_fft_twin.tick_params(axis='y', labelcolor='r')
    period_n_bulk = 1.0 / f_n_bulk_peak if f_n_bulk_peak > 0 else np.inf
    period_p_bulk = 1.0 / f_p_bulk_peak if f_p_bulk_peak > 0 else np.inf
    ax2_fft.set_title(f"Bulk-Averaged Frequency Spectrum\n$T_n={period_n_bulk:.3f}$, $T_p={period_p_bulk:.3f}$", fontsize=14)
    ax2_fft.grid(True, alpha=0.3)
    ax2_fft.legend(loc='upper left', fontsize=12)
    ax2_fft_twin.legend(loc='upper right', fontsize=12)
    
    if len(freqs_bulk) > 0:
        f_max_bulk = min(10.0 * f_n_bulk_peak if f_n_bulk_peak > 0 else freqs_bulk[-1], freqs_bulk[-1])
        ax2_fft.set_xlim(0, f_max_bulk)
        ax2_fft_twin.set_xlim(0, f_max_bulk)
    
    plt.tight_layout(pad=1.5)
    
    output_file2 = "density_temporal_fft_bulk_avg.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Saved bulk-averaged temporal FFT plot → {os.path.abspath(output_file2)}")
    
    output_file2_pdf = "density_temporal_fft_bulk_avg.svg"
    plt.savefig(output_file2_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved bulk-averaged temporal FFT plot (PDF) → {os.path.abspath(output_file2_pdf)}")
    
    print(f"\n{'='*60}")
    print(f"Bulk-averaged frequency analysis:")
    print(f"{'='*60}")
    print(f"  Peak frequency (n): {f_n_bulk_peak:.6f} Hz")
    print(f"  Peak frequency (p): {f_p_bulk_peak:.6f} Hz")
    print(f"  Period (n): {period_n_bulk:.6f}")
    print(f"  Period (p): {period_p_bulk:.6f}")
    
    plt.close(fig2)

# plt.show()

