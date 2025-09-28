#!/usr/bin/env python3
"""
Script to generate specific plots for nu=2.226:
1. spacetime_n_lab_nu2.226.png
2. snapshots_n_nu2.226.png  
3. Time series of n(t) at x=6.0
4. Fourier analysis of n(t) at x=6.0 showing two peaks
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq

# Import the main simulation functions
exec(open('mass injection.py').read().split('if __name__')[0])

def run_simulation_nu2226():
    """
    Run simulation specifically for nu=2.226 and return full time series data
    """
    # Store original values
    old_nu = par.nu
    old_outdir = par.outdir
    
    # Set nu=2.226 (stored as par.nu in the code)
    nu_value = 2.226
    par.nu = nu_value
    par.outdir = f"out_drift_nu{nu_value:g}"
    
    try:
        print(f"\n[nu2226] Running simulation for ν = {nu_value}")
        
        # Run simulation
        t, n_t, p_t = run_once(tag=f"nu{nu_value:g}")
        
        return t, n_t, p_t
        
    finally:
        # Restore original values
        par.nu = old_nu
        par.outdir = old_outdir

def plot_spacetime_lab_nu2226(t, n_t):
    """
    Create spacetime_n_lab_nu2.226.png plot
    """
    nu_value = 2.226
    
    # Create spacetime plot in lab frame
    extent = [x.min(), x.max(), t.min(), t.max()]
    plt.figure(figsize=(9.6, 4.3))
    plt.imshow(n_t.T, origin="lower", aspect="auto", extent=extent, cmap=par.cmap)
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$t$", fontsize=14) 
    plt.title(r"$n(x,t)$, $\nu = 2.226$", fontsize=16)
    plt.colorbar(label=r"$n$")
    
    # Add vertical line at perturbation center
    plt.plot([par.x0, par.x0], [t.min(), t.max()], 'w--', lw=1, alpha=0.7)
    
    # Add vertical line at x=6.0 for reference
    plt.axvline(x=6.0, color='red', linestyle=':', lw=2, alpha=0.8, label='x=6.0')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig(f"spacetime_n_lab_nu{nu_value}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[plot] Saved: spacetime_n_lab_nu{nu_value}.png")

def plot_snapshots_nu2226(t, n_t):
    """
    Create snapshots_n_nu2.226.png plot
    """
    nu_value = 2.226
    
    # Calculate time-averaged density profile (t=[10,50])
    t_start_avg = 30.0
    t_end_avg = 100.0
    i_start = np.argmin(np.abs(t - t_start_avg))
    i_end = np.argmin(np.abs(t - t_end_avg))
    n_avg_time = np.mean(n_t[:, i_start:i_end+1], axis=1)
    
    plt.figure(figsize=(9.6, 3.4))
    
    # Calculate min/max values for display
    n_min_global = n_t.min()
    n_max_global = n_t.max()
    n_min_avg = n_avg_time.min()
    n_max_avg = n_avg_time.max()
    
    # Plot initial and final snapshots
    for frac in [0.0, 1.0]:
        j = int(frac*(len(t)-1))
        n_snapshot = n_t[:,j]
        n_min_snap = n_snapshot.min()
        n_max_snap = n_snapshot.max()
        plt.plot(x, n_snapshot, label=f"$t={t[j]:.1f}$")
    
    # Add time-averaged profile
    plt.plot(x, n_avg_time, 'k--', lw=2, label=r"$\langle n\rangle$")
    
    # Add vertical line at x=6.0
    plt.axvline(x=6.0, color='red', linestyle=':', lw=2, alpha=0.8, label='x=6.0')
    
    plt.legend(fontsize=9)
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$n$", fontsize=14)
    plt.title(r"$n(x)$, $\nu = 2.226$", fontsize=16)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"snapshots_n_nu{nu_value}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[plot] Saved: snapshots_n_nu{nu_value}.png")

def plot_time_series_at_x6(t, n_t):
    """
    Extract and plot n(t) at x=6.0 for nu=2.226
    """
    nu_value = 2.226
    
    # Find the index corresponding to x=6.0
    x_target = 6.0
    i_x = np.argmin(np.abs(x - x_target))
    x_actual = x[i_x]
    
    # Extract time series at this spatial location
    n_at_x6 = n_t[i_x, :]
    
    # Create time series plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, n_at_x6, 'b-', lw=2)
    plt.xlabel(r'$t$', fontsize=14)
    plt.ylabel(r'$n$', fontsize=14)
    plt.title(r'$n(t)$ at $x=6.0$, $\nu = 2.226$', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    n_mean = np.mean(n_at_x6)
    n_std = np.std(n_at_x6)
    n_min = np.min(n_at_x6)
    n_max = np.max(n_at_x6)
    
    stats_text = f"""Statistics:
Mean: {n_mean:.6f}
Std:  {n_std:.6f}
Min:  {n_min:.6f}
Max:  {n_max:.6f}"""
    
    # plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
    #          fontsize=10, verticalalignment='top',
    #          bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"n_time_series_x6_mu{nu_value}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[plot] Saved: n_time_series_x6_mu{nu_value}.png")
    
    return n_at_x6, x_actual

def plot_fourier_analysis_x6(t, n_at_x6, x_actual):
    """
    Perform Fourier analysis of n(t) at x=6.0 and show the two peaks
    """
    nu_value = 2.226
    
    # Remove the mean for Fourier analysis
    n_detrended = n_at_x6 - np.mean(n_at_x6)
    
    # Compute FFT
    N = len(n_detrended)
    dt = t[1] - t[0]  # Time step
    
    # Perform FFT
    n_fft = fft(n_detrended)
    freqs = fftfreq(N, dt)
    
    # Take only positive frequencies
    pos_freqs = freqs[:N//2]
    power_spectrum = np.abs(n_fft[:N//2])**2
    
    # Normalize power spectrum
    power_spectrum = power_spectrum / np.max(power_spectrum)
    
    # Create Fourier analysis plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Time series (top panel)
    plt.subplot(2, 1, 1)
    plt.plot(t, n_at_x6, 'b-', lw=2)
    plt.xlabel(r'$t$', fontsize=12)
    plt.ylabel(r'$n$', fontsize=12)
    plt.title(r'$n(t)$ at $x=6.0$', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Power spectrum (bottom panel)
    plt.subplot(2, 1, 2)
    plt.plot(pos_freqs, power_spectrum, 'r-', lw=2)
    plt.xlabel(r'$f$', fontsize=12)
    plt.ylabel(r'Power', fontsize=12)
    plt.title(r'FFT spectrum, $\nu = 2.226$', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Find and mark the two highest peaks
    # Remove DC component (frequency = 0)
    power_no_dc = power_spectrum.copy()
    power_no_dc[0] = 0  # Remove DC
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(power_no_dc, height=0.01, distance=5)  # Minimum distance between peaks
    
    # Sort peaks by height and take the two strongest
    if len(peaks) >= 2:
        peak_heights = power_spectrum[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]  # Sort in descending order
        top_peaks = peaks[sorted_indices[:2]]  # Take two strongest peaks
        
        # Mark the peaks
        for i, peak_idx in enumerate(top_peaks):
            freq_peak = pos_freqs[peak_idx]
            power_peak = power_spectrum[peak_idx]
            plt.plot(freq_peak, power_peak, 'go', markersize=10, label=f'Peak {i+1}: f={freq_peak:.3f}')
            
            # Add vertical lines
            plt.axvline(freq_peak, color='green', linestyle='--', alpha=0.7)
            
        plt.legend()
        
        # Print peak information
        print(f"\n[fourier] Found peaks in power spectrum:")
        for i, peak_idx in enumerate(top_peaks):
            freq_peak = pos_freqs[peak_idx]
            power_peak = power_spectrum[peak_idx]
            print(f"  Peak {i+1}: frequency = {freq_peak:.6f}, power = {power_peak:.6f}")
    
    else:
        print(f"[fourier] Warning: Found only {len(peaks)} peak(s), expected 2")
        if len(peaks) > 0:
            for i, peak_idx in enumerate(peaks):
                freq_peak = pos_freqs[peak_idx]
                power_peak = power_spectrum[peak_idx]
                plt.plot(freq_peak, power_peak, 'go', markersize=10, label=f'Peak {i+1}: f={freq_peak:.3f}')
                print(f"  Peak {i+1}: frequency = {freq_peak:.6f}, power = {power_peak:.6f}")
            plt.legend()
    
    # Set reasonable frequency range for display
    freq_max = np.max(pos_freqs) * 0.1  # Show only low frequencies where peaks are expected
    plt.xlim(0, freq_max)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"fourier_analysis_x6_mu{nu_value}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[plot] Saved: fourier_analysis_x6_mu{nu_value}.png")
    
    return pos_freqs, power_spectrum, peaks if 'peaks' in locals() else []

def plot_fourier_spectrum_t50(t, n_t):
    """
    Perform spatial Fourier analysis at t=50 to show the spatial spectrum
    """
    nu_value = 2.226
    
    # Find the time index closest to t=50
    t_target = 50.0
    i_t = np.argmin(np.abs(t - t_target))
    t_actual = t[i_t]
    
    # Extract spatial profile at this time
    n_at_t50 = n_t[:, i_t]
    
    # Remove the mean for Fourier analysis
    n_detrended = n_at_t50 - np.mean(n_at_t50)
    
    # Compute spatial FFT
    N = len(n_detrended)
    dx = x[1] - x[0]  # Spatial step
    
    # Perform FFT
    n_fft = fft(n_detrended)
    k_vals = fftfreq(N, dx)
    
    # Take only positive wavenumbers
    pos_k = k_vals[:N//2]
    power_spectrum = np.abs(n_fft[:N//2])**2
    
    # Normalize power spectrum
    power_spectrum = power_spectrum / np.max(power_spectrum)
    
    # Create spatial Fourier analysis plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Spatial profile at t=50 (top panel)
    plt.subplot(2, 1, 1)
    plt.plot(x, n_at_t50, 'b-', lw=2)
    plt.xlabel(r'$x$', fontsize=12)
    plt.ylabel(r'$n$', fontsize=12)
    plt.title(r'$n(x)$ at $t=50$', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at x=6.0
    plt.axvline(x=6.0, color='red', linestyle=':', lw=2, alpha=0.8, label='x=6.0')
    plt.legend()
    
    # Plot 2: Spatial power spectrum (bottom panel)
    plt.subplot(2, 1, 2)
    plt.plot(pos_k, power_spectrum, 'r-', lw=2)
    plt.xlabel(r'$k$', fontsize=12)
    plt.ylabel(r'Power', fontsize=12)
    plt.title(r'Spatial FFT spectrum, $\nu = 2.226$', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Find and mark the two highest peaks
    # Remove DC component (k = 0)
    power_no_dc = power_spectrum.copy()
    power_no_dc[0] = 0  # Remove DC
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(power_no_dc, height=0.01, distance=5)  # Minimum distance between peaks
    
    # Sort peaks by height and take the two strongest
    if len(peaks) >= 2:
        peak_heights = power_spectrum[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]  # Sort in descending order
        top_peaks = peaks[sorted_indices[:2]]  # Take two strongest peaks
        
        # Mark the peaks
        for i, peak_idx in enumerate(top_peaks):
            k_peak = pos_k[peak_idx]
            power_peak = power_spectrum[peak_idx]
            plt.plot(k_peak, power_peak, 'go', markersize=10, label=f'Peak {i+1}: k={k_peak:.3f}')
            
            # Add vertical lines
            plt.axvline(k_peak, color='green', linestyle='--', alpha=0.7)
            
        plt.legend()
        
        # Print peak information
        print(f"\n[spatial_fourier] Found peaks in spatial power spectrum at t={t_actual:.1f}:")
        for i, peak_idx in enumerate(top_peaks):
            k_peak = pos_k[peak_idx]
            power_peak = power_spectrum[peak_idx]
            print(f"  Peak {i+1}: wavenumber k = {k_peak:.6f}, power = {power_peak:.6f}")
    
    else:
        print(f"[spatial_fourier] Warning: Found only {len(peaks)} peak(s) at t={t_actual:.1f}, expected 2")
        if len(peaks) > 0:
            for i, peak_idx in enumerate(peaks):
                k_peak = pos_k[peak_idx]
                power_peak = power_spectrum[peak_idx]
                plt.plot(k_peak, power_peak, 'go', markersize=10, label=f'Peak {i+1}: k={k_peak:.3f}')
                print(f"  Peak {i+1}: wavenumber k = {k_peak:.6f}, power = {power_peak:.6f}")
            plt.legend()
    
    # Set reasonable wavenumber range for display
    k_max = np.max(pos_k) * 0.3  # Show low-to-medium wavenumbers where peaks are expected
    plt.xlim(0, k_max)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"spatial_fourier_t50_nu{nu_value}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"[plot] Saved: spatial_fourier_t50_nu{nu_value}.png")
    
    return pos_k, power_spectrum, peaks if 'peaks' in locals() else []

def main():
    """
    Main function to run all analyses for nu=2.226
    """
    print("=" * 70)
    print("Analysis for ν = 2.226")
    print("=" * 70)
    
    # Run simulation for nu=2.226
    print("\n[1] Running simulation...")
    t, n_t, p_t = run_simulation_nu2226()
    
    # Create spacetime lab frame plot
    print("\n[2] Creating spacetime lab frame plot...")
    plot_spacetime_lab_nu2226(t, n_t)
    
    # Create snapshots plot  
    print("\n[3] Creating snapshots plot...")
    plot_snapshots_nu2226(t, n_t)
    
    # Create time series plot at x=6.0
    print("\n[4] Creating time series at x=6.0...")
    n_at_x6, x_actual = plot_time_series_at_x6(t, n_t)
    
    # Perform Fourier analysis
    print("\n[5] Performing Fourier analysis...")
    freqs, power_spectrum, peaks = plot_fourier_analysis_x6(t, n_at_x6, x_actual)
    
    # Perform spatial Fourier analysis at t=50
    print("\n[6] Performing spatial Fourier analysis at t=50...")
    k_vals, spatial_power_spectrum, spatial_peaks = plot_fourier_spectrum_t50(t, n_t)
    
    print("\n" + "=" * 70)
    print("All plots generated successfully!")
    print("Generated files:")
    print("  - spacetime_n_lab_nu2.226.png")
    print("  - snapshots_n_nu2.226.png") 
    print("  - n_time_series_x6_nu2.226.png")
    print("  - fourier_analysis_x6_nu2.226.png")
    print("  - spatial_fourier_t50_nu2.226.png")
    print("=" * 70)

if __name__ == "__main__":
    main()
