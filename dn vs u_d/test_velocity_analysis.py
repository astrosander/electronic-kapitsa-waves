#!/usr/bin/env python3
"""
Test function to determine u_true from .npz file for out_drift_ud3p9000 case.
This tests the velocity analysis method used in main.py and plot_from_data.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, ifft, fftfreq
from numpy.fft import rfft, irfft, rfftfreq

def _fourier_shift_real(f, shift, L):
    """Circularly shift a real 1D signal by a non-integer amount using FFT."""
    N = f.size
    k = 2*np.pi * np.fft.rfftfreq(N, d=L/N)      # wavenumbers (>=0)
    F = np.fft.rfft(f)
    return np.fft.irfft(F * np.exp(1j*k*shift), n=N)

def estimate_velocity_fourier(n_t1, n_t2, t1, t2, L, power_floor=1e-3):
    """
    Estimate spatial shift and velocity from two snapshots using the
    cross-spectrum phase: Δφ_k = k * shift  (mod 2π).
    """
    N = n_t1.size
    # remove mean to drop k=0
    f1 = n_t1 - n_t1.mean()
    f2 = n_t2 - n_t2.mean()

    k = 2*np.pi * np.fft.rfftfreq(N, d=L/N)            # [0..Nyquist]
    F1 = np.fft.rfft(f1)
    F2 = np.fft.rfft(f2)
    C  = np.conj(F1) * F2                        # cross-spectrum
    phi = np.angle(C)                            # phase diff

    # discard k=0; keep only energetic modes for robustness
    k = k[1:]
    phi = phi[1:]
    w = np.abs(C[1:])                            # weights ~ power
    mask = w > (power_floor * w.max())
    k, phi, w = k[mask], phi[mask], w[mask]

    # unwrap the phase along k so it's linear: phi ≈ k * shift
    phi = np.unwrap(phi)

    # weighted least-squares slope: shift = argmin ||phi - k*shift||_W
    # shift = (Σ w k φ) / (Σ w k^2)
    num = np.sum(w * k * phi)
    den = np.sum(w * k**2)
    shift = num / den

    # Fix sign convention: positive shift = forward motion (rightward drift)
    # The phase difference phi = angle(F2) - angle(F1) should give the correct sign
    # But we need to ensure positive shift means n_t1 shifted right to match n_t2
    shift = -shift  # Flip sign to match spatial convention

    # map to principal interval [-L/2, L/2] (purely cosmetic)
    shift = (shift + 0.5*L) % L - 0.5*L

    dt = float(t2 - t1)
    u = shift / dt
    return u, shift

def find_modulation_period_by_shift(n_t1, n_t2, t1, t2, L):
    """
    Find the drift velocity by comparing two snapshots at different times.
    Uses robust Fourier-based method that works with multiple modes.
    
    Parameters:
    -----------
    n_t1 : array
        Density profile at time t1
    n_t2 : array
        Density profile at time t2
    t1 : float
        First time
    t2 : float
        Second time
    L : float
        Domain length
        
    Returns:
    --------
    u_drift : float
        Drift velocity
    shift_optimal : float
        Optimal spatial shift
    correlation_max : float
        Maximum correlation achieved
    shifts : array
        Array of tested shifts
    correlations : array
        Correlation for each shift
    """
    u, shift = estimate_velocity_fourier(n_t1, n_t2, t1, t2, L)
    
    # Build a correlation-vs-shift curve via FFT xcorr for visualization
    N = n_t1.size
    f1 = n_t1 - n_t1.mean()
    f2 = n_t2 - n_t2.mean()
    F1 = np.fft.rfft(f1)
    F2 = np.fft.rfft(f2)
    xcorr = np.fft.irfft(np.conj(F1) * F2, n=N)        # circular correlation
    dx = L/N
    shifts = dx * (np.arange(N) - (N//2))
    xcorr = np.roll(xcorr, -N//2)
    
    # Simple normalized "correlation"
    correlations = (xcorr - xcorr.min())/(xcorr.max()-xcorr.min() + 1e-15)
    
    # Use Fourier-estimated shift as "optimal"
    corr_max = np.interp(shift, shifts, correlations)
    
    return u, shift, corr_max, shifts, correlations


def plot_velocity_detection(n_t1, n_t2, t1, t2, L, u_drift, shift_opt, corr_max, shifts, correlations, u_d_target, outdir="test_output"):
    """
    Plot the velocity detection method showing correlation vs shift.
    Similar to plot_period_detection in main.py.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Initial and shifted initial vs final
    x = np.linspace(0, L, len(n_t2), endpoint=False)
    
    # Create shifted initial profile at detected shift using Fourier method
    n_t1_shifted = _fourier_shift_real(n_t1, -shift_opt, L)
    
    ax1.plot(x, n_t1, 'b-', label=f'Initial n(x,{t1:.6f})', alpha=0.6, linewidth=1.5)
    ax1.plot(x, n_t2, 'r-', label=f'Final n(x,{t2:.6f})', alpha=0.8, linewidth=2)
    ax1.plot(x, n_t1_shifted, 'g--', label=f'Initial shifted by {shift_opt:.5f}', alpha=0.8, linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('n(x)')
    ax1.set_title('Velocity Detection: Shifted Initial vs Final')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation vs shift
    ax2.plot(shifts, correlations, 'b-', linewidth=2, label='Correlation')
    ax2.axvline(shift_opt, color='r', linestyle='--', linewidth=2, label=f'Optimal shift={shift_opt:.5f}')
    ax2.plot([shift_opt], [corr_max], 'ro', markersize=8, label=f'Max corr={corr_max:.3f}')
    
    ax2.set_xlabel('Spatial shift')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation vs Shift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add text summary
    delta_t = t2 - t1
    fig.suptitle(f'Velocity Analysis: u_drift={u_drift:.3f} (shift={shift_opt:.5f}, Δt={delta_t:.4f}) | u_target={u_d_target:.3f}', 
                 y=0.98, fontsize=10.5)
    
    plt.savefig(f"{outdir}/test_velocity_detection_ud2p8000.png", dpi=160, bbox_inches='tight')
    # plt.savefig(f"{outdir}/test_velocity_detection_ud3p9000.pdf", dpi=160, bbox_inches='tight')
    
    plt.show()
    plt.close()
    
    return u_drift, shift_opt, corr_max

def test_velocity_analysis():
    """
    Test function to determine u_true from the .npz file for out_drift_ud3p9000 case.
    """
    # Path to the .npz file
    # npz_path = "multiple_u_d/modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.001, seed_amp_p=0.001)/out_drift_ud3p9000/data_m07_ud3p9000_ud3.9000000000000012.npz"
    npz_path = "multiple_u_d/modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.001, seed_amp_p=0.001)/out_drift_ud2p8000/data_m07_ud2p8000_ud2.8.npz"
    
    print(f"Loading data from: {npz_path}")
    
    # Check if file exists
    if not os.path.exists(npz_path):
        print(f"Error: File not found: {npz_path}")
        return
    
    # Load the data
    try:
        data = np.load(npz_path)
        print("Data loaded successfully!")
        print(f"Available keys: {list(data.keys())}")
        
        # Extract data
        t = data['t']
        n_t = data['n_t']
        L = float(data['L'])
        Nx = int(data['Nx'])
        
        print(f"Data shape: n_t = {n_t.shape}")
        print(f"Time range: {t[0]:.3f} to {t[-1]:.3f}")
        print(f"Domain length: L = {L}")
        print(f"Grid points: Nx = {Nx}")
        
        # Use the last two time points for velocity measurement (same as main.py)
        idx_t1 = -2  # Second-to-last snapshot
        idx_t2 = -1  # Last snapshot
        
        n_t1 = n_t[:, idx_t1]
        n_t2 = n_t[:, idx_t2]
        t1 = t[idx_t1]
        t2 = t[idx_t2]
        
        print(f"\nUsing time points: t1={t1:.3f}, t2={t2:.3f}")
        print(f"Time difference: Δt = {t2-t1:.3f}")
        
        # Calculate velocity using the exact same method as main.py
        print(f"\n=== FOURIER-BASED VELOCITY ANALYSIS (main.py method) ===")
        u_drift, shift_opt, corr_max, shifts, correlations = find_modulation_period_by_shift(
            n_t1, n_t2, t1, t2, L
        )
        
        print(f"\n=== VELOCITY ANALYSIS RESULTS ===")
        print(f"u_drift = {u_drift:.6f}")
        print(f"Optimal shift = {shift_opt:.6f}")
        print(f"Maximum correlation = {corr_max:.6f}")
        print(f"Time difference = {t2-t1:.6f}")
        
        # Expected u_d from the filename (ud2p8000 = 2.8)
        u_d_expected = 2.8
        print(f"\nExpected u_d = {u_d_expected}")
        print(f"Difference = {abs(u_drift - u_d_expected):.6f}")
        print(f"Relative error = {abs(u_drift - u_d_expected)/u_d_expected*100:.2f}%")
        
        # Create visualization
        plot_velocity_detection(n_t1, n_t2, t1, t2, L, u_drift, shift_opt, corr_max, 
                               shifts, correlations, u_d_expected, outdir="test_output")
        
        print(f"\nVisualization saved to test_output/test_velocity_detection_ud3p9000.png")
        
        # Additional analysis: check if the pattern is actually drifting
        print(f"\n=== PATTERN ANALYSIS ===")
        print(f"Mean density t1: {np.mean(n_t1):.6f}")
        print(f"Mean density t2: {np.mean(n_t2):.6f}")
        print(f"Std density t1: {np.std(n_t1):.6f}")
        print(f"Std density t2: {np.std(n_t2):.6f}")
        
        # Check if there are clear patterns
        n_t1_detrended = n_t1 - np.mean(n_t1)
        n_t2_detrended = n_t2 - np.mean(n_t2)
        
        print(f"Correlation without shift: {np.corrcoef(n_t1_detrended, n_t2_detrended)[0,1]:.6f}")
        print(f"Correlation with optimal shift: {corr_max:.6f}")
        print(f"Improvement: {corr_max - np.corrcoef(n_t1_detrended, n_t2_detrended)[0,1]:.6f}")
        
        # Fourier analysis of patterns
        print(f"\n=== FOURIER ANALYSIS ===")
        fft1 = fft(n_t1_detrended)
        fft2 = fft(n_t2_detrended)
        
        # Power spectra
        power1 = np.abs(fft1)**2
        power2 = np.abs(fft2)**2
        
        # Find dominant modes
        k = 2*np.pi*np.arange(len(n_t1)) / L
        k_pos = k[1:len(k)//2]  # Positive wavenumbers only
        
        # Find peaks in power spectrum
        from scipy.signal import find_peaks
        peaks1, _ = find_peaks(power1[1:len(power1)//2], height=np.max(power1[1:len(power1)//2])*0.1)
        peaks2, _ = find_peaks(power2[1:len(power2)//2], height=np.max(power2[1:len(power2)//2])*0.1)
        
        print(f"Dominant modes t1: {peaks1 + 1} (k = {k_pos[peaks1]})")
        print(f"Dominant modes t2: {peaks2 + 1} (k = {k_pos[peaks2]})")
        
        # Check if patterns are too weak
        max_power1 = np.max(power1[1:len(power1)//2])
        max_power2 = np.max(power2[1:len(power2)//2])
        print(f"Max power t1: {max_power1:.2e}")
        print(f"Max power t2: {max_power2:.2e}")
        
        if max_power1 < 1e-10 or max_power2 < 1e-10:
            print("WARNING: Very weak patterns detected - velocity analysis may be unreliable")
        
        # Suggest improvements
        print(f"\n=== SUGGESTIONS ===")
        if abs(shift_opt) < 1e-6:
            print("1. Pattern may not be moving - check if simulation reached steady state")
            print("2. Try using different time points (earlier in simulation)")
            print("3. Check if pattern amplitude is sufficient")
            print("4. Consider using longer time intervals between snapshots")
        elif corr_max < 0.5:
            print("1. Low correlation suggests pattern may be changing shape, not just translating")
            print("2. Try using shorter time intervals between snapshots")
            print("3. Check if pattern is too weak or noisy")
        else:
            print("1. Good correlation detected - velocity measurement should be reliable")
        
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_velocity_analysis()
