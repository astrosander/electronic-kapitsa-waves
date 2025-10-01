"""
Spectral analysis helper functions for electronic Kapitsa waves simulation.

This module provides functions for loading, analyzing, and visualizing
the time evolution of power spectra from the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_spectral_evolution(filename):
    """
    Load spectral evolution data from npz file.
    
    Parameters:
    -----------
    filename : str
        Path to the .npz file containing spectral data
        
    Returns:
    --------
    data : dict
        Dictionary containing:
        - k_wave: wavenumber array
        - P_all: power spectrum evolution (n_modes, n_times)
        - t: time array
        - m: mode number
        - L: domain length
        - meta: simulation parameters
    """
    data = np.load(filename, allow_pickle=True)
    result = {
        'k_wave': data['k_wave'],
        'P_all': data['P_all'],
        't': data['t'],
        'm': int(data['m']),
        'L': float(data['L']),
        'meta': data['meta'].item() if 'meta' in data else {}
    }
    print(f"[load] Loaded spectral evolution from {filename}")
    print(f"[load]   k_wave: {result['k_wave'].shape}, P_all: {result['P_all'].shape}, t: {result['t'].shape}")
    return result

def plot_spectral_evolution(filename_or_data, k_max=20, tag="spectral_evolution"):
    """
    Plot the time evolution of power spectrum as a 2D colormap.
    
    Parameters:
    -----------
    filename_or_data : str or dict
        Either path to npz file or data dict from load_spectral_evolution
    k_max : float
        Maximum wavenumber to plot
    tag : str
        Output filename tag
    """
    if isinstance(filename_or_data, str):
        data = load_spectral_evolution(filename_or_data)
    else:
        data = filename_or_data
    
    k_wave = data['k_wave']
    P_all = data['P_all']
    t = data['t']
    m = data['m']
    
    # Find k_max index
    k_idx = np.where(k_wave <= k_max)[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Spectral evolution as colormap
    extent = [t[0], t[-1], k_wave[k_idx[0]], k_wave[k_idx[-1]]]
    im1 = ax1.imshow(np.log10(P_all[k_idx, :] + 1e-20), 
                     origin='lower', aspect='auto', extent=extent, 
                     cmap='hot', interpolation='bilinear')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Wavenumber k')
    ax1.set_title(f'Spectral Evolution: log₁₀(P(k,t)) for mode m={m}')
    plt.colorbar(im1, ax=ax1, label='log₁₀(Power)')
    
    # Plot 2: Selected time slices
    time_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
    colors = ['blue', 'green', 'orange', 'red', 'black']
    
    for idx, color in zip(time_indices, colors):
        ax2.plot(k_wave[k_idx], P_all[k_idx, idx], 
                label=f't={t[idx]:.3f}', color=color, linewidth=1.5)
    
    ax2.set_xlabel('Wavenumber k')
    ax2.set_ylabel('Power P(k)')
    ax2.set_title('Power Spectra at Selected Times')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{tag}_m{m}.png", dpi=160, bbox_inches='tight')
    plt.savefig(f"{outdir}/{tag}_m{m}.pdf", dpi=160, bbox_inches='tight')
    print(f"[plot] Saved spectral evolution plot → {outdir}/{tag}_m{m}.png")
    plt.close()

def plot_spectral_growth_rates(data, k_modes=None, tag="growth_rates"):
    """
    Plot growth rates for selected wavenumber modes.
    
    Parameters:
    -----------
    data : dict
        Data from load_spectral_evolution
    k_modes : list, optional
        List of wavenumber indices to analyze. If None, uses peak modes.
    tag : str
        Output filename tag
    """
    k_wave = data['k_wave']
    P_all = data['P_all']
    t = data['t']
    m = data['m']
    
    if k_modes is None:
        # Find peak modes at final time
        P_final = P_all[:, -1]
        peak_indices = np.argsort(P_final)[-5:]  # Top 5 modes
        k_modes = peak_indices
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Time evolution of selected modes
    for i, k_idx in enumerate(k_modes):
        ax1.plot(t, P_all[k_idx, :], 
                label=f'k={k_wave[k_idx]:.2f}', linewidth=2)
    
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Power P(k,t)')
    ax1.set_title('Power Evolution for Selected Modes')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Growth rates (slopes in log space)
    growth_rates = []
    for k_idx in k_modes:
        # Fit exponential growth in log space
        log_P = np.log(P_all[k_idx, :] + 1e-20)
        # Use middle portion to avoid initial transients
        start_idx = len(t) // 4
        end_idx = 3 * len(t) // 4
        if end_idx - start_idx > 5:  # Need enough points
            slope = np.polyfit(t[start_idx:end_idx], log_P[start_idx:end_idx], 1)[0]
            growth_rates.append(slope)
        else:
            growth_rates.append(0.0)
    
    ax2.bar(range(len(k_modes)), growth_rates, 
           tick_label=[f'{k_wave[k_idx]:.2f}' for k_idx in k_modes])
    ax2.set_xlabel('Wavenumber k')
    ax2.set_ylabel('Growth Rate σ')
    ax2.set_title('Growth Rates for Selected Modes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{tag}_m{m}.png", dpi=160, bbox_inches='tight')
    plt.savefig(f"{outdir}/{tag}_m{m}.pdf", dpi=160, bbox_inches='tight')
    print(f"[plot] Saved growth rates plot → {outdir}/{tag}_m{m}.png")
    plt.close()

def compare_spectral_evolution(filenames, k_max=20, tag="spectral_comparison"):
    """
    Compare spectral evolution across multiple simulation runs.
    
    Parameters:
    -----------
    filenames : list
        List of .npz filenames to compare
    k_max : float
        Maximum wavenumber to plot
    tag : str
        Output filename tag
    """
    fig, axes = plt.subplots(2, len(filenames), figsize=(4*len(filenames), 8))
    if len(filenames) == 1:
        axes = axes.reshape(2, 1)
    
    for i, filename in enumerate(filenames):
        data = load_spectral_evolution(filename)
        k_wave = data['k_wave']
        P_all = data['P_all']
        t = data['t']
        m = data['m']
        
        k_idx = np.where(k_wave <= k_max)[0]
        
        # Plot 1: Spectral evolution colormap
        extent = [t[0], t[-1], k_wave[k_idx[0]], k_wave[k_idx[-1]]]
        im = axes[0, i].imshow(np.log10(P_all[k_idx, :] + 1e-20), 
                              origin='lower', aspect='auto', extent=extent, 
                              cmap='hot', interpolation='bilinear')
        axes[0, i].set_xlabel('Time t')
        axes[0, i].set_ylabel('Wavenumber k')
        axes[0, i].set_title(f'Mode m={m}')
        plt.colorbar(im, ax=axes[0, i], label='log₁₀(Power)')
        
        # Plot 2: Final spectrum
        axes[1, i].plot(k_wave[k_idx], P_all[k_idx, -1], 'b-', linewidth=2)
        axes[1, i].set_xlabel('Wavenumber k')
        axes[1, i].set_ylabel('Power P(k)')
        axes[1, i].set_title(f'Final Spectrum (t={t[-1]:.2f})')
        axes[1, i].set_yscale('log')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to first file's output directory
    if filenames:
        data0 = load_spectral_evolution(filenames[0])
        outdir = data0['meta'].get('outdir', 'out_drift')
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/{tag}.png", dpi=160, bbox_inches='tight')
        plt.savefig(f"{outdir}/{tag}.pdf", dpi=160, bbox_inches='tight')
        print(f"[plot] Saved comparison plot → {outdir}/{tag}.png")
    plt.close()

if __name__ == "__main__":
    # Example usage
    print("Spectral analysis helper functions loaded.")
    print("Available functions:")
    print("  - load_spectral_evolution(filename)")
    print("  - plot_spectral_evolution(filename_or_data)")
    print("  - plot_spectral_growth_rates(data)")
    print("  - compare_spectral_evolution(filenames)")


filename = "out_drift/spec_m01_m1.npz"

# load_spectral_evolution()

plot_spectral_evolution(filename)