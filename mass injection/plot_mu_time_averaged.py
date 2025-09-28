#!/usr/bin/env python3
"""
Script to plot time-averaged density profiles for various mu values.
Creates a single plot with only the time-averaged black lines showing their min and max values.
Note: mu corresponds to the driving frequency parameter (par.nu) in the code.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import the main simulation functions
exec(open('mass injection.py').read().split('if __name__')[0])

def run_simulation_for_mu(mu_value, tag=""):
    """
    Run simulation for a specific mu value and return time-averaged density profile
    """
    # Store original values
    old_nu = par.nu
    old_outdir = par.outdir
    
    # Set the mu value (stored as par.nu in the code)
    par.nu = mu_value
    par.outdir = f"out_drift_mu{mu_value:g}"
    
    try:
        print(f"\n[mu_sweep] Running simulation for μ = {mu_value}")
        
        # Run simulation
        t, n_t, p_t = run_once(tag=f"mu{mu_value:g}")
        
        # Calculate time-averaged density profile
        t_start_avg = 10.0
        t_end_avg = 50.0
        i_start = np.argmin(np.abs(t - t_start_avg))
        i_end = np.argmin(np.abs(t - t_end_avg))
        n_avg_time = np.mean(n_t[:, i_start:i_end+1], axis=1)
        
        return n_avg_time, t, n_t
        
    finally:
        # Restore original values
        par.nu = old_nu
        par.outdir = old_outdir

def plot_mu_time_averaged_comparison():
    """
    Plot time-averaged density profiles for a range of mu values
    """
    # Define mu values to test - constant step spacing
    mu_values = np.arange(0.826, 2.426, 0.02)  # From 0.8 to 2.2 with step 0.2
    print(mu_values)
    # This gives: [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    
    # Store results
    results = []
    
    # Run simulations for each mu value
    for mu in mu_values:
        n_avg_time, t, n_t = run_simulation_for_mu(mu)
        results.append((mu, n_avg_time))
        
        # Print min/max values with more decimal places
        n_min = n_avg_time.min()
        n_max = n_avg_time.max()
        print(f"[results] μ = {mu}: min = {n_min:.8f}, max = {n_max:.8f}, Δn = {n_max-n_min:.8f}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Color palette for different mu values (more colors for larger range)
    colors = plt.cm.tab10(np.linspace(0, 1, len(mu_values)))
    
    # Plot each time-averaged profile
    for i, (mu, n_avg_time) in enumerate(results):
        n_min = n_avg_time.min()
        n_max = n_avg_time.max()
        delta_n = n_max - n_min
        
        # Plot the time-averaged profile as black dashed line but with different colors for visibility
        plt.plot(x, n_avg_time, '--', lw=2, color=colors[i], 
                label=f'$\\nu$ = {mu:.3f} [min={n_min:.6f}, max={n_max:.6f}]')
    
    # Formatting
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$\\langle n \\rangle_{time}$', fontsize=14)
    plt.title('Time-Averaged Density Profiles for Different $\\nu$ Values', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, frameon=True, loc='best', ncol=2)
    
    # Add vertical line at perturbation center
    plt.axvline(par.x0, color='gray', linestyle=':', alpha=0.7)
    
    # Add text box with simulation parameters
    param_text = f"""Simulation Parameters:
λ₀ = {par.lambda0}
λ₁ = {par.lambda1}
u_d = {par.u_d}
Time avg: t ∈ [10, 50]
x₀ = {par.x0}"""
    
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig("out_drift/mu_time_averaged_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("out_drift/mu_time_averaged_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n[plot] Saved: out_drift/mu_time_averaged_comparison.png")
    print(f"[plot] Saved: out_drift/mu_time_averaged_comparison.pdf")
    
    return results

def plot_n_analysis_vs_nu():
    """
    Plot both delta_n and sum_n vs frequency nu in a single run
    """
    # Define nu values to test - wider range from 0.2 to 2.4
    nu_values = np.arange(0.2, 2.41, 0.02)  # From 0.2 to 2.4 with step 0.02
    print(f"ν values: {nu_values}")
    print(f"Total simulations to run: {len(nu_values)}")
    
    # Store results
    delta_n_values = []
    sum_n_values = []
    
    # Run simulations for each nu value
    for i, nu in enumerate(nu_values):
        print(f"\n[Progress] {i+1}/{len(nu_values)}: Running ν = {nu:.3f}")
        n_avg_time, t, n_t = run_simulation_for_nu(nu)
        
        # Calculate min, max, difference, and sum
        n_min = n_avg_time.min()
        n_max = n_avg_time.max()
        delta_n = n_max - n_min
        sum_n = n_max + n_min
        
        delta_n_values.append(delta_n)
        sum_n_values.append(sum_n)
        
        print(f"[results] ν = {nu:.3f}: min = {n_min:.8f}, max = {n_max:.8f}, Δn = {delta_n:.8f}, Σn = {sum_n:.8f}")
    
    # Create both plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Delta n vs nu
    ax1.plot(nu_values, delta_n_values, 'o-', linewidth=2, markersize=4, 
             color='blue')
    ax1.set_xlabel('$\\nu$', fontsize=12)
    ax1.set_ylabel('$\\Delta n$', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Density Variation vs Frequency', fontsize=12)
    
    # Plot 2: Sum n vs nu
    ax2.plot(nu_values, sum_n_values, 'o-', linewidth=2, markersize=4, 
             color='red')
    ax2.set_xlabel('$\\nu$', fontsize=12)
    ax2.set_ylabel('$\\max(n) + \\min(n)$', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Density Sum vs Frequency', fontsize=12)
    
    plt.tight_layout()
    
    # Save the combined plot
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig("out_drift/n_analysis_vs_nu.png", dpi=300, bbox_inches='tight')
    plt.savefig("out_drift/n_analysis_vs_nu.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n[plot] Saved: out_drift/n_analysis_vs_nu.png")
    print(f"[plot] Saved: out_drift/n_analysis_vs_nu.pdf")
    
    # Also save individual plots for compatibility
    # Delta n plot
    plt.figure(figsize=(8, 5))
    plt.plot(nu_values, delta_n_values, 'o-', linewidth=2, markersize=4, color='blue')
    plt.xlabel('$\\nu$', fontsize=12)
    plt.ylabel('$\\Delta n$', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("out_drift/delta_n_vs_nu.png", dpi=300, bbox_inches='tight')
    plt.savefig("out_drift/delta_n_vs_nu.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sum n plot
    plt.figure(figsize=(8, 5))
    plt.plot(nu_values, sum_n_values, 'o-', linewidth=2, markersize=4, color='red')
    plt.xlabel('$\\nu$', fontsize=12)
    plt.ylabel('$\\max(n) + \\min(n)$', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("out_drift/sum_n_vs_nu.png", dpi=300, bbox_inches='tight')
    plt.savefig("out_drift/sum_n_vs_nu.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] Also saved individual plots: delta_n_vs_nu.png and sum_n_vs_nu.png")
    
    return nu_values, delta_n_values, sum_n_values

def run_simulation_for_nu(nu_value, tag=""):
    """
    Run simulation for a specific nu value and return time-averaged density profile
    (Renamed from run_simulation_for_mu for consistency with nu notation)
    """
    # Store original values
    old_nu = par.nu
    old_outdir = par.outdir
    
    # Set the nu value
    par.nu = nu_value
    par.outdir = f"out_drift_nu{nu_value:g}"
    
    try:
        print(f"\n[nu_sweep] Running simulation for ν = {nu_value}")
        
        # Run simulation
        t, n_t, p_t = run_once(tag=f"nu{nu_value:g}")
        
        # Calculate time-averaged density profile
        t_start_avg = 10.0
        t_end_avg = 50.0
        i_start = np.argmin(np.abs(t - t_start_avg))
        i_end = np.argmin(np.abs(t - t_end_avg))
        n_avg_time = np.mean(n_t[:, i_start:i_end+1], axis=1)
        
        return n_avg_time, t, n_t
        
    finally:
        # Restore original values
        par.nu = old_nu
        par.outdir = old_outdir

if __name__ == "__main__":
    print("=" * 70)
    print("Density Analysis vs Frequency")
    nu_test_values = np.arange(0.2, 2.41, 0.02)
    print(f"ν values (constant step Δν=0.02): {nu_test_values}")
    print(f"Range: ν ∈ [0.2, 2.4], Total points: {len(nu_test_values)}")
    print("=" * 70)
    
    # Plot both delta_n and sum_n vs nu in a single run
    print("\n[MAIN] Creating combined Δn and max(n)+min(n) vs ν plots...")
    nu_values, delta_n_values, sum_n_values = plot_n_analysis_vs_nu()
    
    print("\n" + "=" * 70)
    print("Summary of Combined Analysis Results:")
    for nu, delta_n, sum_n in zip(nu_values, delta_n_values, sum_n_values):
        print(f"ν = {nu:6.3f}: Δn = {delta_n:.8f}, max(n)+min(n) = {sum_n:.8f}")
    print("=" * 70)
    
    # Optionally also run the time-averaged comparison
    print("\n[MAIN] Creating time-averaged density profiles comparison...")
    results = plot_mu_time_averaged_comparison()
    
    print("\n" + "=" * 70)
    print("Summary of Time-Averaged Results:")
    for mu, n_avg_time in results:
        n_min = n_avg_time.min()
        n_max = n_avg_time.max()
        delta_n = n_max - n_min
        print(f"ν = {mu:6.3f}: min = {n_min:.8f}, max = {n_max:.8f}, Δn = {delta_n:.8f}")
    print("=" * 70)