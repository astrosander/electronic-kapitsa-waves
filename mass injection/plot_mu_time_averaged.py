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
    # Define mu values to test - expanded range including original three values
    mu_values = [0.8, 1.0, 1.2, 1.261, 1.426, 1.5, 1.591, 1.8, 2.0, 2.2]
    
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
                label=f'μ = {mu:.3f} [min={n_min:.6f}, max={n_max:.6f}]')
    
    # Formatting
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$\\langle n \\rangle_{time}$', fontsize=14)
    plt.title('Time-Averaged Density Profiles for Different μ Values', fontsize=16)
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

if __name__ == "__main__":
    print("=" * 70)
    print("Time-Averaged Density Profile Comparison")
    print("μ values: [0.8, 1.0, 1.2, 1.261, 1.426, 1.5, 1.591, 1.8, 2.0, 2.2]")
    print("=" * 70)
    
    # Run the comparison
    results = plot_mu_time_averaged_comparison()
    
    print("\n" + "=" * 70)
    print("Summary of Results (8 decimal places):")
    for mu, n_avg_time in results:
        n_min = n_avg_time.min()
        n_max = n_avg_time.max()
        delta_n = n_max - n_min
        print(f"μ = {mu:6.3f}: min = {n_min:.8f}, max = {n_max:.8f}, Δn = {delta_n:.8f}")
    print("=" * 70)
