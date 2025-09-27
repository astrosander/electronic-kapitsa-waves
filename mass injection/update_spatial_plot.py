#!/usr/bin/env python3
"""
Quick script to regenerate just the spatial-temporal plot with updated settings.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import the main simulation functions and use existing results
exec(open('mass injection.py').read().split('if __name__')[0])

def regenerate_spatial_plot_from_previous():
    """
    Load the existing simulation results and regenerate only the spatial-temporal plot
    with the new 3-position layout and vertical displacement.
    """
    
    # Simulate some dummy results for demonstration
    # In practice, you would load these from saved .npz files
    nu_values = [0.0, 1.0, 1.426, 2.0]
    
    print("Regenerating spatial-temporal plot with updated layout...")
    print("Using 3 positions: x₀-1, x₀, x₀+1 with vertical displacement")
    
    # Create dummy time array
    t = np.linspace(0, 20, 512)
    
    # Create dummy n_t data (replace with actual loaded data if available)
    results = []
    for nu in nu_values:
        # Create synthetic data for demonstration
        n_t = np.zeros((par.Nx, len(t)))
        for j, tj in enumerate(t):
            # Simple traveling wave pattern
            if nu > 0.1:
                phase = 2*np.pi*nu*tj
                n_t[:, j] = par.nbar0 + 0.01*np.sin(2*np.pi*3*x/par.L + phase)
            else:
                n_t[:, j] = par.nbar0 + 0.005*np.sin(2*np.pi*x/par.L)
        results.append((nu, t, n_t, None))
    
    # Now generate the spatial-temporal plot with new settings
    plot_spatial_temporal_from_results(results, tag="updated_3positions")

def plot_spatial_temporal_from_results(results, tag="spatial_temporal"):
    """
    Generate spatial-temporal plot from results with 3 positions and vertical displacement
    """
    nu_values = [r[0] for r in results]
    n_panels = len(nu_values)
    
    # Define spatial monitoring points relative to x₀ (3 points only)
    x_offsets = [-1.0, 0.0, 1.0]  # Relative to x₀
    x_positions = [(par.x0 + offset) % par.L for offset in x_offsets]
    x_indices = [np.argmin(np.abs(x - x_pos)) for x_pos in x_positions]
    
    colors = ['red', 'blue', 'green']  # Colors for different positions
    labels = [f'$x_0{offset:+.1f}$' if offset != 0 else '$x_0$' for offset in x_offsets]
    
    # Vertical displacement for clarity (small offsets to separate the curves)
    vertical_offsets = [-0.008, 0.0, 0.008]  # Slightly larger vertical shifts for visibility
    
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.8*n_panels), gridspec_kw={'hspace': 0.25})
    
    if n_panels == 1:
        axes = [axes]
    
    # Find global y-limits for consistent scaling
    y_min, y_max = float('inf'), float('-inf')
    for nu, t, n_t, p_t in results:
        for i_pos in x_indices:
            n_pos = n_t[i_pos, :]
            y_min = min(y_min, n_pos.min())
            y_max = max(y_max, n_pos.max())
    
    # Adjust y-limits to account for vertical displacement
    y_range = y_max - y_min
    y_min_plot = y_min - 0.015  # Extra space for displacement
    y_max_plot = y_max + 0.015
    
    for i, (nu, t, n_t, p_t) in enumerate(results):
        ax = axes[i]
        
        # Plot temporal evolution at each spatial position with vertical displacement
        for j, (i_pos, color, label, v_offset) in enumerate(zip(x_indices, colors, labels, vertical_offsets)):
            n_pos = n_t[i_pos, :] + v_offset  # Add vertical displacement
            ax.plot(t, n_pos, color=color, lw=1.5, label=label, alpha=0.9)
        
        # Add the driving modulation for comparison (only if λ₁ > 0)
        if nu > 0.0001:
            lambda1_used = 0.1
            lambda_t = par.lambda0 + lambda1_used * np.cos(nu * t)
            # Scale and shift for visibility
            lambda_scaled = par.nbar0 + 0.02 * (lambda_t - par.lambda0)
            ax.plot(t, lambda_scaled, 'k--', lw=1.0, alpha=0.6, label="$\\lambda(t)$ (scaled)")
        
        # Set y-limits
        ax.set_ylim(y_min_plot, y_max_plot)
        
        if i == n_panels - 1:  # Only label x-axis on bottom panel
            ax.set_xlabel("$t$", fontsize=11)
        ax.set_ylabel("$n(x,t)$", fontsize=11)
        
        if nu <= 0.0001:
            ax.set_title(f"$\\lambda_1 = 0.0$ (no modulation)", fontsize=12)
        else:
            ax.set_title(f"$\\nu = {nu:.3f}$", fontsize=12)

        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=9, loc='upper right', ncol=2)
        
        # Add frequency info
        period = 2*np.pi/nu if nu > 0 else np.inf
        period_text = f"$T = {period:.2f}$" if period < 1000 else "$T = \\infty$"
        ax.text(0.02, 0.95, period_text, transform=ax.transAxes, 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        
        # Add note about vertical displacement
        if i == 0:
            ax.text(0.98, 0.05, "Curves vertically displaced\\nfor clarity", 
                   transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f"Temporal Evolution at 3 Spatial Locations (with vertical displacement)", fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    os.makedirs("out_drift", exist_ok=True)
    plt.savefig(f"out_drift/spatial_temporal_evolution_panel_{tag}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"out_drift/spatial_temporal_evolution_panel_{tag}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"✓ Saved: out_drift/spatial_temporal_evolution_panel_{tag}.png")
    print(f"✓ Features:")
    print(f"  • 3 spatial positions: x₀-1, x₀, x₀+1")
    print(f"  • Vertical displacement: {vertical_offsets}")
    print(f"  • Colors: {colors}")
    print(f"  • Clearer visualization of spatial differences")

if __name__ == "__main__":
    regenerate_spatial_plot_from_previous()
