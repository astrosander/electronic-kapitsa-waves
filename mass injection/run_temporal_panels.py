#!/usr/bin/env python3
"""
Script to generate temporal evolution panels for various driving frequencies.

This script demonstrates the temporal evolution of density at the perturbation center
for different driving frequencies ν (nu), organized in a clean panel format.
"""

import numpy as np
import sys
import os

# Import the main simulation functions
exec(open('mass injection.py').read().split('if __name__')[0])

def main():
    print("=" * 60)
    print("Temporal Evolution Panel Generation")
    print("=" * 60)
    
    # Set simulation parameters for faster runs
    par.t_final = 30.0  # Shorter simulation time for panel
    par.n_save = 200    # Fewer time points for efficiency
    
    print(f"System parameters:")
    print(f"  λ₀ (base amplitude) = {par.lambda0}")
    print(f"  λ₁ (modulation amp) = {par.lambda1}")
    print(f"  x₀ (center)         = {par.x0}")
    print(f"  t_final             = {par.t_final}")
    
    # Define frequency values to test
    # Option 1: Around the natural frequency (if you know it approximately)
    nu_values_natural = np.array([0.5, 1.0, 1.426, 2.0, 2.5])  # Include known resonance
    
    # Option 2: Evenly spaced range
    nu_values_range = np.linspace(0.5, 3.0, 5)
    
    # Option 3: Specific interesting frequencies
    nu_values_specific = np.array([0.8, 1.2, 1.426, 1.8, 2.2])
    
    # Choose which set to use
    nu_values = nu_values_specific
    
    print(f"\n1. Running temporal evolution panel for ν values: {nu_values}")
    print(f"   Each panel will show n(x₀,t) vs t for different driving frequencies")
    
    # Run the panel analysis
    results = plot_temporal_evolution_panel(nu_values, lambda1_fixed=0.1, tag="nu_comparison")
    
    print(f"\n2. Panel plot generated!")
    print(f"   Output: out_drift/temporal_evolution_panel_nu_comparison.png")
    print(f"   Layout: {len(nu_values)} rows × 1 column (vertical)")
    
    # Optional: Generate a second panel with different λ₁
    print(f"\n3. Generating second panel with different modulation amplitude...")
    nu_values_subset = nu_values[:3]  # Use fewer frequencies for second panel
    results2 = plot_temporal_evolution_panel(nu_values_subset, lambda1_fixed=0.2, tag="lambda1_02")
    
    print(f"\n4. Second panel generated!")
    print(f"   Output: out_drift/temporal_evolution_panel_lambda1_02.png")
    print(f"   Layout: {len(nu_values_subset)} rows × 1 column (vertical)")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("Panel Generation Completed!")
    print("Generated files:")
    print("  1. temporal_evolution_panel_nu_comparison.png - Various ν with λ₁=0.1")
    print("  2. temporal_evolution_panel_lambda1_02.png - Subset with λ₁=0.2")
    print(f"\nEach panel shows:")
    print(f"  - Blue solid line: n(x₀,t) density evolution")
    print(f"  - Red dashed line: λ(t) driving modulation (scaled)")
    print(f"  - Period T = 2π/ν shown in each panel")
    print(f"  - Consistent y-axis scaling for comparison")
    print("=" * 60)

if __name__ == "__main__":
    main()
