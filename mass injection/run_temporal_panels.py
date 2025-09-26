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
    par.t_final = 10.0  # Shorter simulation time for panel
    par.n_save = 512    # Fewer time points for efficiency
    
    print(f"System parameters:")
    print(f"  λ₀ (base amplitude) = {par.lambda0}")
    print(f"  λ₁ (modulation amp) = {par.lambda1}")
    print(f"  u_d (drift velocity) = {par.u_d}")
    print(f"  x₀ (center)         = {par.x0}")
    print(f"  t_final             = {par.t_final}")
    
    # Define frequency values to test
    # Option 1: Around the natural frequency (if you know it approximately)
    nu_values_natural = np.array([0.5, 1.0, 1.426, 2.0, 2.5])  # Include known resonance
    
    # Option 2: Evenly spaced range
    nu_values_range = np.linspace(0.5, 3.0, 5)
    
    # Option 3: Specific interesting frequencies
    nu_values_specific = np.array([0.8])#, 1.2, 1.426, 1.8, 2.2])
    
    # Choose which set to use
    nu_values = nu_values_specific
    
    print(f"\n1. Running baseline test with static Gaussian potential (λ₁=0)")
    print(f"   - Gaussian potential: λ(x) = {par.lambda0} × exp(-(x-{par.x0})²/(2×{par.sigma_static}²))")
    print(f"   - Time-independent: λ₁=0, so potential doesn't oscillate")
    print(f"   - High drift velocity: u_d = {par.u_d}")
    print(f"   - ν parameter should have NO effect on system behavior")
    
    # Run baseline analysis with no modulation (λ₁=0)
    results_baseline = plot_temporal_evolution_panel(nu_values, lambda1_fixed=0.0, tag="nu_comparison_baseline")
    
    print(f"\n2. Running comparison panel with time-dependent modulation (λ₁=0.1)")
    print(f"   - Time-dependent potential: λ(t) = {par.lambda0} + 0.1×cos(νt)")
    print(f"   - Now ν parameter SHOULD affect system behavior")
    print(f"   - Expect different responses for different ν values")
    
    # Run the panel analysis with modulation
    results = plot_temporal_evolution_panel(nu_values, lambda1_fixed=0.1, tag="nu_comparison")
    
    print(f"\n3. Panel plots generated!")
    print(f"   Baseline (λ₁=0): out_drift/temporal_evolution_panel_nu_comparison_baseline.png")
    print(f"   Modulated (λ₁=0.1): out_drift/temporal_evolution_panel_nu_comparison.png")
    print(f"   Layout: {len(nu_values)} rows × 1 column (vertical)")
    
    # Optional: Generate a third panel with different λ₁
    print(f"\n4. Generating additional panel with stronger modulation...")
    nu_values_subset = nu_values[:3]  # Use fewer frequencies for third panel
    results2 = plot_temporal_evolution_panel(nu_values_subset, lambda1_fixed=0.2, tag="lambda1_02")
    
    print(f"\n5. Third panel generated!")
    print(f"   Output: out_drift/temporal_evolution_panel_lambda1_02.png")
    print(f"   Layout: {len(nu_values_subset)} rows × 1 column (vertical)")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("Panel Generation Completed!")
    print("Generated files:")
    print("  1. temporal_evolution_panel_nu_comparison_baseline.png - Various ν with λ₁=0 (baseline)")
    print("  2. temporal_evolution_panel_nu_comparison.png - Various ν with λ₁=0.1 (modulated)")
    print("  3. temporal_evolution_panel_lambda1_02.png - Subset with λ₁=0.2 (strong modulation)")
    print(f"\nEach panel shows:")
    print(f"  - Blue solid line: n(x₀,t) density evolution")
    print(f"  - Red dashed line: λ(t) driving modulation (scaled)")
    print(f"  - Period T = 2π/ν shown in each panel")
    print(f"  - Consistent y-axis scaling for comparison")
    print("=" * 60)

if __name__ == "__main__":
    main()
