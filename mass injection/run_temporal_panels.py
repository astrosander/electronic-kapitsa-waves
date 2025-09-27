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
    par.t_final = 20.0  # Shorter simulation time for panel
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
    nu_values_specific = np.array([0, 52, 55, 57])#, 1.2, 1.426, 1.8, 2.2])
    print(nu_values_specific)
    # Choose which set to use
    nu_values = nu_values_specific
    
    print(f"\n1. Running baseline test with static Gaussian potential (λ₁=0)")
    print(f"   - Gaussian potential: λ(x) = {par.lambda0} × exp(-(x-{par.x0})²/(2×{par.sigma_static}²))")
    print(f"   - Time-independent: λ₁=0, so potential doesn't oscillate")
    print(f"   - High drift velocity: u_d = {par.u_d}")
    print(f"   - ν parameter should have NO effect on system behavior")
    
    # Run baseline analysis with no modulation (λ₁=0)
    results_baseline = plot_temporal_evolution_panel(nu_values, lambda1_fixed=0.1, tag="nu_comparison_baseline")
    
    print(f"\n2. Plotting temporal evolution at multiple spatial locations (left and right of x₀)")
    print(f"   - Monitor density n(t) at x₀-2, x₀-1, x₀, x₀+1, x₀+2")
    print(f"   - Shows spatial propagation of density variations")
    
    # Plot temporal evolution at multiple spatial locations
    results_spatial = plot_spatial_temporal_evolution_panel(nu_values, lambda1_fixed=0.1, tag="nu_comparison_spatial")
    
    print(f"\n3. Plotting density profiles at t=t_final for each frequency")
    print(f"   - Compare initial vs final density profiles") 
    print(f"   - Shows structural changes after time evolution")
    print(f"   - Quantifies density variations (Δn) for each frequency")
    
    # Plot final density profiles
    results_profiles = plot_final_density_profiles_panel(nu_values, lambda1_fixed=0.1, tag="nu_comparison_profiles")
    
    print(f"\n=" * 60)
    print(f"Analysis complete! Generated plots:")
    print(f"  1. out_drift/temporal_evolution_panel_nu_comparison_baseline.png")
    print(f"  2. out_drift/spatial_temporal_evolution_panel_nu_comparison_spatial.png") 
    print(f"  3. out_drift/final_density_profiles_panel_nu_comparison_profiles.png")
    print(f"=" * 60)
    
    
if __name__ == "__main__":
    main()
