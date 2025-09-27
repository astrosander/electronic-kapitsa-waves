#!/usr/bin/env python3
"""
Efficient script to generate comprehensive temporal evolution analysis.

This script runs each simulation only ONCE and then performs multiple analyses 
on the stored results, avoiding redundant computations.
"""

import numpy as np
import sys
import os

# Import the main simulation functions
exec(open('mass injection.py').read().split('if __name__')[0])

def main():
    print("=" * 70)
    print("EFFICIENT Temporal Evolution Analysis")
    print("(Run simulations once, perform multiple analyses)")
    print("=" * 70)
    
    # Set simulation parameters for faster runs
    par.t_final = 20.0#20.0  # Shorter simulation time for panel
    par.n_save = 200#512    # Fewer time points for efficiency
    
    print(f"System parameters:")
    print(f"  λ₀ (base amplitude) = {par.lambda0}")
    print(f"  λ₁ (modulation amp) = {par.lambda1}")
    print(f"  u_d (drift velocity) = {par.u_d}")
    print(f"  x₀ (center)         = {par.x0}")
    print(f"  t_final             = {par.t_final}")
    print(f"  n_save              = {par.n_save}")
    
    # Define frequency values to test
    # Use the same as yesterday's promising results
    nu_values_interesting = np.array([0, 52, 55, 57])
    
    # Alternative: use smaller values for faster testing
    nu_values_test = np.array([0, 1.0, 1.426, 2.0])
    
    # Choose which set to use
    nu_values = nu_values_interesting  # Change to nu_values_interesting for full analysis
    
    print(f"\nFrequency values to analyze: {nu_values}")
    print(f"Expected simulation time: ~{len(nu_values)} × {par.t_final:.1f}s = {len(nu_values) * par.t_final:.1f}s")
    
    # === STEP 1: Run all simulations once ===
    print(f"\n" + "="*50)
    print(f"STEP 1: Running simulations (once each)")
    print(f"="*50)
    
    # Run simulations with the efficient function
    results = run_simulations_once_and_analyze(
        nu_values=nu_values, 
        lambda1_fixed=par.lambda1,  # Use current parameter value
        tag="efficient_analysis"
    )
    
    # === STEP 2: Generate all analysis plots from stored results ===
    print(f"\n" + "="*50)
    print(f"STEP 2: Generating analysis plots from stored data")
    print(f"="*50)
    
    # Generate all plots from the stored results
    plot_from_stored_results(
        results=results,
        lambda1_used=par.lambda1,  # Use the actual lambda1 that was used
        tag="efficient_analysis"
    )
    
    # === STEP 3: Summary ===
    print(f"\n" + "="*70)
    print(f"ANALYSIS COMPLETE!")
    print(f"="*70)
    print(f"Efficiency gained:")
    print(f"  • Old approach: {len(nu_values)} × 3 = {len(nu_values) * 3} simulations")
    print(f"  • New approach: {len(nu_values)} × 1 = {len(nu_values)} simulations")
    print(f"  • Speedup: {3:.0f}x faster!")
    print(f"")
    print(f"Generated analysis files:")
    print(f"  1. Temporal evolution at x₀:")
    print(f"     → out_drift/temporal_evolution_panel_efficient_analysis.png")
    print(f"  2. Spatial-temporal evolution (multiple x positions):")
    print(f"     → out_drift/spatial_temporal_evolution_panel_efficient_analysis.png")
    print(f"  3. Initial vs final density profiles:")
    print(f"     → out_drift/final_density_profiles_panel_efficient_analysis.png")
    print(f"")
    print(f"Data characteristics:")
    for i, (nu, t, n_t, p_t) in enumerate(results):
        n_final = n_t[:, -1]
        delta_n = n_final.max() - n_final.min()
        if nu <= 0.0001:
            print(f"  • Baseline (λ₁=0):      Δn = {delta_n:.6f}")
        else:
            print(f"  • ν = {nu:6.3f}:          Δn = {delta_n:.6f}")
    
    print(f"="*70)
    
    return results


if __name__ == "__main__":
    results = main()
