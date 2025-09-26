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
    nu_values_specific = np.array([50, 53, 55])#, 1.2, 1.426, 1.8, 2.2])
    
    # Choose which set to use
    nu_values = nu_values_specific
    
    print(f"\n1. Running baseline test with static Gaussian potential (λ₁=0)")
    print(f"   - Gaussian potential: λ(x) = {par.lambda0} × exp(-(x-{par.x0})²/(2×{par.sigma_static}²))")
    print(f"   - Time-independent: λ₁=0, so potential doesn't oscillate")
    print(f"   - High drift velocity: u_d = {par.u_d}")
    print(f"   - ν parameter should have NO effect on system behavior")
    
    # Run baseline analysis with no modulation (λ₁=0)
    results_baseline = plot_temporal_evolution_panel(nu_values, lambda1_fixed=0.05, tag="nu_comparison_baseline")
    
    
if __name__ == "__main__":
    main()
