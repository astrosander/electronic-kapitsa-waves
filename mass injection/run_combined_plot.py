#!/usr/bin/env python3
"""Run velocity analysis for multiple parameter sets."""

from plot_from_data import plot_combined_velocity_analysis, plot_delta_n_vs_ud

if __name__ == "__main__":
    base_dirs = [
        "multiple_u_d/delta n=delta p=0.03(cos3x+cos5x+cos8x+cos13x)",
        "multiple_u_d/delta n=delta p=0.05(cos3x+cos5x+cos8x+cos13x)",
        "multiple_u_d/quadratic;delta n=delta p=0.05(cos3x+cos5x+cos8x+cos13x)",
        "multiple_u_d/delta n=delta p=0.03(cos3x+cos5x+cos8x+cos13x); lin u_d2",
        # "multiple_u_d/cos(5x)+cos(8x)",
        # "multiple_u_d/cos3x+cos5x+cos8x+cos13x",
        # "multiple_u_d/delta n=delta p=cos3x-cos5x+cos8x-cos13x",
    ]
    
    custom_labels = [
        "$\\delta n,\\delta p = 0.03$",
        "$\\delta n,\\delta p = 0.05$ non-unif.",
        "$\\delta n,\\delta p = 0.05$ quad.",
        "$\\delta n,\\delta p = 0.03$, lin. $u_d<2$",
        # "cos(5x)+cos(8x)",
        # "cos3x+cos5x+cos8x+cos13x",
        # "cos3x-cos5x+cos8x-cos13x",
    ]
    
    print("=" * 60)
    print("VELOCITY ANALYSIS: Multiple Parameter Sets")
    print("=" * 60)
    
    # Plot velocity analysis (u_true, n_pulses, frequency)
    print("Generating velocity analysis (u_true, n_pulses, frequency)...")
    velocity_data = plot_combined_velocity_analysis(base_dirs, labels=custom_labels)
    
    # Plot delta n vs u_d
    print("\nGenerating delta n vs u_d plot...")
    # delta_n_data = plot_delta_n_vs_ud(base_dirs, labels=custom_labels)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Generated:")
    print("- velocity_vs_ud_combined.png")
    print("- delta_n_vs_ud.png")
    print("=" * 60)

