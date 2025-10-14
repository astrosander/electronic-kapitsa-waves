#!/usr/bin/env python3
"""Run velocity analysis for multiple parameter sets."""

from plot_from_data import (
    plot_combined_velocity_analysis,
    plot_delta_n_vs_ud,
    plot_multiple_ud_panel,
    plot_multiple_ud_panel_p,
    plot_overlay_final_profiles_n,
    plot_overlay_final_profiles_p,
    plot_n_p_time_series,
)

if __name__ == "__main__":
    base_dirs = [
        # "multiple_u_d/",
        # "multiple_u_d/no_inhomogeneity(lambda=0.0, sigma=-1.0, seed_amp_n=0.01, seed_amp_p=0.01)",
        # "multiple_u_d/no_inhomogeneity(lambda=0.0, sigma=-1.0, seed_amp_n=0.03, seed_amp_p=0.03)",
        # "multiple_u_d/no_inhomogeneity(lambda=0.0, sigma=-1.0, seed_amp_n=0.05, seed_amp_p=0.05)",
        # "multiple_u_d/no_inhomogeneity(lambda=0.0, sigma=-1.0, seed_amp_n=0.07, seed_amp_p=0.07)",

        # "multiple_u_d/modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.001, seed_amp_p=0.001)",# - 10t_final
        # "multiple_u_d/modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.001, seed_amp_p=0.001)-Nx=512; Nt=100"

        "multiple_u_d/2.5L(lambda=0.0, sigma=-1.0, seed_amp_n=0.06, seed_amp_p=0.06)",
        # "multiple_u_d/medium_dissipation_perturbation",
        # "multiple_u_d/small_dissipation_perturbation",

        # "multiple_u_d/dissipation_perturbation(lambda=1.0, sigma=2.0, seed_amp_n=0.0, seed_amp_p=0.0)",
        # "multiple_u_d/medium_dissipation_perturbation(lambda=0.5, sigma=1.0, seed_amp_n=0.0, seed_amp_p=0.0)",
        # "multiple_u_d/small_dissipation_perturbation(lambda=0.25, sigma=0.5, seed_amp_n=0.0, seed_amp_p=0.0)",
        # "multiple_u_d/dissipation_perturbation(lambda=2.0, sigma=1.0, seed_amp_n=0.0, seed_amp_p=0.0)",
        # "multiple_u_d/dissipation_perturbation(lambda=3.0, sigma=1.0, seed_amp_n=0.0, seed_amp_p=0.0)",
        # "multiple_u_d/dissipation_perturbation(lambda=4.0, sigma=1.0, seed_amp_n=0.0, seed_amp_p=0.0)",
        # "multiple_u_d/dissipation_perturbation(lambda=1.0, sigma=2.0, seed_amp_n=0.0, seed_amp_p=0.0)",
        # "multiple_u_d/dissipation_perturbation(lambda=1.0, sigma=4.0, seed_amp_n=0.0, seed_amp_p=0.0)",   
        # "multiple_u_d/dissipation_perturbation(lambda=1.0, sigma=-1.0, seed_amp_n=0.0, seed_amp_p=0.0)",
        # "multiple_u_d/dissipation_perturbation(lambda=1.0, sigma=-2.0, seed_amp_n=0.0, seed_amp_p=0.0)",
    ]
    
    # custom_labels=base_dirs

    custom_labels = [
        "$\\delta n, \\delta p = 0.001$",
        # "$\\sin(x)/x$"#\\delta n, \\delta p = 0.01$",
        # "$\\delta n, \\delta p = 0.03$",
        # "$\\delta n, \\delta p = 0.05$",
        # "$\\delta n, \\delta p = 0.07$",

        # "$\\lambda=2.0, \\sigma=1.0$",
        # "$\\lambda=3.0, \\sigma=1.0$",
        # "$\\lambda=4.0, \\sigma=1.0$",
        # "$\\lambda=1.0, \\sigma=2.0$",
        # "$\\lambda=1.0, \\sigma=4.0$",
        # "$\\lambda=1.0, \\sigma=-1.0$",
        # "$\\lambda=1.0, \\sigma=-2.0$",
    ]
    # custom_labels=base_dirs
    # custom_labels = [
    #     "$\\lambda=0, \\sigma=0.0, (\\delta n, \\delta p = 0.03)$",
    #     "$\\lambda=1.0, \\sigma=2.0, (\\delta n, \\delta p = 0.03)$",
    #     "$\\lambda=0.5, \\sigma=1.0, (\\delta n, \\delta p = 0.03)$",
    #     "$\\lambda=0.25, \\sigma=0.5, (\\delta n, \\delta p = 0.03)$",
    #     "$\\lambda=1.0, \\sigma=2.0, (\\delta n, \\delta p = 0.0)$",
    #     "$\\lambda=0.5, \\sigma=1.0, (\\delta n, \\delta p = 0.0)$",
    #     "$\\lambda=0.25, \\sigma=0.5, (\\delta n, \\delta p = 0.0)$",
    #     "$\\lambda=2.0, \\sigma=1.0, (\\delta n, \\delta p = 0.0)$",
    #     "$\\lambda=3.0, \\sigma=1.0, (\\delta n, \\delta p = 0.0)$",
    #     "$\\lambda=4.0, \\sigma=1.0, (\\delta n, \\delta p = 0.0)$",
    #     "$\\lambda=1.0, \\sigma=2.0, (\\delta n, \\delta p = 0.0)$",
    #     "$\\lambda=1.0, \\sigma=4.0, (\\delta n, \\delta p = 0.0)$",
    #     "$\\lambda=1.0, \\sigma=-1.0, (\\delta n, \\delta p = 0.0)$",
    #     "$\\lambda=1.0, \\sigma=-2.0, (\\delta n, \\delta p = 0.0)$",
    # ]
    
    
    print("=" * 60)
    print("VELOCITY ANALYSIS: Multiple Parameter Sets")
    print("=" * 60)
    
    # Plot velocity analysis (u_true, n_pulses, frequency)
    print("Generating velocity analysis (u_true, n_pulses, frequency)...")
    # velocity_data = plot_combined_velocity_analysis(base_dirs, labels=custom_labels)
    
    # Plot delta n vs u_d
    print("\nGenerating delta n vs u_d plot...")
    delta_n_data = plot_delta_n_vs_ud(base_dirs, labels=custom_labels)
    
    # Plot n(t) and p(t) time series comparison
    print("\nGenerating n(t) and p(t) time series comparison...")
    plot_n_p_time_series(base_dirs, labels=custom_labels)
    
    # Panel plots of final n(x) and p(x) with columns per base_dir
    print("\nGenerating multi-column panel of final n(x) across u_d and datasets...")
    # plot_multiple_ud_panel(base_dirs=base_dirs, labels=custom_labels)
    print("Generating multi-column panel of final p(x) across u_d and datasets...")
    # plot_multiple_ud_panel_p(base_dirs=base_dirs, labels=custom_labels)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Generated:")
    print("- velocity_vs_ud_combined.png")
    print("- delta_n_vs_ud.png")
    print("- n_p_time_series_comparison.png (n(t) and p(t) at x₀)")
    print("- final_profiles_panel_grid_n.png (n panel, columns per dataset)")
    print("- final_profiles_panel_grid_p.png (p panel, columns per dataset)")
    print("=" * 60)

