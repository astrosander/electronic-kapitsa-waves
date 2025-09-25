#!/usr/bin/env python3
"""
Script to run the lambda0 and drift velocity parameter sweep
Creates spacetime plots in a grid format for different lambda0 and u_d values
"""

# Import all functions from the main script
exec(open('mass injection.py').read())

def main():
    print("="*60)
    print("PARAMETER SWEEP: lambda0 vs drift velocity")
    print("="*60)
    
    # Set run mode to bypass the main execution
    import sys
    sys.argv = ["run_lambda_parameter_sweep.py"]  # Override sys.argv to prevent main from running
    
    # Define parameter ranges as requested:
    # lambda0 values all less than 0.3
    drift_velocities = [0.0, 1.0, 2.0, 3.0, 5.0]
    lambda0_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    print(f"Drift velocities: {drift_velocities}")
    print(f"Lambda0 values: {lambda0_values}")
    print(f"Total simulations: {len(drift_velocities) * len(lambda0_values)}")
    print(f"Grid size: {len(lambda0_values)} rows × {len(drift_velocities)} columns")
    print()
    
    # Run the parameter sweep
    run_lambda_drift_parameter_sweep(
        drift_velocities=drift_velocities,
        lambda0_values=lambda0_values,
        tag="seed_modes_1to5"
    )
    
    print("="*60)
    print("PARAMETER SWEEP COMPLETED!")
    print("="*60)
    print("Check the output in: out_drift/spacetime_parameter_grid_seed_modes_1to5.png")

if __name__ == "__main__":
    main()
