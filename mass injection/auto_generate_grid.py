#!/usr/bin/env python3
"""
Auto-generate parameter grid plot from existing simulation data
Automatically detects available parameter combinations and creates the grid
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import re

def auto_detect_parameter_combinations():
    """
    Automatically detect all available parameter combinations from directory names
    """
    # Look for directories matching the lambda parameter pattern
    pattern = r"out_drift_lambda([0-9.]+)_ud([0-9.]+)"
    
    directories = glob.glob("out_drift_lambda*_ud*")
    combinations = []
    
    for dir_path in directories:
        dir_name = os.path.basename(dir_path)
        match = re.match(pattern, dir_name)
        
        if match:
            lambda0 = float(match.group(1))
            u_d = float(match.group(2))
            
            # Check if this directory has spacetime data
            spacetime_files = glob.glob(os.path.join(dir_path, "snapshots_n_*.png"))
            if spacetime_files:
                combinations.append((lambda0, u_d, spacetime_files[0]))
                print(f"✓ Found: λ₀={lambda0:g}, u_d={u_d:g}")
            else:
                print(f"✗ Missing spacetime data: λ₀={lambda0:g}, u_d={u_d:g}")
    
    return combinations

def filter_combinations_by_lambda_range(combinations, lambda_min=None, lambda_max=None):
    """
    Filter combinations to only include specified lambda range
    """
    if lambda_min is None and lambda_max is None:
        return combinations
    
    filtered = []
    for lambda0, u_d, img_path in combinations:
        include = True
        if lambda_min is not None and lambda0 < lambda_min:
            include = False
        if lambda_max is not None and lambda0 > lambda_max:
            include = False
        
        if include:
            filtered.append((lambda0, u_d, img_path))
    
    return filtered

def create_parameter_list(combinations):
    """
    Create formatted parameter lists for easy copying
    """
    lambda_values = sorted(list(set([c[0] for c in combinations])))
    ud_values = sorted(list(set([c[1] for c in combinations])))
    
    print("\n" + "="*50)
    print("DETECTED PARAMETER RANGES")
    print("="*50)
    print(f"Lambda0 values: {lambda_values}")
    print(f"U_d values: {ud_values}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Grid size: {len(lambda_values)} rows × {len(ud_values)} columns")
    
    print("\n" + "="*50)
    print("PYTHON CODE FOR PARAMETER ARRAYS")
    print("="*50)
    print(f"lambda0_values = {lambda_values}")
    print(f"drift_velocities = {ud_values}")
    
    return lambda_values, ud_values

def generate_grid_plot(combinations, output_name="auto_generated_grid"):
    """
    Generate the grid plot from combinations
    """
    if not combinations:
        print("No valid combinations found!")
        return
    
    # Organize data
    lambda_values = sorted(list(set([c[0] for c in combinations])))
    ud_values = sorted(list(set([c[1] for c in combinations])))
    
    n_rows = len(lambda_values)
    n_cols = len(ud_values)
    
    # Create lookup table
    data_lookup = {(c[0], c[1]): c[2] for c in combinations}
    
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(4.35 * n_cols, 2.0 * n_rows),
        gridspec_kw={'hspace': 0.0, 'wspace': 0.01}
    )
    
    # Handle single subplot cases
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for row_idx, lambda0 in enumerate(lambda_values):
        for col_idx, u_d in enumerate(ud_values):
            ax = axes[row_idx][col_idx]
            
            if (lambda0, u_d) in data_lookup:
                # Load and display the spacetime image
                img_path = data_lookup[(lambda0, u_d)]
                try:
                    from matplotlib import image as mpimg
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    
                    # Remove title to save space
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    ax.text(0.5, 0.5, f"Error\n$\\lambda_0={lambda0:g}$\n$u_d={u_d:g}$", 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                # Missing combination
                ax.text(0.5, 0.5, f"Missing\n$\\lambda_0={lambda0:g}$\n$u_d={u_d:g}$", 
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add axis labels (smaller font, tighter spacing)
            if col_idx == 0:
                ax.set_ylabel(f"$\\lambda_0={lambda0:g}$", fontsize=10, rotation=0, 
                             ha='right', va='center', labelpad=5)
            if row_idx == n_rows - 1:
                ax.set_xlabel(f"$u_d={u_d:g}$", fontsize=10, labelpad=2)
    
    # Remove suptitle to save space
    
    # Save the plot
    os.makedirs("out_drift", exist_ok=True)
    png_path = f"out_drift/{output_name}.png"
    pdf_path = f"out_drift/{output_name}.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"\n✓ Grid plot saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

def main():
    print("="*60)
    print("AUTO-GENERATING PARAMETER GRID FROM AVAILABLE DATA")
    print("="*60)
    
    # Auto-detect all available combinations
    all_combinations = auto_detect_parameter_combinations()
    
    if not all_combinations:
        print("No parameter combinations found!")
        print("Make sure you have run simulations with the lambda parameter sweep.")
        return
    
    # Show all available combinations
    print(f"\nFound {len(all_combinations)} total parameter combinations")
    
    # Filter to lambda < 0.3 as currently specified in the code
    filtered_combinations = filter_combinations_by_lambda_range(
        all_combinations, lambda_max=0.3
    )
    
    print(f"Filtered to λ₀ < 0.3: {len(filtered_combinations)} combinations")
    
    if filtered_combinations:
        # Create parameter lists for reference
        lambda_values, ud_values = create_parameter_list(filtered_combinations)
        
        # Generate the grid plot
        generate_grid_plot(filtered_combinations, "auto_spacetime_grid_lambda_lt_0p3")
    
    # Also create a plot with all available data
    if len(all_combinations) > len(filtered_combinations):
        print(f"\n" + "="*50)
        print("CREATING ADDITIONAL PLOT WITH ALL AVAILABLE DATA")
        print("="*50)
        create_parameter_list(all_combinations)
        generate_grid_plot(all_combinations, "auto_spacetime_grid_all_data")
    
    print("\n" + "="*60)
    print("AUTO-GENERATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
