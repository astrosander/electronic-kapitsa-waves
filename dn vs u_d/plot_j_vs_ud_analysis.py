#!/usr/bin/env python3
"""
Script to analyze and plot <j>_t vs u_d for all simulations in multiple_u_d/multiple_w.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import re
import zlib
import matplotlib as mpl

# --- Publication-ready appearance ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "serif",        # serif font for publication
    "font.size": 12,               # standard publication size
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",    # STIX math fonts
    "axes.unicode_minus": False,   # proper minus sign
    "axes.linewidth": 0.8,         # thinner axes
    "xtick.major.size": 4,         # tick size
    "xtick.minor.size": 2,
    "ytick.major.size": 4,
    "ytick.minor.size": 2,
    "xtick.direction": "in",       # ticks inside
    "ytick.direction": "in",
    "xtick.top": True,             # ticks on all sides
    "ytick.right": True,
    "legend.frameon": True,        # legend frame
    "legend.fancybox": False,      # rectangular legend
    "legend.shadow": False,
    "figure.dpi": 300,             # high resolution
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

def load_simulation_data(data_file):
    """Load simulation data from .npz file."""
    try:
        data = np.load(data_file, allow_pickle=True)
        return data
    except (OSError, zlib.error, ValueError, KeyError) as e:
        print(f"Error loading {data_file}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading {data_file}: {e}")
        return None

def extract_parameters_from_path(file_path):
    """Extract w and u_d parameters from the file path."""
    # Extract w from directory name like "w=0.05_modes_3_5_7_L10..."
    w_match = re.search(r'w=(\d+\.\d+)', file_path)
    w = float(w_match.group(1)) if w_match else None
    
    # Extract u_d from directory name like "out_drift_ud1p2"
    ud_match = re.search(r'out_drift_ud(\d+p\d+)', file_path)
    if ud_match:
        ud_str = ud_match.group(1).replace('p', '.')
        u_d = float(ud_str)
    else:
        u_d = None
    
    return w, u_d

def calculate_time_averaged_current(n_t, p_t, t, L, Nx, x0_fraction=0.5):
    """Calculate time-averaged current <j>_t using the same method as plot_from_data.py."""
    # In plot_from_data.py, j = p (momentum), not j = n * p
    # Determine spatial index for measurement
    x0_idx = int(x0_fraction * Nx)
    
    # Extract momentum time series at x0
    p_at_x0 = p_t[x0_idx, :]
    
    # Calculate time-averaged current using Gaussian weighting
    # Gaussian centered at 85% of final time with width 10% of final time
    t_final = t[-1]
    t_center = 0.85 * t_final
    t_width = 0.1 * t_final
    
    # Create Gaussian weights (NOT normalized)
    gaussian_weights = np.exp(-0.5 * ((t - t_center) / t_width)**2)
    
    # Calculate weighted average: sum(w * x) / sum(w)
    j_avg = np.sum(p_at_x0 * gaussian_weights) / np.sum(gaussian_weights)
    
    return j_avg, p_at_x0

def find_all_simulation_files():
    """Find all simulation data files in the multiple_u_d/multiple_w directory."""
    base_dir = "multiple_u_d/multiple_w"
    pattern = os.path.join(base_dir, "**", "data_m07_*.npz")
    files = glob.glob(pattern, recursive=True)
    return files

def main():
    """Main analysis function."""
    print("=" * 80)
    print("ANALYZING <j>_t vs u_d FOR ALL SIMULATIONS")
    print("=" * 80)
    
    # Find all simulation files
    data_files = find_all_simulation_files()
    print(f"Found {len(data_files)} simulation files")
    
    if not data_files:
        print("No simulation files found!")
        return
    
    # Store results
    results = []
    
    # Process each file
    successful_files = 0
    failed_files = 0
    
    for i, data_file in enumerate(data_files):
        print(f"Processing {i+1}/{len(data_files)}: {os.path.basename(data_file)}")
        
        try:
            # Extract parameters
            w, u_d = extract_parameters_from_path(data_file)
            if w is None or u_d is None:
                print(f"  Could not extract parameters from {data_file}")
                failed_files += 1
                continue
            
            # Load data
            data = load_simulation_data(data_file)
            if data is None:
                failed_files += 1
                continue
            
            # Extract fields
            n_t = data['n_t']
            p_t = data['p_t']
            t = data['t']
            L = data['L']
            Nx = data['Nx']
            
            # Calculate time-averaged current
            j_time_avg, j_spatial_avg = calculate_time_averaged_current(n_t, p_t, t, L, Nx)
            
            results.append({
                'w': w,
                'u_d': u_d,
                'j_avg': j_time_avg,  # Rename to match plot_from_data.py convention
                'p_at_x0': j_spatial_avg,  # Rename to match plot_from_data.py convention
                't': t,
                'file': data_file
            })
            
            print(f"  w={w:.2f}, u_d={u_d:.1f}, <j>_t={j_time_avg:.6f}")
            successful_files += 1
            
        except Exception as e:
            print(f"  Error processing {data_file}: {e}")
            failed_files += 1
            continue
    
    print(f"\nSuccessfully processed {len(results)} simulations")
    print(f"Failed files: {failed_files}")
    print(f"Successful files: {successful_files}")
    
    if not results:
        print("No valid results to plot!")
        return
    
    # Convert to arrays for plotting
    w_values = np.array([r['w'] for r in results])
    u_d_values = np.array([r['u_d'] for r in results])
    j_values = np.array([r['j_avg'] for r in results])
    
    # Get unique w values and sort them
    unique_w = np.unique(w_values)
    unique_w = np.sort(unique_w)
    
    print(f"\nUnique w values: {unique_w}")
    print(f"Unique u_d values: {np.unique(u_d_values)}")
    
    # Create publication-ready plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot: <j>_t vs u_d (all data points) colored by w
    import matplotlib.colors as mcolors
    # Use tab20 colormap for distinct colors
    scatter = ax.scatter(u_d_values, j_values, c=w_values, cmap='tab20',
                        s=40, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Publication-ready labels
    ax.set_xlabel('$u_d$', fontsize=14, fontweight='bold')
    ax.set_ylabel('$\\langle j \\rangle_t$', fontsize=14, fontweight='bold')
    
    # Professional grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits with some padding
    ax.set_xlim(u_d_values.min() - 0.05*(u_d_values.max() - u_d_values.min()),
                u_d_values.max() + 0.05*(u_d_values.max() - u_d_values.min()))
    ax.set_ylim(j_values.min() - 0.05*(j_values.max() - j_values.min()),
                j_values.max() + 0.05*(j_values.max() - j_values.min()))
    
    # Add reference line: 0.2 * u_d
    u_d_fit_all = np.linspace(u_d_values.min(), u_d_values.max(), 100)
    j_avg_fit_all = 0.2 * u_d_fit_all
    ax.plot(u_d_fit_all, j_avg_fit_all, 'k--', linewidth=2, alpha=0.8,
            label='$\\langle j \\rangle = 0.2u_d$')
    
    # Publication-ready colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('$w$', fontsize=12, fontweight='bold')
    
    # Format colorbar ticks professionally
    cbar.ax.tick_params(labelsize=10, direction='in')
    # Set tick labels to show only 2 decimal places
    tick_labels = [f'{w:.2f}' for w in unique_w]
    cbar.set_ticks(unique_w)
    cbar.set_ticklabels(tick_labels)
    
    # Publication-ready legend
    ax.legend(loc='upper left', frameon=True, fancybox=False, 
             shadow=False, fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save publication-ready plots
    output_dir = "multiple_u_d/multiple_w"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in multiple formats for publication
    base_name = "j_vs_ud_publication"
    plt.savefig(f"{output_dir}/{base_name}.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(f"{output_dir}/{base_name}.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f"{output_dir}/{base_name}.eps", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"\nPublication-ready plots saved to:")
    print(f"  {output_dir}/{base_name}.png (300 DPI)")
    print(f"  {output_dir}/{base_name}.pdf (vector)")
    print(f"  {output_dir}/{base_name}.eps (vector)")
    
    plt.show()
    
    # Print publication-quality summary statistics
    print(f"\n" + "="*80)
    print("PUBLICATION-READY ANALYSIS SUMMARY")
    print("="*80)
    print(f"Dataset: {len(results)} simulations across {len(unique_w)} w values")
    print(f"Parameter ranges:")
    print(f"  w ∈ [{w_values.min():.2f}, {w_values.max():.2f}]")
    print(f"  u_d ∈ [{u_d_values.min():.1f}, {u_d_values.max():.1f}]")
    print(f"  ⟨j⟩_t ∈ [{j_values.min():.4f}, {j_values.max():.4f}]")
    print(f"Statistical measures:")
    print(f"  ⟨j⟩_t mean ± std: {j_values.mean():.4f} ± {j_values.std():.4f}")
    print(f"  ⟨j⟩_t median: {np.median(j_values):.4f}")
    print(f"  ⟨j⟩_t CV: {j_values.std()/j_values.mean()*100:.1f}%")
    
    # Print results by w value for publication
    print(f"\nResults by w value (for publication):")
    print(f"{'w':>6} {'N':>3} {'⟨j⟩_t':>8} {'±σ':>6} {'u_d range':>12}")
    print("-" * 45)
    for w in unique_w:
        mask = w_values == w
        j_w = j_values[mask]
        u_d_w = u_d_values[mask]
        print(f"{w:6.2f} {len(j_w):3d} {j_w.mean():8.4f} {j_w.std():6.4f} "
              f"{u_d_w.min():4.1f}-{u_d_w.max():4.1f}")
    
    print(f"\nReference line: ⟨j⟩ = 0.2u_d")
    print(f"Figure saved in publication-ready formats (PNG, PDF, EPS)")
    print("="*80)

if __name__ == "__main__":
    main()
