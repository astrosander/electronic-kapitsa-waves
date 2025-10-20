#!/usr/bin/env python3
"""
Script to analyze and plot <j>_t vs u_d for all simulations in multiple_u_d/diffusion_sweep.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import re
import zlib
import matplotlib as mpl

# --- Publication-ready appearance for large dataset ---
mpl.rcParams.update({
    "text.usetex": False,          # use MathText (portable)
    "font.family": "serif",        # serif font for publication
    "font.size": 11,               # slightly smaller for large dataset
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",    # STIX math fonts
    "axes.unicode_minus": False,   # proper minus sign
    "axes.linewidth": 0.8,         # thinner axes
    "xtick.major.size": 3,         # smaller ticks for large dataset
    "xtick.minor.size": 1.5,
    "ytick.major.size": 3,
    "ytick.minor.size": 1.5,
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
    "savefig.pad_inches": 0.15,    # more padding for large dataset
    "axes.spines.top": True,       # show all spines
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
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
    """Extract w, u_d, Dn, Dp parameters from the file path."""
    # Extract w from directory name like "w=0.05_Dn=0p50_Dp=0p05_L10..."
    w_match = re.search(r'w=(\d+\.\d+)', file_path)
    w = float(w_match.group(1)) if w_match else None
    
    # Extract Dn and Dp from directory name
    dn_match = re.search(r'Dn=(\d+p\d+)', file_path)
    if dn_match:
        dn_str = dn_match.group(1).replace('p', '.')
        Dn = float(dn_str)
    else:
        Dn = None
    
    dp_match = re.search(r'Dp=(\d+p\d+)', file_path)
    if dp_match:
        dp_str = dp_match.group(1).replace('p', '.')
        Dp = float(dp_str)
    else:
        Dp = None
    
    # Extract u_d from directory name like "out_drift_ud1p2"
    ud_match = re.search(r'out_drift_ud(\d+p\d+)', file_path)
    if ud_match:
        ud_str = ud_match.group(1).replace('p', '.')
        u_d = float(ud_str)
    else:
        u_d = None
    
    return w, u_d, Dn, Dp

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

def calculate_delta_n(n_t, threshold=0.01):
    """Calculate delta n = n_max - n_min for the final density profile."""
    n_final = n_t[:, -1]  # Last time step
    
    # Calculate n_max and n_min
    n_max = np.max(n_final)
    n_min = np.min(n_final)
    delta_n = n_max - n_min
    
    # Return delta_n and whether it's above threshold
    above_threshold = delta_n > threshold
    
    return delta_n, above_threshold

def find_all_simulation_files():
    """Find all simulation data files in the multiple_u_d/diffusion_sweep directory."""
    base_dir = "multiple_u_d/diffusion_sweep"
    pattern = os.path.join(base_dir, "**", "data_m07_*.npz")
    files = glob.glob(pattern, recursive=True)
    return files

def main():
    """Main analysis function."""
    print("=" * 80)
    print("ANALYZING <j>_t vs u_d FOR DIFFUSION SWEEP SIMULATIONS")
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
            w, u_d, Dn, Dp = extract_parameters_from_path(data_file)
            if w is None or u_d is None or Dn is None or Dp is None:
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
            
            # Calculate delta n and check threshold
            delta_n, above_threshold = calculate_delta_n(n_t, threshold=0.01)
            
            results.append({
                'w': w,
                'u_d': u_d,
                'Dn': Dn,
                'Dp': Dp,
                'j_avg': j_time_avg,
                'p_at_x0': j_spatial_avg,
                'delta_n': delta_n,
                'above_threshold': above_threshold,
                't': t,
                'file': data_file
            })
            
            print(f"  w={w:.2f}, u_d={u_d:.1f}, Dn={Dn:.2f}, Dp={Dp:.2f}, <j>_t={j_time_avg:.6f}")
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
    Dn_values = np.array([r['Dn'] for r in results])
    Dp_values = np.array([r['Dp'] for r in results])
    j_values = np.array([r['j_avg'] for r in results])
    
    # Get unique values and sort them
    unique_w = np.unique(w_values)
    unique_w = np.sort(unique_w)
    unique_Dn = np.unique(Dn_values)
    unique_Dp = np.unique(Dp_values)
    
    print(f"\nUnique w values: {unique_w}")
    print(f"Unique u_d values: {np.unique(u_d_values)}")
    print(f"Unique Dn values: {unique_Dn}")
    print(f"Unique Dp values: {unique_Dp}")
    
    # Create publication-ready plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create diffusion combination labels
    diffusion_labels = []
    for Dn, Dp in zip(Dn_values, Dp_values):
        if Dn == 0.25 and Dp == 0.10:
            diffusion_labels.append("Dn_half")
        elif Dn == 0.50 and Dp == 0.05:
            diffusion_labels.append("Dp_half")
        elif Dn == 0.25 and Dp == 0.05:
            diffusion_labels.append("both_half")
        elif Dn == 1.00 and Dp == 0.10:
            diffusion_labels.append("Dn_double")
        elif Dn == 0.50 and Dp == 0.20:
            diffusion_labels.append("Dp_double")
        elif Dn == 1.00 and Dp == 0.20:
            diffusion_labels.append("both_double")
        else:
            diffusion_labels.append(f"Dn={Dn:.2f},Dp={Dp:.2f}")
    
    # Create color map for diffusion combinations using highly distinct colors
    unique_diffusions = list(set(diffusion_labels))
    # Use manually selected distinct colors for the 6 diffusion combinations
    distinct_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b'   # Brown
    ]
    diffusion_color_map = {diff: distinct_colors[i] for i, diff in enumerate(unique_diffusions)}
    
    # Plot: <j>_t vs u_d colored by diffusion combination
    for i, (w, u_d, Dn, Dp, j_avg, diff_label) in enumerate(zip(w_values, u_d_values, Dn_values, Dp_values, j_values, diffusion_labels)):
        color = diffusion_color_map[diff_label]
        ax.scatter(u_d, j_avg, c=[color], s=30, alpha=0.8, edgecolors='black', linewidth=0.4, zorder=10)
    
    # Publication-ready labels
    ax.set_xlabel('$u_d$', fontsize=16, fontweight='bold')
    ax.set_ylabel('$\\langle j \\rangle_t$', fontsize=16, fontweight='bold')
    ax.set_title('Time-averaged current vs drift velocity\n(Diffusion parameter sweep)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Professional grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits with some padding
    ax.set_xlim(u_d_values.min() - 0.05*(u_d_values.max() - u_d_values.min()),
                u_d_values.max() + 0.05*(u_d_values.max() - u_d_values.min()))
    ax.set_ylim(j_values.min() - 0.05*(j_values.max() - j_values.min()),
                j_values.max() + 0.05*(j_values.max() - j_values.min()))
    
    # Add least squares fit lines for each diffusion combination
    from scipy import stats
    
    # Extract delta_n and threshold information
    delta_n_values = np.array([r['delta_n'] for r in results])
    above_threshold = np.array([r['above_threshold'] for r in results])
    
    print(f"\nDelta n analysis:")
    print(f"  Threshold: 0.01")
    print(f"  Points above threshold: {np.sum(above_threshold)}/{len(above_threshold)} ({np.sum(above_threshold)/len(above_threshold)*100:.1f}%)")
    print(f"  Delta n range: {delta_n_values.min():.4f} - {delta_n_values.max():.4f}")
    
    # Store slopes for each diffusion combination and w value
    slopes_by_diffusion_w = {}
    
    # First pass: calculate slopes for each (diffusion, w) combination
    for diff_label in unique_diffusions:
        slopes_by_diffusion_w[diff_label] = {}
        
        # Get data for this diffusion combination
        mask_diff = np.array(diffusion_labels) == diff_label
        w_diff = w_values[mask_diff]
        u_d_diff = u_d_values[mask_diff]
        j_diff = j_values[mask_diff]
        delta_n_diff = delta_n_values[mask_diff]
        above_thresh_diff = above_threshold[mask_diff]
        
        # Get unique w values for this diffusion combination
        unique_w_diff = np.unique(w_diff)
        
        for w in unique_w_diff:
            # Get data for this w value within this diffusion combination
            mask_w = (w_diff == w) & above_thresh_diff
            u_d_w = u_d_diff[mask_w]
            j_w = j_diff[mask_w]
            
            if len(u_d_w) > 1:  # Need at least 2 points above threshold
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(u_d_w, j_w)
                slopes_by_diffusion_w[diff_label][w] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'u_d_range': (u_d_w.min(), u_d_w.max()),
                    'n_points': len(u_d_w)
                }
                print(f"  {diff_label}, w={w:.2f}: {len(u_d_w)} points above threshold, slope={slope:.4f}, R²={r_value**2:.3f}")
            else:
                print(f"  {diff_label}, w={w:.2f}: {len(u_d_w)} points above threshold - insufficient for fitting")
    
    # Second pass: find min and max slopes for each diffusion combination and plot them
    for diff_label in unique_diffusions:
        if diff_label not in slopes_by_diffusion_w or not slopes_by_diffusion_w[diff_label]:
            continue
            
        slopes_data = slopes_by_diffusion_w[diff_label]
        slopes = [data['slope'] for data in slopes_data.values()]
        w_values_list = list(slopes_data.keys())
        
        if len(slopes) == 0:
            continue
            
        # Find min and max slopes
        min_slope_idx = np.argmin(slopes)
        max_slope_idx = np.argmax(slopes)
        
        min_w = w_values_list[min_slope_idx]
        max_w = w_values_list[max_slope_idx]
        min_slope_data = slopes_data[min_w]
        max_slope_data = slopes_data[max_w]
        
        color = diffusion_color_map[diff_label]
        
        # Plot min slope with bold line
        u_d_fit_min = np.linspace(min_slope_data['u_d_range'][0], min_slope_data['u_d_range'][1], 100)
        j_fit_min = min_slope_data['slope'] * u_d_fit_min + min_slope_data['intercept']
        ax.plot(u_d_fit_min, j_fit_min, color=color, linewidth=4, alpha=0.9,
               linestyle='-', zorder=6, 
               label=f'{diff_label} min: w={min_w:.2f}, slope={min_slope_data["slope"]:.3f}')
        
        # Plot max slope with bold line
        u_d_fit_max = np.linspace(max_slope_data['u_d_range'][0], max_slope_data['u_d_range'][1], 100)
        j_fit_max = max_slope_data['slope'] * u_d_fit_max + max_slope_data['intercept']
        ax.plot(u_d_fit_max, j_fit_max, color=color, linewidth=4, alpha=0.9,
               linestyle='--', zorder=6, 
               label=f'{diff_label} max: w={max_w:.2f}, slope={max_slope_data["slope"]:.3f}')
        
        print(f"  {diff_label}: min slope={min_slope_data['slope']:.4f} at w={min_w:.2f}, max slope={max_slope_data['slope']:.4f} at w={max_w:.2f}")
    
    # Add reference line: 0.2 * u_d
    u_d_fit_all = np.linspace(u_d_values.min(), u_d_values.max(), 100)
    j_avg_fit_all = 0.2 * u_d_fit_all
    ax.plot(u_d_fit_all, j_avg_fit_all, 'k--', linewidth=2, alpha=0.8,
            label='$\\langle j \\rangle = 0.2u_d$')
    
    # Legend - only show reference line and min/max slopes for each diffusion combination
    from matplotlib.lines import Line2D
    legend_handles = []
    
    # Add reference line
    ref_line = Line2D([0], [0], color='k', linestyle='--', linewidth=2, 
                     label='$\\langle j \\rangle = 0.2u_d$')
    legend_handles.append(ref_line)
    
    # Add min/max slope lines for each diffusion combination
    for diff_label in unique_diffusions:
        if diff_label not in slopes_by_diffusion_w or not slopes_by_diffusion_w[diff_label]:
            continue
            
        slopes_data = slopes_by_diffusion_w[diff_label]
        slopes = [data['slope'] for data in slopes_data.values()]
        w_values_list = list(slopes_data.keys())
        
        if len(slopes) == 0:
            continue
            
        # Find min and max slopes
        min_slope_idx = np.argmin(slopes)
        max_slope_idx = np.argmax(slopes)
        
        min_w = w_values_list[min_slope_idx]
        max_w = w_values_list[max_slope_idx]
        min_slope_data = slopes_data[min_w]
        max_slope_data = slopes_data[max_w]
        
        color = diffusion_color_map[diff_label]
        
        # Add min slope line to legend
        min_line = Line2D([0], [0], color=color, linewidth=4, alpha=0.9, linestyle='-',
                         label=f'{diff_label} min: w={min_w:.2f}, slope={min_slope_data["slope"]:.3f}')
        legend_handles.append(min_line)
        
        # Add max slope line to legend
        max_line = Line2D([0], [0], color=color, linewidth=4, alpha=0.9, linestyle='--',
                         label=f'{diff_label} max: w={max_w:.2f}, slope={max_slope_data["slope"]:.3f}')
        legend_handles.append(max_line)
    
    ax.legend(handles=legend_handles, loc='upper left', frameon=True, 
             fancybox=False, shadow=False, fontsize=9, framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    
    # Save publication-ready plots
    output_dir = "multiple_u_d/diffusion_sweep"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in multiple formats for publication
    base_name = "j_vs_ud_diffusion_sweep"
    plt.savefig(f"{output_dir}/{base_name}.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(f"{output_dir}/{base_name}.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"\nPublication-ready plots saved to:")
    print(f"  {output_dir}/{base_name}.png (300 DPI)")
    print(f"  {output_dir}/{base_name}.pdf (vector)")
    
    plt.show()
    
    # Print publication-quality summary statistics
    print(f"\n" + "="*80)
    print("DIFFUSION SWEEP ANALYSIS SUMMARY")
    print("="*80)
    print(f"Dataset: {len(results)} simulations across {len(unique_diffusions)} diffusion combinations")
    print(f"Parameter ranges:")
    print(f"  w ∈ [{w_values.min():.2f}, {w_values.max():.2f}]")
    print(f"  u_d ∈ [{u_d_values.min():.1f}, {u_d_values.max():.1f}]")
    print(f"  Dn ∈ [{Dn_values.min():.2f}, {Dn_values.max():.2f}]")
    print(f"  Dp ∈ [{Dp_values.min():.2f}, {Dp_values.max():.2f}]")
    print(f"  ⟨j⟩_t ∈ [{j_values.min():.4f}, {j_values.max():.4f}]")
    print(f"Statistical measures:")
    print(f"  ⟨j⟩_t mean ± std: {j_values.mean():.4f} ± {j_values.std():.4f}")
    print(f"  ⟨j⟩_t median: {np.median(j_values):.4f}")
    print(f"  ⟨j⟩_t CV: {j_values.std()/j_values.mean()*100:.1f}%")
    
    # Print results by diffusion combination
    print(f"\nResults by diffusion combination (for publication, Δn > 0.01):")
    print(f"{'Diffusion':>12} {'N':>3} {'N_fit':>5} {'⟨j⟩_t':>8} {'±σ':>6} {'u_d range':>12} {'min slope':>10} {'max slope':>10}")
    print("-" * 90)
    for diff_label in unique_diffusions:
        mask = np.array(diffusion_labels) == diff_label
        j_diff = j_values[mask]
        u_d_diff = u_d_values[mask]
        delta_n_diff = delta_n_values[mask]
        above_thresh_diff = above_threshold[mask]
        
        # Get min and max slopes for this diffusion combination
        if diff_label in slopes_by_diffusion_w and slopes_by_diffusion_w[diff_label]:
            slopes_data = slopes_by_diffusion_w[diff_label]
            slopes = [data['slope'] for data in slopes_data.values()]
            min_slope = min(slopes) if slopes else 0.0
            max_slope = max(slopes) if slopes else 0.0
        else:
            min_slope, max_slope = 0.0, 0.0
            
        print(f"{diff_label:>12} {len(j_diff):3d} {np.sum(above_thresh_diff):5d} {j_diff.mean():8.4f} {j_diff.std():6.4f} "
              f"{u_d_diff.min():4.1f}-{u_d_diff.max():4.1f} {min_slope:10.4f} {max_slope:10.4f}")
    
    # Print detailed min/max slope analysis
    print(f"\nMin/Max slope analysis by diffusion combination:")
    print(f"{'Diffusion':>12} {'w_min':>6} {'slope_min':>10} {'w_max':>6} {'slope_max':>10} {'range':>8}")
    print("-" * 70)
    for diff_label in unique_diffusions:
        if diff_label in slopes_by_diffusion_w and slopes_by_diffusion_w[diff_label]:
            slopes_data = slopes_by_diffusion_w[diff_label]
            slopes = [data['slope'] for data in slopes_data.values()]
            w_values_list = list(slopes_data.keys())
            
            if len(slopes) > 0:
                min_slope_idx = np.argmin(slopes)
                max_slope_idx = np.argmax(slopes)
                
                min_w = w_values_list[min_slope_idx]
                max_w = w_values_list[max_slope_idx]
                min_slope = slopes[min_slope_idx]
                max_slope = slopes[max_slope_idx]
                slope_range = max_slope - min_slope
                
                print(f"{diff_label:>12} {min_w:6.2f} {min_slope:10.4f} {max_w:6.2f} {max_slope:10.4f} {slope_range:8.4f}")
            else:
                print(f"{diff_label:>12} {'N/A':>6} {'N/A':>10} {'N/A':>6} {'N/A':>10} {'N/A':>8}")
        else:
            print(f"{diff_label:>12} {'N/A':>6} {'N/A':>10} {'N/A':>6} {'N/A':>10} {'N/A':>8}")
    
    print(f"\nReference line: ⟨j⟩ = 0.2u_d")
    print(f"Figure saved in publication-ready formats (PNG, PDF)")
    print("="*80)

if __name__ == "__main__":
    main()
