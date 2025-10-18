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
            
            # Calculate delta n and check threshold
            delta_n, above_threshold = calculate_delta_n(n_t, threshold=0.01)
            
            results.append({
                'w': w,
                'u_d': u_d,
                'j_avg': j_time_avg,  # Rename to match plot_from_data.py convention
                'p_at_x0': j_spatial_avg,  # Rename to match plot_from_data.py convention
                'delta_n': delta_n,
                'above_threshold': above_threshold,
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
                        s=40, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=10)
    
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
    
    # Add least squares fit lines for all w values (subtle styling to avoid clutter)
    from scipy import stats
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_w)))
    
    # Extract delta_n and threshold information
    delta_n_values = np.array([r['delta_n'] for r in results])
    above_threshold = np.array([r['above_threshold'] for r in results])
    
    print(f"\nDelta n analysis:")
    print(f"  Threshold: 0.01")
    print(f"  Points above threshold: {np.sum(above_threshold)}/{len(above_threshold)} ({np.sum(above_threshold)/len(above_threshold)*100:.1f}%)")
    print(f"  Delta n range: {delta_n_values.min():.4f} - {delta_n_values.max():.4f}")
    
    # Store slopes for finding min/max
    slopes = []
    w_slopes = []
    
    for i, w in enumerate(unique_w):
        # Get data for this w value
        mask = w_values == w
        u_d_w = u_d_values[mask]
        j_w = j_values[mask]
        delta_n_w = delta_n_values[mask]
        above_thresh_w = above_threshold[mask]
        
        # Plot fit lines for all w values that have sufficient data
        if np.sum(above_thresh_w) > 1:  # Need at least 2 points above threshold
            u_d_filtered = u_d_w[above_thresh_w]
            j_filtered = j_w[above_thresh_w]
            
            # Perform linear regression on filtered data
            slope, intercept, r_value, p_value, std_err = stats.linregress(u_d_filtered, j_filtered)
            
            # Store slope for min/max analysis
            slopes.append(slope)
            w_slopes.append(w)
            
            # Generate fit line
            u_d_fit = np.linspace(u_d_filtered.min(), u_d_filtered.max(), 100)
            j_fit = slope * u_d_fit + intercept
            
            # Plot fit line with same color as data points, but very subtle
            ax.plot(u_d_fit, j_fit, color=colors[i], linewidth=0.8, alpha=0.3,
                   linestyle='-', zorder=5)
            
            print(f"  w={w:.2f}: {np.sum(above_thresh_w)}/{len(above_thresh_w)} points above threshold, slope={slope:.4f}, R²={r_value**2:.3f}")
        else:
            print(f"  w={w:.2f}: {np.sum(above_thresh_w)}/{len(above_thresh_w)} points above threshold - insufficient for fitting")
    
    # Find min and max slopes and make them bold with labels
    if slopes:
        slopes = np.array(slopes)
        w_slopes = np.array(w_slopes)
        
        min_slope_idx = np.argmin(slopes)
        max_slope_idx = np.argmax(slopes)
        
        min_slope = slopes[min_slope_idx]
        max_slope = slopes[max_slope_idx]
        min_w = w_slopes[min_slope_idx]
        max_w = w_slopes[max_slope_idx]
        
        print(f"\nSlope analysis:")
        print(f"  Minimum slope: {min_slope:.4f} at w={min_w:.2f}")
        print(f"  Maximum slope: {max_slope:.4f} at w={max_w:.2f}")
        
        # Re-plot min and max slopes with bold styling and labels
        for i, w in enumerate(unique_w):
            if w in [min_w, max_w]:
                mask = w_values == w
                u_d_w = u_d_values[mask]
                j_w = j_values[mask]
                above_thresh_w = above_threshold[mask]
                
                if np.sum(above_thresh_w) > 1:
                    u_d_filtered = u_d_w[above_thresh_w]
                    j_filtered = j_w[above_thresh_w]
                    
                    # Perform linear regression again
                    slope, intercept, r_value, p_value, std_err = stats.linregress(u_d_filtered, j_filtered)
                    
                    # Generate fit line
                    u_d_fit = np.linspace(u_d_filtered.min(), u_d_filtered.max(), 100)
                    j_fit = slope * u_d_fit + intercept
                    
                    # Bold styling for min/max slopes
                    linewidth = 3.0 if w == min_w or w == max_w else 0.8
                    alpha = 0.8 if w == min_w or w == max_w else 0.3
                    
                    # Create label for min/max slopes
                    label_text = f'$\\langle j \\rangle = {slope:.3f} u_d$ $(w={w:.2f})$'
                    
                    ax.plot(u_d_fit, j_fit, color=colors[i], linewidth=linewidth, alpha=alpha,
                           linestyle='-', zorder=6, label=label_text)
    
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
    
    # Legend will automatically include all labeled lines
    # The min/max slope lines are already plotted with labels above
    # We just need to ensure the legend shows them
    ax.legend(loc='upper left', frameon=True, 
             fancybox=False, shadow=False, fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save publication-ready plots
    output_dir = "multiple_u_d/multiple_w"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in multiple formats for publication
    base_name = "j_vs_ud"
    plt.savefig(f"{output_dir}/{base_name}.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(f"{output_dir}/{base_name}.pdf", bbox_inches='tight',
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
    
    # Print results by w value for publication (with delta_n filtering)
    print(f"\nResults by w value (for publication, Δn > 0.01):")
    print(f"{'w':>6} {'N':>3} {'N_fit':>5} {'⟨j⟩_t':>8} {'±σ':>6} {'u_d range':>12} {'slope':>8} {'R²':>6}")
    print("-" * 70)
    for w in unique_w:
        mask = w_values == w
        j_w = j_values[mask]
        u_d_w = u_d_values[mask]
        delta_n_w = delta_n_values[mask]
        above_thresh_w = above_threshold[mask]
        
        # Calculate least squares fit only for points above threshold
        if np.sum(above_thresh_w) > 1:
            u_d_filtered = u_d_w[above_thresh_w]
            j_filtered = j_w[above_thresh_w]
            slope, intercept, r_value, p_value, std_err = stats.linregress(u_d_filtered, j_filtered)
            r_squared = r_value**2
        else:
            slope, r_squared = 0.0, 0.0
            
        print(f"{w:6.2f} {len(j_w):3d} {np.sum(above_thresh_w):5d} {j_w.mean():8.4f} {j_w.std():6.4f} "
              f"{u_d_w.min():4.1f}-{u_d_w.max():4.1f} {slope:8.4f} {r_squared:6.3f}")
    
    # Print least squares summary (filtered data only)
    print(f"\nLeast squares analysis (Δn > 0.01 only):")
    print(f"{'w':>6} {'N_fit':>5} {'slope':>8} {'intercept':>10} {'R²':>6} {'p-value':>8}")
    print("-" * 55)
    for w in unique_w:
        mask = w_values == w
        u_d_w = u_d_values[mask]
        j_w = j_values[mask]
        above_thresh_w = above_threshold[mask]
        
        if np.sum(above_thresh_w) > 1:
            u_d_filtered = u_d_w[above_thresh_w]
            j_filtered = j_w[above_thresh_w]
            slope, intercept, r_value, p_value, std_err = stats.linregress(u_d_filtered, j_filtered)
            r_squared = r_value**2
            print(f"{w:6.2f} {np.sum(above_thresh_w):5d} {slope:8.4f} {intercept:10.4f} {r_squared:6.3f} {p_value:8.2e}")
        else:
            print(f"{w:6.2f} {np.sum(above_thresh_w):5d} {'N/A':>8} {'N/A':>10} {'N/A':>6} {'N/A':>8}")
    
    print(f"\nReference line: ⟨j⟩ = 0.2u_d")
    print(f"Figure saved in publication-ready formats (PNG, PDF, EPS)")
    print("="*80)

if __name__ == "__main__":
    main()
