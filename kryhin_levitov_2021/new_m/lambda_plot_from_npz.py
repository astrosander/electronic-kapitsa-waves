#!/usr/bin/env python3
"""
Plot generalized decay rates gamma_m(T) from saved CSV data.

This script loads the CSV file created by lambda_plot.py and reproduces
the plot exactly, allowing for complete reproducibility without recomputing
the eigenvalues.
"""

import numpy as np
import csv
from matplotlib import pyplot as plt
import argparse
import os

# --- plot style (matching lambda_plot.py) ---
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 20

# Default input/output filenames
DEFAULT_IN_CSV = "Eigenvals_bruteforce.csv"
DEFAULT_OUT_PNG = "Eigenvals_bruteforce_generalized_from_csv.png"
DEFAULT_OUT_SVG = "Eigenvals_bruteforce_generalized_from_csv.svg"


def load_data(csv_path: str):
    """
    Load data from CSV file.
    Expected format: T, m0, m1, m2, ... (as created by Integrals_eig_bruteforce.py)
    Returns (T, modes, gammas_dict, T_requested) where gammas_dict[m] = array of gamma_m values.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    T = []
    T_requested = []
    gammas = {}
    modes_set = set()
    
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Determine which modes are present from headers
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV file has no headers")
        
        # Check for formats: "m0c, m1c" (conserved), "m0r, m1r" (relaxing), "m2, m3, ..." (regular), 
        # and old formats: "m0, m1, ..." and "gamma_0, gamma_1, ..."
        for field in fieldnames:
            field_stripped = field.strip()
            # New format: m0r, m1r (relaxing modes) - treat as m=0, m=1 for plotting
            if field_stripped in ['m0r', 'm1r', 'm0']:
                m = 0 if field_stripped == 'm0r' else 1
                modes_set.add(m)
                gammas[m] = []
            # New format: m0c, m1c (conserved) - skip these (always 0)
            elif field_stripped in ['m0c', 'm1c', 'm1']:
                continue  # Skip conserved modes
            # New format: m2, m3, ... (regular modes)
            elif field_stripped.startswith('m') and len(field_stripped) > 1:
                try:
                    m = int(field_stripped[1:])  # Extract number after 'm'
                    # Skip m0, m1 if they exist (use m0r, m1r instead)
                    if m >= 2:
                        modes_set.add(m)
                        gammas[m] = []
                except (ValueError, IndexError):
                    pass
            # Old format: gamma_0, gamma_1, ...
            elif field_stripped.startswith('gamma_'):
                try:
                    m = int(field_stripped.split('_')[1])
                    modes_set.add(m)
                    gammas[m] = []
                except (ValueError, IndexError):
                    pass
        
        if len(modes_set) == 0:
            raise ValueError("No mode columns found in CSV. Expected columns like 'm0', 'm1', ... or 'gamma_0', 'gamma_1', ...")
        
        # Read data rows
        for row in reader:
            # Read T
            if 'T' in row and row['T'].strip():
                try:
                    T.append(float(row['T']))
                except (ValueError, TypeError):
                    continue
            else:
                continue
            
            # Read T_requested if available
            if 'T_requested' in row and row['T_requested'].strip():
                try:
                    T_requested.append(float(row['T_requested']))
                except (ValueError, TypeError):
                    T_requested.append(T[-1])  # Use T as fallback
            else:
                T_requested.append(T[-1])  # Use T as fallback
            
            # Read gammas for each mode
            for m in modes_set:
                # For m=0, m=1: try m0r, m1r first (relaxing modes)
                if m in [0, 1]:
                    gamma_key = f'm{m}r'  # m0r, m1r
                    if gamma_key in row and row[gamma_key].strip():
                        try:
                            gammas[m].append(float(row[gamma_key]))
                            continue
                        except (ValueError, TypeError):
                            pass
                    # Fallback to old format: m0, m1
                    gamma_key = f'm{m}'
                    if gamma_key in row and row[gamma_key].strip():
                        try:
                            gammas[m].append(float(row[gamma_key]))
                            continue
                        except (ValueError, TypeError):
                            pass
                else:
                    # For m>=2: try m2, m3, ...
                    gamma_key = f'm{m}'
                    if gamma_key in row and row[gamma_key].strip():
                        try:
                            gammas[m].append(float(row[gamma_key]))
                            continue
                        except (ValueError, TypeError):
                            pass
                
                # Try old format: gamma_0, gamma_1, ...
                gamma_key = f'gamma_{m}'
                if gamma_key in row and row[gamma_key].strip():
                    try:
                        gammas[m].append(float(row[gamma_key]))
                        continue
                    except (ValueError, TypeError):
                        pass
                
                # If none of the formats worked, append NaN
                gammas[m].append(np.nan)
    
    # Convert to numpy arrays
    T = np.array(T, dtype=np.float64)
    T_requested = np.array(T_requested, dtype=np.float64) if T_requested else None
    modes = np.array(sorted(modes_set), dtype=np.int32)
    
    # Convert gammas to numpy arrays
    for m in gammas:
        gammas[m] = np.array(gammas[m], dtype=np.float64)
    
    return T, modes, gammas, T_requested


def get_color_and_alpha(m, max_m=8):
    """
    Get color and alpha for mode m with custom color scheme for visibility on white background.
    - m=0,1: darker gray with small alpha
    - m=3,5,7... (odd): blue to darker/more contrast blue, gradient with m increase
    - m=2,4,6,8,10,12,14... (even): red to redder colors, gradient with m increase
    """
    if m <= 1:
        # m=0,1: use darker gray with reduced alpha (dashed lines)
        color = 'black' if m == 0 else 'violet'  # Darker grays
        return color, 1.0  # Reduced alpha for dashed lines
    elif m % 2 == 0:
        # Even modes (2,4,6,8,10,12,14...): red to redder colors
        # Start with bright red, get darker/more saturated red as m increases
        even_modes = sorted([m_val for m_val in range(2, max_m + 1, 2)])
        if m in even_modes:
            idx = even_modes.index(m)
            normalized = idx / max(len(even_modes) - 1, 1)  # Normalize to [0, 1]
        else:
            # For even modes beyond max_m, extrapolate
            normalized = min(1.0, (m - 2) / (2 * max(len(even_modes), 1)))
        
        # Red colors: wider range from orange-red to deep maroon for more distinct colors
        # Use a wider color range to make different even modes more distinguishable
        if normalized < 0.33:
            # First third: orange-red to bright red
            t = normalized / 0.33
            r = 1.0
            g = 0.4 + t * (-0.2)  # 0.4 -> 0.2
            b = 0.0
        elif normalized < 0.67:
            # Second third: bright red to crimson
            t = (normalized - 0.33) / 0.34
            r = 1.0 + t * (-0.2)  # 1.0 -> 0.8
            g = 0.2 + t * (-0.15)  # 0.2 -> 0.05
            b = 0.0
        else:
            # Last third: crimson to deep maroon
            t = (normalized - 0.67) / 0.33
            r = 0.8 + t * (-0.3)  # 0.8 -> 0.5
            g = 0.05 + t * (-0.05)  # 0.05 -> 0.0
            b = 0.0 + t * 0.1  # 0.0 -> 0.1 (slight blue tint for maroon)
        
        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))
        
        color = (r, g, b)
        return color, 1.0
    else:
        # Odd modes (3,5,7,9,11,13...): blue to darker/more contrast blue
        # Start with medium blue, get darker/more contrasted blue as m increases
        odd_modes = sorted([m_val for m_val in range(3, max_m + 1, 2)])
        if m in odd_modes:
            idx = odd_modes.index(m)
            normalized = idx / max(len(odd_modes) - 1, 1)  # Normalize to [0, 1]
        else:
            # For odd modes beyond max_m, extrapolate
            normalized = min(1.0, (m - 3) / (2 * max(len(odd_modes), 1)))
        
        # Blue colors: from bright blue to darker/more contrast blue
        # Interpolate between bright blue and deep blue for better visibility on white
        r_start, g_start, b_start = 0.2, 0.5, 1.0  # Bright blue (good visibility on white)
        r_end, g_end, b_end = 0.0, 0.2, 0.6  # Darker blue (more contrast, deeper)
        
        r = r_start + normalized * (r_end - r_start)
        g = g_start + normalized * (g_end - g_start)
        b = b_start + normalized * (b_end - b_start)
        
        color = (r, g, b)
        return color, 1.0


def plot_from_data(T, modes, gammas, out_png=None, out_svg=None):
    """
    Plot gamma_m(T) from loaded data (without dividing by T^2).
    """
    # --- plot gamma_m(T) ---
    fig, ax = plt.subplots(figsize=(8 * 0.9, 6 * 0.9))
    fig.patch.set_facecolor('white')  # Ensure white background
    ax.set_facecolor('white')

    # Collect valid data points for setting limits (before plotting reference lines)
    T_valid = []
    gamma_valid = []
    
    # Get max mode for colormap normalization
    max_m = int(np.max(modes)) if len(modes) > 0 else 8
    
    for m in modes:
        m_int = int(m)
        if m_int in gammas:
            gm = np.array(gammas[m_int], dtype=np.float64)
            mask = np.isfinite(gm) & (gm > 0.0) & (T > 0.0)
            if np.any(mask):
                T_plot = T[mask]
                gm_plot = gm[mask]
                T_valid.extend(T_plot)
                gamma_valid.extend(gm_plot)
                
                color, alpha = get_color_and_alpha(m_int, max_m)
                linestyle = '-' if m_int <= 1 else '-'
                # Use thinner lines for red (even) modes to improve separation when close
                # Keep thicker lines for blue (odd) modes and special modes
                if m_int <= 1:
                    linewidth = 3.5#1.8
                elif m_int % 2 == 0:
                    linewidth = 1.3  # Thinner for red (even) modes
                else:
                    linewidth = 2.0  # Thicker for blue (odd) modes
                if m_int <= 1:
                    # For m=0, m=1: label as relaxing modes
                    label = fr"$m={m_int}$ (relax)" if m_int in [0, 1] else fr"$m={m_int}$"
                    ax.loglog(T_plot, gm_plot, label=label, 
                             linewidth=linewidth, color=color, alpha=alpha, linestyle=linestyle)
                else:
                    ax.loglog(T_plot, gm_plot, #label=fr"$m={m_int}$", 
                                 linewidth=linewidth, color=color, alpha=alpha, linestyle=linestyle)

    # Add T^2 and T^4 reference lines
    # Normalize to match a typical data point for visual reference
    T_ref = T[np.isfinite(T) & (T > 0)]
    if len(T_ref) > 0:
        T_mid = np.sqrt(T_ref.min() * T_ref.max())  # geometric mean
        
        # Find a typical gamma value to normalize the reference lines
        gamma_ref = None
        for m_int in sorted(gammas.keys()):
            gm = np.array(gammas[m_int], dtype=np.float64)
            mask = np.isfinite(gm) & (gm > 0.0) & (T > 0.0)
            if np.any(mask):
                # Use gamma at T closest to T_mid
                idx = np.argmin(np.abs(T[mask] - T_mid))
                gamma_ref = gm[mask][idx]
                break
        
        if gamma_ref is not None and gamma_ref > 0:
            # For T^1: gamma = C_T1 * T^1
            # For T^2: gamma = C_T2 * T^2
            # For T^4: gamma = C_T4 * T^4
            C_T1 = gamma_ref / (T_mid ** 1)
            C_T2 = gamma_ref / (T_mid ** 2)
            C_T4 = gamma_ref / (T_mid ** 4)
            
            # T^1 reference: C_T1 * T^1
            ref_T1 = C_T1 * (T_ref ** 1)
            # T^2 reference: C_T2 * T^2
            ref_T2 = C_T2 * (T_ref ** 2)
            # T^4 reference: C_T4 * T^4
            ref_T4 = C_T4 * (T_ref ** 4)
            
            ax.loglog(T_ref, ref_T1, ':', color='darkgreen', linewidth=3.5, alpha=0.8, label=r"$\propto T^1$")
            ax.loglog(T_ref, ref_T2, '--', color='darkblue', linewidth=3.5, alpha=0.8, label=r"$\propto T^2$")
            ax.loglog(T_ref, ref_T4, '-.', color='darkred', linewidth=3.5, alpha=0.8, label=r"$\propto T^4$")

    # Set limits based on data curves only (not reference lines)
    if len(T_valid) > 0 and len(gamma_valid) > 0:
        T_valid = np.array(T_valid)
        gamma_valid = np.array(gamma_valid)
        ax.set_xlim([T_valid.min(), T_valid.max()])
        ax.set_ylim([gamma_valid.min(), gamma_valid.max()])

    ax.set_xlabel(r"Temperature, $T/T_F$")
    ax.set_ylabel(r"Decay rate (eigenvalue), $\gamma_m$")
    ax.legend()

    fig.tight_layout()
    
    if out_svg:
        fig.savefig(out_svg)
        print(f"Saved: {out_svg}")
    
    if out_png:
        fig.savefig(out_png, dpi=300)
        print(f"Saved: {out_png}")
    
    # Don't show here, will show both figures together at the end


def plot_logarithmic_derivative(T, modes, gammas, out_png=None, out_svg=None):
    """
    Plot logarithmic derivative of the decay rate gamma_m to extract the local scaling exponent.
    
    For odd m, the crossover from T^2 to T^4 scaling is clearly visible, where the crossover
    temperature T* decreases with increasing m.
    """
    # Small square figure for logarithmic derivative plot
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Get max mode for colormap normalization
    max_m = int(np.max(modes)) if len(modes) > 0 else 8
    
    # Reference lines for T^2 and T^4 scaling
    T_ref = T[np.isfinite(T) & (T > 0)]
    if len(T_ref) > 0:
        ax.axhline(y=2.0, color='blue', linestyle='--', linewidth=1.0, alpha=0.5, label=r"$T^2$ scaling")
        ax.axhline(y=4.0, color='red', linestyle='-.', linewidth=1.0, alpha=0.5, label=r"$T^4$ scaling")
    
    for m in modes:
        m_int = int(m)
        # Skip m=0 and m=1 for logarithmic derivative plot
        if m_int <= 1:
            continue
            
        if m_int in gammas:
            gm = np.array(gammas[m_int], dtype=np.float64)
            mask = np.isfinite(gm) & (gm > 0.0) & (T > 0.0)
            if np.any(mask):
                T_plot = T[mask]
                gm_plot = gm[mask]
                
                # Compute logarithmic derivative: d(log(γ)) / d(log(T))
                # For discrete data: log(γ_{i+1}/γ_i) / log(T_{i+1}/T_i)
                if len(T_plot) > 1:
                    # Compute differences
                    log_T_diff = np.diff(np.log(T_plot))
                    log_gamma_diff = np.diff(np.log(gm_plot))
                    
                    # Avoid division by zero
                    valid = np.abs(log_T_diff) > 1e-12
                    if np.any(valid):
                        exponent = log_gamma_diff[valid] / log_T_diff[valid]
                        # Use midpoints of temperature intervals for plotting
                        T_mid = np.sqrt(T_plot[:-1] * T_plot[1:])[valid]
                        exponent_plot = exponent
                        
                        # Filter out extreme outliers (likely numerical noise)
                        exponent_median = np.median(exponent_plot)
                        exponent_std = np.std(exponent_plot)
                        outlier_mask = np.abs(exponent_plot - exponent_median) < 5 * exponent_std
                        
                        if np.any(outlier_mask):
                            T_mid = T_mid[outlier_mask]
                            exponent_plot = exponent_plot[outlier_mask]
                            
                            color, alpha = get_color_and_alpha(m_int, max_m)
                            linestyle = '-'
                            
                            # Use thinner lines for red (even) modes
                            if m_int % 2 == 0:
                                linewidth = 1.3
                            else:
                                linewidth = 2.0
                            
                            ax.semilogx(T_mid, exponent_plot, 
                                      linewidth=linewidth, color=color, alpha=alpha, linestyle=linestyle)
    
    ax.set_xlabel(r"Temperature, $T/T_F$")
    ax.set_ylabel(r"Local scaling exponent, $\frac{d\log(\gamma_m)}{d\log(T)}$")
    ax.set_ylim([0, 6])  # Reasonable range for exponents (0 to 6)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    if out_svg:
        fig.savefig(out_svg)
        print(f"Saved: {out_svg}")
    
    if out_png:
        fig.savefig(out_png, dpi=300)
        print(f"Saved: {out_png}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot decay rates gamma_m(T) from CSV data file"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_IN_CSV,
        help=f"Input CSV file (default: {DEFAULT_IN_CSV})"
    )
    parser.add_argument(
        "--output-png", "-o",
        type=str,
        default=None,
        help=f"Output PNG file (default: {DEFAULT_OUT_PNG} or auto-generated from input name)"
    )
    parser.add_argument(
        "--output-svg",
        type=str,
        default=None,
        help=f"Output SVG file (default: {DEFAULT_OUT_SVG} or auto-generated from input name)"
    )
    
    args = parser.parse_args()
    
    # Auto-generate output names if not provided
    if args.output_png is None:
        base = os.path.splitext(args.input)[0]
        args.output_png = f"{base}_from_csv.png"
    
    if args.output_svg is None:
        base = os.path.splitext(args.input)[0]
        args.output_svg = f"{base}_from_csv.svg"
    
    print(f"Loading data from: {args.input}")
    T, modes, gammas, T_requested = load_data(args.input)
    
    print(f"Loaded {len(T)} temperature points")
    print(f"Modes: {modes}")
    print(f"Temperature range: {T.min():.6g} to {T.max():.6g}")
    
    if T_requested is not None:
        print(f"Requested temperature range: {T_requested.min():.6g} to {T_requested.max():.6g}")
    
    # Plot main figure: gamma_m / T^2
    plot_from_data(T, modes, gammas, args.output_png, args.output_svg)
    
    # Plot logarithmic derivative figure
    # Use the same base name as the main plot
    if args.output_png is not None:
        base = os.path.splitext(args.output_png)[0]
    else:
        base = os.path.splitext(args.input)[0]
    
    logderiv_png = f"{base}_logderiv.png"
    logderiv_svg = f"{base}_logderiv.svg"
    
    plot_logarithmic_derivative(T, modes, gammas, logderiv_png, logderiv_svg)
    print("Plotting complete.")


if __name__ == "__main__":
    main()