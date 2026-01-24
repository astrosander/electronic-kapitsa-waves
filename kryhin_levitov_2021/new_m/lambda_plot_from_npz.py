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
        
        # Check for both formats: "m0, m1, ..." (new format) and "gamma_0, gamma_1, ..." (old format)
        for field in fieldnames:
            field_stripped = field.strip()
            # New format: m0, m1, m2, ...
            if field_stripped.startswith('m') and len(field_stripped) > 1:
                try:
                    m = int(field_stripped[1:])  # Extract number after 'm'
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
            
            # Read gammas for each mode (try both formats)
            for m in modes_set:
                # Try new format first: m0, m1, ...
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
                
                # If neither format worked, append NaN
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
    Get color and alpha for mode m using rainbow colormap (adjusted for visibility on white).
    - m=0,1: darker gray with small alpha
    - m=2,4,6,8 (even): use darker blue-green part of rainbow
    - m=3,5,7 (odd): use darker red-orange part of rainbow
    """
    # Use rainbow colormap but adjust positions to avoid light colors
    colormap = plt.cm.rainbow
    
    if m <= 1:
        # m=0,1: use darker gray with reduced alpha (dashed lines)
        color = 'black' if m == 0 else '#34495E'  # Darker grays
        return color, 0.4  # Reduced alpha for dashed lines
    elif m % 2 == 0:
        # Even modes (2,4,6,8): map to darker blue-green part of rainbow
        # Avoid light cyan, use darker blues and greens (0.55 to 0.75)
        even_modes = [2, 4, 6, 8]
        if m in even_modes:
            idx = even_modes.index(m)
            normalized = idx / (len(even_modes) - 1) if len(even_modes) > 1 else 0.0
            # Map to darker blue-green range: 0.55 (darker blue) to 0.75 (darker green)
            rainbow_pos = 0.55 + normalized * 0.20
        else:
            # Fallback for other even modes
            rainbow_pos = 0.65
        color = colormap(rainbow_pos)
        return color, 1.0
    else:
        # Odd modes (3,5,7): map to darker red-orange part of rainbow
        # Avoid light yellow, use darker reds and oranges (0.0 to 0.25)
        odd_modes = [3, 5, 7]
        if m in odd_modes:
            idx = odd_modes.index(m)
            normalized = idx / (len(odd_modes) - 1) if len(odd_modes) > 1 else 0.0
            # Map to darker red-orange range: 0.0 (red) to 0.25 (darker orange, avoiding yellow)
            rainbow_pos = 0.0 + normalized * 0.25
        else:
            # Fallback for other odd modes
            rainbow_pos = 0.125
        color = colormap(rainbow_pos)
        return color, 1.0


def plot_from_data(T, modes, gammas, out_png=None, out_svg=None):
    """
    Plot gamma_m(T)/T^2 from loaded data.
    """
    # --- plot gamma_m(T)/T^2 ---
    fig, ax = plt.subplots(figsize=(8 * 0.9, 6 * 0.9))
    fig.patch.set_facecolor('white')  # Ensure white background
    ax.set_facecolor('white')

    # Collect valid data points for setting limits (before plotting reference lines)
    T_valid = []
    gamma_over_T2_valid = []
    
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
                gm_over_T2 = gm_plot / (T_plot ** 2)
                T_valid.extend(T_plot)
                gamma_over_T2_valid.extend(gm_over_T2)
                
                color, alpha = get_color_and_alpha(m_int, max_m)
                linestyle = '-' if m_int <= 1 else '-'
                # Use slightly thicker lines for better visibility on white background
                linewidth =1.8 if m_int <= 1 else 2.0
                ax.loglog(T_plot, gm_over_T2, label=fr"$m={m_int}$", 
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
            # Normalize so ref lines pass through (T_mid, gamma_ref/T_mid^2)
            # For T^2: gamma = C_T2 * T^2, so gamma/T^2 = C_T2 (constant)
            # For T^4: gamma = C_T4 * T^4, so gamma/T^2 = C_T4 * T^2
            C_T2 = gamma_ref / (T_mid ** 2)
            C_T4 = gamma_ref / (T_mid ** 4)
            
            # T^2 reference: constant line at C_T2
            ref_T2_normalized = np.full_like(T_ref, C_T2)*10
            # T^4 reference: C_T4 * T^2
            ref_T4_normalized = C_T4 * (T_ref ** 2)
            
            ax.loglog(T_ref, ref_T2_normalized*1000, '--', color='blue', linewidth=1.0, label=r"$\propto T^2$")
            ax.loglog(T_ref, ref_T4_normalized*1000, '-.', color='red', linewidth=1.0, label=r"$\propto T^4$")

    # Set limits based on data curves only (not reference lines)
    if len(T_valid) > 0 and len(gamma_over_T2_valid) > 0:
        T_valid = np.array(T_valid)
        gamma_over_T2_valid = np.array(gamma_over_T2_valid)
        ax.set_xlim([T_valid.min(), T_valid.max()])
        ax.set_ylim([gamma_over_T2_valid.min(), gamma_over_T2_valid.max()])

    # ax.set_xlim([T_valid.min(), T_valid.max()])
    ax.set_ylim(1e-7, 1e2)

    ax.set_xlabel(r"Temperature, $T/T_F$")
    ax.set_ylabel(r"Decay rate (eigenvalue), $\gamma_m / T^2$")
    ax.legend()

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
    
    plot_from_data(T, modes, gammas, args.output_png, args.output_svg)
    print("Plotting complete.")


if __name__ == "__main__":
    main()

