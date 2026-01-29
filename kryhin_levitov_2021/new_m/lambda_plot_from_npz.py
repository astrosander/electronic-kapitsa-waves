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
DEFAULT_IN_CSV = r"D:\Downloads\28_1\test\10.csv"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\kryhin_levitov_2021\collision_integral_direct\Matrixes_bruteforce\mu001\gamma_vs_mu_T0.1.csv"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\kryhin_levitov_2021\collision_integral_direct\Matrixes_bruteforce\mu\gamma_vs_mu_T0.1.csv"#"D:\Рабочая папка\GitHub\electronic-kapitsa-waves\kryhin_levitov_2021\collision_integral_direct\Matrixes_bruteforce\gamma_vs_T_mu0p1_U1.csv"#"gamma_vs_T_mu0p1_U1.csv"
BASE_DIR = r"D:\Downloads\28_1\test"

DEFAULT_OUT_PNG = f"{DEFAULT_IN_CSV.replace(BASE_DIR, '')}.png"
DEFAULT_OUT_SVG = f"{DEFAULT_IN_CSV.replace(BASE_DIR, '')}.svg"

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
        control_key = None  # "T", "Theta", or "mu" depending on CSV
        
        # First pass: prioritize temperature columns (T or Theta) over mu
        for field in fieldnames:
            field_stripped = field.strip()
            low = field_stripped.lower()
            # Prioritize temperature columns
            if low in ("t", "theta"):
                control_key = field_stripped
                break  # Found temperature column, use it
        
        # Second pass: if no temperature column found, look for mu
        if control_key is None:
            for field in fieldnames:
                field_stripped = field.strip()
                low = field_stripped.lower()
                if low == "mu":
                    control_key = field_stripped
                    break
        
        # Now process all fields for mode detection
        for field in fieldnames:
            field_stripped = field.strip()
            low = field_stripped.lower()

            # New format: m0r, m1r (relaxing modes) - treat as m=0, m=1 for plotting
            if field_stripped in ['m0r', 'm1r']:
                m = 0 if field_stripped == 'm0r' else 1
                modes_set.add(m)
                gammas[m] = []
            # New format: m0c, m1c (conserved) - skip these (always 0)
            elif field_stripped in ['m0c', 'm1c']:
                continue  # Skip conserved modes
            # Regular format: m0, m1, m2, m3, ... (all modes)
            elif field_stripped.startswith('m') and len(field_stripped) > 1:
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
        if control_key is None:
            raise ValueError("No control parameter column found in CSV. Expected 'T', 'Theta', or 'mu'.")
        
        # Read data rows
        for row in reader:
            # Read temperature / control parameter from detected column (T or mu)
            raw_val = row.get(control_key, "").strip()
            try:
                val_T = float(raw_val) if raw_val else None
            except (ValueError, TypeError):
                val_T = None

            if val_T is None:
                # Skip rows without a valid control parameter
                continue

            T.append(val_T)
            
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
                # Try different formats in order of preference
                # 1. Try m0r, m1r (relaxing modes) for m=0,1
                if m in [0, 1]:
                    gamma_key = f'm{m}r'  # m0r, m1r
                    if gamma_key in row and row[gamma_key].strip():
                        try:
                            gammas[m].append(float(row[gamma_key]))
                            continue
                        except (ValueError, TypeError):
                            pass
                
                # 2. Try standard format: m0, m1, m2, m3, ...
                gamma_key = f'm{m}'
                if gamma_key in row and row[gamma_key].strip():
                    try:
                        gammas[m].append(float(row[gamma_key]))
                        continue
                    except (ValueError, TypeError):
                        pass
                
                # 3. Try old format: gamma_0, gamma_1, ...
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
    
    # control_key tells us whether T is actually temperature or mu
    return T, modes, gammas, T_requested, control_key


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


def plot_from_data(T, modes, gammas, out_png=None, out_svg=None, x_label=r"Temperature, $T/T_F$", control_key="T"):
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
                    label = fr"$m={m_int}$" if m_int in [0, 1] else fr"$m={m_int}$"
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
            # Determine if this is a mu-sweep or T-sweep
            is_mu_sweep = (control_key.lower() == "mu")
            
            if is_mu_sweep:
                # For mu-sweeps: show mu^1, mu^2, mu^3, mu^4 reference lines
                C_mu1 = gamma_ref / (T_mid ** 1)
                C_mu2 = gamma_ref / (T_mid ** 2)
                C_mu3 = gamma_ref / (T_mid ** 3)
                C_mu4 = gamma_ref / (T_mid ** 4)
                
                ref_mu1 = C_mu1 * (T_ref ** 1)
                ref_mu2 = C_mu2 * (T_ref ** 2)
                ref_mu3 = C_mu3 * (T_ref ** 3)
                ref_mu4 = C_mu4 * (T_ref ** 4)
                
                ax.loglog(T_ref, ref_mu1, ':', color='darkgreen', linewidth=3.5, alpha=0.8, label=r"$\propto \mu^1$")
                ax.loglog(T_ref, ref_mu2, '--', color='darkblue', linewidth=3.5, alpha=0.8, label=r"$\propto \mu^2$")
                ax.loglog(T_ref, ref_mu3, '-.', color='darkorange', linewidth=3.5, alpha=0.8, label=r"$\propto \mu^3$")
                ax.loglog(T_ref, ref_mu4, '--', color='darkred', linewidth=3.5, alpha=0.8, label=r"$\propto \mu^4$")
            else:
                # For T-sweeps or Theta-sweeps: show T^2, T^4, and T^8 reference lines
                C_T2 = gamma_ref / (T_mid ** 2)
                C_T4 = gamma_ref / (T_mid ** 4)
                C_T8 = gamma_ref / (T_mid ** 8)
                
                ref_T2 = C_T2 * (T_ref ** 2)
                ref_T4 = C_T4 * (T_ref ** 4)
                ref_T8 = C_T8 * (T_ref ** 8)
                
                if control_key.lower() == "theta":
                    ax.loglog(T_ref, ref_T2, '--', color='darkblue', linewidth=3.5, alpha=0.8, label=r"$\propto T^2$")
                    ax.loglog(T_ref, ref_T4, '-.', color='darkred', linewidth=3.5, alpha=0.8, label=r"$\propto T^4$")
                    ax.loglog(T_ref, ref_T8, ':', color='purple', linewidth=3.5, alpha=0.8, label=r"$\propto T^8$")
                else:
                    ax.loglog(T_ref, ref_T2, '--', color='darkblue', linewidth=3.5, alpha=0.8, label=r"$\propto T^2$")
                    ax.loglog(T_ref, ref_T4, '-.', color='darkred', linewidth=3.5, alpha=0.8, label=r"$\propto T^4$")
                    ax.loglog(T_ref, ref_T8, ':', color='purple', linewidth=3.5, alpha=0.8, label=r"$\propto T^8$")

    # Set limits based on data curves only (not reference lines)
    if len(T_valid) > 0 and len(gamma_valid) > 0:
        T_valid = np.array(T_valid)
        gamma_valid = np.array(gamma_valid)
        ax.set_xlim([T_valid.min(), T_valid.max()])
        ax.set_ylim([gamma_valid.min(), gamma_valid.max()])

    ax.set_xlabel(x_label)
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


def plot_logarithmic_derivative(T, modes, gammas, out_png=None, out_svg=None, x_label=r"Temperature, $T/T_F$", control_key="T"):
    """
    Plot logarithmic derivative of the decay rate gamma_m to extract the local scaling exponent.
    
    For odd m, the crossover from T^2 to T^4 scaling is clearly visible, where the crossover
    temperature T* decreases with increasing m.
    For mu-sweeps, shows d(log(gamma))/d(log(mu)) to extract mu-scaling exponents.
    """
    # Compact square figure for logarithmic derivative plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Get max mode for colormap normalization
    max_m = int(np.max(modes)) if len(modes) > 0 else 8
    
    # Reference lines - adjust based on control parameter
    T_ref = T[np.isfinite(T) & (T > 0)]
    if len(T_ref) > 0:
        if control_key.lower() == "mu":
            # For mu-sweeps: show only mu^2 reference line
            ax.axhline(y=2.0, color='red', linestyle='-.', linewidth=1.0, alpha=0.5, label=r"$\mu^2$")
        else:
            # For T-sweeps or Theta-sweeps: show T^2 and T^4 reference lines
            ax.axhline(y=2.0, color='blue', linestyle='--', linewidth=1.0, alpha=0.5, label=r"$\Theta^2$")
            ax.axhline(y=4.0, color='red', linestyle='-.', linewidth=1.0, alpha=0.5, label=r"$\Theta^4$")
    
    for m in modes:
        m_int = int(m)
        # Skip m=0 for logarithmic derivative plot (include m=1)
        if m_int == 0:
            continue
            
        if m_int in gammas:
            gm = np.array(gammas[m_int], dtype=np.float64)
            mask = np.isfinite(gm) & (gm > 0.0) & (T > 0.0)
            if np.any(mask):
                T_plot = T[mask]
                gm_plot = gm[mask]
                
                # Compute logarithmic derivative: d(log(γ)) / d(log(T))
                # Use span of 5 points to reduce noise: log(γ_{i+span}/γ_i) / log(T_{i+span}/T_i)
                # Use smaller span if we don't have enough points
                span = min(6, max(2, len(T_plot) - 1))
                if len(T_plot) > 1:
                    T_mid_list = []
                    exponent_list = []
                    
                    # Compute derivative over intervals of size span
                    for i in range(len(T_plot) - span):
                        j = i + span
                        log_T_diff = np.log(T_plot[j]) - np.log(T_plot[i])
                        log_gamma_diff = np.log(gm_plot[j]) - np.log(gm_plot[i])
                        
                        # Avoid division by zero
                        if np.abs(log_T_diff) > 1e-12:
                            exponent = log_gamma_diff / log_T_diff
                            # Use geometric mean of endpoints for x-coordinate
                            T_mid = np.sqrt(T_plot[i] * T_plot[j])
                            T_mid_list.append(T_mid)
                            exponent_list.append(exponent)
                    
                    if len(T_mid_list) > 0:
                        T_mid = np.array(T_mid_list)
                        exponent_plot = np.array(exponent_list)
                        
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
                            
                            # Add label for m=1 (current mode)
                            label = fr"$m={m_int}$" if m_int == 1 else None
                            
                            ax.semilogx(T_mid, exponent_plot, 
                                      linewidth=linewidth, color=color, alpha=alpha, linestyle=linestyle, label=label)
    
    ax.set_xlabel(x_label)
    # Update y-axis label based on control parameter
    if control_key.lower() == "mu":
        ax.set_ylabel(r"$\frac{d\log(\gamma_m)}{d\log(\mu)}$")
    elif control_key.lower() == "theta":
        ax.set_ylabel(r"$\frac{d\log(\gamma_m)}{d\log(\Theta)}$")
    else:
        ax.set_ylabel(r"$\frac{d\log(\gamma_m)}{d\log(T)}$")
    ax.set_ylim([0, 10])  # Reasonable range for exponents (0 to 6)
    ax.legend(loc='best', fontsize=12, frameon=False)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout(pad=1.5)
    
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
    T, modes, gammas, T_requested, control_key = load_data(args.input)
    
    print(f"Loaded {len(T)} points")
    print(f"Modes: {modes}")
    if len(T) > 0:
        print(f"Control range ({control_key}): {T.min():.6g} to {T.max():.6g}")
    
    if T_requested is not None:
        print(f"Requested temperature range: {T_requested.min():.6g} to {T_requested.max():.6g}")
    
    # Restrict to modes m=1..4 as requested
    modes_plot = np.array([m for m in modes if 1 <= int(m) <= 30], dtype=np.int32)
    if modes_plot.size == 0:
        modes_plot = modes

    # Choose x-axis label depending on control parameter
    if control_key.lower() == "mu":
        x_label = r"$\mu$"
    elif control_key.lower() == "theta":
        x_label = r"$\Theta = T/T_F$"
    else:
        x_label = r"Temperature, $T/T_F$"

    # Plot main figure
    plot_from_data(T, modes_plot, gammas, args.output_png, args.output_svg, x_label=x_label, control_key=control_key)
    
    # Plot logarithmic derivative figure (for both T-scan and mu-scan)
    if args.output_png is not None:
        base = os.path.splitext(args.output_png)[0]
    else:
        base = os.path.splitext(args.input)[0]
    
    logderiv_png = f"{base}_logderiv.png"
    logderiv_svg = f"{base}_logderiv.svg"
    
    plot_logarithmic_derivative(T, modes_plot, gammas, logderiv_png, logderiv_svg, x_label=x_label, control_key=control_key)
    print("Plotting complete.")


if __name__ == "__main__":
    main()