#!/usr/bin/env python3
"""
Plot generalized decay rates gamma_m(T) from saved NPZ data.

This script loads the NPZ file created by lambda_plot.py and reproduces
the plot exactly, allowing for complete reproducibility without recomputing
the eigenvalues.
"""

import numpy as np
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
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20

# Default input/output filenames
DEFAULT_IN_NPZ = "Eigenvals_bruteforce_generalized.npz"
DEFAULT_OUT_PNG = "Eigenvals_bruteforce_generalized_from_npz.png"
DEFAULT_OUT_SVG = "Eigenvals_bruteforce_generalized_from_npz.svg"


def load_data(npz_path: str):
    """
    Load data from NPZ file.
    Returns (T, modes, gammas_dict) where gammas_dict[m] = array of gamma_m values.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    
    T = data["T"]
    modes = data["modes"]
    
    # Extract gamma_m for each mode
    gammas = {}
    for m in modes:
        key = f"gamma_{m}"
        if key in data:
            gammas[int(m)] = data[key]
        else:
            print(f"Warning: {key} not found in NPZ file")
            gammas[int(m)] = np.full_like(T, np.nan)
    
    # Optional: also load T_requested if available
    T_requested = data.get("T_requested", None)
    
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
        color = '#2C3E50' if m == 0 else '#34495E'  # Darker grays
        return color, 0.3  # Reduced alpha for dashed lines
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
                linestyle = '--' if m_int <= 1 else '-'
                # Use slightly thicker lines for better visibility on white background
                linewidth = 2.0 if m_int <= 1 else 2.0
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
            ref_T2_normalized = np.full_like(T_ref, C_T2)
            # T^4 reference: C_T4 * T^2
            ref_T4_normalized = C_T4 * (T_ref ** 2)
            
            ax.loglog(T_ref, ref_T2_normalized, '--', color='blue', linewidth=1.0, label=r"$\propto T^2$")
            ax.loglog(T_ref, ref_T4_normalized, '-.', color='red', linewidth=1.0, label=r"$\propto T^4$")

    # Set limits based on data curves only (not reference lines)
    if len(T_valid) > 0 and len(gamma_over_T2_valid) > 0:
        T_valid = np.array(T_valid)
        gamma_over_T2_valid = np.array(gamma_over_T2_valid)
        ax.set_xlim([T_valid.min(), T_valid.max()])
        ax.set_ylim([gamma_over_T2_valid.min(), gamma_over_T2_valid.max()])

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


def main():
    parser = argparse.ArgumentParser(
        description="Plot decay rates gamma_m(T) from NPZ data file"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_IN_NPZ,
        help=f"Input NPZ file (default: {DEFAULT_IN_NPZ})"
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
        args.output_png = f"{base}_from_npz.png"
    
    if args.output_svg is None:
        base = os.path.splitext(args.input)[0]
        args.output_svg = f"{base}_from_npz.svg"
    
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

