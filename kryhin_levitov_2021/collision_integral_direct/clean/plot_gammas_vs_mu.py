import os
import csv

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# Use non-interactive backend so the script can run in batch mode
matplotlib.use("Agg")

# Configure matplotlib for PRL publication quality with LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "New Century Schoolbook", "Bookman", "Computer Modern Roman"],
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 18,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 2.0,
    "patch.linewidth": 1.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def read_gammas_csv(csv_path: str):
    """
    Read CSV with columns:
        mu, gamma1, gamma2, ..., gamma8
    Returns:
        mus: (N,) array
        gammas: dict m -> (N,) array
    """
    mus = []
    gammas_cols = {m: [] for m in range(1, 21)}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        expected_cols = ["mu"] + [f"gamma{m}" for m in range(1, 21)]
        missing = [c for c in expected_cols if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV {csv_path} is missing expected columns: {missing}")

        for row in reader:
            try:
                mu_val = float(row["mu"])
            except Exception:
                continue
            mus.append(mu_val)

            for m in range(1, 21):
                key = f"gamma{m}"
                try:
                    gammas_cols[m].append(float(row[key]))
                except Exception:
                    gammas_cols[m].append(np.nan)

    mus = np.asarray(mus, dtype=float)
    for m in range(1, 21):
        gammas_cols[m] = np.asarray(gammas_cols[m], dtype=float)

    # Sort by mu
    order = np.argsort(mus)
    mus_sorted = mus[order]
    gammas_sorted = {m: gammas_cols[m][order] for m in range(1, 21)}

    return mus_sorted, gammas_sorted


def plot_gammas_vs_mu(csv_path: str, out_prefix: str = "gammas_vs_mu"):
    mus, gammas = read_gammas_csv(csv_path)

    # PRL single column width is ~3.4 inches, but for log-log plots use wider figure
    # Common practice: ~6-7 inches wide for detailed plots
    # Increased size for larger fonts
    plt.figure(figsize=(8, 6.5))

    # Build separate color gradients with more variation at small m:
    #  - odd m: from bluish/cyan to colder (deeper) blue, small m more separated, large m closer together
    #  - even m: from orangish to red, same pattern
    odd_ms = [m for m in range(1, 21) if m % 2 == 1]
    even_ms = [m for m in range(1, 21) if m % 2 == 0]

    # Create custom color gradients for higher contrast
    # Odd: from light blue/cyan to deep navy blue
    # Even: from orange to deep red
    
    # Nonlinear spacing: sqrt gives larger steps at small indices, smaller at large
    n_odd = len(odd_ms)
    n_even = len(even_ms)
    
    # Map indices with sqrt spacing for more variation at small m
    indices_odd = np.arange(n_odd)
    indices_even = np.arange(n_even)
    
    # Normalize to [0, 1] with sqrt spacing
    t_odd = np.sqrt(indices_odd / (n_odd - 1)) if n_odd > 1 else np.array([0.0])
    t_even = np.sqrt(indices_even / (n_even - 1)) if n_even > 1 else np.array([0.0])
    
    # Odd: from light blue/cyan (0.0) to deep navy blue (1.0)
    # Use a wider range for more contrast: light cyan-blue to deep blue
    blue_colors = []
    for t in t_odd:
        # Interpolate from light cyan-blue (0.6, 0.8, 1.0) to deep navy (0.0, 0.0, 0.5)
        r = 0.6 * (1 - t) + 0.0 * t
        g = 0.8 * (1 - t) + 0.0 * t
        b = 1.0 * (1 - t) + 0.5 * t
        blue_colors.append((r, g, b, 1.0))
    blue_colors = np.array(blue_colors)
    
    # Even: from orange to deep red
    # Interpolate from orange (1.0, 0.5, 0.0) to deep red (0.8, 0.0, 0.0)
    red_colors = []
    for t in t_even:
        r = 1.0 * (1 - t) + 0.8 * t
        g = 0.5 * (1 - t) + 0.0 * t
        b = 0.0 * (1 - t) + 0.0 * t
        red_colors.append((r, g, b, 1.0))
    red_colors = np.array(red_colors)

    color_map = {}
    for m, c in zip(odd_ms, blue_colors):
        color_map[m] = c
    for m, c in zip(even_ms, red_colors):
        color_map[m] = c

    # Collect all valid y values to determine ylim from data only
    all_y_valid = []
    for m in range(2, 21):
        y = gammas[m]
        mask = np.isfinite(y) & (y > 0)
        if np.any(mask):
            all_y_valid.extend(y[mask].tolist())

    for m in range(2, 21):
        y = gammas[m]
        plt.plot(
            mus,
            y,
            # marker="o",
            linestyle="-",
            color=color_map[m],
            # label=f"m={m}",
        )

    # Add reference power-law slopes: mu^{-1}, mu^{-3}, mu^{-5}
    # Normalize them to pass through a representative point of m=2 for visual comparison
    mask_pos = mus > 0
    if np.any(mask_pos):
        mus_pos = mus[mask_pos]
        idx_mid = np.where(mask_pos)[0][len(mus_pos) // 2]
        mu_ref = mus[idx_mid]

        y2 = gammas[2]
        if np.isfinite(y2[idx_mid]) and y2[idx_mid] > 0:
            gamma_ref = y2[idx_mid]
        else:
            # Fallback: use median positive finite value of gamma_2
            y2_valid = y2[np.isfinite(y2) & (y2 > 0)]
            gamma_ref = np.median(y2_valid) if y2_valid.size > 0 else 1.0

        # Use distinct colors for reference lines
        for p, ls, lab, col in [(-1, ":",  r"$\mu^{-1}$", "black"),
                                (-3, "--", r"$\mu^{-3}$", "red"),
                                (-5, "-.", r"$\mu^{-5}$", "blue")]:
            ref = gamma_ref * (mus / mu_ref) ** p
            plt.plot(
                mus,
                ref,
                linestyle=ls,
                color=col,
                linewidth=2.5,
                alpha=0.8,
                label=lab,
            )

    # Use log-log scale
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1e-2, 1e5)
    
    # Set ylim based on data curves only (min/max of valid gamma values)
    if all_y_valid:
        y_min = np.min(all_y_valid)
        y_max = np.max(all_y_valid)
        plt.ylim(y_min, y_max)

    plt.xlabel(r"$\mu$", fontsize=18)
    plt.ylabel(r"$\gamma_m$", fontsize=18)
    plt.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    png_path = f"{out_prefix}.png"
    svg_path = f"{out_prefix}.svg"

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(svg_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Saved plots: {png_path}, {svg_path}")

    # ---- Log-derivative slope figure: d log(gamma_m) / d log(mu) ----
    plt.figure(figsize=(8, 6.5))
    
    def compute_smooth_slope(log_mu, log_y, window=10):
        """
        Compute derivative using a window of points for smoothing.
        For each point, use a linear fit over the surrounding window points.
        """
        n = len(log_mu)
        if n < window:
            # Fallback to simple gradient if not enough points
            return np.gradient(log_y, log_mu)
        
        slope = np.full(n, np.nan)
        half_window = window // 2
        
        for i in range(n):
            # Determine window bounds
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            # Ensure we have at least 2 points
            if end - start < 2:
                continue
            
            # Extract window
            mu_window = log_mu[start:end]
            y_window = log_y[start:end]
            
            # Fit linear regression: y = a * x + b
            # Slope is the derivative
            if len(mu_window) > 1:
                # Use polyfit for linear fit
                coeffs = np.polyfit(mu_window, y_window, 1)
                slope[i] = coeffs[0]
        
        return slope
    
    for m in range(2, 21):
        y = gammas[m]
        mask = np.isfinite(y) & (y > 0) & (mus > 0)
        if np.count_nonzero(mask) < 10:  # Need at least 10 points for smoothing
            continue
        mu_m = mus[mask]
        log_mu = np.log(mu_m)
        log_y = np.log(y[mask])
        slope = compute_smooth_slope(log_mu, log_y, window=20)
        
        # Only plot points where slope is valid
        valid_slope = np.isfinite(slope)
        if np.any(valid_slope):
            plt.plot(
                mu_m[valid_slope],
                slope[valid_slope],
                linestyle="-",
                color=color_map[m],
                # label=f"m={m}",
            )

    plt.xscale("log")
    plt.xlabel(r"$\mu$", fontsize=18)
    plt.ylabel(r"$d \log \gamma_m / d \log \mu$", fontsize=18)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Reference horizontal lines corresponding to slopes mu^{-1}, mu^{-3} and mu^{-5}
    plt.axhline(-1, color="black", linestyle=":", linewidth=2.5, alpha=0.7, label=r"$-1$")
    plt.axhline(-3, color="red", linestyle="--", linewidth=2.5, alpha=0.7, label=r"$-3$")
    plt.axhline(-5, color="blue", linestyle="-.", linewidth=2.5, alpha=0.7, label=r"$-5$")

    plt.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=0.9)
    plt.xlim(1e-2, 1e5)

    slope_svg_path = f"{out_prefix}_log_slope.svg"
    plt.tight_layout()
    plt.savefig(slope_svg_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Saved log-slope plot: {slope_svg_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "gammas_vs_mu.csv")
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    plot_gammas_vs_mu(csv_file)


