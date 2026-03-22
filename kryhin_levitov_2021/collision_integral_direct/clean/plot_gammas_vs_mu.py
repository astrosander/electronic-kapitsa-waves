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

    has_neg = np.any(mus < 0)
    has_pos = np.any(mus > 0)
    plt.yscale("log")
    if has_neg and has_pos:
        plt.xscale("symlog", linthresh=np.maximum(1e-4, 0.01 * np.min(mus[mus > 0])))
    elif has_neg:
        plt.xscale("symlog", linthresh=np.maximum(1e-4, 0.01 * np.min(np.abs(mus[mus != 0]))))
    else:
        plt.xscale("log")

    mu_finite = mus[np.isfinite(mus)]
    if mu_finite.size > 0:
        pad = 0.05 * (np.max(mu_finite) - np.min(mu_finite) + 1e-300)
        plt.xlim(np.min(mu_finite) - pad, np.max(mu_finite) + pad)
        plt.xlim(-10000.0, -5)
        print(np.min(mu_finite) - pad, np.max(mu_finite) + pad)
        # plt.xlim()

    # plt.xlim(1e-4,1e-0)

    if all_y_valid:
        y_min = np.min(all_y_valid)
        y_max = np.max(all_y_valid)
        plt.ylim(0.5 * y_min, 2.0 * y_max)

    # Power-law guides: γ ∝ (|μ|/μ₀)^p with μ₀=γ₀=1e-9, scaled so each curve hits y_anchor at μ_mid.
    # Raw templates often span many decades on μ∈[-1e4,-5] and disappear under log-y clipping; we (i) build
    # μ on [max(xmin,μmin), min(xmax,μmax)] from the CSV, (ii) only draw points with ymin<ref_y<ymax.
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if not (np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0 and ymax > ymin):
        ymin, ymax = 1e-30, 1.0
    mus_f = mus[np.isfinite(mus)]
    if mus_f.size > 0:
        lo = max(xmin, float(np.min(mus_f)))
        hi = min(xmax, float(np.max(mus_f)))
        if lo > hi:
            lo, hi = xmin, xmax
        mu_line = np.linspace(lo, hi, 1200)
    else:
        mu_line = np.linspace(xmin, xmax, 1200)

    REF_MU = 1e-9
    REF_GAMMA = 1e-9
    ref_mag = np.maximum(np.abs(mu_line), REF_MU)
    mu_mid = 0.5 * (float(mu_line[0]) + float(mu_line[-1]))
    ref_mag_mid = max(abs(mu_mid), REF_MU)
    if ymin > 0 and ymax > 0 and np.isfinite(ymin) and np.isfinite(ymax):
        y_anchor = float(np.sqrt(ymin * ymax))
    else:
        y_anchor = 1e-9

    y_lo = ymin * 0.98
    y_hi = ymax * 1.02

    for p, ls, lab, col in [
        (-2, ":", r"$\mu^{-2}$", "black"),
        (-3, "--", r"$\mu^{-3}$", "red"),
        (-5, "-.", r"$\mu^{-5}$", "blue"),
        # (1, "-", r"$\mu^{1}$", "forestgreen"),
        # (3, "--", r"$\mu^{3}$", "darkorange"),
        # (5, "-.", r"$\mu^{5}$", "purple"),
    ]:
        ref_raw = REF_GAMMA * (ref_mag / REF_MU) ** p
        ref_raw_mid = REF_GAMMA * (ref_mag_mid / REF_MU) ** p
        if not np.isfinite(ref_raw_mid) or ref_raw_mid <= 0:
            continue
        ref_y = ref_raw * (y_anchor / ref_raw_mid)
        ok = np.isfinite(ref_y) & (ref_y > 0) & (ref_y >= y_lo) & (ref_y <= y_hi)
        if np.count_nonzero(ok) < 2:
            ok = np.isfinite(ref_y) & (ref_y > 0)
        if np.count_nonzero(ok) < 2:
            continue
        ax.plot(
            mu_line[ok],
            ref_y[ok],
            linestyle=ls,
            color=col,
            linewidth=2.5,
            alpha=0.9,
            label=lab,
            zorder=5,
            clip_on=False,
        )

    plt.xlabel(r"$\mu$", fontsize=18)
    plt.ylabel(r"$\gamma_m$", fontsize=18)
    plt.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    png_path = f"{out_prefix}.png"
    svg_path = f"{out_prefix}.svg"
    # plt.xlim(1e-4, 1e5)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(svg_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Saved plots: {png_path}, {svg_path}")

    # ---- Log-derivative slope: d log(gamma_m) / d log|mu| on each sign (classical vs degenerate) ----
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

    def plot_slope_panel(ax, mask, xvals, ylab, title, xscale, xlim, xlabel):
        ax.set_title(title, fontsize=14)
        for m in range(2, 21):
            y = gammas[m]
            msk = np.isfinite(y) & (y > 0) & mask
            if np.count_nonzero(msk) < 10:
                continue
            x_plot = xvals[msk]
            log_mu = np.log(np.maximum(x_plot, 1e-300))
            log_y = np.log(y[msk])
            slope = compute_smooth_slope(log_mu, log_y, window=min(40, max(10, np.count_nonzero(msk) // 2)))
            valid_slope = np.isfinite(slope)
            if np.any(valid_slope):
                ax.plot(
                    x_plot[valid_slope],
                    slope[valid_slope],
                    linestyle="-",
                    color=color_map[m],
                )
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylab, fontsize=18)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.axhline(-1, color="black", linestyle=":", linewidth=2.5, alpha=0.7, label=r"$-1$")
        ax.axhline(-3, color="red", linestyle="--", linewidth=2.5, alpha=0.7, label=r"$-3$")
        ax.axhline(-5, color="blue", linestyle="-.", linewidth=2.5, alpha=0.7, label=r"$-5$")
        ax.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=0.9)
        ax.set_ylim(-5, -1)
        if xlim is not None:
            ax.set_xlim(*xlim)

    mask_neg = mus < 0
    mask_pos = mus > 0
    if np.any(mask_neg) and np.any(mask_pos):
        fig, (axn, axp) = plt.subplots(1, 2, figsize=(11, 4.5))
        abs_mu = np.abs(mus)
        plot_slope_panel(
            axn,
            mask_neg,
            abs_mu,
            r"$d \log \gamma_m / d \log |\mu|$",
            r"$\mu < 0$ (non-degenerate / classical branch)",
            "log",
            None,
            r"$|\mu|$",
        )
        mus_neg = mus[mask_neg]
        if mus_neg.size:
            axn.set_xlim(1.1 * np.min(abs_mu[mask_neg]), 0.9 * np.max(abs_mu[mask_neg]))
        plot_slope_panel(
            axp,
            mask_pos,
            mus,
            r"$d \log \gamma_m / d \log \mu$",
            r"$\mu > 0$ (degenerate branch)",
            "log",
            None,
            r"$\mu$",
        )
        mus_pos_only = mus[mask_pos]
        if mus_pos_only.size:
            axp.set_xlim(0.9 * np.min(mus_pos_only), 1.1 * np.max(mus_pos_only))
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_log_slope_panels.svg", bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Saved log-slope panels: {out_prefix}_log_slope_panels.svg")
    elif np.any(mask_pos):
        plt.figure(figsize=(5, 4.5))
        plot_slope_panel(
            plt.gca(),
            mask_pos,
            mus,
            r"$d \log \gamma_m / d \log \mu$",
            "",
            "log",
            (1e-2, 1e5),
            r"$\mu$",
        )
        slope_svg_path = f"{out_prefix}_log_slope.svg"
        plt.tight_layout()
        plt.savefig(slope_svg_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Saved log-slope plot: {slope_svg_path}")
    elif np.any(mask_neg):
        plt.figure(figsize=(5, 4.5))
        abs_mu = np.abs(mus)
        plot_slope_panel(
            plt.gca(),
            mask_neg,
            abs_mu,
            r"$d \log \gamma_m / d \log |\mu|$",
            "",
            "log",
            None,
            r"$|\mu|$",
        )
        mus_neg = mus[mask_neg]
        if mus_neg.size:
            plt.xlim(1.1 * np.min(abs_mu[mask_neg]), 0.9 * np.max(abs_mu[mask_neg]))
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


