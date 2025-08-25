import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- User settings ----------------
gammas = [0.50, 1.00, 2.00]   # Γ₀ values to compare
n0 = 1.0                      # reference density n0
mask_below = -0.1            # clamp values with log10(n/n0) < mask_below to mask_below
xlim = (3, 9)                 # set to None to disable, or (xmin, xmax)
cmap = "RdBu_r"
outfile = "compare_gamma_n_clamped.png"
dpi = 150
# ------------------------------------------------


def load_field(csv_path):
    """Load CSV into x, t, n_t arrays."""
    df = pd.read_csv(csv_path, index_col=0)
    # Columns are times
    try:
        t = np.array(df.columns, dtype=float)
    except ValueError:
        t = np.array([float(str(c)) for c in df.columns])
    x = df.index.to_numpy(dtype=float)
    n_t = df.to_numpy(dtype=float)
    return x, t, n_t


def main():
    data = {}
    # Load density CSVs for the requested gammas
    for g in gammas:
        # Hardcoded Windows path as requested
        fname = f"plot_from_csv\\results_n_gamma{g:.2f}.csv"
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Missing file: {fname}")
        x, t, n_t = load_field(fname)
        data[g] = {"x": x, "t": t, "n_t": n_t}

    # Compute log10(n/n0) and clamp below threshold
    clamped_logs = {}
    mins, maxs = [], []
    for g in gammas:
        n_t = data[g]["n_t"]
        log_ratio = np.log10(n_t / n0)
        clamped = np.maximum(log_ratio, mask_below)  # values below threshold set to mask_below
        clamped_logs[g] = clamped
        mins.append(clamped.min())
        maxs.append(clamped.max())

    vmin = min(mins)
    vmax = max(maxs)

    # Plot
    fig, axes = plt.subplots(1, len(gammas), figsize=(18, 5), constrained_layout=True)
    plt.rcParams.update({
    "font.size": 20,         # default text
    "axes.titlesize": 18,    # subplot titles
    "axes.labelsize": 16,    # x and y labels
    "xtick.labelsize": 14,   # x tick labels
    "ytick.labelsize": 14,   # y tick labels
    "legend.fontsize": 14,   # legend text
})


    im = None
    for i, g in enumerate(gammas):
        ax = axes[i] if len(gammas) > 1 else axes
        x = data[g]["x"]
        t = data[g]["t"]
        clamped = clamped_logs[g]

        extent = [float(x.min()), float(x.max()), float(t.min()), float(t.max())]
        im = ax.imshow(
            clamped.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(f"$ \\Gamma_0= {g:.2f}$")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label(r"$\log_{10}(n/n_0)$")

    # Save and show
    plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
