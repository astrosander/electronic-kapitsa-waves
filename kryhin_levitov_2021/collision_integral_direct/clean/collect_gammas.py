import os
import csv
import pickle
from typing import List, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (produces PNG files)

# We reuse the eigen-solver from plot_ring.py
from plot_ring import compute_eigenfunctions_by_mode


def extract_mu_from_meta_or_name(meta: Dict, filename: str) -> float:
    """
    Try to get mu from the pickle metadata; if not available,
    fall back to parsing it from the filename, assuming a pattern like
    ..._mu2.15443_U1_...
    """
    for key in ("mu_phys", "mu"):
        if key in meta:
            try:
                return float(meta[key])
            except Exception:
                pass

    # Fallback: parse from filename
    base = os.path.basename(filename)
    # look for '_mu' then take up to next '_'
    if "_mu" in base:
        try:
            after_mu = base.split("_mu", 1)[1]
            mu_part = after_mu.split("_", 1)[0]
            return float(mu_part)
        except Exception:
            pass

    raise ValueError(f"Cannot determine mu from meta or filename: {filename}")


def compute_gammas_for_file(pkl_path: str, ms: List[int]) -> Tuple[float, Dict[int, float]]:
    """
    Load a single .pkl file, compute eigenvalues gamma_m for given ms.
    Returns (mu, {m: gamma_m or np.nan})
    """
    with open(pkl_path, "rb") as fp:
        Ma, meta = pickle.load(fp)

    mu = extract_mu_from_meta_or_name(meta, pkl_path)

    # compute eigenfunctions / eigenvalues for requested ms
    eigenfunctions, eigenvalues, px, py = compute_eigenfunctions_by_mode(Ma, meta, ms=ms)

    # Make sure we always return a float (or NaN) for each m
    gammas_by_m: Dict[int, float] = {}
    for m in ms:
        val = eigenvalues.get(m, np.nan)
        try:
            gammas_by_m[m] = float(val)
        except Exception:
            gammas_by_m[m] = float("nan")

    return mu, gammas_by_m


def main():
    # Angular modes we want in the CSV: gamma1..gamma20
    ms = list(range(1, 21))

    # Directory with .pkl files – match the default used in plot_ring.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    matrices_dir = os.path.join(base_dir, "Matrixes_bruteforce")

    if not os.path.isdir(matrices_dir):
        raise FileNotFoundError(f"Directory not found: {matrices_dir}")

    # Collect all .pkl files
    pkl_files = [
        os.path.join(matrices_dir, f)
        for f in os.listdir(matrices_dir)
        if f.lower().endswith(".pkl")
    ]

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {matrices_dir}")

    rows = []
    for pkl_path in sorted(pkl_files):
        print(f"Processing {os.path.basename(pkl_path)} ...", flush=True)
        try:
            mu, gammas_by_m = compute_gammas_for_file(pkl_path, ms)
        except Exception as e:
            print(f"  Failed to process {pkl_path}: {e}", flush=True)
            continue

        row = [mu] + [gammas_by_m.get(m, float("nan")) for m in ms]
        rows.append(row)

    # Sort rows by mu
    rows.sort(key=lambda r: r[0])

    # Output CSV in the base directory
    csv_path = os.path.join(base_dir, "gammas_vs_mu.csv")
    header = ["mu"] + [f"gamma{m}" for m in ms]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Written CSV: {csv_path}")

    # ---- Plot gamma_m(mu) curves ----
    rows_arr = np.array(rows, dtype=float)
    mu_vals = rows_arr[:, 0]

    plt.figure(figsize=(8, 6))
    for i, m in enumerate(ms, start=1):
        gamma_vals = rows_arr[:, i]
        plt.plot(mu_vals, gamma_vals, marker="o", linestyle="-", label=f"γ{m}")

    plt.xlabel("μ")
    plt.ylabel("γ_m")
    plt.title("Angular relaxation rates γ_m(μ)")
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.grid(True, linestyle="--", alpha=0.3)

    plot_path = os.path.join(base_dir, "gammas_vs_mu.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Written plot: {plot_path}")


if __name__ == "__main__":
    main()


