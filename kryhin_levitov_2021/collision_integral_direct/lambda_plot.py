#!/usr/bin/env python3

import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

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


IN_DIR = "Matrixes_bruteforce"
Thetas = np.geomspace(0.0025, 1.28, 30).astype(float).tolist()

ms = [0, 1, 2, 3, 4, 5, 6]
OUT_PNG = "Eigenvals_bruteforce.png"
OUT_SVG = "Eigenvals_bruteforce.svg"


def load_matrix(theta: float):
    """
    Load matrix for given temperature.
    Returns (M, meta, path) if found, None if file is missing.
    """
    tstr = f"{theta:.10g}"
    candidates = [fn for fn in os.listdir(IN_DIR) if fn.endswith(f"_T{tstr}.pkl")]
    if not candidates:
        return None
    path = os.path.join(IN_DIR, sorted(candidates)[0])
    with open(path, "rb") as fp:
        M, meta = pickle.load(fp)
    return M, meta, path


def harmonic_vector(meta, m: int):
    px = meta["px"]
    py = meta["py"]
    f  = meta["f"]

    theta = np.arctan2(py, px)
    theta = np.where((px == 0.0) & (py == 0.0), 0.0, theta)

    w = np.sqrt(f * (1.0 - f))
    v = w * np.cos(m * theta)
    return v.astype(np.float64)


def rayleigh_lambda(M: np.ndarray, v: np.ndarray) -> float:
    denom = float(v @ v)
    if denom == 0.0:
        return 0.0
    Mv = M @ v
    num = float(v @ Mv)
    return -num / denom


def main():
    lambdas = {m: [] for m in ms}
    Ts_loaded = []

    print("=== Computing harmonic decay eigenvalues (Rayleigh) ===")
    for T in Thetas:
        result = load_matrix(float(T))
        if result is None:
            print(f"Warning: No matrix file found for Theta={T:.10g}, skipping...")
            continue
        
        M, meta, path = result
        print(f"Loaded Theta={T:.6g} from {path}")

        lam0 = None
        for m in ms:
            v = harmonic_vector(meta, m)
            lam = rayleigh_lambda(M, v)
            if m == 0:
                lam0 = lam
            lambdas[m].append(lam)

        if lam0 is None:
            lam0 = 0.0
        
        Ts_loaded.append(T)

    if not Ts_loaded:
        print("Error: No matrix files were successfully loaded!")
        return

    print(f"\nSuccessfully processed {len(Ts_loaded)} out of {len(Thetas)} temperatures.")

    plt.rcParams["legend.frameon"] = False
    fig, ax = plt.subplots(figsize=(8 * 0.9, 6 * 0.9))

    Ts = np.array(Ts_loaded, dtype=np.float64)
    lam0s = np.array(lambdas[0], dtype=np.float64)

    for m in ms:
        if m == 0:
            continue
        y = (np.array(lambdas[m], dtype=np.float64) - lam0s) / (Ts ** 2)
        ax.loglog(Ts, np.abs(y), label=f"m = {m}", linewidth=1.5)

    T_ref = Ts[len(Ts) // 3]
    y_anchor = 1.0

    ax.loglog(Ts, y_anchor * (Ts / T_ref) ** 0.0, linestyle="--", alpha=1, label=r"$T^2$")

    ax.loglog(Ts, y_anchor * (Ts / T_ref) ** 1.3, linestyle="-.", alpha=1, label=r"$T^{3.3}$")
    ax.loglog(Ts, y_anchor * (Ts / T_ref) ** 1.9, linestyle=":",  alpha=1, label=r"$T^{3.9}$")
    ax.loglog(Ts, y_anchor * (Ts / T_ref) ** 2.0, linestyle=":",  alpha=1, label=r"$T^{4}$")

    ax.set_xlabel(r"Temperature, $T/T_F$")
    ax.set_ylabel(r"Scaled decay, $(\lambda_m-\lambda_0)/T^{2}$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    print(f"Saved: {OUT_SVG}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
