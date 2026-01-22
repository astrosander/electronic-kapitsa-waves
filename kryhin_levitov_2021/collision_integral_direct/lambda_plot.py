#!/usr/bin/env python3
"""
Plot generalized decay rates gamma_m(T) for angular harmonics.

We use the generalized eigenproblem:
    (-M) eta = gamma * W eta,   W = diag(f(1-f))
and approximate gamma_m by the generalized Rayleigh quotient for trial functions:
    eta_m ~ cos(m theta), sin(m theta)

This script supports:
- active-only matrices saved as CSR (shape Nactive x Nactive)
- legacy full matrices saved dense (will be restricted to active)

Outputs:
- Scaled plot: (gamma_m - gamma_0)/T^2  vs T  (log-log)
"""

import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from scipy.sparse import csr_matrix, isspmatrix_csr

# --- plot style (keep your choices) ---
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
Thetas_req = np.geomspace(0.0025, 1.28, 30).astype(float).tolist()

# angular harmonics m to plot
ms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

OUT_PNG = "Eigenvals_bruteforce_generalized.png"
OUT_SVG = "Eigenvals_bruteforce_generalized.svg"


def _as_csr(X):
    if isspmatrix_csr(X):
        return X
    return csr_matrix(X)


def find_matrix_file(theta_req: float):
    """
    Pick the *_T{...}.pkl file whose T in the filename is closest to theta_req.
    """
    if not os.path.isdir(IN_DIR):
        raise FileNotFoundError(f"Directory not found: {IN_DIR}")

    files = [fn for fn in os.listdir(IN_DIR) if fn.endswith(".pkl") and "_T" in fn]
    if not files:
        raise FileNotFoundError(f"No .pkl files with '_T' found in {IN_DIR}")

    Ts = []
    paths = []
    for fn in files:
        try:
            tpart = fn.split("_T", 1)[1].rsplit(".pkl", 1)[0]
            Tval = float(tpart)
        except Exception:
            continue
        Ts.append(Tval)
        paths.append(os.path.join(IN_DIR, fn))

    if not Ts:
        raise FileNotFoundError(f"Could not parse any temperatures from filenames in {IN_DIR}")

    Ts = np.array(Ts, dtype=float)
    idx = int(np.argmin(np.abs(Ts - theta_req)))
    return float(Ts[idx]), paths[idx]


def load_matrix_nearest(theta_req: float):
    """
    Load matrix closest to theta_req.
    Returns (M, meta, path, theta_used).
    """
    theta_used, path = find_matrix_file(theta_req)
    with open(path, "rb") as fp:
        M, meta = pickle.load(fp)
    return M, meta, path, theta_used


def get_active_operator(M, meta):
    """
    Returns:
      Ma : csr_matrix, shape (Nactive, Nactive)
      active : np.ndarray of global indices
    """
    active = meta.get("active", None)
    if active is None or len(active) == 0:
        # fallback: treat everything active
        Nstates = int(meta["nx"].size)
        active = np.arange(Nstates, dtype=np.int32)
    else:
        active = np.asarray(active, dtype=np.int32)

    Nactive = int(active.size)

    # New format: active-only CSR saved directly
    if bool(meta.get("active_only", False)) and getattr(M, "shape", None) == (Nactive, Nactive):
        Ma = _as_csr(M)
        return Ma, active

    # Legacy format: full matrix (dense), restrict
    # NOTE: this is expensive for large Nstates
    Ma = _as_csr(M[np.ix_(active, active)])
    return Ma, active


def harmonic_vectors(meta, active, m: int):
    """
    Build trial vectors on the ACTIVE subspace:
      eta_cos = cos(m theta)
      eta_sin = sin(m theta)

    theta = atan2(py, px), with theta(0,0)=0.
    """
    px = np.asarray(meta["px"], dtype=np.float64)[active]
    py = np.asarray(meta["py"], dtype=np.float64)[active]

    theta = np.arctan2(py, px)
    theta = np.where((px == 0.0) & (py == 0.0), 0.0, theta)

    if m == 0:
        eta = np.ones_like(theta, dtype=np.float64)
        return eta, None

    eta_cos = np.cos(m * theta).astype(np.float64)
    eta_sin = np.sin(m * theta).astype(np.float64)
    return eta_cos, eta_sin


def generalized_rayleigh_gamma(Ma: csr_matrix, meta, active, eta: np.ndarray) -> float:
    """
    gamma(eta) = (eta^T (-M) eta) / (eta^T W eta),  W = diag(f(1-f))
    """
    f_full = np.asarray(meta["f"], dtype=np.float64)
    w = f_full[active] * (1.0 - f_full[active])  # W diagonal on active
    w = np.clip(w, 0.0, None)

    denom = float(np.dot(eta, w * eta))
    if denom <= 0.0:
        return 0.0

    # numerator: eta^T (-M) eta
    M_eta = Ma.dot(eta)
    num = float(np.dot(eta, -M_eta))
    return num / denom


def main():
    gammas = {m: [] for m in ms}
    Ts_used = []     # actual loaded temperatures
    Ts_req_used = [] # requested temperatures (for reference)

    print("=== Computing harmonic decay rates gamma_m(T) (generalized Rayleigh) ===")

    for Treq in Thetas_req:
        M, meta, path, Tused = load_matrix_nearest(float(Treq))
        if len(Ts_used) > 0 and np.isclose(Tused, Ts_used[-1], rtol=0, atol=0):
            continue
        Ma, active = get_active_operator(M, meta)

        print(f"[load] requested Theta={Treq:.6g}, using nearest Theta={Tused:.6g}  |  {os.path.basename(path)}  |  shape={Ma.shape}")

        # m=0 baseline
        eta0, _ = harmonic_vectors(meta, active, 0)
        gamma0 = generalized_rayleigh_gamma(Ma, meta, active, eta0)

        for m in ms:
            eta_cos, eta_sin = harmonic_vectors(meta, active, m)

            if m == 0:
                gamma_m = gamma0
            else:
                gcos = generalized_rayleigh_gamma(Ma, meta, active, eta_cos)
                gsin = generalized_rayleigh_gamma(Ma, meta, active, eta_sin)
                # average cos/sin to reduce grid anisotropy effects
                gamma_m = 0.5 * (gcos + gsin)

            gammas[m].append(gamma_m)

        Ts_used.append(Tused)
        Ts_req_used.append(Treq)

    Ts = np.array(Ts_used, dtype=np.float64)
    print("Ts=", Ts)
    if Ts.size == 0:
        print("Error: no matrices loaded.")
        return

    # --- plot scaled (gamma_m - gamma_0)/T^2 ---
    fig, ax = plt.subplots(figsize=(8 * 0.9, 6 * 0.9))

    gamma0s = np.array(gammas[0], dtype=np.float64)

    for m in ms:
        if m == 0:
            continue
        gm = np.array(gammas[m], dtype=np.float64)
        y = (gm - gamma0s) / (Ts ** 2)
        ax.loglog(Ts, np.abs(y), label=fr"$m={m}$", linewidth=1.5)
        
        print(f"m={m}, y={y}")
        print("================================================")

    # reference slopes in the scaled plot:
    # y ~ const  => (gamma_m - gamma_0) ~ T^2
    T_ref = Ts[len(Ts) // 3]
    y_anchor = 1.0

    ax.loglog(Ts, y_anchor * (Ts / T_ref) ** 0.0, linestyle="--", alpha=1.0, label=r"$\propto T^{2}$")
    ax.loglog(Ts, y_anchor * (Ts / T_ref) ** 1.3, linestyle="-.", alpha=1.0, label=r"$\propto T^{3.3}$")
    ax.loglog(Ts, y_anchor * (Ts / T_ref) ** 1.9, linestyle=":",  alpha=1.0, label=r"$\propto T^{3.9}$")
    ax.loglog(Ts, y_anchor * (Ts / T_ref) ** 2.0, linestyle=":",  alpha=1.0, label=r"$\propto T^{4}$")

    ax.set_xlabel(r"Temperature, $T/T_F$")
    ax.set_ylabel(r"Scaled decay, $(\gamma_m-\gamma_0)/T^{2}$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=300)
    print(f"Saved: {OUT_SVG}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
