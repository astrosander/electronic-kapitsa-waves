#!/usr/bin/env python3
import os, argparse
import numpy as np

def power_spectrum_1d(n_slice, L):
    """Match your _power_spectrum_1d(): one–sided k>0, mean removed, / (N*N)."""
    N = n_slice.size
    dn = n_slice - np.mean(n_slice)
    nhat = np.fft.fft(dn)
    P = (nhat * np.conj(nhat)).real / (N*N)
    m = np.arange(N//2 + 1)
    kpos = 2*np.pi*m / L
    return kpos[1:], P[1:N//2+1]   # drop k=0

def synth_field(x, modes, coeffs=None, phase=None):
    """
    Build n(x) = sum_j a_j * sin(m_j * 2π x / L + φ_j).
    modes: list of integers m_j
    coeffs: same length or None -> all ones
    phase:  same length or None -> all zeros
    """
    L = x[-1] - x[0] + (x[1]-x[0])
    modes = np.atleast_1d(modes).astype(int)
    M = len(modes)
    if coeffs is None: coeffs = np.ones(M, float)
    if phase  is None: phase  = np.zeros(M, float)
    n = np.zeros_like(x)
    for mj, aj, ph in zip(modes, coeffs, phase):
        k = 2*np.pi*mj / L
        n += aj * np.sin(k*x + ph)
    return n

def main():
    ap = argparse.ArgumentParser(description="Generate test spec_m*.npz for plot_saved_spectra.py")
    ap.add_argument("--outdir", default="out_drift1", help="where to save NPZ")
    ap.add_argument("--L", type=float, default=10.0, help="domain length")
    ap.add_argument("--Nx", type=int, default=812, help="grid points")
    ap.add_argument("--tfinal", type=float, default=1.0, help="fake t_final to store")
    ap.add_argument("--tag", default="synthetic", help="tag in filenames")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    x = np.linspace(0.0, args.L, args.Nx, endpoint=False)

    # Define a few simple cases (m is just a label for the file & legend order)
    cases = [
        dict(m=1, modes=[7],      coeffs=[1.0],  phase=[0.0]),
        dict(m=2, modes=[13],      coeffs=[1.0],  phase=[0.0]),
        dict(m=3, modes=[0],      coeffs=[1.0],  phase=[0.0]),
        dict(m=4, modes=[0],    coeffs=[1.0], phase=[0.0]),
        dict(m=5, modes=[0],    coeffs=[1.0], phase=[0.0]),
        dict(m=6, modes=[0], coeffs=[1.0], phase=[0.0]),
    ]

    for C in cases:
        m_label = int(C["m"])

        # "Initial" and "Final" fields (can differ if you want)
        n0 = synth_field(x, C["modes"], coeffs=C["coeffs"], phase=C["phase"])
        # Example: small change at “final” to make plots non-trivial
        coeffs_final = [a*1.05 for a in C["coeffs"]]
        nF = synth_field(x, C["modes"], coeffs=coeffs_final, phase=C["phase"])

        # Spectra (match your normalization/one-sided selection)
        k0, P0 = power_spectrum_1d(n0, args.L)
        kF, PF = power_spectrum_1d(nF, args.L)

        out = os.path.join(args.outdir, f"spec_m{m_label:02d}_{args.tag}.npz")
        np.savez_compressed(out,
                            m=m_label,
                            t_final=float(args.tfinal),
                            L=float(args.L),
                            Nx=int(args.Nx),
                            k0=k0, P0=P0,   # initial spectrum
                            k=kF, P=PF,     # final spectrum
                            meta=dict(note="synthetic sines"))
        print(f"[write] {out}")

if __name__ == "__main__":
    main()
