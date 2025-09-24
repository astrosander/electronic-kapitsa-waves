#!/usr/bin/env python3
import os, glob
import numpy as np
import matplotlib.pyplot as plt

def load_spectra(pattern):
    files = sorted(glob.glob(pattern))
    data = []
    for f in files:
        try:
            Z = np.load(f, allow_pickle=True)
            m  = int(Z['m'])
            k  = Z['k'];   P  = Z['P']
            k0 = Z['k0'];  P0 = Z['P0']
            tf = float(Z['t_final'])
            data.append(dict(file=f, m=m, k=k, P=P, k0=k0, P0=P0, t_final=tf))
        except Exception as e:
            print(f"[warn] skip {f}: {e}")
    return data

def plot_overlay_final(data, normalize=False, title="Final spectra", outdir=".", tag="final_overlay"):
    if not data:
        print("[plot] nothing to plot")
        return

    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
              '#FF7F00', '#A65628', '#F781BF', '#999999']

    plt.figure(figsize=(7.0, 4.0))
    for i, d in enumerate(sorted(data, key=lambda z: z['m'])):
        k, P = d['k'], d['P']
        if normalize and np.max(P) > 0:
            P = P / np.max(P)

        color = colors[i % len(colors)]
        plt.plot(k, P, lw=1.8, color=color, label=f"m={d['m']}")
        plt.xlim(0,50)
        ip = np.argmax(P)
        plt.plot([k[ip]], [P[ip]], marker='o', ms=5, color=color,
                     markeredgecolor='white', markeredgewidth=1.0)



    plt.xlabel("wavenumber k", fontsize=12)
    plt.ylabel("|n̂(k)|²" + (" (norm.)" if normalize else ""), fontsize=12)
    plt.title(title, fontsize=12)
    plt.grid(True, which="both", alpha=0.3, linestyle='--')
    plt.legend(frameon=False, ncol=2, fontsize=9)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"fft_final_overlay_{tag}.png")
    pdf = os.path.join(outdir, f"fft_final_overlay_{tag}.pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"[plot] saved {png} and {pdf}")

def plot_overlay_initial(data, normalize=False, title="Initial spectra", outdir=".", tag="initial_overlay"):
    if not data:
        print("[plot] nothing to plot")
        return

    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
              '#FF7F00', '#A65628', '#F781BF', '#999999']

    plt.figure(figsize=(7.0, 4.0))
    for i, d in enumerate(sorted(data, key=lambda z: z['m'])):
        k0, P0 = d['k0'], d['P0']
        if normalize and np.max(P0) > 0:
            P0 = P0 / np.max(P0)

        color = colors[i % len(colors)]
        plt.semilogy(k0, P0, lw=1.8, color=color, label=f"m={d['m']}")

        ip = np.argmax(P0)
        plt.semilogy([k0[ip]], [P0[ip]], marker='s', ms=5, color=color,
                     markeredgecolor='white', markeredgewidth=1.0)

    plt.xlabel("wavenumber k", fontsize=12)
    plt.ylabel("|n̂(k)|²" + (" (norm.)" if normalize else ""), fontsize=12)
    plt.title(title, fontsize=12)
    plt.grid(True, which="both", alpha=0.3, linestyle='--')
    plt.legend(frameon=False, ncol=2, fontsize=9)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"fft_initial_overlay_{tag}.png")
    pdf = os.path.join(outdir, f"fft_initial_overlay_{tag}.pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"[plot] saved {png} and {pdf}")

def main():
    data_dir = "out_drift1"
    pattern = "spec_*.npz"
    normalize = False   
    tag = "saved"
    
    pattern_path = os.path.join(data_dir, pattern)
    data = load_spectra(pattern_path)

    plot_overlay_initial(data, normalize=normalize, title="Initial spectra (t=0)", outdir=data_dir, tag=tag)
    plot_overlay_final(data, normalize=normalize, title="Final spectra (t=t_final)", outdir=data_dir, tag=tag)

if __name__ == "__main__":
    main()
