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

def plot_overlay_final(data, normalize=False, title="Final spectra", outdir=".", tag="final_panels"):
    if not data:
        print("[plot] nothing to plot")
        return

    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
              '#FF7F00', '#A65628', '#F781BF', '#999999']

    sorted_data = sorted(data, key=lambda z: z['m'])
    n_panels = len(sorted_data)
    
    fig, axes = plt.subplots(1, n_panels, figsize=(15.0, 3.5), sharey=True)
    
    if n_panels == 1:
        axes = [axes]
    
    for i, d in enumerate(sorted_data):
        ax = axes[i]
        k, P = d['k'], d['P']
        if normalize and np.max(P) > 0:
            P = P / np.max(P)

        color = colors[i % len(colors)]
        
        if d['m'] == 1:
            label = f"$\\cos(3x) + \\cos(5x)$"
        elif d['m'] == 2:
            label = f"$\\cos(5x) + \\cos(8x)$"
        elif d['m'] == 3:
            label = f"$\\cos(8x) + \\cos(13x)$"
        elif d['m'] == 4:
            label = f"$\\cos(13x) + \\cos(21x)$"
        elif d['m'] == 5:
            label = f"$\\cos(21x) + \\cos(34x)$"
        elif d['m'] == 6:
            label = f"$\\cos(34x) + \\cos(55x)$"
        else:
            label = f"$\\cos(ax) + \\cos(bx)$"
        
        ax.plot(k, P, lw=1.8, color=color)
        ax.set_xlim(0, 50)
        
        ip = np.argmax(P)
        ax.plot([k[ip]], [P[ip]], marker='o', ms=5, color=color,
                markeredgecolor='white', markeredgewidth=1.0)
        
        ax.set_xlabel("$k$", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[0].set_ylabel("|n̂(k)|²" + (" (norm.)" if normalize else ""), fontsize=10)
    
    fig.suptitle(title, fontsize=12, y=0.98)

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"fft_final_panels_{tag}.png")
    pdf = os.path.join(outdir, f"fft_final_panels_{tag}.pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[plot] saved {png} and {pdf}")

def plot_overlay_initial(data, normalize=False, title="Initial spectra", outdir=".", tag="initial_panels"):
    if not data:
        print("[plot] nothing to plot")
        return

    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3',
              '#FF7F00', '#A65628', '#F781BF', '#999999']

    sorted_data = sorted(data, key=lambda z: z['m'])
    n_panels = len(sorted_data)
    
    fig, axes = plt.subplots(1, n_panels, figsize=(15.0, 3.5), sharey=True)
    
    if n_panels == 1:
        axes = [axes]
    
    for i, d in enumerate(sorted_data):
        ax = axes[i]
        k0, P0 = d['k0'], d['P0']
        if normalize and np.max(P0) > 0:
            P0 = P0 / np.max(P0)

        color = colors[i % len(colors)]
        
        if d['m'] == 1:
            label = f"$\\cos(3x) + \\cos(5x)$"
        elif d['m'] == 2:
            label = f"$\\cos(5x) + \\cos(8x)$"
        elif d['m'] == 3:
            label = f"$\\cos(8x) + \\cos(13x)$"
        elif d['m'] == 4:
            label = f"$\\cos(13x) + \\cos(21x)$"
        elif d['m'] == 5:
            label = f"$\\cos(21x) + \\cos(34x)$"
        elif d['m'] == 6:
            label = f"$\\cos(34x) + \\cos(55x)$"
        else:
            label = f"$\\cos(ax) + \\cos(bx)$"
        
        ax.plot(k0, P0, lw=1.8, color=color)
        ax.set_xlim(0, 50)
        
        ip = np.argmax(P0)
        ax.plot([k0[ip]], [P0[ip]], marker='s', ms=5, color=color,
                markeredgecolor='white', markeredgewidth=1.0)
        
        ax.set_xlabel("$k$", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[0].set_ylabel("$|\\hat{n}(k)|^2$" + (" (norm.)" if normalize else ""), fontsize=10)
    
    fig.suptitle(title, fontsize=12, y=0.98)

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"fft_initial_panels_{tag}.png")
    pdf = os.path.join(outdir, f"fft_initial_panels_{tag}.pdf")
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"[plot] saved {png} and {pdf}")

def main():
    data_dir = "out_drift"
    pattern = "spec_*.npz"
    normalize = False   
    tag = "saved"
    
    pattern_path = os.path.join(data_dir, pattern)
    data = load_spectra(pattern_path)

    plot_overlay_initial(data, normalize=normalize, title="Initial spectra (t=0)", outdir=data_dir, tag=tag)
    plot_overlay_final(data, normalize=normalize, title="Final spectra (t=t_final)", outdir=data_dir, tag=tag)

if __name__ == "__main__":
    main()
