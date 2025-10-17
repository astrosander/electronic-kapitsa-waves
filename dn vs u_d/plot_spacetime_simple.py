#!/usr/bin/env python3
"""
Simple script to plot spacetime diagrams for n and p from simulation data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Set up matplotlib for better appearance
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral", 
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

def main():
    """Main function to plot spacetime diagrams."""
    
    # Specify the data file path
    data_file = "multiple_u_d/w=0.05_modes_3_5_7_L10(lambda=0.0, sigma=-1.0, seed_amp_n=0.03, seed_amp_p=0.03)/out_drift_ud1p9000/data_m07_ud1p9000_ud1.8999999999999995.npz"
    
    print("=" * 60)
    print("SPACETIME DIAGRAM PLOTTING")
    print("=" * 60)
    print(f"Data file: {data_file}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        print("Please check the file path and try again.")
        return
    
    # Load data with allow_pickle=True
    try:
        print("Loading data...")
        data = np.load(data_file, allow_pickle=True)
        print(f"Successfully loaded data!")
        print(f"Available keys: {list(data.keys())}")
        
        # Extract data arrays
        n_t = data['n_t']  # Shape: (Nx, Nt)
        p_t = data['p_t']  # Shape: (Nx, Nt)
        t = data['t']      # Time array
        L = float(data['L'])  # Domain size
        Nx = int(data['Nx'])  # Number of spatial points
        
        print(f"Data shape: n_t={n_t.shape}, p_t={p_t.shape}")
        print(f"Time range: {t[0]:.3f} to {t[-1]:.3f}")
        print(f"Domain size: L={L:.3f}")
        print(f"Spatial resolution: Nx={Nx}")
        
        # Create spatial grid
        x = np.linspace(0, L, Nx, endpoint=False)
        print(f"Spatial grid: x with {len(x)} points, range: [{x[0]:.3f}, {x[-1]:.3f}]")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    u_d = 1.9  # From the filename
    
    # Create output directory
    output_dir = os.path.dirname(data_file)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating plots...")
    
    # Plot 1: n(x,t) spacetime diagram (rotated 90 degrees)
    print("  Creating n(x,t) spacetime diagram...")
    plt.figure(figsize=(12, 8))
    plt.imshow(n_t.T, aspect='auto', origin='lower', 
               extent=[x[0], x[-1], t[0], t[-1]], 
               cmap='inferno', interpolation='bilinear')
    plt.colorbar(label='$n(x,t)$')
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(f'Density $n(x,t)$ spacetime diagram (u_d = {u_d:.3f})')
    plt.tight_layout()
    
    n_plot_file = os.path.join(output_dir, f'spacetime_n_ud{u_d:.3f}.png')
    plt.savefig(n_plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {n_plot_file}")
    plt.close()
    
    # Plot 2: p(x,t) spacetime diagram (rotated 90 degrees)
    print("  Creating p(x,t) spacetime diagram...")
    plt.figure(figsize=(12, 8))
    plt.imshow(p_t.T, aspect='auto', origin='lower', 
               extent=[x[0], x[-1], t[0], t[-1]], 
               cmap='viridis', interpolation='bilinear')
    plt.colorbar(label='$p(x,t)$')
    plt.ylabel('$t$')
    plt.xlabel('$x$')
    plt.title(f'Momentum $p(x,t)$ spacetime diagram (u_d = {u_d:.3f})')
    plt.tight_layout()
    
    p_plot_file = os.path.join(output_dir, f'spacetime_p_ud{u_d:.3f}.png')
    plt.savefig(p_plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {p_plot_file}")
    plt.close()
    
    # Plot 3: Combined n and p spacetime diagrams (side by side)
    print("  Creating combined spacetime diagram...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # n(x,t) plot (rotated 90 degrees)
    im1 = ax1.imshow(n_t.T, aspect='auto', origin='lower', 
                     extent=[x[0], x[-1], t[0], t[-1]], 
                     cmap='inferno', interpolation='bilinear')
    ax1.set_ylabel('$t$')
    ax1.set_xlabel('$x$')
    ax1.set_title(f'Density $n(x,t)$ (u_d = {u_d:.3f})')
    plt.colorbar(im1, ax=ax1, label='$n(x,t)$')
    
    # p(x,t) plot (rotated 90 degrees)
    im2 = ax2.imshow(p_t.T, aspect='auto', origin='lower', 
                     extent=[x[0], x[-1], t[0], t[-1]], 
                     cmap='viridis', interpolation='bilinear')
    ax2.set_ylabel('$t$')
    ax2.set_xlabel('$x$')
    ax2.set_title(f'Momentum $p(x,t)$ (u_d = {u_d:.3f})')
    plt.colorbar(im2, ax=ax2, label='$p(x,t)$')
    
    plt.tight_layout()
    
    combined_plot_file = os.path.join(output_dir, f'spacetime_combined_ud{u_d:.3f}.png')
    plt.savefig(combined_plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {combined_plot_file}")
    plt.close()
    
    # Plot 4: Co-moving frame (rotated 90 degrees)
    print("  Creating co-moving frame diagram...")
    # Create co-moving coordinates: xi = x - u_d * t
    xi_coords = np.zeros_like(n_t)
    for i, t_val in enumerate(t):
        xi_coords[:, i] = x - u_d * t_val
    
    plt.figure(figsize=(12, 8))
    plt.imshow(n_t.T, aspect='auto', origin='lower', 
               extent=[xi_coords.min(), xi_coords.max(), t[0], t[-1]], 
               cmap='inferno', interpolation='bilinear')
    plt.colorbar(label='$n(\\xi,t)$')
    plt.xlabel('$t$')
    plt.ylabel('$\\xi = x - u_d t$')
    plt.title(f'Density $n(\\xi,t)$ in co-moving frame (u_d = {u_d:.3f})')
    plt.tight_layout()
    
    comoving_plot_file = os.path.join(output_dir, f'spacetime_n_comoving_ud{u_d:.3f}.png')
    plt.savefig(comoving_plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {comoving_plot_file}")
    plt.close()
    
    # Plot 5: 2x2 panel with spacetime diagrams and final profiles
    print("  Creating 2x2 panel plot...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: n(x,t) spacetime diagram
    im1 = ax1.imshow(n_t.T, aspect='auto', origin='lower', 
                     extent=[x[0], x[-1], t[0], t[-1]], 
                     cmap='inferno', interpolation='bilinear')
    ax1.set_xlabel('$x$', fontsize=24)
    ax1.set_ylabel('$t$', fontsize=24)
    ax1.set_title('$n(x,t)$', fontsize=24)
    plt.colorbar(im1, ax=ax1, label='$n(x,t)$')
    
    # Panel 2: p(x,t) spacetime diagram
    im2 = ax2.imshow(p_t.T, aspect='auto', origin='lower', 
                     extent=[x[0], x[-1], t[0], t[-1]], 
                     cmap='viridis', interpolation='bilinear')
    ax2.set_xlabel('$x$', fontsize=24)
    ax2.set_ylabel('$t$', fontsize=24)
    ax2.set_title('$p(x,t)$', fontsize=24)
    plt.colorbar(im2, ax=ax2, label='$p(x,t)$')
    
    # Panel 3: n(x) at t=t_final
    n_final = n_t[:, -1]  # Last time step
    ax3.plot(x, n_final, 'b-', linewidth=2, label=f'$n(x)$ at $t={t[-1]:.3f}$')
    ax3.set_xlabel('$x$', fontsize=24)
    ax3.set_ylabel('$n(x)$', fontsize=24)
    # ax3.set_title(f'Final density profile at $t={t[-1]:.3f}$')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Panel 4: p(x) at t=t_final
    p_final = p_t[:, -1]  # Last time step
    ax4.plot(x, p_final, 'r-', linewidth=2, label=f'$p(x)$ at $t={t[-1]:.3f}$')
    ax4.set_xlabel('$x$', fontsize=24)
    ax4.set_ylabel('$p(x)$', fontsize=24)
    # ax4.set_title(f'Final momentum profile at $t={t[-1]:.3f}$')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    final_profiles_file = os.path.join(output_dir, f'final_profiles_ud{u_d:.3f}.png')
    plt.savefig(final_profiles_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {final_profiles_file}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("PLOTTING COMPLETE!")
    print("=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("Generated files:")
    print(f"  - spacetime_n_ud{u_d:.3f}.png")
    print(f"  - spacetime_p_ud{u_d:.3f}.png") 
    print(f"  - spacetime_combined_ud{u_d:.3f}.png")
    print(f"  - spacetime_n_comoving_ud{u_d:.3f}.png")
    print(f"  - final_profiles_ud{u_d:.3f}.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
