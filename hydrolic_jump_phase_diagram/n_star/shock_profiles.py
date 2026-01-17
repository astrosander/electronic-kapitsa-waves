"""
% !TEX program = pdflatex

Publication-ready script for generating shock profiles n(x) with different 
momentum values p. The figure is optimized for LaTeX inclusion.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import os

# ========================================================================
# Publication-quality matplotlib settings
# ========================================================================
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman']
plt.rcParams['legend.frameon'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 11
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for LaTeX compatibility
plt.rcParams['ps.fonttype'] = 42

# ========================================================================
# Physical parameters
# ========================================================================
U = 1.0  # Velocity parameter
w = 1.0  # Width parameter for gamma function
p_values = [0.9, 1.0, 1.1]  # Momentum values
n_points = 2500  # Number of points for numerical integration

# ========================================================================
# Color palette for distinct, publication-ready colors
# ========================================================================
# Using distinct colors that are colorblind-friendly and work in grayscale
COLORS = [
    '#2E86AB',  # Blue
    '#A23B72',  # Magenta/Purple
    '#F18F01',  # Orange
]

# ========================================================================
# Mathematical functions
# ========================================================================
def gamma(n, w):
    """
    Damping function: gamma(n) = exp(-|n|/w)
    
    Parameters:
    -----------
    n : array_like
        Density values
    w : float
        Width parameter
        
    Returns:
    --------
    array_like
        Gamma values
    """
    return np.exp(-np.abs(n) / w)


def dx_dn(n, U, p, w):
    """
    Derivative dx/dn for the shock profile equation.
    
    Parameters:
    -----------
    n : array_like
        Density values
    U : float
        Velocity parameter
    p : float
        Momentum parameter
    w : float
        Width parameter
        
    Returns:
    --------
    array_like
        dx/dn values
    """
    return -(U * n - p**2 / n**2) / (gamma(n, w) * p)


def n_star(U, p):
    """
    Singular density point: n* = (p^2/U)^(1/3)
    
    Parameters:
    -----------
    U : float
        Velocity parameter
    p : float
        Momentum parameter
        
    Returns:
    --------
    float
        Singular density value
    """
    return (p**2 / U) ** (1/3)


# ========================================================================
# Generate and save figure
# ========================================================================
def create_figure(save_path=None, show_plot=False):
    """
    Create publication-ready figure of shock profiles.
    
    Parameters:
    -----------
    save_path : str, optional
        Path to save the figure (default: None, no saving)
    show_plot : bool, optional
        Whether to display the plot (default: False)
    """
    # Create figure with appropriate size for LaTeX (column width ~3.5-4 in)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    
    # Plot profiles for each momentum value with distinct colors
    for i, p in enumerate(p_values):
        ns = n_star(U, p)
        
        # Density range scaled to the singular point
        n_min = 1.001 * ns
        n_max = 1.1 * ns
        n = np.linspace(n_min, n_max, n_points)
        
        # Integrate x(n) using cumulative trapezoidal rule
        x = cumtrapz(dx_dn(n, U, p, w), n, initial=0.0)
        
        # Shift x so the singular point aligns visually
        x -= x[0]
        
        # Plot profile with distinct color
        color = COLORS[i % len(COLORS)]
        ax.plot(x, n, label=f"$p={p}$", linewidth=1.2, color=color)
        
        # Mark singular point with matching color (lighter shade)
        ax.axhline(ns, linestyle="--", linewidth=0.8, color=color, alpha=0.4)
    
    # Vertical reference line at x=0
    ax.axvline(0, linestyle="--", color="black", linewidth=0.8, alpha=0.5)
    
    # Axis labels
    ax.set_xlabel("$x$")
    ax.set_ylabel("$n(x)$")
    ax.set_xlim(-0.03, 0.01)
    
    # Legend
    ax.legend(loc='best', frameon=False)
    
    # Tight layout for publication
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
        print(f"Figure saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


# ========================================================================
# Main execution
# ========================================================================
if __name__ == "__main__":
    # Determine output directory (same as script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "shock_profiles.pdf")
    
    # Create and save figure
    create_figure(save_path=output_path, show_plot=True)
