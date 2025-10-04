import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the mass injection directory to the path
sys.path.append('mass injection')
from plot_from_data import load_data

def plot_u_true_vs_time(filename):
    """Plot u_true vs time for a single data file"""
    data = load_data(filename)
    
    # Extract data
    n_t = data['n_t']
    t = data['t']
    L = data['L']
    u_d = data['meta'].get('u_d', 0.0)
    
    # Calculate u_true for each time step
    u_true_values = []
    time_values = []
    
    # Use consecutive time steps for correlation
    for i in range(len(t) - 1):
        t1 = t[i]
        t2 = t[i + 1]
        
        n1 = n_t[:, i]
        n2 = n_t[:, i + 1]
        
        # Calculate spatial correlation to find optimal shift
        max_shift = int(L // 4)
        shifts = np.arange(-max_shift, max_shift + 1)
        correlations = []
        
        for shift in shifts:
            shift = int(shift)  # Ensure shift is an integer
            if shift == 0:
                corr = np.corrcoef(n1, n2)[0, 1]
            elif shift > 0:
                corr = np.corrcoef(n1[shift:], n2[:-shift])[0, 1]
            else:
                corr = np.corrcoef(n1[:shift], n2[-shift:])[0, 1]
            correlations.append(corr)
        
        correlations = np.array(correlations)
        max_idx = np.argmax(correlations)
        shift_opt = shifts[max_idx]
        u_true = shift_opt / (t2 - t1)
        
        u_true_values.append(u_true)
        time_values.append(t1)  # Use the earlier time point
    
    u_true_values = np.array(u_true_values)
    time_values = np.array(time_values)
    
    # Plot u_true vs time
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, np.abs(u_true_values), 'b-', linewidth=2, label=f'|u_true| (u_d = {u_d:.1f})')
    # plt.plot(time_values, u_true_values, 'r--', linewidth=1, alpha=0.7, label=f'u_true (u_d = {u_d:.1f})')
    plt.xlabel('Time t')
    plt.ylabel('$u_{\\text{true}}$')
    plt.title(f'u_true vs Time (u_d = {u_d:.1f})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    outdir = data['meta'].get('outdir', 'out_drift')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/u_true_vs_time.png", dpi=160)
    # plt.show()
    plt.close()
    
    # Plot correlation at final time step
    plt.figure(figsize=(12, 5))
    
    # Get final time step data
    n1_final = n_t[:, -2]  # Second to last time step
    n2_final = n_t[:, -1]  # Last time step
    t1_final = t[-2]
    t2_final = t[-1]
    
    # Calculate correlation for different shifts
    max_shift = int(L // 1)  # Increased range
    # Use very small and dense shifts for high resolution
    shifts = np.arange(-max_shift, max_shift + 0.1, 0.1)
    correlations = []
    
    for shift in shifts:
        shift_int = int(shift)
        shift_frac = shift - shift_int
        
        # Calculate correlation for integer shift
        if shift_int == 0:
            corr_int = np.corrcoef(n1_final, n2_final)[0, 1]
        elif shift_int > 0:
            corr_int = np.corrcoef(n1_final[shift_int:], n2_final[:-shift_int])[0, 1]
        else:
            corr_int = np.corrcoef(n1_final[:shift_int], n2_final[-shift_int:])[0, 1]
        
        # If fractional part is significant, interpolate
        if abs(shift_frac) > 1e-6:
            if shift_int == 0:
                if shift_frac > 0:
                    corr_next = np.corrcoef(n1_final[1:], n2_final[:-1])[0, 1]
                else:
                    corr_next = np.corrcoef(n1_final[:-1], n2_final[1:])[0, 1]
            elif shift_int > 0:
                corr_next = np.corrcoef(n1_final[shift_int+1:], n2_final[:-shift_int-1])[0, 1]
            else:
                corr_next = np.corrcoef(n1_final[:shift_int-1], n2_final[-shift_int+1:])[0, 1]
            
            # Linear interpolation
            corr = corr_int + shift_frac * (corr_next - corr_int)
        else:
            corr = corr_int
            
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    # Plot correlation vs shift
    plt.subplot(1, 2, 1)
    # With dense data, just plot the line without markers for clarity
    plt.plot(shifts, correlations, 'b-', linewidth=2, label='Correlation')
    
    # Mark every 10th point for reference
    step = max(1, len(shifts) // 20)  # Show ~20 points
    plt.plot(shifts[::step], correlations[::step], 'ro', markersize=3, alpha=0.7, label='Sample points')
    
    plt.xlabel('Shift')
    plt.ylabel('Correlation')
    plt.title(f'Correlation vs Shift (t={t1_final:.2f} to {t2_final:.2f})')
    plt.grid(True)
    
    # Find and mark optimal shift
    max_idx = np.argmax(correlations)
    shift_opt = shifts[max_idx]
    plt.axvline(shift_opt, color='red', linestyle='--', alpha=0.7, label=f'Optimal shift = {shift_opt}')
    
    # Find and mark minimum correlation
    min_idx = np.argmin(correlations)
    shift_min = shifts[min_idx]
    plt.axvline(shift_min, color='green', linestyle='--', alpha=0.7, label=f'Min shift = {shift_min}')
    
    plt.legend()
    
    # Plot n1 and n2 profiles
    plt.subplot(1, 2, 2)
    x = np.linspace(0, L, len(n1_final), endpoint=False)
    plt.plot(x, n1_final, 'b-', linewidth=2, label=f'n(t={t1_final:.2f})')
    plt.plot(x, n2_final, 'r-', linewidth=2, label=f'n(t={t2_final:.2f})')
    plt.xlabel('Position x')
    plt.ylabel('Density n')
    plt.title('Density profiles at final time steps')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/correlation_analysis_t_final.png", dpi=160)
    plt.show()
    
    print(f"Final time correlation analysis:")
    print(f"  Time range: {t1_final:.3f} to {t2_final:.3f}")
    print(f"  Optimal shift: {shift_opt}")
    print(f"  Max correlation: {correlations[max_idx]:.4f}")
    print(f"  Calculated u_true: {shift_opt / (t2_final - t1_final):.4f}")
    
    return time_values, u_true_values

def main():
    """Example usage of plot_u_true_vs_time"""
    filename = "mass injection/multiple_u_d/out_drift_ud2.0250/data_m01_ud2.025.npz"
    print(f"Analyzing file: {filename}")
    
    try:
        time_values, u_true_values = plot_u_true_vs_time(filename)
        print(f"Calculated u_true for {len(time_values)} time points")
        print(f"Time range: {time_values[0]:.2f} to {time_values[-1]:.2f}")
        print(f"u_true range: {np.min(u_true_values):.4f} to {np.max(u_true_values):.4f}")
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    main()
