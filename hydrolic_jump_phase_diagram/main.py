import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['figure.titlesize'] = 30

params = {
    'L_device': 2e8,
    'L_lambda': 2e8,
    'U': 1,
    'm': 1.0,
    'echarge': 1.0,

    # keep lambda condition barely satisfied near n~22
    'gamma0': 5.95e-7,      # gamma(n0_ref)=2.975e-7 (tight but OK)
    'beta': 90.0,          # lowers I_min noticeably vs 120
    'n0_ref': 22.0,

    # show the whole wedge
    'nmin': 21.6,
    'nmax': 22.4,
    'Imax': 6.0,

    'grid_n': 401,
    'grid_I': 401,

    'alpha_jump': 8.0,
    'delta_factor': 0,#0.7,
    'alpha_nu': 0.0,
    'nu': 0.0,
}

L_device = params['L_device']
L_lambda = params['L_lambda']
U = params['U']
m = params['m']
echarge = params['echarge']
gamma0 = params['gamma0']
beta = params['beta']
n0_ref = params['n0_ref']
nmin = params['nmin']
nmax = params['nmax']
Imax = params['Imax']

def p_of_I(I):
    return m * I / echarge

def gamma_fn(n):
    """Power-law gamma function: gamma(n) = gamma0 / (1 + n^beta / n0_ref^beta)."""
    return gamma0 / (1.0 + (n / n0_ref)**beta)

def dgamma_dn(n):
    """Derivative of gamma_fn with respect to n."""
    n_power = (n / n0_ref)**beta
    denominator = (1.0 + n_power)**2
    return -gamma0 * beta * (n / n0_ref)**(beta - 1) / (n0_ref * denominator)


def distance_to_sonic(n0, I, gamma_fn, npts=2000):
    if I == 0:
        return np.inf, np.nan, "no-flow"

    p = abs(p_of_I(I))
    n_star = (p * p / (m * U)) ** (1.0 / 3.0)

    if n0 <= n_star:
        return 0.0, n_star, "inlet-sonic-or-supersonic"

    # Use delta_n to define sonic layer width (surgical regularization at singularity)
    delta_n = params.get('delta_n', 0.0)
    n1 = n_star + delta_n
    if n1 >= n0:
        return 0.0, n_star, "inlet-sonic-or-supersonic"
    
    n_grid = np.linspace(n1, n0, npts)

    numerator = U * n_grid - (p * p) / (m * n_grid**2)
    denom = gamma_fn(n_grid) * p
    integrand = numerator / denom

    x_star = np.trapezoid(integrand, n_grid)
    
    return x_star, n_star, "ok"

def shock_required(n0, I, gamma_fn):
    x_star, n_star, status = distance_to_sonic(n0, I, gamma_fn)
    if status == "no-flow":
        return False
    if status == "inlet-sonic-or-supersonic":
        return True
    return x_star < L_device

def get_x_star(n0, I, gamma_fn):
    x_star, n_star, status = distance_to_sonic(n0, I, gamma_fn)
    if status == "no-flow":
        return np.inf
    if status == "inlet-sonic-or-supersonic":
        return 0.0
    return x_star

n0_vals = np.linspace(nmin, nmax, params['grid_n'])

fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.5))

I_vals = np.linspace(0, Imax, params['grid_I'])

N, Igrid = np.meshgrid(n0_vals, I_vals)

gamma_vals = gamma_fn(N)
dgamma_dn_vals = dgamma_dn(N)

u0 = np.sqrt(U * N / m)

dgamma_nonzero = np.abs(dgamma_dn_vals) > 1e-20
threshold = np.zeros_like(N)
threshold[dgamma_nonzero] = echarge * u0[dgamma_nonzero] * gamma_vals[dgamma_nonzero] / np.abs(dgamma_dn_vals[dgamma_nonzero])
condition_mask = np.where((dgamma_nonzero) & (Igrid > threshold), 1.0, 0.0)

# Calculate x_star directly from exact solver
x_star_vec = np.vectorize(lambda n, I: get_x_star(n, I, gamma_fn), otypes=[float])
x_star = x_star_vec(N, Igrid)

# Shock observability: use local damping length with soft transition
# Use local damping length: ell_jump = alpha * u0/gamma + alpha_nu * nu/u0
alpha_jump = params.get('alpha_jump', 1.0)
alpha_nu = params.get('alpha_nu', 0.0)
nu = params.get('nu', 0.0)
delta_factor = params.get('delta_factor', 0.5)

ell_jump_grid = alpha_jump * (u0 / gamma_vals)
if alpha_nu > 0 and nu > 0:
    ell_jump_grid = ell_jump_grid + alpha_nu * (nu / u0)
delta_ell = delta_factor * (u0 / gamma_vals)

# Probability shock is observable (soft transition instead of hard cutoff)
# Clip z to avoid overflow in exp
z = (ell_jump_grid - (L_device - x_star)) / delta_ell
z = np.clip(z, -60.0, 60.0)  # avoids overflow; 60 is plenty for exp
Pshock = 1.0 / (1.0 + np.exp(z))

# Binary masks for phase diagram (using 0.5 threshold)
shock_mask = ((x_star < L_device) & (Pshock > 0.5)).astype(float)
# Everything else counts as "no observable shock"
no_shock_mask = (shock_mask == 0).astype(float)

# Expected "no-shock" weight (probability that shock is NOT observable)
w_no_shock = 1.0 - Pshock

lambda_star = 4 * np.pi * u0 / gamma_vals
x_condition_mask = np.where(lambda_star < L_lambda, 1.0, 0.0)

combined_mask = shock_mask + 2 * condition_mask + 4 * x_condition_mask

# Check intersection of interest: condition1 AND condition2 AND no_shock
# Hard version (binary threshold)
interest_hard = (condition_mask == 1) & (x_condition_mask == 1) & (shock_mask == 0)
interest_fraction_hard = interest_hard.mean()

# Soft/weighted version (weighted by probability that shock is NOT observable)
mask12 = (condition_mask == 1) & (x_condition_mask == 1)
interest_fraction_soft = np.mean(mask12 * w_no_shock)

print(f"Interest region fraction (hard): {interest_fraction_hard:.6f}")
print(f"Interest region fraction (soft-weighted): {interest_fraction_soft:.6f}")

# Diagnostic: mean observability inside cond1&2
print(f"\nObservability diagnostics inside cond1&2:")
print(f"  Mean (1-Pshock) = {np.mean(w_no_shock[mask12]):.6f}")
print(f"  Mean Pshock = {np.mean(Pshock[mask12]):.6f}")
print(f"  Pure cond1&2 area fraction = {mask12.mean():.6f}")

# Diagnostic code to verify inequalities
R = (N / gamma_vals) * np.abs(dgamma_dn_vals)

jmin = echarge * N * u0 / R
sigma = N * echarge**2 / (m * gamma_vals)
jmax = U * N * sigma / (L_device * echarge)

# Find R at n=22 (interpolate from grid - use middle row for representative I)
n_test_diag = 22.0
mid_I_idx = len(I_vals) // 2
R_at_n22 = np.interp(n_test_diag, n0_vals, R[mid_I_idx, :])
print(f"\nDiagnostics:")
print(f"  R at n={n_test_diag} ~ {R_at_n22:.6f}")
print(f"  min(jmax - jmin) over grid = {np.min(jmax - jmin):.10e}")
print(f"  fraction where jmin<j<jmax = {np.mean((Igrid > jmin) & (Igrid < jmax)):.6f}")
print(f"  fraction where lambda*<L = {np.mean(lambda_star < L_lambda):.6f}")
print(f"  fraction no shock = {np.mean(no_shock_mask == 1):.6f}")

# Print exact numeric values at specific points
n0_test = 22.0
I_test1 = 2.2
I_test2 = 3.0

gamma_n0 = gamma_fn(n0_test)
print(f"\nAt n₀ = {n0_test}:")
print(f"  γ(n₀) = {gamma_n0:.10e}")

x_star_1 = get_x_star(n0_test, I_test1, gamma_fn)
x_star_2 = get_x_star(n0_test, I_test2, gamma_fn)
print(f"\nAt n₀ = {n0_test}, I = {I_test1}:")
print(f"  x_* = {x_star_1:.10e}")
print(f"\nAt n₀ = {n0_test}, I = {I_test2}:")
print(f"  x_* = {x_star_2:.10e}")

# Diagnostic: show effect of alpha_jump on observable shock criterion
print(f"\nShock suppression diagnostic (alpha_jump sweep):")
alpha_jump_original = params['alpha_jump']
mask12 = (condition_mask == 1) & (x_condition_mask == 1)
delta_factor = params.get('delta_factor', 0.5)

for a in [0.0, 0.5, 1.0, 2.0, 3.0]:
    params['alpha_jump'] = a
    ell_jump_grid_test = a * (u0 / gamma_vals)
    if alpha_nu > 0 and nu > 0:
        ell_jump_grid_test = ell_jump_grid_test + alpha_nu * (nu / u0)
    
    shock_eff = ((x_star < L_device) & ((L_device - x_star) > ell_jump_grid_test))
    interest_eff = mask12 & (~shock_eff)
    print(f"  alpha_jump={a:0.1f}: interest_fraction={interest_eff.mean():.6f}, shock_fraction={shock_eff.mean():.6f}")
# Restore original value
params['alpha_jump'] = alpha_jump_original

# Comprehensive diagnostic: Mean(Pshock) inside cond1&2 vs alpha_jump
print(f"\nComprehensive alpha_jump diagnostic (Mean Pshock in cond1&2):")
print(f"{'alpha':>6}  {'meanPshock_in12':>15}  {'hard_interest':>13}  {'soft_interest':>13}")
print("-" * 60)
alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

for a in alphas:
    ell_jump_grid_test = a * (u0 / gamma_vals)
    if alpha_nu > 0 and nu > 0:
        ell_jump_grid_test = ell_jump_grid_test + alpha_nu * (nu / u0)
    delta_ell_test = delta_factor * (u0 / gamma_vals)
    
    z_test = (ell_jump_grid_test - (L_device - x_star)) / delta_ell_test
    z_test = np.clip(z_test, -60.0, 60.0)
    Pshock_test = 1.0 / (1.0 + np.exp(z_test))
    
    hard_no_shock = ~((x_star < L_device) & ((L_device - x_star) > ell_jump_grid_test))
    interest_hard = np.mean(mask12 & hard_no_shock)
    
    interest_soft = np.mean(mask12 * (1.0 - Pshock_test))
    
    mean_P_in12 = np.mean(Pshock_test[mask12]) if np.any(mask12) else np.nan
    
    print(f"{a:>6.1f}  {mean_P_in12:>15.3f}  {interest_hard:>13.6f}  {interest_soft:>13.6f}")

# Create discrete colormap with exactly 8 colors matching the legend
base_cmap = plt.cm.get_cmap('rainbow')
colors = [base_cmap(i / 7.0) for i in range(8)]
cmap_discrete = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap_discrete.N)

pcm = ax.pcolormesh(
    n0_vals, I_vals, combined_mask,
    shading="auto",
    cmap=cmap_discrete,
    norm=norm
)
ax.set_xlim(n0_vals.min(), n0_vals.max())
ax.set_ylim(I_vals.min(), I_vals.max())

ax.contour(N, Igrid, shock_mask, levels=[0.5], linewidths=3, colors='black')
ax.contour(N, Igrid, condition_mask, levels=[0.5], linewidths=2, colors='white', linestyles='--')
ax.contour(N, Igrid, x_condition_mask, levels=[0.5], linewidths=2, colors='yellow', linestyles='-.')

# Boundary where L - x_star = ell_jump (observable shock boundary)
# ax.contour(N, Igrid, (L_device - x_star) - ell_jump_grid, levels=[0.0],
#            linewidths=2, colors='cyan', linestyles=':')
# ax.contour(N, Igrid, Pshock, levels=[0.5], linewidths=2, colors='cyan', linestyles='--', alpha=0.7)

# Orange contour for relaxed condition (don't reuse x_condition_mask variable)
# x_condition_mask_relaxed = np.where(lambda_star < L_lambda * 2, 1.0, 0.0)
# ax.contour(N, Igrid, x_condition_mask_relaxed, levels=[0.5], linewidths=2, colors='orange', linestyles='-.')


code2_fraction = np.sum(combined_mask == 2.0) / combined_mask.size
print(f"Code 2 fraction = {code2_fraction:.3f}")

ax.set_ylabel("current $I$")
ax.set_xlabel("density $n_0$")
plt.tight_layout(h_pad=0)
plt.savefig("phase_diagram.png", dpi=300, bbox_inches="tight")
plt.savefig("phase_diagram.pdf", dpi=300, bbox_inches="tight")

fig_legend = plt.figure(figsize=(10, 6))
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis('off')

# Use the same discrete colormap as the plot
base_cmap = plt.cm.get_cmap('rainbow')
colors = [base_cmap(i / 7.0) for i in range(8)]

table_data = []
for code in range(8):
    # Encoding: code = shock_mask*1 + condition_mask*2 + x_condition_mask*4
    # Extract bits: bit0=shock, bit1=condition1, bit2=condition2
    is_shock = (code & 1) == 1  # bit 0
    condition1 = (code & 2) == 2  # bit 1
    condition2 = (code & 4) == 4  # bit 2
    color = colors[code]  # Use exact color from discrete colormap
    table_data.append({
        'code': code,
        'color': color,
        'condition1': condition1,
        'condition2': condition2,
        'is_shock': is_shock
    })

y_start = 0.95
y_step = 0.11
x_color = 0.05
x_code = 0.18
x_cond1 = 0.28
x_cond2 = 0.45
x_shock = 0.62
col_width = 0.12

ax_legend.text(x_color, y_start, 'Color', fontsize=14, fontweight='bold', ha='center')
ax_legend.text(x_code, y_start, 'Code', fontsize=14, fontweight='bold', ha='center')
ax_legend.text(x_cond1, y_start, 'Condition 1', fontsize=14, fontweight='bold', ha='center')
ax_legend.text(x_cond2, y_start, 'Condition 2', fontsize=14, fontweight='bold', ha='center')
ax_legend.text(x_shock, y_start, 'Is Shock', fontsize=14, fontweight='bold', ha='center')

ax_legend.text(x_cond1, y_start - 0.04, r'$(I > e \cdot u_0 \cdot \gamma / |d\gamma/dn|)$', 
              fontsize=10, ha='center', style='italic')
ax_legend.text(x_cond2, y_start - 0.04, r'$(\lambda_* < L_\lambda)$', 
              fontsize=10, ha='center', style='italic')
ax_legend.text(x_shock, y_start - 0.04, r'$(x_* < L_{device})$', 
              fontsize=10, ha='center', style='italic')

for i, data in enumerate(table_data):
    y_pos = y_start - 0.08 - i * y_step
    
    rect = plt.Rectangle((x_color - col_width/2, y_pos - 0.03), col_width, 0.06, 
                        facecolor=data['color'], edgecolor='black', linewidth=1)
    ax_legend.add_patch(rect)
    
    ax_legend.text(x_code, y_pos, str(data['code']), fontsize=24, ha='center', va='center')
    ax_legend.text(x_cond1, y_pos, 'Yes' if data['condition1'] else 'No', 
                  fontsize=24, ha='center', va='center')
    ax_legend.text(x_cond2, y_pos, 'Yes' if data['condition2'] else 'No', 
                  fontsize=24, ha='center', va='center')
    ax_legend.text(x_shock, y_pos, 'Yes' if data['is_shock'] else 'No', 
                  fontsize=24, ha='center', va='center')

ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)
ax_legend.set_title("Color Code Legend", fontsize=16, pad=20)

plt.tight_layout()
plt.savefig("color_legend.png", dpi=300, bbox_inches="tight")
plt.savefig("color_legend.svg", dpi=300, bbox_inches="tight")
plt.show()
