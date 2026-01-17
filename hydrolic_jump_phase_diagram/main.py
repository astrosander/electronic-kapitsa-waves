import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# plt.rcParams['text.usetex'] = True
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
    'L_device': 2e8,      # used in shock: x_* < L_device
    'L_lambda': 2e8,      # used in lambda condition: L_lambda < lambda_*
    'U': 1,
    'm': 1.0,
    'echarge': 1.0,
    'gamma0': 1e3,
    'w': 1,             # for gamma_exp
    'nmin': 20,
    'nmax': 25,
    'Imax': 10,
    'grid_n': 401,
    'grid_I': 401,
}

L_device = params['L_device']
L_lambda = params['L_lambda']
U = params['U']
m = params['m']
echarge = params['echarge']
gamma0 = params['gamma0']
w = params['w']
nmin = params['nmin']
nmax = params['nmax']
Imax = params['Imax']

def p_of_I(I):
    return m * I / echarge

def gamma_exp(n):
    return gamma0 * np.exp(-n / w)

def dgamma_exp_dn(n):
    return -gamma0 / w * np.exp(-n / w)


def distance_to_sonic(n0, I, gamma_fn, npts=2000):
    if I == 0:
        return np.inf, np.nan, "no-flow"

    p = abs(p_of_I(I))
    n_star = (p * p / (m * U)) ** (1.0 / 3.0)

    if n0 <= n_star:
        return 0.0, n_star, "inlet-sonic-or-supersonic"

    eps = 1e-8
    n1 = n_star * (1.0 + eps)
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

I_vals = np.linspace(0.0, Imax, params['grid_I'])

N, Igrid = np.meshgrid(n0_vals, I_vals)

gamma_vals = gamma_exp(N)
dgamma_dn_vals = dgamma_exp_dn(N)

u0 = np.sqrt(U * N / m)

dgamma_nonzero = np.abs(dgamma_dn_vals) > 1e-10
threshold = np.zeros_like(N)
threshold[dgamma_nonzero] = echarge * u0[dgamma_nonzero] * gamma_vals[dgamma_nonzero] / np.abs(dgamma_dn_vals[dgamma_nonzero])
condition_mask = np.where((dgamma_nonzero) & (Igrid > threshold), 1.0, 0.0)

shock_required_vec = np.vectorize(lambda n, I: shock_required(n, I, gamma_exp), otypes=[float])
shock_mask = shock_required_vec(N, Igrid).astype(float)

lambda_star = 4 * np.pi * u0 / gamma_vals
x_condition_mask = np.where(lambda_star < L_lambda, 1.0, 0.0)

combined_mask = shock_mask + 2 * condition_mask + 4 * x_condition_mask
norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), 256)
pcm = ax.pcolormesh(
    n0_vals, I_vals, combined_mask,
    shading="auto",
    cmap="rainbow",
    norm=norm
)
ax.set_xlim(n0_vals.min(), n0_vals.max())
ax.set_ylim(I_vals.min(), I_vals.max())

ax.contour(N, Igrid, shock_mask, levels=[0.5], linewidths=3, colors='black')
ax.contour(N, Igrid, condition_mask, levels=[0.5], linewidths=2, colors='white', linestyles='--')
ax.contour(N, Igrid, x_condition_mask, levels=[0.5], linewidths=2, colors='yellow', linestyles='-.')

code2_fraction = np.sum(combined_mask == 2.0) / combined_mask.size
print(f"Code 2 fraction = {code2_fraction:.3f}")

ax.set_ylabel("current $I$")
ax.set_xlabel("density $n_0$")
plt.tight_layout(h_pad=0)
plt.savefig("phase_diagram.png", dpi=300, bbox_inches="tight")
plt.savefig("phase_diagram.svg", dpi=300, bbox_inches="tight")

fig_legend = plt.figure(figsize=(10, 6))
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis('off')

cmap = plt.cm.get_cmap('rainbow')

table_data = []
for code in range(8):
    is_shock = (code % 2) == 1
    condition1 = ((code // 2) % 2) == 1
    condition2 = ((code // 4) % 2) == 1
    color = cmap(code / 7.0)
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
ax_legend.text(x_cond2, y_start - 0.04, r'$(L < \lambda_* = 4\pi u_0/\gamma(n))$', 
              fontsize=10, ha='center', style='italic')
ax_legend.text(x_shock, y_start - 0.04, r'$(x_* < L)$', 
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
