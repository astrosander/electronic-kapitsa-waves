import numpy as np
import matplotlib.pyplot as plt

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

L = 1000.0
U = 100
m = 1.0
echarge = 1.0

def p_of_I(I):
    return m * I / echarge

gamma0 = 1.0

def gamma_const(n):
    return gamma0 * np.ones_like(n)

w = 0.5
def gamma_exp(n):
    return gamma0 * np.exp(-n / w)

nref = 1.0
def gamma_power(n):
    return gamma0 / (1 + n**2 / nref**2)

def dgamma_const_dn(n):
    return np.zeros_like(n)

def dgamma_exp_dn(n):
    return -gamma0 / w * np.exp(-n / w)

def dgamma_power_dn(n):
    return -2 * gamma0 * n / (nref**2 * (1 + n**2 / nref**2)**2)

GAMMAS = [
    (r"$\gamma = \gamma_0$", gamma_const, dgamma_const_dn),
    (rf"$\gamma = \gamma_0 e^{{-n/w}}$", gamma_exp, dgamma_exp_dn),
    (rf"$\gamma = \frac{{\gamma_0}}{{1+n^2/n_0^2}}$", gamma_power, dgamma_power_dn),
]

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
    return x_star < L

def get_x_star(n0, I, gamma_fn):
    x_star, n_star, status = distance_to_sonic(n0, I, gamma_fn)
    if status == "no-flow":
        return np.inf
    if status == "inlet-sonic-or-supersonic":
        return 0.0
    return x_star

nmax=5.0
# Imax=5.0
n0_vals = np.linspace(0, nmax, 26*4)
I_vals  = np.linspace(0.0, 100, 24*4)

N, Igrid = np.meshgrid(n0_vals, I_vals)

fig, axes = plt.subplots(len(GAMMAS), 1, figsize=(6.5, 3.5 * len(GAMMAS)), sharex=True)

if len(GAMMAS) == 1:
    axes = [axes]

for ax, (label, gfn, dgfn) in zip(axes, GAMMAS):
    gamma_vals = gfn(N)
    dgamma_dn_vals = dgfn(N)
    
    u0 = np.sqrt(U * N / m)
    
    dgamma_nonzero = np.abs(dgamma_dn_vals) > 1e-10
    threshold = np.zeros_like(N)
    threshold[dgamma_nonzero] = u0[dgamma_nonzero] * gamma_vals[dgamma_nonzero] / np.abs(dgamma_dn_vals[dgamma_nonzero])
    condition_mask = np.where((dgamma_nonzero) & (Igrid > threshold), 1.0, 0.0)
    
    shock_required_vec = np.vectorize(lambda n, I: shock_required(n, I, gfn), otypes=[float])
    shock_mask = shock_required_vec(N, Igrid).astype(float)
    
    lambda_star = 4 * np.pi * u0 / gamma_vals
    print(lambda_star)
    x_condition_mask = np.where(L < lambda_star, 1.0, 0.0)

    combined_mask = shock_mask + 2 * condition_mask + 4 * x_condition_mask
    pcm = ax.pcolormesh(
        n0_vals, I_vals, combined_mask,
        shading="auto",
        cmap="rainbow"
    )
    ax.set_title(label)
    ax.set_xlim(n0_vals.min(), n0_vals.max())
    ax.set_ylim(I_vals.min(), I_vals.max())

    ax.contour(N, Igrid, shock_mask, levels=[0.5], linewidths=3, colors='black')
    ax.contour(N, Igrid, condition_mask, levels=[0.5], linewidths=2, colors='white', linestyles='--')
    ax.contour(N, Igrid, x_condition_mask, levels=[0.5], linewidths=2, colors='yellow', linestyles='-.')

for ax in axes:
    ax.set_ylabel("current $I$")
axes[-1].set_xlabel("density $n_0$")
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

ax_legend.text(x_cond1, y_start - 0.04, r'$(I > u_0 \cdot \gamma / |d\gamma/dn|)$', 
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
    
    ax_legend.text(x_code, y_pos, str(data['code']), fontsize=12, ha='center', va='center')
    ax_legend.text(x_cond1, y_pos, 'Yes' if data['condition1'] else 'No', 
                  fontsize=12, ha='center', va='center')
    ax_legend.text(x_cond2, y_pos, 'Yes' if data['condition2'] else 'No', 
                  fontsize=12, ha='center', va='center')
    ax_legend.text(x_shock, y_pos, 'Yes' if data['is_shock'] else 'No', 
                  fontsize=12, ha='center', va='center')

ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)
ax_legend.set_title("Color Code Legend", fontsize=16, pad=20)

plt.tight_layout()
plt.savefig("color_legend.png", dpi=300, bbox_inches="tight")
plt.savefig("color_legend.svg", dpi=300, bbox_inches="tight")
plt.show()
