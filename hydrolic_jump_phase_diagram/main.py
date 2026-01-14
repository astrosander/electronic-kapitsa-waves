import numpy as np
import matplotlib.pyplot as plt

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

L = 1.0
U = 100000
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

nmax=1.0
n0_vals = np.linspace(0, nmax, 26*4)
I_vals  = np.linspace(0.0, np.sqrt(nmax*U), 24*4)

N0, Igrid = np.meshgrid(n0_vals, I_vals)

fig, axes = plt.subplots(len(GAMMAS), 1, figsize=(6.5, 3.5 * len(GAMMAS)), sharex=True)

if len(GAMMAS) == 1:
    axes = [axes]

for ax, (label, gfn, dgfn) in zip(axes, GAMMAS):
    shock_mask = np.zeros_like(N0, dtype=float)
    condition_mask = np.zeros_like(N0, dtype=float)

    for iy in range(Igrid.shape[0]):
        for ix in range(N0.shape[1]):
            n_val = N0[iy, ix]
            I_val = Igrid[iy, ix]
            shock_mask[iy, ix] = 1.0 if shock_required(n_val, I_val, gfn) else 0.0
            
            gamma_val = gfn(n_val)
            dgamma_dn = dgfn(n_val)
            if np.abs(dgamma_dn) > 1e-10:
                threshold = np.sqrt(n_val * U / m) * np.sqrt(n_val) * n_val * gamma_val / np.abs(dgamma_dn)
                condition_mask[iy, ix] = 1.0 if I_val > threshold else 0.0
            else:
                condition_mask[iy, ix] = 0.0

    combined_mask = shock_mask + 2 * condition_mask
    pcm = ax.pcolormesh(
        n0_vals, I_vals, combined_mask,
        shading="auto",
        cmap="rainbow"
    )
    ax.set_title(label)
    ax.set_xlim(n0_vals.min(), n0_vals.max())
    ax.set_ylim(I_vals.min(), I_vals.max())

    ax.contour(N0, Igrid, shock_mask, levels=[0.5], linewidths=2.5, colors='black')
    ax.contour(N0, Igrid, condition_mask, levels=[0.5], linewidths=2.5, colors='white', linestyles='--')

for ax in axes:
    ax.set_ylabel("current $I$")
axes[-1].set_xlabel("density $n_0$")
plt.tight_layout(h_pad=0)
plt.savefig("phase_diagram.png", dpi=300, bbox_inches="tight")
plt.savefig("phase_diagram.svg", dpi=300, bbox_inches="tight")
plt.show()
