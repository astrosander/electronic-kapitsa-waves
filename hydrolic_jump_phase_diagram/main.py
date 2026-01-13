import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['figure.titlesize'] = 22

L = 1.0
U = 1.0
m = 1.0
echarge = 1.0

def p_of_I(I):
    return m * I / echarge

gamma0 = 1.0

def gamma_const(n):
    return gamma0 * np.ones_like(n)

w = 1.0
def gamma_exp(n):
    return gamma0 * np.exp(-n / w)

nref = 1.0
def gamma_power(n):
    return gamma0 / (1 + n**2 / nref**2)

GAMMAS = [
    (r"$\gamma = \gamma_0$", gamma_const),
    (rf"$\gamma = \gamma_0 e^{{-n/w}}$", gamma_exp),
    (rf"$\gamma = \frac{{\gamma_0}}{{1+n^2/n_0^2}}$", gamma_power),
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

n0_vals = np.linspace(0.2, 5.0, 260*2)
I_vals  = np.linspace(0.0, 6.0, 240*2)

N0, Igrid = np.meshgrid(n0_vals, I_vals)

fig, axes = plt.subplots(1, len(GAMMAS), figsize=(4.8 * len(GAMMAS), 4.4), sharey=True)

if len(GAMMAS) == 1:
    axes = [axes]

for ax, (label, gfn) in zip(axes, GAMMAS):
    shock_mask = np.zeros_like(N0, dtype=float)

    for iy in range(Igrid.shape[0]):
        for ix in range(N0.shape[1]):
            shock_mask[iy, ix] = 1.0 if shock_required(N0[iy, ix], Igrid[iy, ix], gfn) else 0.0

    pcm = ax.pcolormesh(
        n0_vals, I_vals, shock_mask,
        shading="auto",
        cmap="rainbow"
    )
    ax.set_title(label)
    ax.set_xlabel("density $n_0$")
    ax.set_xlim(n0_vals.min(), n0_vals.max())
    ax.set_ylim(I_vals.min(), I_vals.max())

    ax.contour(N0, Igrid, shock_mask, levels=[0.5], linewidths=1.5)

axes[0].set_ylabel("injected current $I$")
plt.tight_layout(w_pad=0.5)
plt.savefig("phase_diagram.png", dpi=300, bbox_inches="tight")
plt.savefig("phase_diagram.svg", dpi=300, bbox_inches="tight")
plt.show()
