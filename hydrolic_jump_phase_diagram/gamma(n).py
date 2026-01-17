import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20


# parameters
L = 2e8
U = 1.0
m = 1.0
e = 1.0

nmin = 21.6
nmax = 22.0
grid_n = 1000

# n grid
n_vals = np.linspace(nmin, nmax, grid_n)

def u0_of(n):
    return np.sqrt(U * n / m)

# gamma parameters
gamma_min = 3e-7   # ~ 4π u0 / L at n ~ 22
Delta = 9.0        # gamma(0) ~ 10x gamma_min
nc = 21.8
s = 0.05

def gamma_fn(n):
    n = np.asarray(n)
    return gamma_min * (1 + 0.5 * Delta * (1 - np.tanh((n - nc) / s)))

# compute gamma
gamma_vals = gamma_fn(n_vals)

# plot
plt.figure(figsize=(6*1.2, 4))
plt.plot(n_vals, gamma_vals, lw=2, color="blue")
plt.xlabel(r"$n$")
plt.ylabel(r"$\gamma(n)$")
plt.title(r"$\gamma(n)=\gamma_{\min}\left[1+\frac{\Delta}{2}\left(1-\tanh\left(\frac{n-n_c}{s}\right)\right)\right]$", fontsize=16)
plt.savefig("img.pdf")
plt.grid(True)
plt.tight_layout()
plt.show()
