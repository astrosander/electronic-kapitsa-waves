import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

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

U = 1.0
m = 1.0
L = 1.0
eps = 1e-12

n_min, n_max = 1e-4, 0.1
I_min, I_max = 0.0, 0.1

Nn, NI = 700, 550

def gamma_of_n(n, gamma0=2.5, w=0.04):
    return gamma0 * np.exp(-np.abs(n) / w)

def dgamma_dn(n, gamma0=2.5, w=0.04):
    g = gamma_of_n(n, gamma0=gamma0, w=w)
    return -(np.sign(n) / w) * g

n = np.linspace(n_min, n_max, Nn)
I = np.linspace(I_min, I_max, NI)
NN, II = np.meshgrid(n, I, indexing="xy")

g = gamma_of_n(NN)
gp = dgamma_dn(NN)

u0 = np.sqrt(U * NN / m)                         
lambda_star = (4.0 * np.pi / (g + eps)) * u0     

I_crit = (g * m * u0) / (np.abs(gp) + eps)

unstable = (lambda_star < L) & (II > I_crit)

I_shock = np.sqrt(U) * n**1.5

I_crit_line = (gamma_of_n(n) * m * np.sqrt(U * n / m)) / (np.abs(dgamma_dn(n)) + eps)
lambda_line = (4.0 * np.pi / (gamma_of_n(n) + eps)) * np.sqrt(U * n / m)

plt.figure(figsize=(7.8, 5.8))

norm = BoundaryNorm(np.arange(-0.5, 2.5, 1), 256)
plt.contourf(NN, II, unstable.astype(float), levels=[-0.5, 0.5, 1.5], 
             cmap="rainbow", norm=norm, alpha=0.85)

plt.plot(n, I_crit_line, linewidth=2, linestyle="--", color='white',
         label=r"$\dfrac{\gamma\,m\,u_0}{|\gamma'|}$")

plt.plot(n, I_shock, linewidth=3, linestyle="-", color='orange',
         label=r"$\sqrt{U}\,n^{3/2}$")

plt.plot(n, lambda_line, linewidth=2, linestyle="-.", color='yellow',
         label=r"$\lambda(n)$")

plt.xlim(n_min, n_max)
plt.ylim(I_min, I_max)
plt.xlabel(r"$n$")
plt.ylabel(r"$I$")

legend = plt.legend()
for text in legend.get_texts():
    text.set_color('white')

plt.tight_layout()
plt.show()