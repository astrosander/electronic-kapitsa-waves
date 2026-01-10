import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False
# Publication-ready font sizes

plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20


U = 1.0
m = 1.0
gamma0 = 6.0
w = 1.0

p_list = [0.8, 1.0, 1.2]
n_init = 2.0
x_max = 10.0

def gamma_of_n(n):
    return gamma0 * np.exp(-n / w)

def dn_dx(x, n, p):
    n = float(n)
    denom = U * n - (p**2) / (m * n**2)
    return -(gamma_of_n(n) * p) / denom

def integrate_profile(p):
    n_star = (p**2 / (m * U))**(1/3)

    def event_sonic(x, y):
        n = y[0]
        return U * n**3 - (p**2 / m)
    event_sonic.terminal = True
    event_sonic.direction = -1

    sol = solve_ivp(
        fun=lambda x, y: [dn_dx(x, y[0], p)],
        t_span=(0.0, x_max),
        y0=[n_init],
        events=event_sonic,
        max_step=0.02,
        rtol=1e-8,
        atol=1e-10,
    )
    x_sing = sol.t_events[0][0] if len(sol.t_events[0]) else None
    return sol.t, sol.y[0], n_star, x_sing

plt.figure()
for p in p_list:
    x, n, n_star, x_sing = integrate_profile(p)
    plt.plot(x, n, label=f"p={p:g}")
    plt.axhline(n_star, linewidth=1, color='black', linestyle='--')
    if x_sing is not None:
        plt.axvline(x_sing, linewidth=1, color='purple', linestyle='-.')

plt.xlabel("$x$")
plt.ylabel("$n(x)$")
plt.legend()
plt.tight_layout()
plt.show()

