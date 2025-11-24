import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- appearance ---
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 16,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# --- physical constants (SI) ---
e   = 1.602176634e-19      # C
m_e = 9.10938356e-31       # kg
m   = 0.04 * m_e           # effective mass
hbar = 1.054e-34           # J·s

# =========================================================
# Density mapping (corrected):
#   black -> n1 = 10^12 cm^-2  (lowest density, highest rho)
#   red   -> n2 = 10^13 cm^-2
#   blue  -> n3 = 10^14 cm^-2  (highest density, lowest rho)
# =========================================================
n1_cm2 = 1e12   # black
n2_cm2 = 1e13   # red
n3_cm2 = 1e14   # blue

# convert to m^-2: 1 cm^-2 = 1e4 m^-2
n1 = n1_cm2 * 1e4
n2 = n2_cm2 * 1e4
n3 = n3_cm2 * 1e4

# --- load rho(x) in ohms (sheet resistance) ---
blue  = pd.read_csv(r"D:\Downloads\blue.csv",  sep=",", header=None, names=["x", "rho"])   # n3
red   = pd.read_csv(r"D:\Downloads\red.csv",   sep=",", header=None, names=["x", "rho"])   # n2
black = pd.read_csv(r"D:\Downloads\black.csv", sep=",", header=None, names=["x", "rho"])   # n1

for df in [blue, red, black]:
    df["x"]   = df["x"].astype(float)
    df["rho"] = df["rho"].astype(float)

# --- compute gamma = rho * n * e^2 / m for each dataset ---
black["gamma"] = black["rho"] * n1 * e**2 / m  # n1 = 1e12 cm^-2
red["gamma"]   = red["rho"]   * n2 * e**2 / m  # n2 = 1e13 cm^-2
blue["gamma"]  = blue["rho"]  * n3 * e**2 / m  # n3 = 1e14 cm^-2

# ---------------------------------------------------------
# 1) rho(x) plot (same as before, optional)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.loglog(black["x"], black["rho"], color="black",
           label=r"$\rho,\ n=10^{12}\ \mathrm{cm^{-2}}$")
plt.loglog(red["x"],   red["rho"],   color="red",
           label=r"$\rho,\ n=10^{13}\ \mathrm{cm^{-2}}$")
plt.loglog(blue["x"],  blue["rho"],  color="blue",
           label=r"$\rho,\ n=10^{14}\ \mathrm{cm^{-2}}$")

plt.ylim(1e-6, 1e2)
plt.xlim(5, 200)
plt.xlabel(r"$x$")
plt.ylabel(r"$\rho\ [\Omega]$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rho_plot_corrected.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 2) gamma(x) plot (same as before, optional)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.loglog(black["x"], black["gamma"], color="black",
           label=r"$\gamma,\ n=10^{12}\ \mathrm{cm^{-2}}$")
plt.loglog(red["x"],   red["gamma"],   color="red",
           label=r"$\gamma,\ n=10^{13}\ \mathrm{cm^{-2}}$")
plt.loglog(blue["x"],  blue["gamma"],  color="blue",
           label=r"$\gamma,\ n=10^{14}\ \mathrm{cm^{-2}}$")

plt.xlabel(r"$x$")
plt.ylabel(r"$\gamma\ [\mathrm{s^{-1}}]$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gamma_plot_corrected.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 3) lambda_*(x) from Eq. (8): lambda_* = 4 pi u0 / gamma
# ---------------------------------------------------------

def u0_from_n(n):
    """
    u0(n) = v_F / sqrt(2),
    v_F = ħ k_F / m, k_F = sqrt(pi n)
    """
    kF = np.sqrt(np.pi * n)
    vF = hbar * kF / m
    # print(vF)
    return vF / np.sqrt(2.0)

u0_n1 = u0_from_n(n1)
u0_n2 = u0_from_n(n2)
u0_n3 = u0_from_n(n3)

# lambda_* (in meters)
lambda1 = 4.0 * np.pi * u0_n1 / black["gamma"].to_numpy()
lambda2 = 4.0 * np.pi * u0_n2 / red["gamma"].to_numpy()
lambda3 = 4.0 * np.pi * u0_n3 / blue["gamma"].to_numpy()

# convert to micrometers for nicer numbers
lambda1_um = lambda1 * 1e6
lambda2_um = lambda2 * 1e6
lambda3_um = lambda3 * 1e6

plt.figure(figsize=(8, 6))
plt.loglog(black["x"], lambda1_um, color="black",
           label=r"$\lambda^\ast,\ n=10^{12}\ \mathrm{cm^{-2}}$", linewidth=2)
plt.loglog(red["x"],   lambda2_um, color="red",
           label=r"$\lambda^\ast,\ n=10^{13}\ \mathrm{cm^{-2}}$", linewidth=2)
plt.loglog(blue["x"],  lambda3_um, color="blue",
           label=r"$\lambda^\ast,\ n=10^{14}\ \mathrm{cm^{-2}}$", linewidth=2)

print("lambda1_um=", lambda1_um)

plt.ylim(1e2, 1e8)
plt.xlabel(r"$x$")
plt.ylabel(r"$\lambda^\ast\ [\mu\mathrm{m}]$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lambda_star_vs_x.png", dpi=300)
# plt.show()
plt.close()
