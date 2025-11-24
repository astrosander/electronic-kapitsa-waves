import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- appearance ---
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "font.size": 18,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# --- physical constants (SI) ---
e   = 1.602176634e-19      # C
m_e = 9.10938356e-31       # kg
m   = 0.04 * m_e           # effective mass

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

# log n for readability
logn1, logn2, logn3 = np.log([n1, n2, n3])

# --- u0 for three different densities (set your values here) ---
# these are just placeholders; put physical values you want
u0_n1 = 1.0   # u0 at n = 1e12 cm^-2
u0_n2 = 1.0   # u0 at n = 1e13 cm^-2
u0_n3 = 1.0   # u0 at n = 1e14 cm^-2

# --- load rho(x) in ohms (sheet resistance) ---
# file colors reflect how they were digitized, but their densities are as above
blue  = pd.read_csv(r"D:\Downloads\blue.csv",  sep=",", header=None, names=["x", "rho"])   # n3
red   = pd.read_csv(r"D:\Downloads\red.csv",   sep=",", header=None, names=["x", "rho"])   # n2
black = pd.read_csv(r"D:\Downloads\black.csv", sep=",", header=None, names=["x", "rho"])   # n1

for df in [blue, red, black]:
    df["x"]   = df["x"].astype(float)
    df["rho"] = df["rho"].astype(float)

# --- compute gamma = rho * n * e^2 / m for each dataset ---
black["gamma"] = black["rho"] * n1 * e**2 / m  # lowest n
red["gamma"]   = red["rho"]   * n2 * e**2 / m
blue["gamma"]  = blue["rho"]  * n3 * e**2 / m  # highest n

# ---------------------------------------------------------
# 1) rho(x) plot
# ---------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.loglog(black["x"], black["rho"], color="black",
           label=r"$\rho,\ n=10^{12}\ \mathrm{cm^{-2}}$")
plt.loglog(red["x"],   red["rho"],   color="red",
           label=r"$\rho,\ n=10^{13}\ \mathrm{cm^{-2}}$")
plt.loglog(blue["x"],  blue["rho"],  color="blue",
           label=r"$\rho,\ n=10^{14}\ \mathrm{cm^{-2}}$")

plt.ylim(1e-6, 1e2)
plt.xlim(5, 200)
plt.xlabel("$x$")
plt.ylabel(r"$\rho\ [\Omega]$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rho_plot_corrected.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 2) gamma(x) plot
# ---------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.loglog(black["x"], black["gamma"], color="black",
           label=r"$\gamma,\ n=10^{12}\ \mathrm{cm^{-2}}$")
plt.loglog(red["x"],   red["gamma"],   color="red",
           label=r"$\gamma,\ n=10^{13}\ \mathrm{cm^{-2}}$")
plt.loglog(blue["x"],  blue["gamma"],  color="blue",
           label=r"$\gamma,\ n=10^{14}\ \mathrm{cm^{-2}}$")

plt.xlabel("$x$")
plt.ylabel(r"$\gamma\ [\mathrm{s^{-1}}]$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("gamma_plot_corrected.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 3) slopes d log gamma / d log n (n1->n2, n2->n3, n1->n3)
#    using pairwise overlaps in x
# ---------------------------------------------------------

def pair_slopes(df_low, n_low, df_high, n_high):
    """
    Compute x_common and d log(gamma) / d log(n) between
    densities n_low and n_high, using only the overlap of
    df_low and df_high in x.
    """
    # overlap in x for this pair only
    xmin = max(df_low["x"].min(), df_high["x"].min())
    xmax = min(df_low["x"].max(), df_high["x"].max())

    mask = (df_low["x"] >= xmin) & (df_low["x"] <= xmax)
    x_common = np.sort(df_low.loc[mask, "x"].values)

    # sort and interpolate gamma onto x_common
    low_sorted  = df_low.sort_values("x")
    high_sorted = df_high.sort_values("x")

    g_low = np.interp(
        x_common,
        low_sorted["x"].values,
        low_sorted["gamma"].values
    )
    g_high = np.interp(
        x_common,
        high_sorted["x"].values,
        high_sorted["gamma"].values
    )

    logg_low  = np.log(g_low)
    logg_high = np.log(g_high)

    logn_low  = np.log(n_low)
    logn_high = np.log(n_high)

    slopes = (logg_high - logg_low) / (logn_high - logn_low)

    return x_common, slopes

# n1 < n2 < n3  (black, red, blue)
x12, slope_12 = pair_slopes(black, n1, red,  n2)  # n1 -> n2
x23, slope_23 = pair_slopes(red,   n2, blue, n3)  # n2 -> n3
x13, slope_13 = pair_slopes(black, n1, blue, n3)  # n1 -> n3 (full range)
# ---------------------------------------------------------
# 4) u_c(x) = u_0(n) / |d log gamma / d log n|
#     use full-range slope (n1->n3) as R(x)
# ---------------------------------------------------------

# Fermi velocity-based u0(n): u0 = v_F / sqrt(2), v_F = ħ k_F / m, k_F = sqrt(pi n)
hbar = 1.054e-34  # J*s

def u0_from_n(n):
    kF = np.sqrt(np.pi * n)
    vF = hbar * kF / m
    print(vF)
    return vF / np.sqrt(2.0)

def vF_from_n(n):
    kF = np.sqrt(np.pi * n)
    vF = hbar * kF / m
    return vF 

u0_n1 = u0_from_n(n1)  # for n = 1e12 cm^-2
u0_n2 = u0_from_n(n2)  # for n = 1e13 cm^-2
u0_n3 = u0_from_n(n3)  # for n = 1e14 cm^-2

# R(x) = |d log gamma / d log n| over n1->n3
R_x = np.abs(slope_13)

# avoid division by zero
eps = 1e-12
R_safe = np.where(R_x > eps, R_x, np.nan)

u_c_n1 = u0_n1 / R_safe
u_c_n2 = u0_n2 / R_safe
u_c_n3 = u0_n3 / R_safe

plt.figure(figsize=(8, 6))
plt.loglog(x13, u_c_n1, label=r"$u_c,\ n=10^{12}\ \mathrm{cm^{-2}}$", color="black", linewidth=2)
plt.loglog(x13, u_c_n2, label=r"$u_c,\ n=10^{13}\ \mathrm{cm^{-2}}$", color="red", linewidth=2)
plt.loglog(x13, u_c_n3, label=r"$u_c,\ n=10^{14}\ \mathrm{cm^{-2}}$", color="blue", linewidth=2)

plt.loglog(x13, np.full_like(x13, vF_from_n(n1)), label=r"$u_F,\ n=10^{12}\ \mathrm{cm^{-2}}$", color="gray", linewidth=2)
plt.loglog(x13, np.full_like(x13, vF_from_n(n2)), label=r"$u_F,\ n=10^{13}\ \mathrm{cm^{-2}}$", color="orange", linewidth=2)
plt.loglog(x13, np.full_like(x13, vF_from_n(n3)), label=r"$u_F,\ n=10^{14}\ \mathrm{cm^{-2}}$", color="purple", linewidth=2)


plt.ylim(1e5, 1e9)

plt.xlabel(r"$x$")
plt.ylabel(r"$u_c ~[m/s]$")
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("u_c_vs_x.png", dpi=300)
# plt.show()
plt.close()
