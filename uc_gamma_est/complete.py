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

# for derivatives we want n in ascending order
n_array  = np.array([n1, n2, n3], dtype=float)
logn_arr = np.log(n_array)
logn1, logn2, logn3 = logn_arr  # readability

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
# plt.show()

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
# plt.show()

# ---------------------------------------------------------
# 3) pairwise slopes d log gamma / d log n (n1->n2, n2->n3)
# ---------------------------------------------------------

# choose a common x-grid: overlapping range, use (say) red's x,
# restricted to overlap of all 3 datasets
xmin = max(black["x"].min(), red["x"].min(), blue["x"].min())
xmax = min(black["x"].max(), red["x"].max(), blue["x"].max())

mask_red = (red["x"] >= xmin) & (red["x"] <= xmax)
x_common = np.sort(red.loc[mask_red, "x"].values)

# sort for safe interpolation
black_sorted = black.sort_values("x")
red_sorted   = red.sort_values("x")
blue_sorted  = blue.sort_values("x")

gamma_black_common = np.interp(
    x_common,
    black_sorted["x"].values,
    black_sorted["gamma"].values
)
gamma_red_common = np.interp(
    x_common,
    red_sorted["x"].values,
    red_sorted["gamma"].values
)
gamma_blue_common = np.interp(
    x_common,
    blue_sorted["x"].values,
    blue_sorted["gamma"].values
)

# compute pairwise slopes
slope_12 = []  # between n1 (black) and n2 (red)
slope_23 = []  # between n2 (red)   and n3 (blue)

for g1, g2, g3 in zip(gamma_black_common,
                      gamma_red_common,
                      gamma_blue_common):
    logg1, logg2, logg3 = np.log([g1, g2, g3])

    s12 = (logg2 - logg1) / (logn2 - logn1)  # n1 -> n2
    s23 = (logg3 - logg2) / (logn3 - logn2)  # n2 -> n3

    slope_12.append(s12)
    slope_23.append(s23)
# compute pairwise slopes
slope_12 = []  # between n1 (black) and n2 (red)
slope_23 = []  # between n2 (red)   and n3 (blue)
slope_13 = []  # between n1 (black) and n3 (blue)

for g1, g2, g3 in zip(gamma_black_common,
                      gamma_red_common,
                      gamma_blue_common):
    logg1, logg2, logg3 = np.log([g1, g2, g3])

    # d log gamma / d log n between (n1,n2), (n2,n3), (n1,n3)
    s12 = (logg2 - logg1) / (logn2 - logn1)  # n1 -> n2
    s23 = (logg3 - logg2) / (logn3 - logn2)  # n2 -> n3
    s13 = (logg3 - logg1) / (logn3 - logn1)  # n1 -> n3 (full range)

    slope_12.append(s12)
    slope_23.append(s23)
    slope_13.append(s13)

slope_12 = np.array(slope_12)
slope_23 = np.array(slope_23)
slope_13 = np.array(slope_13)

# pack if you want to inspect
combined = pd.DataFrame({
    "x": x_common,
    "slope_12": slope_12,
    "slope_23": slope_23,
    "slope_13": slope_13,
})
print(combined.head())

# ---------------------------------------------------------
# 3) pairwise slopes d log gamma / d log n (n1->n2, n2->n3, n1->n3)
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
# Combined figure: rho(x), gamma(x), and slopes in 3x1 layout
# ---------------------------------------------------------

# fig, (ax1, ax2, ax3) = plt.subplots(
#     3, 1, figsize=(6, 10), sharex=True  # 3 rows, 1 column
# )

# # --- 1) rho(x) ---
# ax1.loglog(black["x"], black["rho"], color="black",
#            label=r"$\rho,\ n=10^{12}\ \mathrm{cm^{-2}}$")
# ax1.loglog(red["x"],   red["rho"],   color="red",
#            label=r"$\rho,\ n=10^{13}\ \mathrm{cm^{-2}}$")
# ax1.loglog(blue["x"],  blue["rho"],  color="blue",
#            label=r"$\rho,\ n=10^{14}\ \mathrm{cm^{-2}}$")

# ax1.set_ylim(1e-6, 1e2)
# ax1.set_xlim(5, 200)
# ax1.set_ylabel(r"$\rho\ [\Omega]$")
# ax1.grid(True)
# ax1.legend(loc="best")
# ax1.set_title(r"$\rho(x),\ \gamma(x),\ \frac{d\log\gamma}{d\log n}(x)$")

# # --- 2) gamma(x) ---
# ax2.loglog(black["x"], black["gamma"], color="black",
#            label=r"$\gamma,\ n=10^{12}\ \mathrm{cm^{-2}}$")
# ax2.loglog(red["x"],   red["gamma"],   color="red",
#            label=r"$\gamma,\ n=10^{13}\ \mathrm{cm^{-2}}$")
# ax2.loglog(blue["x"],  blue["gamma"],  color="blue",
#            label=r"$\gamma,\ n=10^{14}\ \mathrm{cm^{-2}}$")

# ax2.set_ylabel(r"$\gamma\ [\mathrm{s^{-1}}]$")
# ax2.grid(True)
# ax2.legend(loc="best")

# # --- 3) pairwise slopes d log gamma / d log n ---
# ax3.semilogx(x12, slope_12,
#              color="purple", linewidth=2,
#              label=r"$n=10^{12}\rightarrow10^{13}\ \mathrm{cm^{-2}}$")
# ax3.semilogx(x23, slope_23,
#              color="orange", linewidth=2,
#              label=r"$n=10^{13}\rightarrow10^{14}\ \mathrm{cm^{-2}}$")
# ax3.semilogx(x13, slope_13,
#              color="green", linewidth=2, linestyle="--",
#              label=r"$n=10^{12}\rightarrow10^{14}\ \mathrm{cm^{-2}}$")

# # optional theory line γ ∝ n^{-3/2} → slope = -3/2
# # ax3.axhline(-1.5, color="k", linestyle=":", label=r"theory $=-3/2$")

# ax3.set_xlabel(r"$x$")
# ax3.set_ylabel(r"$\frac{d \log \gamma}{d \log n}$")
# ax3.grid(True)
# ax3.legend(loc="best")

# plt.tight_layout()
# plt.savefig("combined_rho_gamma_slopes_3x1.png", dpi=300)
# plt.show()


# slope_12 = np.array(slope_12)
# slope_23 = np.array(slope_23)

# # pack if you want to inspect
# combined = pd.DataFrame({
#     "x": x_common,
#     "slope_12": slope_12,
#     "slope_23": slope_23,
# })
# print(combined.head())

# # plot slopes vs x
# plt.figure(figsize=(6, 6))
# plt.semilogx(x_common, slope_12,
#              color="purple", linewidth=2, label=r"$n=10^{12}$ and $n=10^{13}$")
# plt.semilogx(x_common, slope_23,
#              color="orange", linewidth=2,
#              label=r"$n=10^{13}$ and $n=10^{14}$")

# # optional theory line for γ ∝ n^{-3/2}  → slope = -3/2
# # plt.axhline(-1.5, color="k", linestyle=":", label=r"theory $=-3/2$")

# plt.xlabel("$x$")
# plt.ylabel(r"$\frac{d \log \gamma}{d \log n}$")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("dloggamma_dlogn_pairwise_corrected.png", dpi=300)
# # plt.show()
# plt.close()

# ---------------------------------------------------------
# Combined figure: rho(x), gamma(x), and slopes in 3x1 layout
# ---------------------------------------------------------

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(18, 6), sharex=True  # 3 rows, 1 column
)

# --- 1) rho(x) ---
ax1.loglog(black["x"], black["rho"], color="black",
           label=r"$\rho,\ n=10^{12}\ \mathrm{cm^{-2}}$")
ax1.loglog(red["x"],   red["rho"],   color="red",
           label=r"$\rho,\ n=10^{13}\ \mathrm{cm^{-2}}$")
ax1.loglog(blue["x"],  blue["rho"],  color="blue",
           label=r"$\rho,\ n=10^{14}\ \mathrm{cm^{-2}}$")

ax1.set_ylim(1e-6, 1e2)
ax1.set_xlim(5, 200)
ax1.set_ylabel(r"$\rho\ [\Omega]$")
ax1.grid(True)
ax1.legend(loc="best")
# ax1.set_title(r"$\rho(x)$")
# ax2.set_title(r"$\gamma(x)$")
# ax3.set_title(r"$\frac{d\log\gamma}{d\log n}(x)$")#,\ 
ax1.set_xlabel(r"$x$")


# --- 2) gamma(x) ---
ax2.loglog(black["x"], black["gamma"], color="black",
           label=r"$\gamma,\ n=10^{12}\ \mathrm{cm^{-2}}$")
ax2.loglog(red["x"],   red["gamma"],   color="red",
           label=r"$\gamma,\ n=10^{13}\ \mathrm{cm^{-2}}$")
ax2.loglog(blue["x"],  blue["gamma"],  color="blue",
           label=r"$\gamma,\ n=10^{14}\ \mathrm{cm^{-2}}$")

ax2.set_ylabel(r"$\gamma\ [\mathrm{s^{-1}}]$")
ax2.grid(True)
ax2.legend(loc="best")
ax2.set_xlabel(r"$x$")



# --- 3) pairwise slopes d log gamma / d log n ---
# ax3.semilogx(x_common, slope_12,
#              color="purple", linewidth=2,
#              label=r"$n=10^{12}$ and $n=10^{13}\ \mathrm{cm^{-2}}$")
# ax3.semilogx(x_common, slope_23,
#              color="orange", linewidth=2,
#              label=r"$n=10^{13}$ and $n=10^{14}\ \mathrm{cm^{-2}}$")
ax3.semilogx(x12, slope_12,
             color="purple", linewidth=2,
             label=r"$n=10^{12}$ and $n=10^{13}\ \mathrm{cm^{-2}}$")
ax3.semilogx(x23, slope_23,
             color="orange", linewidth=2,
             label=r"$n=10^{13}$ and $n=10^{14}\ \mathrm{cm^{-2}}$")
ax3.semilogx(x13, slope_13,
             color="green", linewidth=2, 
             label=r"$n=10^{12}$ and $n=10^{14}\ \mathrm{cm^{-2}}$")

# optional theory line
# ax3.axhline(-1.5, color="k", linestyle=":", label=r"theory $=-3/2$")

ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$\frac{d \log \gamma}{d \log n}$")
ax3.grid(True)
ax3.legend(loc="best")

plt.tight_layout()
plt.savefig("combined_rho_gamma_slopes_3x1.png", dpi=300)
# plt.show()


# --- ensure columns named uniquely ---
black_g = black.rename(columns={"x": "x_n1", "rho": "rho_n1", "gamma": "gamma_n1"})
red_g   = red.rename(columns={"x": "x_n2", "rho": "rho_n2", "gamma": "gamma_n2"})
blue_g  = blue.rename(columns={"x": "x_n3", "rho": "rho_n3", "gamma": "gamma_n3"})

# --- reset index so they align row-by-row ---
black_g = black_g.reset_index(drop=True)
red_g   = red_g.reset_index(drop=True)
blue_g  = blue_g.reset_index(drop=True)

# --- concatenate horizontally (wide format) ---
gamma_wide = pd.concat([black_g[[ "x_n1","rho_n1","gamma_n1" ]],
                        red_g  [[ "x_n2","rho_n2","gamma_n2" ]],
                        blue_g [[ "x_n3","rho_n3","gamma_n3" ]]],
                       axis=1)

gamma_wide.to_csv("gamma_wide.csv", index=False)
print("Saved gamma_wide.csv")
print(gamma_wide.head())

# --- construct dataframes for each slope set ---
df12 = pd.DataFrame({"x_12": x12, "slope_12": slope_12})
df23 = pd.DataFrame({"x_23": x23, "slope_23": slope_23})
df13 = pd.DataFrame({"x_13": x13, "slope_13": slope_13})

# reset index
df12 = df12.reset_index(drop=True)
df23 = df23.reset_index(drop=True)
df13 = df13.reset_index(drop=True)

# concatenate side by side
slopes_wide = pd.concat([df12, df23, df13], axis=1)
slopes_wide.to_csv("slopes_wide.csv", index=False)

print("Saved slopes_wide.csv")
print(slopes_wide.head())
