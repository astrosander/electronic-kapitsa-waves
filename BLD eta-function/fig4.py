import numpy as np
import matplotlib.pyplot as plt

# n in cm^-2
n = np.array([1, 2, 3, 4, 5]) * 1e12

# ---- data read off the attached figure (Fig. 4) ----
# critical velocity u_c in m/s
uc = {
    10:  np.array([0.48, 0.69, 0.88, 1.04, 1.18]) * 1e6,
    50:  np.array([0.51, 0.73, 0.91, 1.07, 1.21]) * 1e6,
    150: np.array([0.60, 0.83, 1.00, 1.15, 1.29]) * 1e6,
}

# instability wavelength lambda in microns (µm)
lam_um = {
    10:  np.array([150, 400, 700, 1000, 1400]),
    50:  np.array([7, 17, 30, 42, 56]),
    150: np.array([1.0, 2.0, 3.5, 5.0, 6.5]),
}

temps = [10, 50, 150]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 9), sharex=True)

# --- top: u_c vs n ---
for T in temps:
    ax1.plot(n, uc[T], marker="o", linewidth=1.8, label=f"T={T} K")

ax1.set_title("Critical velocity vs density")
ax1.set_ylabel("Critical velocity u_c (m/s)")
ax1.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left")

# --- bottom: lambda vs n (log y) ---
for T in temps:
    ax2.plot(n, lam_um[T], marker="o", linewidth=1.8, label=f"T={T} K")

ax2.set_title("Instability wavelength vs density")
ax2.set_xlabel(r"n (cm$^{-2}$)")
ax2.set_ylabel("Wavelength λ (µm)")
ax2.set_yscale("log")
ax2.ticklabel_format(axis="x", style="sci", scilimits=(12, 12))
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.legend(loc="lower right")

plt.tight_layout()
plt.show()
