import numpy as np
import matplotlib.pyplot as plt

hbar = 1.054571817e-34
kB   = 1.380649e-23
eV   = 1.602176634e-19

vF     = 1.0e6
gamma1 = 0.39 * eV
g      = 4

mstar = gamma1 / (2 * vF**2)
U     = 2 * np.pi * hbar**2 / (g * mstar)

a_interp = 1.0

def kF_from_n(n_m2: np.ndarray) -> np.ndarray:
    return np.sqrt(4 * np.pi * n_m2 / g)

def eps_low_band(k: np.ndarray) -> np.ndarray:
    return 0.5 * (np.sqrt(gamma1**2 + 4 * (hbar * vF * k)**2) - gamma1)

def v_group(k: np.ndarray) -> np.ndarray:
    return (2 * hbar * vF**2 * k) / np.sqrt(gamma1**2 + 4 * (hbar * vF * k)**2)

def eta_of_n(n_m2: np.ndarray) -> np.ndarray:
    kF = kF_from_n(n_m2)
    v  = v_group(kF)
    v_gal = (hbar * kF) / mstar
    chi = v - v_gal
    return np.abs(chi) / np.maximum(np.abs(v), 1e-30)

def gamma_ee(T: float, mu: np.ndarray, a: float = 1.0) -> np.ndarray:
    x = kB * T
    return (x / hbar) * (x / (x + a * mu))

def gamma_total(n_m2: np.ndarray, T: float, a: float = 1.0) -> np.ndarray:
    mu = eps_low_band(kF_from_n(n_m2))
    eta = eta_of_n(n_m2)
    return gamma_ee(T, mu, a=a) * eta**2

def u0_of_n(n_m2: np.ndarray) -> np.ndarray:
    return np.sqrt(U * n_m2 / mstar)

temps = [10, 50, 150]

n_cm2 = np.linspace(1e12, 5e12, 300)
n_m2  = n_cm2 * 1e4

u0 = u0_of_n(n_m2)

uc_dict = {}
lam_um_dict = {}

for T in temps:
    gam = gamma_total(n_m2, T, a=a_interp)
    R = np.gradient(np.log(gam), np.log(n_m2))
    uc = u0 / np.maximum(np.abs(R), 1e-30)
    lam = 4 * np.pi * u0 / np.maximum(gam, 1e-30)

    uc_dict[T] = uc
    lam_um_dict[T] = lam * 1e6

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 9.0), sharex=True)

for T in temps:
    ax1.plot(n_cm2, uc_dict[T], linewidth=2, label=f"T={T} K")
ax1.set_title("Critical velocity vs density")
ax1.set_ylabel("Critical velocity $u_c$ (m/s)")
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1.legend(loc="upper left")

for T in temps:
    ax2.plot(n_cm2, lam_um_dict[T], linewidth=2, label=f"T={T} K")
ax2.set_title("Instability wavelength vs density")
ax2.set_xlabel(r"$n$ (cm$^{-2}$)")
ax2.set_ylabel(r"Wavelength $\lambda$ ($\mu$m)")
ax2.set_yscale("log")
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax2.legend(loc="lower right")

plt.tight_layout()
plt.show()
