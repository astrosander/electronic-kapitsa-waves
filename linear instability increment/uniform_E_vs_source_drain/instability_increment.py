import numpy as np
import matplotlib.pyplot as plt

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

def pressure_derivatives_parabolic(n, p, m, U):
    Pi_p = 2.0 * p / (m * n)
    Pi_n = U * n - (p**2) / (m * n**2)
    return Pi_n, Pi_p


def sqrt_continuous(z):
    w = np.sqrt(z.astype(np.complex128))
    for i in range(1, len(w)):
        if abs(w[i] - w[i - 1]) > abs(-w[i] - w[i - 1]):
            w[i] = -w[i]
    return w


def omega_plus(k, n0, u0_drift, m, U, gamma, dgamma_dn, Dn=0.0, Dp=0.0, setup="paper"):
    p0 = m * n0 * u0_drift
    Pi_n, Pi_p = pressure_derivatives_parabolic(n0, p0, m, U)

    gamma_tilde = gamma + (Dp - Dn) * k**2

    if setup.lower() == "paper":
        Lambda = (dgamma_dn - gamma / n0) * p0
    elif setup.lower() in ("contact", "updated", "source_drain"):
        Lambda = dgamma_dn * p0
    else:
        raise ValueError("setup must be 'paper' or 'contact'")

    Delta = (gamma_tilde + 1j * k * Pi_p)**2 + (4j * k / m) * Lambda - (4.0 * k**2 / m) * Pi_n

    sqrtD = sqrt_continuous(Delta)
    omega_p = 0.5 * (k * Pi_p - 1j * gamma_tilde) + 0.5j * sqrtD - 1j * Dn * k**2
    return omega_p


def growth_rate(k, **kwargs):
    return np.imag(omega_plus(k, **kwargs))


def main():
    m = 1.0
    U = 1.0

    n0 = 0.2
    u_drift = 0.55

    gamma = 0.3
    dgamma_dn = -2.0

    Dn = 0.1
    Dp = 0.1

    k = np.linspace(-2.0, 2.0, 2000)

    g_paper = growth_rate(
        k,
        n0=n0, u0_drift=u_drift, m=m, U=U,
        gamma=gamma, dgamma_dn=dgamma_dn,
        Dn=Dn, Dp=Dp,
        setup="paper",
    )

    g_contact = growth_rate(
        k,
        n0=n0, u0_drift=u_drift, m=m, U=U,
        gamma=gamma, dgamma_dn=dgamma_dn,
        Dn=Dn, Dp=Dp,
        setup="contact",
    )

    plt.figure()
    plt.plot(k, g_paper, label=r"uniform $E$", color="blue")
    plt.plot(k, g_contact, label=r"source, $E=0$", color="red")

    plt.axhline(0.0, linewidth=1, color="black")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\mathrm{Im}[\omega_+(k)]$")
    # plt.title(r"Instability increment vs $k$")
    plt.legend()
    plt.tight_layout()
    plt.xlim(np.min(k), np.max(k))
    plt.show()


if __name__ == "__main__":
    main()
