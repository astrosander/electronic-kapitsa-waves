import numpy as np

HBAR = 1.054_571_817e-34  # J*s
PI = np.pi

def lorentz_delta(x, lam):
    """Approximate delta(x) by a Lorentzian of width lam."""
    return (1.0 / PI) * (lam / (x*x + lam*lam))

def energy_parabolic(p, m_eff=1.0):
    """Example dispersion: eps = |p|^2 / (2 m_eff). Replace if needed."""
    return (p[..., 0]**2 + p[..., 1]**2) / (2.0 * m_eff)

def fermi_dirac(eps, mu, T):
    """Equilibrium Fermi function."""
    return 1.0 / (np.exp((eps - mu) / T) + 1.0)

def build_lattice_states(Nmax, dp, centered=True):
    """
    Build 2D momentum states labeled by integer pairs (nx, ny).
    - If centered=True, nx,ny in [-Nmax//2, ..., Nmax//2-1] (better for momentum conservation).
    - If centered=False, nx,ny in [0, ..., Nmax-1] (matches 0<n<Nmax style but breaks symmetry at edges).
    Returns:
      nvecs: (Ns, 2) int array of lattice indices
      pvecs: (Ns, 2) float array of momenta p = dp * n
    """
    if centered:
        n0 = -Nmax // 2
        ns = np.arange(n0, n0 + Nmax, dtype=int)
    else:
        ns = np.arange(0, Nmax, dtype=int)

    nvecs = np.array([(nx, ny) for nx in ns for ny in ns], dtype=int)
    pvecs = dp * nvecs.astype(float)
    return nvecs, pvecs

def collision_operator_bruteforce(
    eta,                 # (Ns,) array, eta at each lattice momentum state
    Nmax,
    dp,
    lam,                 # lambda (energy broadening width) in same units as eps
    V2,                  # |V|^2
    mu,
    T,
    m_eff=1.0,
    centered=True,
    measure_factor=None  # optional: include discretization measure (e.g. dp**4, dp**6/(2π)^6, etc.)
):
    """
    Compute Iee[eta] on a 2D momentum grid by brute force:
      Iee_1 = sum_{2,1',2'} (2π/ħ)|V|^2 F δ_eps δ_p (eta_1' + eta_2' - eta_1 - eta_2)

    Discrete momentum delta:
      δ_p = 1 if (n1 + n2 == n1' + n2') componentwise, else 0.

    WARNING: This is extremely expensive:
      Ns = Nmax^2 states, cost ~ O(Ns^4) = O(Nmax^8).
    """
    nvecs, pvecs = build_lattice_states(Nmax, dp, centered=centered)
    Ns = nvecs.shape[0]

    eta = np.asarray(eta, dtype=float).reshape(Ns)

    # Precompute eps and f0 for all states
    eps = energy_parabolic(pvecs, m_eff=m_eff)
    f0 = fermi_dirac(eps, mu=mu, T=T)

    pref = (2.0 * PI / HBAR) * V2
    if measure_factor is None:
        # If you want to mimic a plain lattice sum with an overall Δp power,
        # plug in your choice here (many conventions exist). Example:
        # measure_factor = dp**4
        measure_factor = 1.0

    Iee = np.zeros(Ns, dtype=float)

    # Brute-force loops
    for i1 in range(Ns):
        n1 = nvecs[i1]
        eps1 = eps[i1]
        f1 = f0[i1]
        eta1 = eta[i1]

        for i2 in range(Ns):
            n2 = nvecs[i2]
            eps2 = eps[i2]
            f2 = f0[i2]
            eta2 = eta[i2]

            n_sum = n1 + n2  # total incoming lattice momentum

            for i1p in range(Ns):
                n1p = nvecs[i1p]
                eps1p = eps[i1p]
                f1p = f0[i1p]
                eta1p = eta[i1p]

                for i2p in range(Ns):
                    n2p = nvecs[i2p]

                    # δ_p: enforce momentum conservation on the lattice
                    if (n1p[0] + n2p[0] != n_sum[0]) or (n1p[1] + n2p[1] != n_sum[1]):
                        continue

                    eps2p = eps[i2p]
                    f2p = f0[i2p]
                    eta2p = eta[i2p]

                    # δ_eps: Lorentzian-broadened energy conservation
                    de = (eps1 + eps2) - (eps1p + eps2p)
                    delta_eps = lorentz_delta(de, lam)

                    # F = f1 f2 (1-f1') (1-f2')
                    F = f1 * f2 * (1.0 - f1p) * (1.0 - f2p)

                    # (eta_1' + eta_2' - eta_1 - eta_2)
                    d_eta = (eta1p + eta2p) - (eta1 + eta2)

                    Iee[i1] += pref * measure_factor * F * delta_eps * d_eta

    return Iee

# -------------------------
# Example usage (tiny Nmax only!)
if __name__ == "__main__":
    Nmax = 12#8
    dp = 1.0
    lam = 0.1
    V2 = 1.0
    mu = 0.0
    T = 1.0
    eta0 = np.random.randn(Nmax * Nmax)

    Ieta = collision_operator_bruteforce(
        eta=eta0, Nmax=Nmax, dp=dp, lam=lam, V2=V2, mu=mu, T=T,
        m_eff=1.0, centered=True,
        measure_factor=dp**4  # if you want an overall Δp^4 factor as in your notes
    )
    print(Ieta)
