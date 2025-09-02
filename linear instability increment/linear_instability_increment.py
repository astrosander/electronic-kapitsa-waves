import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Par:
    m: float = 1.0
    U0: float = 0.5          
    n: float = 1.0           
    w: float = 0.5           
    Gamma0: float = 10.0     
    Dn: float = 0.01         
    Dp: float = 0.01         
    N: int = 200
    kmin: float = -8.0
    kmax: float = 8.0

par = Par()

def filter_23(f):
    fh = np.fft.fft(f)
    N = f.shape[0]
    kc = N//3
    fh[kc:-kc] = 0.0
    return np.fft.ifft(fh).real

def Gamma(n):            
    return par.Gamma0 * np.exp(-n/par.w)

def dGamma_dn(n):        
    return -Gamma(n)/par.w

def Pi_n(n, p, m, U):    
    return U*n - (p**2)/(m*n**2)

def Pi_p(n, p, m):       
    return 2.0*p/(m*n)

def Lambda(n, p):        
    g = Gamma(n)
    gp = dGamma_dn(n)
    return (gp - g/n)*p

def omega_pm(k, u):
    m = par.m
    n = par.n
    U = par.U0
    Dn = par.Dn
    Dp = par.Dp

    p = n*m*u
    G = Gamma(n)
    Lam = Lambda(n, p)          
    Pin = Pi_n(n, p, m, U)      
    Pip = Pi_p(n, p, m)         

    k = np.asarray(k, dtype=np.float64)
    front = -1j*(G + (Dp + Dn)*k**2) + (k*Pip)

    Disc = (G + (Dp - Dn)*k**2 + 1j*k*Pip)**2 + (4j*k*Lam)/m - (4*k**2*Pin)/m

    sqrtDisc = np.sqrt(Disc.astype(np.complex128))  

    omega_plus  = 0.5*(front + 1j*sqrtDisc)
    omega_minus = 0.5*(front - 1j*sqrtDisc)
    return omega_plus, omega_minus

def plot_increment_vs_k(u_list, title="Instability increment vs k with $D_n, D_p$"):
    k_out = np.linspace(par.kmin, par.kmax, par.N)
    plt.figure(figsize=(8.8,4.6))
    for u in u_list:
        _, om_minus = omega_pm(k_out, u)
        zeta = np.imag(om_minus)
        plt.plot(k_out, zeta, lw=1.8, label=fr"$u={u:g}$")

    L = 10.0

    colors = ["green","red","blue","orange", "purple"]
    mirror_vals = [9,10,15,16,15]

    for i in range(1,6):
        # print(i)
        k_line = 2*i * np.pi / L
        plt.axvline(k_line, color=colors[i-1], linestyle="--", linewidth=1.2, label=f"$m = {i}$")

    for i in range(1,6):
        k_line = 2*mirror_vals[i-1] * np.pi / L
        plt.axvline(k_line, color=colors[i-1], linestyle="-.", linewidth=1.2, label=f"$m = {i}$")

    mask = k_out >= 0
    k_right = k_out[mask]
    im1_right = np.imag(om_minus)[mask]
    k_max_right = k_right[np.argmax(im1_right)]
    plt.axvline(k_max_right, color="black", linestyle="-", linewidth=2.0,
               label=f"max at k={k_max_right:.2f}")

    plt.axhline(0, color="k", lw=0.8, alpha=0.6)
    plt.grid(True, ls=":", alpha=0.5)
    plt.xlabel("wavenumber $k$")
    plt.ylabel("instability increment  $\\zeta(k)=\\mathrm{Im}\\,\\omega_-(k)$")
    plt.title(title)
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig("linear_instability_increment.png")
    plt.show()

if __name__ == "__main__":
    par.U0   = 0.5
    par.n    = 1.0
    par.w    = 5.0
    par.Gamma0 = 2.5
    par.Dp   = 0.1
    par.Dn   = 0.5
    par.N    = 200
    par.kmin = -12.0
    par.kmax = 12.0

    plot_increment_vs_k(u_list=[20.0])
