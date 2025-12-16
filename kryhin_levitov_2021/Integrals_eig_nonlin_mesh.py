# Integrals_eig_nonlin_mesh.py  (FASTER but SAME LOGIC)
import os
import numpy as np
import pickle
from scipy import integrate
from matplotlib import pyplot as plt

# Must match the matrix generator
N_p   = 40
N_th  = 100
N0_th = 201
N = 1

Thetas = np.geomspace(0.0025, 1.28, 30).tolist()
# Thetas = [0.0025, 0.0035, 0.005, 0.007, 0.01, 0.014, 0.02, 0.028, 0.04,
#           0.056, 0.08, 0.112, 0.16, 0.224, 0.32, 0.448, 0.64, 0.896, 1.28]

ms = [0, 1, 2, 3, 4, 5, 6]
k = 0  # Figure-1 style (no "-a2" files)

plt.rcParams['text.usetex'] = False#True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False


def _theta_str(theta: float) -> str:
    return f"{theta:.10g}"


def Fourier_transform2(func, th_i, dV_th, m):
    exp = np.cos(th_i * m)
    integ = func * exp
    return np.sum(integ)


Orig_Matrixes = {}
p_is = {}
th_is = {}
dVp_s = {}
dVth_s = {}
I1s = {}
eigs = {}

print("=== Eigenvalue / angular-distribution postprocessing ===", flush=True)

# Load matrices
for Theta in Thetas:
    for n in range(N):
        name = f"matrix_p-{N_p}_th-{N_th}_th0-{N0_th}"
        if k == 0:
            file_name = f'./Matrixes/{name}/{name}_T-{_theta_str(Theta)}-{n}.p'
        else:
            file_name = f'./Matrixes/{name}/{name}_T-{_theta_str(Theta)}-{n}-a{k}.p'

        (Theta_loaded, matrn, I1, p_i, th_in, dV_p, dV_thn) = pickle.load(open(file_name, 'rb'))

        if n == 0:
            matr = matrn
            th_i = th_in
            dV_th = dV_thn
        else:
            matr = np.append(matr, matrn, axis=2)
            th_i = np.append(th_i, th_in)
            dV_th = np.append(dV_th, dV_thn)

    Orig_Matrixes[Theta] = matr
    p_is[Theta] = p_i
    th_is[Theta] = th_i
    dVp_s[Theta] = dV_p
    dVth_s[Theta] = dV_th
    I1s[Theta] = np.diag(I1)

# Compute eigenvalues
for Theta in Thetas:
    print(f"[compute] Theta={Theta}", flush=True)
    p_i = p_is[Theta]
    th_i = th_is[Theta]
    dV_p = dVp_s[Theta]
    dV_th = dVth_s[Theta]
    eta = np.sqrt(p_i * dV_p)
    eta_E = eta * p_i**2

    I = Orig_Matrixes[Theta]

    # dist[i] = sqrt(dV_th[0])*sqrt(dV_th[i]) * (eta^T I[:,:,i] eta)
    core = np.einsum("p,pqi,q->i", eta, I, eta, optimize=True)
    dist = core * np.sqrt(dV_th[0]) * np.sqrt(dV_th)

    coreE = np.einsum("p,pqi,q->i", eta_E, I, eta_E, optimize=True)
    dist_E = coreE * np.sqrt(dV_th[0]) * np.sqrt(dV_th)

    eigs[Theta] = {}

    # Same normalization as your original code
    dist_normed = np.array(dist[1:]) / dV_th[1:] / dV_th[0]
    for m in ms:
        eigs[Theta][m] = Fourier_transform2(dist_normed * dV_th[1:], th_i[1:], dV_th, m)

# ---- Plot eigenvalues as in your original script (Figure-1 style) ----
f10, ax10 = plt.subplots()

for m in ms:
    y = []
    y0 = []
    for Theta in Thetas:
        y.append(eigs[Theta][m])
        if k == 0:
            y0.append(eigs[Theta][0])
        else:
            y0.append(eigs[Theta][1])

    power = 2  # same choices you had

    # Only plot m != 0,1 prominently (same logic as your code)
    if k == 0 and m != 0 and m != 1:
        ax10.plot(np.log(Thetas)[6:], 3*np.log(2*np.pi) +
                  np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[6:],
                  label=f"m = {m}", linewidth=1.5)

# m=1 dashed special line (as in your original)
if k == 0:
    m = 1
    y = []
    y0 = []
    for Theta in Thetas:
        y.append(eigs[Theta][m])
        y0.append(eigs[Theta][0])
    power = 2
    ax10.plot(np.log(Thetas)[5:], 3*np.log(2*np.pi) +
              np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[5:],
              label="m = 1", linestyle="--", linewidth=1.5, color="gray")

ax10.set_xlim(-4.7, 0.5)
ax10.set_ylim(-3.0, 2.2)
ax10.set_xlabel(r'Temperature, $\ln (T/T_F)$')
ax10.set_ylabel(r'Eigenvalues, $\ln(\lambda_m T_F^2/T^2)$')
ax10.legend()

f10.tight_layout()
f10.savefig('./Eigenvals.svg')
f10.savefig('./Eigenvals.png')
print("Saved: ./Eigenvals.svg", flush=True)
