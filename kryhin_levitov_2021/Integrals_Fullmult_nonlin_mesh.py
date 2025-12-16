# Integrals_Fullmult_nonlin_mesh.py  (FIXED + High-fidelity FAST)
# - Fixes the negative-radicand bug: uses (1-f(0)) = exp(-1/Theta), NOT 1.
# - Uses original-like midpoint grids for I1 loss (100 x 50 x 50).
# - Uses dense fixed-grid integration for I2 and I3+I4 with NPHI_GAIN points.
# - Optional numba acceleration (strongly recommended): pip install numba
# - Prints realtime progress

import os
import time
import math
import pickle
import multiprocessing as mp
import numpy as np

# ----------------------- Parameters (match original as close as possible) -----------------------
N_p   = 40
N_th  = 100
N0_th = 101          # NOTE: original Fig-1 script uses 101; use 201 if you want that variant.

# Temperatures used in the eigenvalue figure script
Thetas = [0.0025, 0.0035, 0.005, 0.007, 0.01, 0.014, 0.02, 0.028, 0.04,
          0.056, 0.08, 0.112, 0.16, 0.224, 0.32, 0.448, 0.64, 0.896, 1.28]

# I1 loss-term midpoint grid (same as your original int_I1_1)
NP_LOSS   = 100
NTHP_LOSS = 50
NTHV_LOSS = 50

# Gain-term angular quadrature replacing quad (increase to 2048 for even closer match)
NPHI_GAIN = 1024

# Mesh precision (same as your original)
DIM1MESH_P = 1000

BASE_DIR = os.path.join(os.getcwd(), "Matrixes")
# ----------------------------------------------------------------------------------------------


# Optional numba
USE_NUMBA = False
try:
    import numba
    from numba import njit, prange
    USE_NUMBA = True
except Exception:
    USE_NUMBA = False


def _theta_str(theta: float) -> str:
    return f"{theta:.10g}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.empty_like(x)
    out[0] = 0.0
    out[1:] = np.cumsum(area)
    return out


def dim1Mesh(rho, N=100, xmin=0, xmax=1, p=200):
    points = np.linspace(xmin, xmax, num=p * N + 1, endpoint=True)
    Int = cumulative_trapezoid(rho(points), points)
    Norm = Int[-1]
    if Norm <= 0:
        raise ValueError("Bad rho: integral is non-positive.")

    Int *= (N + 1e-3) / Norm

    result = []
    volume = []
    iprev = 0
    k = 1
    for i in range(p * N + 1):
        if Int[i] >= k:
            if Int[i] >= k + 1:
                raise ValueError("Given distribution is too sharp for chosen p.")
            result.append(0.5 * (points[i] + points[iprev]))
            volume.append(points[i] - points[iprev])
            iprev = i
            k += 1

    if not volume:
        raise ValueError("dim1Mesh failed (check bounds).")

    volume[-1] += xmax - points[iprev]
    return np.array(result, dtype=np.float64), np.array(volume, dtype=np.float64)


def AngleMesh(N_th, N0_th, Theta):
    th_i = np.linspace(0, 2 * np.pi, num=N_th, endpoint=False)
    th_i_front = np.linspace(-4 * Theta, 4 * Theta, num=N0_th, endpoint=True)
    th_i_front = np.where(th_i_front < 0, th_i_front + 2 * np.pi, th_i_front)

    x = Theta * 10
    if x > np.pi / 2:
        x = np.pi / 2
    if Theta < 0.015:
        x = np.sqrt(Theta)

    th_i_back = np.linspace(np.pi - x, np.pi + x, num=N0_th, endpoint=True)

    ang = np.unique(np.concatenate([th_i, th_i_front, th_i_back]))
    ang.sort()

    ang_right = np.append(ang, ang[0] + 2 * np.pi)
    ang_left = np.append(ang[-1] - 2 * np.pi, ang)

    d_right = ang_right[1:] - ang_right[:-1]
    d_left = ang_left[1:] - ang_left[:-1]
    dV = 0.5 * (d_right + d_left)
    return ang.astype(np.float64), dV.astype(np.float64)


def midpoint_grid_0_2pi(n: int):
    h = 2.0 * np.pi / n
    x = (np.arange(n, dtype=np.float64) + 0.5) * h
    return x, h


def midpoint_grid_0_a(n: int, a: float):
    h = a / n
    x = (np.arange(n, dtype=np.float64) + 0.5) * h
    return x, h


if USE_NUMBA:
    @njit(cache=True, fastmath=True)
    def f_scalar(P: float, Theta: float) -> float:
        invT = 1.0 / Theta
        em = math.exp(-invT)
        num = 1.0 - em
        den = math.exp((P * P - 1.0) * invT) + 1.0 - em
        return num / den

    @njit(cache=True, fastmath=True)
    def int_I1_loss_for_p(P_i: float,
                          Theta: float,
                          pj: np.ndarray, hp: float,
                          thp_cos: np.ndarray, thp_sin: np.ndarray, hthp: float,
                          thv_cos: np.ndarray, thv_sin: np.ndarray, hthv: float) -> float:
        # Correct handling: if val<0 => momentum=0 => (1-f)= (1-f(0)) = exp(-1/Theta)
        invT = 1.0 / Theta
        om_f0 = math.exp(-invT)   # 1 - f(0) = exp(-1/Theta)
        twopi = 2.0 * math.pi
        pref = f_scalar(P_i, Theta) / (Theta * (twopi ** 4))

        res = 0.0
        Pi2 = P_i * P_i

        for ip in range(pj.shape[0]):
            P_j = pj[ip]
            fj = f_scalar(P_j, Theta)
            Pj2 = P_j * P_j

            for a in range(thp_cos.shape[0]):
                cthp = thp_cos[a]
                sthp = thp_sin[a]

                tmp = Pi2 + Pj2 - 2.0 * P_i * P_j * cthp
                X2 = math.sqrt(tmp) if tmp > 0.0 else 0.0

                for b in range(thv_cos.shape[0]):
                    cthv = thv_cos[b]
                    sthv = thv_sin[b]

                    cos_diff = cthp * cthv + sthp * sthv
                    X3 = P_i * cthv + P_j * cos_diff

                    val1 = 0.5 * (Pi2 + Pj2) + 0.5 * X2 * X3
                    om1 = om_f0 if val1 <= 0.0 else (1.0 - f_scalar(math.sqrt(val1), Theta))

                    val2 = 0.5 * (Pi2 + Pj2) - 0.5 * X2 * X3
                    om2 = om_f0 if val2 <= 0.0 else (1.0 - f_scalar(math.sqrt(val2), Theta))

                    res += (P_j * fj * om1 * om2)

        dV = hp * hthp * hthv
        return pref * res * dV

    @njit(cache=True, fastmath=True, parallel=True)
    def compute_gain_slices(out: np.ndarray,
                            p_i: np.ndarray,
                            dV: np.ndarray,
                            th: np.ndarray,
                            dV_th: np.ndarray,
                            Theta: float,
                            phi_cos: np.ndarray,
                            phi_sin: np.ndarray,
                            hphi: float,
                            k0: int,
                            k1: int):
        invT = 1.0 / Theta
        om_f0 = math.exp(-invT)   # (1 - f(0))
        twopi = 2.0 * math.pi
        scale = 1.0 / (Theta * (twopi ** 4))

        dVth0 = dV_th[0]
        Np = p_i.shape[0]
        Nphi = phi_cos.shape[0]

        for kk in prange(k0, k1):
            ang = th[kk]
            dv_ang = dV_th[kk]
            cpk = math.cos(ang)
            spk = math.sin(ang)

            for i in range(Np):
                p1 = p_i[i]
                f1 = f_scalar(p1, Theta)

                for j in range(i, Np):
                    p2 = p_i[j]
                    f2 = f_scalar(p2, Theta)
                    om_f2 = 1.0 - f2

                    avg_dV = math.sqrt((dV[i] * dVth0) * (dV[j] * dv_ang))

                    # I2 common
                    X1 = p1 * p1 + p2 * p2
                    tmp = X1 - 2.0 * p1 * p2 * cpk
                    X2 = math.sqrt(tmp) if tmp > 0.0 else 0.0

                    # I3+I4 common
                    xk_x = p2 * cpk
                    xk_y = p2 * spk
                    dx_x = xk_x - p1
                    dx_y = xk_y
                    norm2 = dx_x * dx_x + dx_y * dx_y

                    I2_int = 0.0
                    I34_int = 0.0

                    for n in range(Nphi):
                        c = phi_cos[n]
                        s = phi_sin[n]

                        # ---- I2 ----
                        cos_ang_minus_t = cpk * c + spk * s
                        X3 = p1 * c + p2 * cos_ang_minus_t

                        val1 = 0.5 * X1 + 0.5 * X2 * X3
                        om1 = om_f0 if val1 <= 0.0 else (1.0 - f_scalar(math.sqrt(val1), Theta))

                        val2 = 0.5 * X1 - 0.5 * X2 * X3
                        om2 = om_f0 if val2 <= 0.0 else (1.0 - f_scalar(math.sqrt(val2), Theta))

                        I2_int += (om1 * om2)

                        # ---- I3+I4 ----
                        dotv = dx_x * c + dx_y * s
                        if dotv == 0.0 or norm2 == 0.0:
                            continue

                        alpha = norm2 / dotv

                        xj_x = 2.0 * xk_x - p1 - alpha * c
                        xj_y = 2.0 * xk_y - alpha * s
                        rj = math.sqrt(xj_x * xj_x + xj_y * xj_y)

                        xj1_x = xk_x - alpha * c
                        xj1_y = xk_y - alpha * s
                        rj1 = math.sqrt(xj1_x * xj1_x + xj1_y * xj1_y)

                        Det = 2.0 * alpha / dotv

                        fj = f_scalar(rj, Theta)
                        omfj1 = 1.0 - f_scalar(rj1, Theta)

                        I34_int += fj * omfj1 * Det

                    # multiply by dphi
                    I2_int *= hphi
                    I34_int *= hphi

                    I2  = I2_int  * (f1 * f2)   * scale
                    I34 = I34_int * (f1 * om_f2) * scale

                    val = (I2 - I34) * avg_dV
                    out[i, j, kk] += val
                    if i != j:
                        out[j, i, kk] += val


def _compute_one_theta(Theta: float, idx: int, total: int):
    t0 = time.time()

    a = 1e-3
    p_max = math.sqrt(1.0 - Theta * math.log(a))
    p_min_tmp = 1.0 + Theta * math.log(a)
    if p_min_tmp <= 0.01:
        p_min = 0.1
    else:
        p_min = math.sqrt(p_min_tmp)

    def rho(p):
        return np.ones_like(p)

    p_i, dV_p = dim1Mesh(rho, N=N_p, xmin=p_min, xmax=p_max, p=DIM1MESH_P)
    th_i, dV_th = AngleMesh(N_th, N0_th, Theta)
    K = th_i.shape[0]
    dV = p_i * dV_p

    computed_ints = np.zeros((N_p, N_p, K), dtype=np.float64)
    computed_I1 = np.zeros(N_p, dtype=np.float64)

    name = f"matrix_p-{N_p}_th-{N_th}_th0-{N0_th}"
    out_dir = os.path.join(BASE_DIR, name)
    _ensure_dir(out_dir)
    file_name = os.path.join(out_dir, f"{name}_T-{_theta_str(Theta)}-0.p")

    # I1 grids (midpoint, as in your original)
    pj_max = math.sqrt(1.0 + Theta * 4.0 * math.log(10.0))
    pj, hp = midpoint_grid_0_a(NP_LOSS, pj_max)
    thp, hthp = midpoint_grid_0_2pi(NTHP_LOSS)
    thv, hthv = midpoint_grid_0_2pi(NTHV_LOSS)
    thp_cos = np.cos(thp); thp_sin = np.sin(thp)
    thv_cos = np.cos(thv); thv_sin = np.sin(thv)

    # gain quadrature
    phi, hphi = midpoint_grid_0_2pi(NPHI_GAIN)
    phi_cos = np.cos(phi); phi_sin = np.sin(phi)

    print(f"[{idx+1}/{total}] Theta={Theta}  K={K}  NPHI_GAIN={NPHI_GAIN}", flush=True)

    # ---- I1 (loss) ----
    if not USE_NUMBA:
        raise RuntimeError("Install numba for the intended speed/accuracy balance: pip install numba")

    for i in range(N_p):
        computed_I1[i] = int_I1_loss_for_p(
            float(p_i[i]), float(Theta),
            pj, float(hp),
            thp_cos, thp_sin, float(hthp),
            thv_cos, thv_sin, float(hthv)
        )
        if (i + 1) % max(1, N_p // 5) == 0 or (i + 1) == N_p:
            print(f"    I1 progress: {i+1}/{N_p}", flush=True)

    # add I1 to diagonal (k=0)
    for i in range(N_p):
        computed_ints[i, i, 0] += computed_I1[i]

    # ---- Gain terms (I2 - I3/I4) ----
    chunk = max(8, K // 25)
    done = 0
    while done < K:
        k0 = done
        k1 = min(K, done + chunk)
        compute_gain_slices(computed_ints, p_i, dV, th_i, dV_th, float(Theta),
                            phi_cos, phi_sin, float(hphi), int(k0), int(k1))
        done = k1
        print(f"    Gain progress: {done}/{K} angles", flush=True)

    save_data = (float(Theta), computed_ints, computed_I1, p_i, th_i, dV_p, dV_th)
    with open(file_name, "wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    dt = time.time() - t0
    print(f"    Saved: {file_name}")
    print(f"    Done Theta={Theta} in {dt:.2f}s\n", flush=True)


def main():
    _ensure_dir(BASE_DIR)

    print("=== Matrix generation (FIXED + High-fidelity FAST) ===")
    print(f"USE_NUMBA={USE_NUMBA}")
    print(f"CPU count: {mp.cpu_count()}")
    print(f"N_p={N_p}, N_th={N_th}, N0_th={N0_th}, len(Thetas)={len(Thetas)}")
    print(f"I1 grid: NP_LOSS={NP_LOSS}, NTHP_LOSS={NTHP_LOSS}, NTHV_LOSS={NTHV_LOSS}")
    print(f"Gain grid: NPHI_GAIN={NPHI_GAIN}\n", flush=True)

    if not USE_NUMBA:
        raise RuntimeError("Please install numba: pip install numba")

    try:
        numba.set_num_threads(mp.cpu_count())
    except Exception:
        pass

    t_all = time.time()
    for idx, Theta in enumerate(Thetas):
        _compute_one_theta(float(Theta), idx, len(Thetas))

    print(f"=== All done in {time.time() - t_all:.2f}s ===", flush=True)


if __name__ == "__main__":
    if os.name == "nt":
        mp.freeze_support()
    main()
