import numpy as np
from numpy import pi
from scipy.integrate import quad, cumtrapz

import multiprocessing as mp
import pickle
import os
#############################################################################

#p_min = 0.5    #minimal momentum
#p_max = 1.5    #maximal momentum

N_p  = 40     #number of points in momentum
N_th = 100     #number of points in angle
N0_th = 201
N = 1

#Thetas = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
#Thetas = [0.005, 0.0025, 0.01, 0.02, 0.04]
#Thetas = [0.08, 0.16, 0.32, 0.64, 1.28]
#Thetas = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
Thetas = [0.0035, 0.007, 0.014, 0.028, 0.056, 0.112, 0.224, 0.448, 0.896]

#############################################################################

def dot(A, B):
    return A[0]*B[0] + A[1]*B[1]

def norm(A):
    return np.sqrt(A[0]**2 + A[1]**2)

def f(P, Theta):
    return (1 - np.exp(-1/Theta)) / (np.exp((P**2 - 1) / Theta) + 1 - np.exp(-1/Theta))

def fP_i1(P_i, P_j, Th_pj, Th_v1):
    X1 = P_i**2 + P_j**2
    X2 = np.sqrt(P_i**2 + P_j**2 - 2 * P_i * P_j * np.cos(Th_pj))
    X3 = P_i * np.cos(Th_v1) + P_j * np.cos(Th_pj - Th_v1)

    if X1 / 2 + X2 * X3 / 2 < 0:
        return 0

    return np.sqrt(X1 / 2 + X2 * X3 / 2)



def fP_j1(P_i, P_j, Th_pj, Th_v1):
    X1 = P_i**2 + P_j**2
    X2 = np.sqrt(P_i**2 + P_j**2 - 2 * P_i * P_j * np.cos(Th_pj))
    X3 = P_i * np.cos(Th_v1) + P_j * np.cos(Th_pj - Th_v1)

    if X1 / 2 - X2 * X3 / 2 < 0:
        return 0

    return np.sqrt(X1 / 2 - X2 * X3 / 2)


#first integral good version
def int_I1_1(P_i, Theta):

    def integrand(Th_pj, Th_v1, P_j):
        P_i1 = fP_i1(P_i, P_j, Th_pj, Th_v1)
        P_j1 = fP_j1(P_i, P_j, Th_pj, Th_v1)
        return P_j * f(P_j, Theta) * (1 - f(P_i1, Theta)) * (1 - f(P_j1, Theta))

    p_max = np.sqrt(1 + Theta * 4 * np.log(10))

    p_j = np.linspace(0, p_max, num = 100)
    th_pj = np.linspace(0, 2 * np.pi, num = 50)
    th_v = np.linspace(0, 2 * np.pi, num = 50)

    step_p = p_j[1] - p_j[0]
    step_thp = th_pj[1] - th_pj[0]
    step_thv = th_v[1] - th_v[0]

    p_j = p_j[:-1] + step_p/2
    th_pj = th_pj[:-1] + step_thp/2
    th_v = th_v[:-1] + step_thv/2

    dV = step_p * step_thp * step_thv

    res = 0
    for x in th_pj:
        for y in th_v:
            for z in p_j:
                res += integrand(x, y, z) * dV

    res *= f(P_i, Theta) / (Theta * (2 * np.pi)**4) 
    return res


#second integral
def int_I2(P_i, P_k, Th_pk, Theta):

    def integrand(Th_v1):
        P_i1 = fP_i1(P_i, P_k, Th_pk, Th_v1)
        P_j1 = fP_j1(P_i, P_k, Th_pk, Th_v1)
        return (1 - f(P_i1, Theta)) * (1 - f(P_j1, Theta))

    res = quad(integrand, 0, 2*pi, epsrel = 1e-9)
    res = res[0]
    res *= f(P_i, Theta) * f(P_k, Theta)  / (Theta * (2 * np.pi)**4)
    return res


#third and forth integrals
def int_I3_and_I4(P_i, P_k, Th_pk, Theta):


    def alpha(xi, xk, n):
        if dot(xk-xi, n) == 0:
            return 0
        return (norm(xk-xi)**2) / dot(xk-xi, n)


    def integrand(Th_v1):
        xi = np.array([P_i, 0])
        xk = np.array([P_k * np.cos(Th_pk), P_k * np.sin(Th_pk)])
        n  = np.array([np.cos(Th_v1), np.sin(Th_v1)])

        dot_val = dot(xk-xi, n)
        norm_val = norm(xk-xi)**2

        if dot_val == 0 or norm_val == 0 or dot_val == 0.0 or norm_val == 0.0:
            return 0

        alp = alpha(xi, xk, n)
        xj  = 2 * xk - xi - alp * n
        xj1 = xk - alp * n

        Det = 2 * alp / dot_val

        if not isinstance(alp, float) and not isinstance(alp, int):
            raise ValueError("Your alp is broken: " + str(alp))

        return f(norm(xj), Theta) * (1 - f(norm(xj1), Theta)) * Det



    res = quad(integrand, 0, 2*pi, epsrel = 1e-9)
    res = res[0]
    return res * f(P_i, Theta) * (1 - f(P_k, Theta))  / (Theta * (2 * np.pi)**4)

#############################################################################

def dim1Mesh(rho, N = 100, xmin = 0, xmax = 1, p = 100):
    """
    Function constructs a 1D mesh,
    where point density is proportional to the distribution (rho).

    Inputs:
        rho - function that takes one variable and able to process numpy array
        N - int, number of points in the mesh
        xmin - float, minimum value of the region
        xmax - float, maximum value of the region
        p - int, precision coefficient; as bigger p,
            as more precise computed coordinates, but T = O(p)

    Outputs:
        results - 1D array of coordinates of points in the mesh sorted from min to max
        volumes - 1D array of volumes of space corresponding to each point
    """
    #Initial mesh
    points = np.linspace(xmin, xmax, num = p * N + 1, endpoint = 'True')
    Int = cumtrapz(rho(points), points, initial = 0)
    Norm = Int[-1]

    Int *= (N + 1e-3) / Norm

    #calculating final mesh
    result = []
    volume = []
    iprev = 0
    icurr = 0
    k = 1
    for i in range(p * N+1):
        icurr = i
        if Int[icurr] >= k:
            if Int[icurr] >= k+1:
                raise ValueError("Given distribution is too sharp.")

            result.append(points[icurr] / 2 + points[iprev] / 2)
            volume.append(points[icurr] - points[iprev])
            iprev = icurr
            k += 1

    volume[-1] += xmax - points[iprev]
    result = np.array(result)
    volume = np.array(volume)

    return result, volume

#############################################################################

def AngleMesh(N_th, N0_th, Theta):
    th_i = np.linspace(0, 2 * np.pi, num = N_th, endpoint = False)
    th_i_front = np.linspace(-4 * Theta, 4 * Theta, num = N0_th, endpoint = True)
        
    for i in range(th_i_front.shape[0]):
        if th_i_front[i] < 0:
            th_i_front[i] += 2 * np.pi
    
    x = Theta * 10
    if x > np.pi / 2:
        x = np.pi / 2
    
    if Theta < 0.015:
        x = np.sqrt(Theta)
    
    th_i_back  = np.linspace(np.pi - x, np.pi + x, num = N0_th, endpoint = True)
    
    ang = np.append(th_i, th_i_front)
    ang = np.append(ang, th_i_back)
    ang = np.unique(ang)
    ang = np.sort(ang)
    ang_right = np.append(ang, [ang[0] + 2 * np.pi,])
    ang_left  = np.append([ang[-1] - 2 * np.pi,], ang)
    ang_diff_right = ang_right[1:] - ang_right[:-1]
    ang_diff_left  = ang_left[1:] - ang_left[:-1]
    dV = (ang_diff_right + ang_diff_left) / 2
    return ang, dV


##############################################################################
    

#Wrapping function for collision operator matrix computation
def Integral(params):

    '''
    Function assumes homogenious radial and angular distribution of the grid.
    '''

    def func(p):
        #return np.ones(p.shape[0]) + np.sin(np.pi/2 * (p-p_min)/(1 - p_min))
        return np.ones(p.shape[0])
        
    (N_p, N_th, N0_th, Theta, N, n) = params

    a = 1e-3

    p_max = np.sqrt(1 - Theta * np.log(a))
    #trying to additionally lower the lower border
    p_min = 1 + Theta * np.log(a)

    if p_min <= 0.01:
        p_min = 0.1
    else:
        p_min = np.sqrt(p_min)

    #in grid generation
    #p_i = np.linspace(p_min, p_max, num = N_p)
    p_i, dV_p = dim1Mesh(func, N_p, p_min, p_max, p = 1000)
    th_i, dV_th = AngleMesh(N_th, N0_th, Theta)
    N_th_new = th_i.shape[0]
    
    if n == N-1:
        th_i = th_i[(N_th_new//N)*n: ]
        dV_th = dV_th[(N_th_new//N)*n: ]
    else:
        th_i = th_i[(N_th_new//N)*n: (N_th_new//N)*(n+1)]
        dV_th = dV_th[(N_th_new//N)*n: (N_th_new//N)*(n+1)]
    
    
    
    dV = p_i * dV_p

    computed_ints = np.zeros((p_i.shape[0], p_i.shape[0], th_i.shape[0]))
    computed_I1 = np.zeros(p_i.shape[0])

    for i in range(p_i.shape[0]):
        p1 = p_i[i]
        dV1 = dV[i] * dV_th[0]
        I1_calculated = int_I1_1(p1, Theta)
        if n == 0:
            computed_ints[i, i, 0] += I1_calculated
            computed_ints[i, i, 0] += int_I2(p1, p1, 0, Theta) * dV1
            computed_ints[i, i, 0] -= int_I3_and_I4(p1, p1, 0, Theta) * dV1

        computed_I1[i] += I1_calculated

        for j in range(i, p_i.shape[0]):
            if i == j and n == 0:
                st = 1
            else:
                st = 0

            for k in range(st, th_i.shape[0]):
                p2 = p_i[j]
                ang = th_i[k]
                dV1 = dV[i] * dV_th[0]
                dV2 = dV[j] * dV_th[k]
                avg_dV = np.sqrt(dV1 * dV2)
                I2  =        int_I2(p1, p2, ang, Theta) * avg_dV
                I34 = int_I3_and_I4(p1, p2, ang, Theta) * avg_dV
                if i != j:
                    computed_ints[i, j, k] += I2
                    computed_ints[j, i, k] += I2
                    computed_ints[i, j, k] -= I34
                    computed_ints[j, i, k] -= I34
                if i == j:
                    computed_ints[i, i, k] += I2
                    computed_ints[i, i, k] -= I34
    

    current = os.getcwd()
    current_matr_path = current + '/Matrixes/matrix_p-' + str(N_p) + '_th-' + str(N_th) + '_th0-' + str(N0_th)

    if not os.path.exists(current_matr_path):
        os.mkdir(current_matr_path)

    file_name = current_matr_path + '/matrix_p-' + str(N_p) + '_th-' + str(N_th) + '_th0-' + str(N0_th) + '_T-' + str(Theta) + '-'+ str(n) +'.p'
    save_data = (Theta, computed_ints, computed_I1, p_i, th_i, dV_p, dV_th)
    pickle.dump(save_data , open(file_name, 'wb'))
    #return computed_ints, computed_I1, p_i


##############################################################################

if __name__ == '__main__':

    #Test:

    #print(int_I1_1(1.5, 0.5))
    #print(int_I2(1, 2, np.pi/2, 0.5))
    #print(int_I3_and_I4(1, 1.5, np.pi/2, 0.5))


    #in grid generation
    #p_i = np.linspace(p_min, p_max, num = N_p)
    #th_i = np.linspace(0, 2 * np.pi, num = N_th)

    #diff = (th_i[1] - th_i[0]) * (p_i[1] - p_i[0])
    #grid_in  = []

    #for x in p_i:
    #    for y in th_i:
    #        grid_in.append([x, y])

    #grid_in = np.array(grid_in)


    #out grid generation (the same as in, but potentially could be different)
    #p_k = p_i[:]
    #th_k = th_i[:]

    #grid_out = []

    #for x in p_k:
    #    for y in th_k:
    #        grid_out.append([x, y])

    #grid_out = np.array(grid_out)



    #matr, grin, grout = Integral((N_p, N_th, 0.5))
    #print(matr)
    
    current = os.getcwd()
    matrix_path = current + '/Matrixes/'
    if not os.path.exists(matrix_path):
        os.mkdir(matrix_path)

    pool = mp.Pool(len(Thetas) * N)

    print('Calculation started')
    params = []
    for k in range(N):
        paramsk = [(N_p, N_th, N0_th, Theta, N, k) for Theta in Thetas]
        params += paramsk

    params = tuple(params)
    results = pool.map(Integral, params)

    pool.close()
    pool.join()
    
    #print(AngleMesh(4, 5, 0.1)[0])
    #print(AngleMesh(4, 5, 0.1)[1])
    
    #print(Integral(params[0])[2])

    #print('Calculation started')
    #current = os.getcwd()
    #current_matr_path = current + '/Matrixes/matrix_p-' + str(N_p) + '_th-' + str(N_th)

    #if not os.path.exists(current_matr_path):
    #    os.mkdir(current_matr_path)

    #Matrix generation
    #for i in range(len(Thetas)):
    #    Theta = Thetas[i]
    #    result = results[i][0]
    #    grid_in = results[i][1]
    #    grid_out = results[i][2]

        #print(grid_in)

    #    file_name = current_matr_path + '/matrix_p-' + str(N_p) + '_th-' + str(N_th) + '_T-' + str(Theta) + '.p'
    #    save_data = (grid_in, grid_out, Theta, result)
    #    pickle.dump(save_data , open(file_name, 'wb'))
