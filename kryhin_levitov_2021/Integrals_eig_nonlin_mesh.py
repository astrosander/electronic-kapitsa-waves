import numpy as np
#from scipy.linalg import eigh
from scipy import integrate
import pickle
from matplotlib import pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import matplotlib as mp
##############################################################################

#p_min = 0.5    #minimal momentum
#p_max = 1.5    #maximal momentum

N_p  = 40     #number of points in momentum
N_th = 100     #number of points in angle
N0_th = 101
N = 1


#Thetas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
#Thetas = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]
#Thetas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
#Thetas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

Thetas = [0.0025, 0.0035, 0.005, 0.007, 0.01, 0.014, 0.02, 0.028, 0.04, 
          0.056, 0.08, 0.112, 0.16, 0.224, 0.32, 0.448, 0.64, 0.896, 1.28]

#Thetas = [0.0025, 0.005, 0.01, 0.02]

ms = [0, 1, 2, 3, 4, 5, 6]

k = 2

##############################################################################

plt.rcParams['text.usetex'] = True
#plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.size'] = '18'
#plt.rcParams['figure.autolayout'] = True
#plt.rcParams["legend.frameon"] = False
#plt.rcParams["errorbar.capsize"]=5

##############################################################################

def f(P, Theta):
    return (1 - np.exp(-1/Theta)) / (np.exp((P**2 - 1) / Theta) + 1 - np.exp(-1/Theta))

##############################################################################




##############################################################################

def Fourier_transform(func, th_i, m):
    exp = np.cos(th_i * m)
    integ = func * exp
    shift = integrate.trapz(func, th_i)
    return integrate.trapz(integ, th_i)# - shift

def Fourier_transform2(func, th_i, dV_th, m):
    exp = np.cos(th_i * m) 
    integ = func * exp
    #shift = integrate.trapz(func, th_i)
    return np.sum(integ)# - shift

##############################################################################

#w_odd = []
#w_even = []

Orig_Matrixes = {}
Matrixes = {}
Results = {}
p_is = {}
th_is = {}
dVp_s = {}
dVth_s = {}
I1s = {}

eigs = {}

for Theta in Thetas:
    for n in range(N):
        name = 'matrix_p-' + str(N_p) + '_th-' + str(N_th) + '_th0-' + str(N0_th)
        if k == 0:
            file_name = './Matrixes/' + name + '/' + name + '_T-' + str(Theta) + '-' + str(n) + '.p'
        else:
            file_name = './Matrixes/' + name + '/' + name + '_T-' + str(Theta) + '-' + str(n) + '-a' + str(k) + '.p'

        (Theta, matrn, I1, p_i, th_in, dV_p, dV_thn) = pickle.load(open(file_name, 'rb'))
        if n == 0:
            matr = matrn
            th_i = th_in
            dV_th = dV_thn
        else:
            matr = np.append(matr, matrn, axis = 2)
            th_i = np.append(th_i, th_in)
            dV_th = np.append(dV_th, dV_thn)
    
    #print(Theta)
    Matrixes[Theta] = {}
    Results[Theta] = {}
    p_is[Theta] = p_i
    th_is[Theta] = th_i
    dVp_s[Theta] = dV_p
    dVth_s[Theta] = dV_th
    #ms = [0, 1, 2, 3, 4, 5, 6]

    I1_diag = np.diag(I1)
    I1s[Theta] = I1_diag
    
    Orig_Matrixes[Theta] = matr
        
    #for m in ms:
    #    Matrixes[Theta][m] = Fourier_transform(matr, p_i, m, th_i, dV_th)

    

f6, ax6 = plt.subplots()
f7, ax7 = plt.subplots()

max_dist = []

part_generator = []
E_generator = []
for Theta in Thetas:
    p_i = p_is[Theta]
    th_i = th_is[Theta]
    dV_p = dVp_s[Theta]
    dV_th = dVth_s[Theta]
    eta = np.sqrt(p_i * dV_p)
    eta_E = np.sqrt(p_i * dV_p) * p_i**2

    I = Orig_Matrixes[Theta]
    
    dist = []
    dist_E = []
    for i in range(dV_th.shape[0]):
        It = I[:,:,i]
        a = np.matmul(It, eta * np.sqrt(dV_th[i]))
        dist.append(np.sum(eta * a * np.sqrt(dV_th[0])))
        
        a_E = np.matmul(It, eta_E * np.sqrt(dV_th[i]))
        dist_E.append(np.sum(eta * a_E * np.sqrt(dV_th[0])))
        
    #print(th_i/ np.pi)
    part_generator.append(np.abs(np.sum(dist[1:]) / dist[0]))
    E_generator.append(np.abs(np.sum(dist_E[1:]) / dist_E[0]))
  
    
    eigs[Theta] = {}
    for m in ms:
        #exp = np.cos(th_i * m)
        #dist_normed = np.array(dist)
        #dist_normed[0] *= np.abs(part_generator[-1])
        #eigs[Theta][m] = np.sum(exp * dist_normed)
        dist_normed = np.array(dist[1:]) / dV_th[1:] / dV_th[0]
        #eigs[Theta][m] = Fourier_transform(dist_normed, th_i[1:], m)
        eigs[Theta][m] = Fourier_transform2(dist_normed * dV_th[1:], th_i[1:], dV_th, m)

    #print(eigs[Theta])
        
    dist = np.array(dist) / dV_th**1 / dV_th[0]
    #print(dist)
    mask = th_i > 3/2*np.pi
    dist = np.append(dist[mask],  dist[~mask])
    dist = - np.array(dist)
    th_i = np.append((th_i[mask] - 2 * np.pi), th_i[~mask])

    max_dist.append(-np.max(np.abs(dist[5 * dist.shape[0]//8:7 * dist.shape[0]//8]))*Theta)
    ax6.plot( th_i / np.pi, dist / (Theta**2), label = 'T = ' + str(Theta))
    
    
ax6.legend()
    
Thetas_log = np.log(np.array(Thetas))
max_dist = np.array(max_dist) / np.array(Thetas)**2
ax7.plot(Thetas_log, max_dist)
    
############################################################################

f8, ax8 = plt.subplots()

part_generator = np.array(part_generator)
E_generator = np.array(E_generator)
ax8.plot(Thetas, E_generator / part_generator)
ax8.set_xscale('log')
ax8.set_xlabel('T')
ax8.set_ylabel('dissipation / production')

############################################################################

f9, ax9 = plt.subplots()

for m in ms:
    y = []
    for Theta in Thetas:
        y.append(eigs[Theta][m])
    if k == 0:
        power = 2
    if k == 1:
        power = 2
    if k == 2:
        power = 2
    if k == 3:
        power = 2
    if k == 4:
        power = 2
    ax9.plot(np.log(Thetas), np.array(y)/(np.array(Thetas)**power), label = 'm = ' + str(m))
    
ax9.legend()
#ax9.set_xscale('log')

f10, ax10 = plt.subplots()


def star(cent, r, cr, m, phi):
    t = np.linspace(0, 2 * np.pi, num = 1000, endpoint = True)
    x = cent[0] + (r + cr * np.cos(m * t + phi)) * np.cos(t)
    y = cent[1] + (r + cr * np.cos(m * t + phi)) * np.sin(t)
    return x, y

for m in ms:
    y = []
    y0 = []
    for Theta in Thetas:
        y.append(eigs[Theta][m])
        if k == 0:
            y0.append(eigs[Theta][0])
        else:
            y0.append(eigs[Theta][1])
    if k == 0:
        power = 2
    if k == 1:
        power = 2
    if k == 2:
        power = 2
    if k == 3:
        power = 2
    if k == 4:
        power = 2
    
    if k == 0:
        ax10.plot([-3.5, -2], [3 * np.log(2 * np.pi)-9, 3 * np.log(2 * np.pi)-6], linewidth = 0.5, color = 'k', linestyle = '--')
    
    if k == 2:
        ax10.plot([-3, -1.25], [3 * np.log(2 * np.pi)-10, 3 * np.log(2 * np.pi)-6.5], color = 'k', linestyle = '--', linewidth = 0.5)
    
    
    if k == 0 and m != 0 and m!= 1:
        if  m == 3:
            ya = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[7]
            xa = np.log(Thetas)[7]
            ya1 = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[7]
            yb1 = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[8]
            xa1 = np.log(Thetas)[7]
            xb1 = np.log(Thetas)[8]
            ya2 = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[8]
            yb2 = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[9]
            xa2 = np.log(Thetas)[8]
            xb2 = np.log(Thetas)[9]
            coef31 = (ya1-yb1)/(xa1-xb1)
            coef32 = (ya2-yb2)/(xa2-xb2)
            coef3 = (coef31 + coef32)/2
            print('m = 3', coef3)
            #ax10.plot(np.log(Thetas)[7:], np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[7:], label = 'm = ' + str(m), linewidth = 1.5)
            ax10.plot([-4.5, xb2], [3 * np.log(2 * np.pi) + coef3 * (-xb2-4.5) + yb2, 3 * np.log(2 * np.pi) + yb2], linewidth = 0.5, color = 'k', linestyle = '--')
        if  m == 5:
            ya = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[7]
            yb = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[8]
            xa = np.log(Thetas)[7]
            xb = np.log(Thetas)[8]
            ya2 = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[8]
            yb2 = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[9]
            xa2 = np.log(Thetas)[8]
            xb2 = np.log(Thetas)[9]
            coef51 = (ya-yb)/(xa-xb)
            coef52 = (ya2-yb2)/(xa2-xb2)
            coef5 = (coef51 + coef52)/2
            print('m = 5', coef5)
            #ax10.plot(np.log(Thetas)[7:], np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[7:], label = 'm = ' + str(m), linewidth = 1.5)
            ax10.plot([-4.5, xb2], [3 * np.log(2 * np.pi) + coef5 * (-xb2-4.5) + yb2, 3 * np.log(2 * np.pi) + yb2], linewidth = 0.5, color = 'k', linestyle = '--')

    if k == 0 and m != 0 and m!= 1:
        if  m == 3:
            ax10.plot(np.log(Thetas)[7:], 3 * np.log(2 * np.pi) + np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[7:], label = 'm = ' + str(m), linewidth = 1.5)
        elif  m == 5:
            ax10.plot(np.log(Thetas)[7:], 3 * np.log(2 * np.pi) + np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[7:], label = 'm = ' + str(m), linewidth = 1.5)
        else:
            ax10.plot(np.log(Thetas)[6:], 3 * np.log(2 * np.pi) + np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[6:], label = 'm = ' + str(m), linewidth = 1.5)
    
    if k == 2 and m != 0 and m != 1 and m % 2 == 0:
        ax10.plot(np.log(Thetas)[6:], 3 * np.log(2 * np.pi) + np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[6:], label = 'm = ' + str(m))
    
    if k == 2 and m == 3:
        ya = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[8]
        yb = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[9]
        xa = np.log(Thetas)[8]
        xb = np.log(Thetas)[9]
        ya2 = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[9]
        yb2 = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[10]
        xa2 = np.log(Thetas)[9]
        xb2 = np.log(Thetas)[10]
        coef31 = (ya-yb)/(xa-xb)
        coef32 = (ya2-yb2)/(xa2-xb2)
        coef3 = (coef31 + coef32) / 2 
        print('m = 3', coef3)
        ax10.plot([-4.5, xb2], [3 * np.log(2 * np.pi) + coef3 * (-xb2-4.5) + yb2, 3 * np.log(2 * np.pi) + yb2], linewidth = 0.5, color = 'k', linestyle = '--')
        ax10.plot(np.log(Thetas)[8:], 3 * np.log(2 * np.pi) + np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[8:], label = 'm = ' + str(m))    
            
    if k == 2 and m == 5:
            ya = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[8]
            yb = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[9]
            xa = np.log(Thetas)[8]
            xb = np.log(Thetas)[9]
            ya2 = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[9]
            yb2 = np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[10]
            xa2 = np.log(Thetas)[9]
            xb2 = np.log(Thetas)[10]
            coef51 = (ya-yb)/(xa-xb)
            coef52 = (ya2-yb2)/(xa2-xb2)
            coef5 = (coef51 + coef52) / 2
            print('m = 5', coef5)
            #print([-4.5, xa], [coef5 * (-xa-4.5) + ya, ya])
            ax10.plot([-5, xa], [3 * np.log(2 * np.pi) + coef5 * (-xa-5) + ya, 3 * np.log(2 * np.pi) + ya], linewidth = 0.5, color = 'k', linestyle = '--')
            ax10.plot(np.log(Thetas)[8:], 3 * np.log(2 * np.pi) + np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[8:], label = 'm = ' + str(m))    


if k == 0:
    m = 1
    y = []
    y0 = []
    for Theta in Thetas:
        y.append(eigs[Theta][m])
        if k == 0:
            y0.append(eigs[Theta][0])
        else:
            y0.append(eigs[Theta][1])
    ax10.plot(np.log(Thetas)[5:],  3 * np.log(2 * np.pi) + np.log((np.array(y) - np.array(y0))/(np.array(Thetas)**power))[5:], label = 'm = ' + str(m), 
              linestyle = '--', linewidth = 1.5)
    
if k == 0:   
    ax10.set_xlim(-4.7, 0.5)
    ax10.set_ylim(-3.0, 2.2)
if k == 2:
    ax10.set_xlim(-4.7, 0.5)
    ax10.set_ylim(-3.1, 1.9)



#if k == 2:
#    ax10.legend()
#if k == 0:
#    ax10.legend(bbox_to_anchor=(0.55,0.5), loc = 'upper left')

ax10.set_xlabel(r'Temperature, $\ln (T/T_F)$')
ax10.set_ylabel(r'Eigenvalues, $\ln(\lambda_m T_F^2/T^2)$')

#if k == 2:
#    ax10.text(-4.3, -5.1, r'$\lambda_m \sim T^2$', fontsize = 12)
#    ax10.text(-4.3, -7, r'$\lambda_m \sim T^4$', fontsize = 12)

if k == 0:
    ax10.text(-3.4, 3 * np.log(2 * np.pi)-5.8, r'$\sim T^{' + str(np.ceil(coef5 * 10)/10 + 2) + r'}$', fontsize = 12)
    ax10.text(-3.5, 3 * np.log(2 * np.pi)-7.4, r'$\sim T^{' + str(np.ceil(coef3 * 10)/10 + 2) + r'}$', fontsize = 12)
    ax10.text(-2.1, 3 * np.log(2 * np.pi)-6.5, r'$\sim T^{4}$', fontsize = 12)
    ax10.text(-4.0, 3 * np.log(2 * np.pi)-4.65, r'$\sim T^{2}$', fontsize = 12)
    ax10.text(-3.5, 3 * np.log(2 * np.pi)-3.8, r'$m = 2, 4, 6$', fontsize = 11)
    ax10.text(-4.5, 3 * np.log(2 * np.pi)-6.0, r'$m = 5$', fontsize = 11)
    ax10.text(-4.5, 3 * np.log(2 * np.pi)-7.7, r'$m = 3$', fontsize = 11)
    ax10.text(-1.8, 3 * np.log(2 * np.pi)-7.2, r'$m = 1$', fontsize = 11)
    #x5, y5 = star([-3.3, -4.9], 0.5, 0.05, 5, 0)
    #plt.plot(x5, y5, color = 'k', linewidth = 0.5)
    
if k == 2:
    ax10.text(-3.5, 3 * np.log(2 * np.pi)-6.4, r'$\sim T^{' + str(np.ceil(coef5 * 10)/10 + 2) + r'}$', fontsize = 12)
    ax10.text(-2.8, 3 * np.log(2 * np.pi)-7.1, r'$\sim T^{' + str(np.ceil(coef3 * 10)/10 + 2) + r'}$', fontsize = 12)
    ax10.text(-1.8, 3 * np.log(2 * np.pi)-8.0, r'$\sim T^{4}$', fontsize = 12)
    ax10.text(-4.5, 3 * np.log(2 * np.pi)-5.1, r'$\sim T^{2}$', fontsize = 12)
    ax10.text(-4,   3 * np.log(2 * np.pi)-4.2, r'$m = 2, 4, 6$', fontsize = 11)
    ax10.text(-4.2, 3 * np.log(2 * np.pi)-7.1, r'$m = 5$', fontsize = 11)
    ax10.text(-3.9, 3 * np.log(2 * np.pi)-8.2, r'$m = 3$', fontsize = 11)

if k == 0:
    f10.savefig('./Eigenvals.svg')
if k == 2:
    f10.savefig('./Eigenvals_no_forw.svg')


############################################################################
   
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams["legend.frameon"] = False

 
def rose_plot(ax):
    #ax.plot([1, 2])
    #fontsize = 12
    
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cnorm = mcol.Normalize(vmin = np.min(-np.log(Thetas)), vmax = np.max(-np.log(Thetas)))
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
    cpick.set_array([])

    #ax.set_theta_zero_location("N")
    ax.set_yticklabels([])
    
    for Theta in np.flip(Thetas):
        p_i = p_is[Theta]
        th_i = th_is[Theta]
        dV_p = dVp_s[Theta]
        dV_th = dVth_s[Theta]
        eta = np.sqrt(p_i * dV_p)
        I = Orig_Matrixes[Theta]
        
        dist = []
        for i in range(dV_th.shape[0]):
            It = I[:,:,i]
            a = np.matmul(It, eta * np.sqrt(dV_th[0]))
            dist.append(np.sum(eta * a * np.sqrt(dV_th[i])))
        
        dist = np.array(dist) / dV_th**1
        
        if k == 0:
            norm = 5e3
        if k == 1:
            norm = 2e4
        if k == 2:
            norm = 2e4
        if k == 3:
            #norm = 1e4
            norm = 2e4
        if k == 4:
            #norm = 1
            norm = 2e4
            
        #ax.plot(th_i[1:], 1 - dist[1:] * norm, \
        #        color = cpick.to_rgba(-np.log(Theta)), label = 'T = ' + str(Theta),\
        #        linewidth = 1)
        if k == 0:
            ax.plot(th_i[2:-1], 1 - dist[2:-1] / Theta * norm, \
                            color = cpick.to_rgba(-np.log(Theta)), label = 'T = ' + str(Theta),\
                            linewidth = 1)            
        if k == 2:
            ax.plot(th_i[1:], 1 - dist[1:] / Theta * norm, \
                    color = cpick.to_rgba(-np.log(Theta)), label = 'T = ' + str(Theta),\
                    linewidth = 1)
        
        if k == 0:
            ax.text(0.09,  0.3, 'Forward', fontsize = 14)
            ax.text(-0.31, 0.3, 'scattering', fontsize = 14)
        
        if k == 2:
            ax.text(0.05,  0.5, 'Forward', fontsize = 14)
            ax.text(-0.15, 0.5, 'scattering', fontsize = 14)
        
        if k == 0:
            ax.text(np.pi + 0.04, 0.9, 'Backscattering', fontsize = 14)
            #ax.text(np.pi + 0.11, 0.8, 'scattering', fontsize = 14)

        if k == 2:
            ax.text(np.pi + 0.05, 0.5, 'Backscattering', fontsize = 14)
            #ax.text(np.pi + 0.15, 0.5, 'scattering', fontsize = 14)

    #ax.add_patch(mp.patches.Rectangle((1.5, 1.5), 1, 1, edgecolor = 'k', facecolor = 'none'))


def pi_plot(ax):

    #max_dist = []

        #color map
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cnorm = mcol.Normalize(vmin = np.min(-np.log(Thetas)), vmax = np.max(-np.log(Thetas)))
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
    cpick.set_array([])


    if k == 0:
        ax.plot([0, 0.0003], [0, ((2*np.pi)**3 )*(-0.0013)], color = 'k', linestyle = '--', linewidth = 1)
    if k == 2:
        ax.plot([0, 0.0000], [0, ((2*np.pi)**3 )*(-0.00090)], color = 'k', linestyle = '--', linewidth = 1)
        
    for Theta in Thetas:
        p_i = p_is[Theta]
        th_i = th_is[Theta]
        dV_p = dVp_s[Theta]
        dV_th = dVth_s[Theta]
        eta = np.sqrt(p_i * dV_p)
        I = Orig_Matrixes[Theta]
        
        dist = []
        for i in range(dV_th.shape[0]):
            It = I[:,:,i]
            a = np.matmul(It, eta * np.sqrt(dV_th[0]))
            dist.append(np.sum(eta * a * np.sqrt(dV_th[i])))
        
        #print(th_i/ np.pi)
        dist = - np.array(dist) / dV_th**1 / dV_th[0]
        #mask = (th_i > 3/2*np.pi) 
        #dist = np.append(dist[mask],  dist[~mask])
        #dist = - np.array(dist)
        #th_i = np.append((th_i[mask] - 2 * np.pi), th_i[~mask])
        
        mask1 = (th_i > np.pi/20) & (th_i <  39 * np.pi / 20)
        th_i = th_i[mask1]
        dist = dist[mask1]
        
        #max_dist.append(np.max(np.abs(dist[5 * dist.shape[0]//8:7 * dist.shape[0]//8])))
        
        if k == 0:
            norm = 5e3
        if k == 1:
            norm = 2e4
        if k == 2:
            norm = 0.8e4
        if k == 4:
            #norm = 1e3
            norm = 2e4
        if k == 3:
            #norm = 2e3
            norm = 2e4
        
        norm = 1
        
        ax.plot(1 / Theta * (th_i / np.pi - 1), ((2*np.pi)**3) * dist / Theta**1 * norm, \
                color = cpick.to_rgba(-np.log(Theta)), label = 'T = ' + str(Theta),\
                linewidth = 1)
    
    
    #ax.arrow(-3, -7, 0, 3, width = 0.07, head_width=0.2, head_length=0.5, fc='k', ec='k')
    #ax.text(-2.9, -5.5, 'Temperature')
    #ax.text(-2.8, -6.4, 'Increases')
    
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    if k == 0:
        ax.set_ylim([((2*np.pi)**3 )*-0.0017, ((2*np.pi)**3 )*0.001])
        ax.arrow(-3, ((2*np.pi)**3 )*-0.0015, 0, ((2*np.pi)**3 )*0.0007, width = 0.07, head_width=0.2, head_length=((2*np.pi)**3 )*0.00015, fc='k', ec='k')
        ax.text(-2.9, ((2*np.pi)**3 )*(-0.0011), 'Temperature')
        ax.text(-2.8, ((2*np.pi)**3 )*(-0.0013), 'increases')

    #ax.locator_params(nbins=3)
    fontsize = 12
    #ax.set_xlabel(r'Angle, $\frac{\theta - 180^\circ}{180^\circ}$', fontsize=fontsize)
    ax.set_xlabel(r'Angle, $(\theta - 180^\circ)/180^\circ$', fontsize=fontsize)
    #ax.set_ylabel(r'Angular distribution $\sigma(\theta)\frac{T_F^2}{T^2}$', fontsize=fontsize)
    ax.set_ylabel(r'Angular distribution $\sigma_\theta \times T_F^2/T^2$', fontsize=fontsize)
    #ax.set_xlim([0.56, 1.44])
    ax.set_xlim([-3.5, 3.5])

    
    if k == 2:
        pass
        ax.set_ylim([((2*np.pi)**3 )*(-0.00095), ((2*np.pi)**3 )*0.0003])
        ax.arrow(-3, ((2*np.pi)**3 )*-0.0008, 0, ((2*np.pi)**3 )*0.00035, width = 0.07, head_width=0.2, head_length=((2*np.pi)**3)*0.00007, fc='k', ec='k')
        ax.text(-2.9, ((2*np.pi)**3 )*(-0.0006), 'Temperature')
        ax.text(-2.8, ((2*np.pi)**3 )*(-0.0007), 'increases')
    #ax.legend(ncol = 4, loc = 'lower center')

    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels([r'$-3\theta_T$', r'$-2\theta_T$', r'$-\theta_T$', \
                      r'$0$', r'$\theta_T$', r'$2\theta_T$', r'$3\theta_T$'])

def Tsq_plot(ax):
    global max_dist
    
    Thetas_log = np.log(np.array(Thetas))
    #max_dist = np.log(np.array(max_dist))
    x0 = np.min(Thetas_log)
    y0 = np.min(max_dist)
    a = 5
    ax.plot([x0, x0 + a], [((2*np.pi)**3 )*y0, ((2*np.pi)**3 )*y0], color = 'k', linestyle = '--', linewidth = 1)
    ax.plot(Thetas_log, ((2*np.pi)**3 )*max_dist, color = 'b', linewidth = 1)

    fontsize = 12
    
    #ax.set_xlabel(r'Temperature, $\ln \frac{T}{T_F}$', fontsize=fontsize)
    ax.set_xlabel(r'Temperature, $\ln (T/T_F)$', fontsize=fontsize)
    #ax.set_ylabel(r'$\sigma(\theta = 180^\circ)\frac{T_F^2}{T^2}$', fontsize=fontsize)
    ax.set_ylabel(r'$\sigma_{\theta = 180^\circ}\times T_F^2/T^2$', fontsize=fontsize)
    #ax.set_xlim([-5, 2])
    #ax.set_ylim([-12, -6])

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    if k == 0:
        ax.set_ylim([((2*np.pi)**3 )*-1.3e-3, ((2*np.pi)**3 )*0.1e-3])
    
fig = plt.figure(figsize = (10, 5))

ax1 = plt.subplot2grid((5, 9), (0, 4), colspan = 5, rowspan = 5, projection = 'polar')
ax2 = plt.subplot2grid((5, 9), (0, 0), colspan = 4, rowspan = 3)
ax3 = plt.subplot2grid((5, 9), (3, 0), colspan = 4, rowspan = 2)


rose_plot(ax1)
pi_plot(ax2)
Tsq_plot(ax3)

plt.tight_layout()

if k == 0:
    fig.savefig('./Angular_dist_3plots.svg')
if k == 2:
    fig.savefig('./Angular_dist_3plots_no_forw.svg')