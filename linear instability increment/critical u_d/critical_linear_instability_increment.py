import numpy as np, matplotlib.pyplot as plt

def find_udrift(u0,u1,kmin=-1,kmax=1,N=20000,eps=1e-120,tol=1e-80,max_iter=80,**p):
    k=np.linspace(kmin,kmax,N)
    def ok(u):
        U0, Dp, n, w, g0 = p['U0'], p['Dp'], p['n'], p['w'], p['gamma0']
        g=g0*np.exp(-n/w); q=n*u; Pn=n*U0-(q*q)/n**2; Pp=2*q/n; La=(-(g/w)-g/n)*q
        D=(g+1j*k*Pp)**2+4j*k*La-4*(k*k)*Pn
        om=(-1j*g+k*Pp+1j*np.sqrt(D))/2-1j*Dp*(k*k)
        return np.max(np.imag(om))<-eps
    a,b=u0,u1
    for _ in range(max_iter):
        if b-a<=tol: break
        m=(a+b)/2
        if ok(m): a=m
        else: b=m
    return (a+b)/2

def u_star_for(params, scan=(0.0,20.0,0.01)):
    lo,hi,st=scan
    def ok(u):
        return find_udrift(u,u+st,**params,eps=1e-12,tol=st/4) if False else None  # placeholder to avoid lint
    # quick bracket by coarse scan
    k=np.linspace(params.get('kmin',-1),params.get('kmax',1),params.get('N',20000))
    def all_neg(u):
        U0, Dp, n, w, g0 = params['U0'], params['Dp'], params['n'], params['w'], params['gamma0']
        g=g0*np.exp(-n/w); q=n*u; Pn=n*U0-(q*q)/n**2; Pp=2*q/n; La=(-(g/w)-g/n)*q
        D=(g+1j*k*Pp)**2+4j*k*La-4*(k*k)*Pn
        om=(-1j*g+k*Pp+1j*np.sqrt(D))/2-1j*Dp*(k*k)
        return np.max(np.imag(om))< -1e-12
    prev=True
    u=lo
    while u<=hi:
        cur=all_neg(u)
        if prev and not cur:
            return find_udrift(u-st,u, **params)
        prev=cur; u+=st
    # fallback: try wider range
    return find_udrift(0.2,0.6, **params)

# base params
base=dict(U0=0.5, Dp=0.2, Dn=0.2, n=1.0, w=0.5, gamma0=2.5, kmin=-1, kmax=1, N=20000)

# sweeps
Dn_vals=np.linspace(0.05,0.6,50)
Dp_vals=np.linspace(0.05,0.6,50)
G0_vals=np.linspace(1.0,5.0,50)
U_vals =np.linspace(0.1,1.0,50)
w_vals =np.linspace(0.1,1.0,50)
n_vals =np.linspace(0.1,2.0,50)

def sweep(name, vals):
    us=[]
    for v in vals:
        p=base.copy()
        if name=='Dn': p['Dn']=v   # unused in this model
        if name=='Dp': p['Dp']=v
        if name=='G0': p['gamma0']=v
        if name=='U':  p['U0']=v
        if name=='w':  p['w']=v
        if name=='n':  p['n']=v
        us.append(u_star_for(p))
    return np.array(us)

u_Dn=sweep('Dn',Dn_vals)
u_Dp=sweep('Dp',Dp_vals)
u_G0=sweep('G0',G0_vals)
u_U =sweep('U', U_vals)
u_w =sweep('w', w_vals)
u_n =sweep('n', n_vals)

fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
axs=axs.ravel()
axs[0].plot(Dn_vals,u_Dn,'-o',ms=3); axs[0].set_xlabel(r'$D_n$'); axs[0].set_ylabel(r'$u_\ast$'); axs[0].set_title(r'$u_\ast(D_n)$')
axs[1].plot(Dp_vals,u_Dp,'-o',ms=3); axs[1].set_xlabel(r'$D_p$'); axs[1].set_title(r'$u_\ast(D_p)$')
axs[2].plot(G0_vals,u_G0,'-o',ms=3); axs[2].set_xlabel(r'$\Gamma_0$'); axs[2].set_title(r'$u_\ast(\Gamma_0)$')
axs[3].plot(U_vals,u_U,'-o',ms=3);   axs[3].set_xlabel(r'$U$'); axs[3].set_title(r'$u_\ast(U)$')
axs[4].plot(w_vals,u_w,'-o',ms=3);   axs[4].set_xlabel(r'$w$'); axs[4].set_title(r'$u_\ast(w)$')
axs[5].plot(n_vals,u_n,'-o',ms=3);   axs[5].set_xlabel(r'$\bar n$'); axs[5].set_title(r'$u_\ast(\bar n)$')
for ax in axs: ax.grid(True)

plt.tight_layout(); plt.savefig(f"u_star_parameters.pdf", dpi=160); plt.savefig(f"u_star_parameters.png", dpi=160); plt.show()#plt.close()
