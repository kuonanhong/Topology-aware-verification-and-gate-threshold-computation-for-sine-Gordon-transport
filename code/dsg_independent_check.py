#!/usr/bin/env python3
"""Independent audit of dsg_continuation outputs; no production-module import.

Uses adaptive QUADPACK quadrature and DOP853 spatial integration, a finite-
difference parameter derivative in fixed x, and independent adaptive BDF time
integration. The model is identical; algorithms differ from production.
"""
from pathlib import Path
import csv,json,time,hashlib
import numpy as np
from scipy.integrate import quad,solve_ivp,simpson
from scipy.optimize import brentq,minimize_scalar
from scipy.special import ellipk,ellipe
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';PI=np.pi

def V(u,e,c):return 1-np.cos(u)+e*(1-np.cos(2*u+c))
def Csolve(e,c,L):
    samples=np.linspace(0,2*PI,1001);i=np.argmin(V(samples,e,c));du=samples[1]-samples[0]
    r=minimize_scalar(lambda u:V(u,e,c),bounds=(samples[i]-du,samples[i]+du),method='bounded')
    def length(C):return quad(lambda u:1/np.sqrt(2*(V(u,e,c)+C)),0,2*PI,epsabs=3e-12,epsrel=3e-12,limit=200)[0]
    excess=1e-3
    while length(-r.fun+excess)<L:
        excess/=10
        if excess<1e-12: raise ValueError("Length too close to separatrix for this double-precision audit")
    lo=-r.fun+excess
    hi=max(1.,lo+1)
    while length(hi)>L:hi*=2
    return brentq(lambda C:length(C)-L,lo,hi,xtol=5e-15,rtol=5e-15)

def profile(e,c,L):
    C=Csolve(e,c,L)
    sol=solve_ivp(lambda x,u:np.sqrt(2*(V(u,e,c)+C)),(0,L),[PI],method='DOP853',rtol=3e-12,atol=3e-13,dense_output=True)
    def lifted(x):
        z=np.asarray(x);return sol.sol(np.mod(z,L))[0]+2*PI*np.floor(z/L)
    return C,lifted

def finite_parameter_connection(e,c,L,gauge=False):
    # No analytic C_chi or x_chi formula appears in this independent calculation.
    x=np.linspace(0,L,1025);h=2e-5
    C,q=profile(e,c,L);_,qp=profile(e,c+h,L);_,qm=profile(e,c-h,L)
    shift=lambda cc:.17*np.sin(cc) if gauge else 0.
    xx=x+shift(c);u=q(xx);ux=np.sqrt(2*(V(u,e,c)+C))
    d=(qp(x+shift(c+h))-qm(x+shift(c-h)))/(2*h)
    return simpson(ux*d,x=x)/simpson(ux*ux,x=x),C

def adaptive_bdf_case(e=.2,L=8.,N=128,T=100.,relax=50.):
    x=np.arange(N)*L/N;base=2*PI*x/L;freq=2*PI*np.fft.fftfreq(N,d=L/N)
    # Dense spectral differentiation constructed separately, then adaptive BDF.
    lap=np.fft.ifft(-freq[:,None]**2*np.fft.fft(np.eye(N),axis=0),axis=0).real
    _,q=profile(e,0,L);u0=q(x);w0=u0-base
    def chi(t):
        s=np.clip(t/T,0,1);return 2*PI*(3*s*s-2*s*s*s)
    def rhs(t,w):
        u=base+w;return lap@w-np.sin(u)-2*e*np.sin(2*u+chi(t))
    def jac(t,w):
        u=base+w;return lap-np.diag(np.cos(u)+4*e*np.cos(2*u+chi(t)))
    sol=solve_ivp(rhs,(0,T+relax),w0,method='BDF',jac=jac,rtol=2e-10,atol=2e-12,max_step=.25)
    Q=-np.mean(sol.y[:,-1]-w0)/(2*PI)
    return dict(charge_average=Q,nfev=sol.nfev,njev=sol.njev,nlu=sol.nlu,accepted_steps=len(sol.t)-1,success=bool(sol.success),final_rhs_norm=float(np.linalg.norm(rhs(sol.t[-1],sol.y[:,-1]))),rtol=2e-10,atol=2e-12,max_step=.25)

def compressibility_check():
    rows=[]
    for k in [.2,.5,.8,.95,.99]:
        K=ellipk(k*k);E=ellipe(k*k);L=2*k*K;C=2/k**2-2
        A=quad(lambda u:np.sqrt(2*(1-np.cos(u)+C)),0,2*PI,epsabs=1e-11)[0]
        B=quad(lambda u:(2*(1-np.cos(u)+C))**-1.5,0,2*PI,epsabs=1e-11)[0]
        exact=(1-k*k)*K*K/(E*E);quadval=L*L/(A*B)
        # Independent energy-versus-length curvature using analytic SG E=A-C L
        h=1e-3*L
        def energy(l):
            kk=brentq(lambda kk:2*kk*ellipk(kk*kk)-l,.001,1-1e-14,xtol=4e-15)
            return 8*ellipe(kk*kk)/kk-(2/kk**2-2)*l
        ELL=(energy(L+h)-2*energy(L)+energy(L-h))/h**2
        # Long-wave Bloch slope, independently sampled at two small phases.
        N=64;x=np.arange(N)*L/N
        from scipy.special import ellipj
        p=2*ellipj(x/k,k*k)[0]**2-1;F=np.fft.fft(np.eye(N),axis=0,norm='ortho')
        slopes=[]
        for th in [.02,.01]:
            freqs=2*PI*np.fft.fftfreq(N,d=L/N)+th/L
            H=(F.conj().T*(freqs*freqs))@F+np.diag(p)
            lam=eigh((H+H.conj().T)/2,subset_by_index=[0,0],eigvals_only=True)[0]
            slopes.append(lam/(th/L)**2)
        extrap=(4*slopes[1]-slopes[0])/3
        rows.append(dict(k=k,L=L,A=A,B=B,c2_analytic=exact,c2_quadrature=quadval,c2_Bloch_extrapolated=extrap,c2_quadrature_relative_error=abs(quadval/exact-1),c2_Bloch_absolute_error=abs(extrap-exact),energy_curvature_FD=ELL,energy_curvature_exact=1/B,energy_curvature_relative_error=abs(ELL*B-1)))
    return rows

def main():
    start=time.perf_counter()
    with open(OUT/'dsg_static_branches.csv') as f:prod=list(csv.DictReader(f))
    rows=[]
    for target in [0.,PI/2,PI,3*PI/2,2*PI]:
        p=min((r for r in prod if abs(float(r['eta'])-.2)<1e-10),key=lambda r:abs(float(r['chi'])-target))
        c=float(p['chi']);con,C=finite_parameter_connection(.2,c,8.);cg,_=finite_parameter_connection(.2,c,8.,True)
        rows.append(dict(chi=c,connection_independent=con,connection_production=float(p['connection']),connection_absolute_error=abs(con-float(p['connection'])),stress_independent=C,stress_production=float(p['stress_C']),stress_absolute_error=abs(C-float(p['stress_C'])),gauge_shift_amplitude=.17,gauge_connection=cg,gauge_covariance_error=abs(cg-con-.17*np.cos(c))))
    bdf=adaptive_bdf_case()
    with open(OUT/'dsg_pde_cycles.csv') as f:dyn=list(csv.DictReader(f))
    p=next(r for r in dyn if float(r['eta'])==.2 and float(r['T'])==100. and int(r['orientation'])==1)
    bdf['production_main_charge']=float(p['charge_average']);bdf['main_absolute_charge_difference']=abs(bdf['charge_average']-float(p['charge_average']))
    with open(OUT/'dsg_pde_refinement.csv') as f:ref=list(csv.DictReader(f))
    finest=min((r for r in ref if int(r['N'])==128),key=lambda r:float(r['dt']))
    bdf['production_finest_dt']=float(finest['dt']);bdf['production_finest_charge']=float(finest['charge_average']);bdf['finest_absolute_charge_difference']=abs(bdf['charge_average']-float(finest['charge_average']))
    comp=compressibility_check()
    for name,data in [('dsg_independent_connections.csv',rows),('elliptic_compressibility.csv',comp)]:
        with open(OUT/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    checks=dict(connection=max(r['connection_absolute_error'] for r in rows)<1e-7,stress=max(r['stress_absolute_error'] for r in rows)<1e-10,gauge=max(r['gauge_covariance_error'] for r in rows)<1e-7,adaptive_BDF=bdf['success'] and bdf['finest_absolute_charge_difference']<2e-8,compressibility=max(r['c2_quadrature_relative_error'] for r in comp)<1e-10,energy_curvature=max(r['energy_curvature_relative_error'] for r in comp)<2e-5,Bloch_speed=max(r['c2_Bloch_absolute_error'] for r in comp)<2e-6)
    checks={k:bool(v) for k,v in checks.items()}
    data=dict(description=__doc__,production_module_imported=False,connection_cases=len(rows),connection_max_error=max(r['connection_absolute_error'] for r in rows),gauge_max_error=max(r['gauge_covariance_error'] for r in rows),adaptive_BDF=bdf,compressibility_cases=len(comp),checks=checks,all_passed=all(checks.values()),runtime_seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'dsg_independent_verification.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2));assert data['all_passed']
if __name__=='__main__':main()
