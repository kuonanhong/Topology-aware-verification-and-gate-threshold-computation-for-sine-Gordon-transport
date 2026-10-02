#!/usr/bin/env python3
"""Reproducible branch-aware SG / theta / Landen / Lame verification.

All special-function API calls use parameter m=k**2; manuscript formulas use k.
This script verifies classical identities and a specifically stated genus-two
parity reduction. It does not prove generic genus-two SG admissibility or a
quantum spin-pump invariant. Run: python code/elliptic_verification.py
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import scipy
from scipy.special import ellipj, ellipk
from scipy.integrate import solve_ivp, simpson
from scipy.linalg import eigh
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)


def mstr(x):
    return mp.nstr(x, 65)


def theta(n, z, tau):
    """Normalized argument: theta_n(z|tau)=mp.jtheta(n,pi*z,e^(i*pi*tau))."""
    return mp.jtheta(n, mp.pi*z, mp.exp(mp.j*mp.pi*tau))


def jacobi_theta(u,k):
    K=mp.ellipk(k*k); tau=mp.j*mp.ellipk(1-k*k)/K; z=u/(2*K)
    t4=theta(4,z,tau)
    return (theta(3,0,tau)*theta(1,z,tau)/(theta(2,0,tau)*t4),
            theta(4,0,tau)*theta(2,z,tau)/(theta(2,0,tau)*t4),
            theta(4,0,tau)*theta(3,z,tau)/(theta(3,0,tau)*t4))


def descend(u,k):
    kp=mp.sqrt(1-k*k); l=(1-kp)/(1+kp);v=u/(1+l)
    s=mp.ellipfun('sn',v,l*l);c=mp.ellipfun('cn',v,l*l);d=mp.ellipfun('dn',v,l*l)
    den=1+l*s*s
    return ((1+l)*s/den, c*d/den, (1-l*s*s)/den)


def riemann_theta(z,B,N):
    return mp.fsum(mp.exp(mp.pi*mp.j*(B[0,0]*i*i+2*B[0,1]*i*j+B[1,1]*j*j+2*i*z[0]+2*j*z[1])) for i in range(-N,N+1) for j in range(-N,N+1))


def theta_tail_bound(beta,N):
    """Absolute cube tail bound for real z and Im(B)>=beta I.
    Union bound on two 1D Gaussian tails; valid for all beta>0.
    Arithmetic roundoff is separate from this mathematical truncation bound.
    """
    return 4*(1+1/mp.sqrt(beta))*mp.exp(-mp.pi*beta*(N+1)**2)/(1-mp.exp(-mp.pi*beta*(2*N+3)))


def high_precision_verification(out):
    rows=[]; residual=[]; taurows=[]
    for ks in ['0.05','0.2','0.5','0.8','0.95','0.99','0.9999']:
        k=mp.mpf(ks); K=mp.ellipk(k*k);l=(1-mp.sqrt(1-k*k))/(1+mp.sqrt(1-k*k))
        tau=mp.j*mp.ellipk(1-k*k)/K
        tau_l=mp.j*mp.ellipk(1-l*l)/mp.ellipk(l*l)
        taurows.append(dict(k=ks,l=mstr(l),K_identity_error=mstr(abs(K-(1+l)*mp.ellipk(l*l))),tau_doubling_error=mstr(abs(tau_l-2*tau))))
        for j in range(-8,9):
            u=K*mp.mpf(j)/3+mp.mpf('0.017')
            direct=[mp.ellipfun(a,u,k*k) for a in ['sn','cn','dn']]
            lt=descend(u,k);tt=jacobi_theta(u,k)
            rows.append(dict(k=ks,u_over_K=mstr(u/K),landen_max_error=mstr(max(abs(a-b) for a,b in zip(direct,lt))),theta_max_error=mstr(max(abs(a-b) for a,b in zip(direct,tt)))))
        for xs in ['0.17','0.43','1.17','2.71']:
            x=mp.mpf(xs)*k*K
            f=lambda y:mp.pi+2*mp.atan2(mp.ellipfun('sn',y/k,k*k),mp.ellipfun('cn',y/k,k*k))
            q=f(x);d1=mp.diff(f,x);d2=mp.diff(f,x,2)
            C=d1*d1/2+mp.cos(q)
            residual.append(dict(k=ks,x_over_kK=xs,static_residual=mstr(abs(d2-mp.sin(q))),first_integral_error=mstr(abs(C-(2/(k*k)-1)))))
    write_csv(out/'elliptic_identities.csv',rows);write_csv(out/'elliptic_periods.csv',taurows);write_csv(out/'elliptic_residuals.csv',residual)
    # Independent integral of dn for an amplitude crossing multiple principal branches.
    quadrature=[]
    for ks in ['0.2','0.8','0.99']:
        k=mp.mpf(ks);K=mp.ellipk(k*k)
        I=mp.quad(lambda t:mp.ellipfun('dn',t,k*k),[0,K,2*K])
        quadrature.append(dict(k=ks,charge_error=mstr(abs(I/mp.pi-1))))
    write_csv(out/'elliptic_winding_quadrature.csv',quadrature)
    return dict(identity_samples=len(rows),residual_samples=len(residual),max_landen_error=max(float(r['landen_max_error']) for r in rows),max_theta_error=max(float(r['theta_max_error']) for r in rows),max_static_residual=max(float(r['static_residual']) for r in residual),max_first_integral_error=max(float(r['first_integral_error']) for r in residual),max_K_identity_error=max(float(r['K_identity_error']) for r in taurows),max_tau_doubling_error=max(float(r['tau_doubling_error']) for r in taurows),max_winding_quadrature_error=max(float(r['charge_error']) for r in quadrature))


def parity_and_integer_checks(out):
    # This genus-two identity holds for arbitrary real z and Im(tau+/-)>0.
    params=[('0.13','0.35','-0.21','0.55'),('0','0.7','0','1.1'),('0.4','1.2','0.17','0.42')]
    zlist=[('0','0'),('0.13','-0.23'),('0.31','0.47'),('-0.61','0.08')]
    rows=[]
    for ir,(a,b,c,d) in enumerate(params):
        tp=mp.mpc(a,b);tm=mp.mpc(c,d);B=mp.matrix([[(tp+tm)/2,(tp-tm)/2],[(tp-tm)/2,(tp+tm)/2]])
        for iz,(z1,z2) in enumerate(zlist):
            z1=mp.mpf(z1);z2=mp.mpf(z2)
            fact=theta(3,z1+z2,2*tp)*theta(3,z1-z2,2*tm)+theta(2,z1+z2,2*tp)*theta(2,z1-z2,2*tm)
            for N in [2,4,6,8,10,12]:
                brute=riemann_theta((z1,z2),B,N);err=abs(brute-fact);bound=theta_tail_bound(min(mp.im(tp),mp.im(tm)),N)
                rows.append(dict(parameter_case=ir,z_case=iz,N=N,min_imag_eigenvalue=mstr(min(mp.im(tp),mp.im(tm))),absolute_error=mstr(err),absolute_truncation_bound=mstr(bound),roundoff_allowance='1e-70',within_bound=bool(err<=bound+mp.mpf('1e-70'))))
    A=np.array([[2,-1],[0,1]],dtype=np.int64);D=np.array([[1,0],[1,2]],dtype=np.int64);zero=np.zeros((2,2),dtype=np.int64);eye=np.eye(2,dtype=np.int64)
    M=np.block([[A,zero],[zero,D]]);J=np.block([[zero,eye],[-eye,zero]])
    # determinant exactly det(A)*det(D)=2*2, no floating determinant used.
    data=dict(M=M.tolist(),J=J.tolist(),MTJM=(M.T@J@M).tolist(),determinant=4,multiplier=2,is_symplectic=False,is_integral_symplectic_similitude=True,
              source='N2 13-page uploaded draft, p.5 Eq.(25); p.6 printed sigma_c has an inconsistent entry and must not replace Eq.(25).')
    assert np.array_equal(M.T@J@M,2*J)
    # Check period transformation against independent explicitly built Ba.
    with mp.workdps(80):
        tp=mp.mpc('.13','.8');tm=mp.mpc('-.21','1.1');Bb=mp.matrix([[tp,tp],[tp,tp+tm]]);Ba=mp.matrix([[(tp+tm)/2,(tp-tm)/2],[(tp-tm)/2,(tp+tm)/2]])
        got=mp.matrix(A.tolist())*Bb*mp.matrix(D.tolist())**-1
        data['period_transform_max_error']=mstr(max(abs(got[i,j]-Ba[i,j]) for i in range(2) for j in range(2)))
    write_csv(out/'elliptic_theta2_parity.csv',rows);(out/'elliptic_integer_matrix.json').write_text(json.dumps(data,indent=2))
    return dict(parity_samples=len(rows),parity_N12_max_error=max(float(r['absolute_error']) for r in rows if r['N']==12),all_truncation_bounds_pass=all(r['within_bound'] for r in rows),integer_matrix=data)


def branch_and_ode_checks(out):
    rows=[];arrays={}; k=.8;K=ellipk(k*k);L=2*k*K
    for ncell in [64,128,256,512,1024]:
        x=np.linspace(0,4*L,4*ncell+1);h=x[1]-x[0]
        sn,cn,dn,_=ellipj(x/k,k*k)
        rot=2*np.unwrap(np.arctan2(sn,cn));correct=np.pi+rot;folded=np.pi+2*np.arcsin(np.clip(sn,-1,1))
        def resid(u):return (u[2:]-2*u[1:-1]+u[:-2])/h**2-np.sin(u[1:-1])
        rc=resid(correct);rf=resid(folded)
        dq=(correct[2:]-correct[:-2])/(2*h);dqf=(folded[2:]-folded[:-2])/(2*h)
        rows.append(dict(points_per_cell=ncell,h=h,lift_winding=(correct[-1]-correct[0])/(2*np.pi),folded_winding=(folded[-1]-folded[0])/(2*np.pi),lift_residual_inf=np.max(np.abs(rc)),folded_residual_inf=np.max(np.abs(rf)),lift_first_integral_error=np.max(np.abs(.5*dq*dq+np.cos(correct[1:-1])-(2/k**2-1))),folded_first_integral_error=np.max(np.abs(.5*dqf*dqf+np.cos(folded[1:-1])-(2/k**2-1)))))
        if ncell==256: arrays.update(x=x,lift=correct,folded=folded,residual_x=x[1:-1],lift_residual=rc,folded_residual=rf)
    # Genuine differential equation benchmark: integrate q_tt+sin(q)=0.
    ode=[]
    for k in [.2,.5,.8,.95,.99]:
        K=ellipk(k*k);t=np.linspace(0,8*k*K,1025)
        sn,cn,dn,ph=ellipj(t/k,k*k);exact=2*np.unwrap(np.arctan2(sn,cn));exactv=2*dn/k
        for tol in [1e-7,1e-9,1e-11]:
            sol=solve_ivp(lambda t,y:[y[1],-np.sin(y[0])],(t[0],t[-1]),[0,2/k],t_eval=t,method='DOP853',rtol=tol,atol=tol*.01)
            energy=.5*sol.y[1]**2+1-np.cos(sol.y[0])
            ode.append(dict(k=k,rtol=tol,nfev=sol.nfev,max_phase_error=np.max(np.abs(sol.y[0]-exact)),max_velocity_error=np.max(np.abs(sol.y[1]-exactv)),max_energy_error=np.max(np.abs(energy-2/k**2)),winding_exact=4.0,winding_numerical=(sol.y[0,-1]-sol.y[0,0])/(2*np.pi)))
    write_csv(out/'elliptic_branch_convergence.csv',rows);write_csv(out/'elliptic_ode_convergence.csv',ode)
    np.savez_compressed(out/'elliptic_branch_profiles.npz',**arrays)
    return dict(branch_k=.8,branch_cells=4,branch_finest=rows[-1],branch_rows=rows,ode_cases=len(ode),ode_max_phase_error_finest=max(r['max_phase_error'] for r in ode if r['rtol']==1e-11))


def bloch_operator(k,N,theta_b):
    L=2*k*ellipk(k*k);x=np.arange(N)*L/N
    s=ellipj(x/k,k*k)[0];V=2*s*s-1
    freq=2*np.pi*np.fft.fftfreq(N,d=L/N)+theta_b/L
    # Unitary Fourier transform gives a Hermitian spectral kinetic operator.
    F=np.fft.fft(np.eye(N),axis=0,norm='ortho')
    H=(F.conj().T*(freq*freq))@F+np.diag(V)
    return (H+H.conj().T)/2


def lame_checks(out):
    rows=[];bandrows=[];kvals=[.2,.5,.8,.95,.99,.999]
    for k in kvals:
        for N in [16,32,64,128]:
            p=eigh(bloch_operator(k,N,0),subset_by_index=[0,2],eigvals_only=True)
            ap=eigh(bloch_operator(k,N,np.pi),subset_by_index=[0,2],eigvals_only=True)
            exact=np.array([0,(1-k*k)/(k*k),1/(k*k)])
            computed=np.array([p[0],ap[0],ap[1]])
            rows.append(dict(k=k,N=N,period=2*k*ellipk(k*k),periodic_lowest=p[0],antiperiodic_lowest=ap[0],antiperiodic_second=ap[1],exact_edge0=0,exact_edge1=exact[1],exact_edge2=exact[2],max_edge_absolute_error=np.max(np.abs(computed-exact)),computed_first_gap=ap[1]-ap[0],exact_first_gap=1.0))
        for th in np.linspace(0,np.pi,25):
            vals=eigh(bloch_operator(k,64,th),subset_by_index=[0,3],eigvals_only=True)
            bandrows.append(dict(k=k,theta_over_pi=th/np.pi,band0=vals[0],band1=vals[1],band2=vals[2],band3=vals[3]))
    write_csv(out/'elliptic_lame_convergence.csv',rows);write_csv(out/'elliptic_lame_bands.csv',bandrows)
    return dict(lame_mesh_cases=len(rows),bloch_cases=len(bandrows),max_edge_error_N128=max(r['max_edge_absolute_error'] for r in rows if r['N']==128),min_sampled_bloch_eigenvalue=min(r['band0'] for r in bandrows),negative_eigenvalue_note='Tiny negative values close to zero are floating-point/truncation errors of the translational mode, not instability.')


def make_plots(out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    def read(name):
        with open(out/name) as f:return list(csv.DictReader(f))
    z=np.load(out/'elliptic_branch_profiles.npz');br=read('elliptic_branch_convergence.csv');lr=read('elliptic_lame_convergence.csv');pa=read('elliptic_theta2_parity.csv')
    fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    xx=z['x']/(2*.8*ellipk(.8**2))
    ax[0,0].plot(xx,z['lift']/(2*np.pi),label='Continuous rotational lift');ax[0,0].plot(xx,z['folded']/(2*np.pi),label='Principal arcsin reconstruction',lw=1.5);ax[0,0].set(xlabel='Position / period',ylabel='Field / (2π)',title='(a) Branch choice changes the winding');ax[0,0].legend(fontsize=8)
    h=np.array([float(r['h']) for r in br]);ec=np.array([float(r['lift_residual_inf']) for r in br]);ef=np.array([float(r['folded_residual_inf']) for r in br]);ax[0,1].loglog(h,ec,'o-',label='Lift');ax[0,1].loglog(h,ef,'s-',label='Folded arcsin');ax[0,1].set(xlabel='Grid spacing',ylabel='Max static SG residual',title='(b) Finite-difference refinement exposes cusps');ax[0,1].legend(fontsize=8)
    for k,col in [(.2,'#4477AA'),(.8,'#228833'),(.99,'#CCBB44'),(.999,'#AA3377')]:
        r=[r for r in lr if float(r['k'])==k];ax[1,0].loglog([int(a['N']) for a in r],[float(a['max_edge_absolute_error']) for a in r],'o-',color=col,label=f'k={k}')
    ax[1,0].set(xlabel='Fourier collocation points',ylabel='Max band-edge absolute error',title='(c) Three exact Lamé spectral edges');ax[1,0].legend(fontsize=8)
    for case,col in [(0,'#4477AA'),(1,'#228833'),(2,'#AA3377')]:
        r=[r for r in pa if int(r['parameter_case'])==case and int(r['z_case'])==1];ax[1,1].semilogy([int(a['N']) for a in r],[max(float(a['absolute_error']),1e-85) for a in r],'o-',color=col,label=f'Period pair {case+1}');ax[1,1].semilogy([int(a['N']) for a in r],[float(a['absolute_truncation_bound']) for a in r],'--',color=col,alpha=.6)
    ax[1,1].set(xlabel='Genus-two lattice cutoff N',ylabel='Absolute theta error',ylim=(1e-85,1),title='(d) Parity reduction; dashed: truncation bounds');ax[1,1].legend(fontsize=8)
    fig.savefig(out/'elliptic_verification.png');fig.savefig(out/'elliptic_verification.pdf');plt.close(fig)
    bands=read('elliptic_lame_bands.csv');fig,ax=plt.subplots(1,3,figsize=(11,3.3),constrained_layout=True)
    for a,k in zip(ax,[.5,.8,.99]):
        r=[r for r in bands if float(r['k'])==k]
        for b in range(3): a.plot([float(z['theta_over_pi']) for z in r],[float(z[f'band{b}']) for z in r],lw=2)
        a.axhspan((1-k*k)/k**2,1/k**2,color='#DDCC77',alpha=.4);a.set(xlabel='Bloch phase / π',ylabel='Squared perturbation frequency',title=f'k={k}; one-cell phase twist')
    fig.savefig(out/'elliptic_lame_bands.png');fig.savefig(out/'elliptic_lame_bands.pdf');plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results');args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();mp.mp.dps=80
    result={'scope':'Classical N1 SG lattice identities and stability; special genus-two theta parity reduction. No generic N2 SG reconstruction or quantum pump claim.','precision_decimal_digits':80,'conventions':'k is modulus; scipy/mpmath parameter m=k^2; theta(z|tau) uses normalized z and jtheta argument pi*z.'}
    result.update(high_precision_verification(out));print('high precision checks finished',flush=True)
    result.update(parity_and_integer_checks(out));print('parity and exact integer checks finished',flush=True)
    result.update(branch_and_ode_checks(out));print('branch and independent ODE checks finished',flush=True)
    result.update(lame_checks(out));print('Lame checks finished',flush=True)
    make_plots(out)
    result['verification_passed']=bool(result['max_landen_error']<1e-60 and result['max_theta_error']<1e-60 and result['max_static_residual']<1e-60 and result['parity_N12_max_error']<1e-60 and result['all_truncation_bounds_pass'] and result['max_edge_error_N128']<1e-8 and result['ode_max_phase_error_finest']<1e-7)
    result['environment']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mp.__version__};result['runtime_seconds']=time.perf_counter()-start;result['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'elliptic_summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2));assert result['verification_passed']

if __name__=='__main__':main()
