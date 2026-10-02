#!/usr/bin/env python3
"""Conservative second-order solver for a coordinate-designed classical SG gate.

All coordinates, times and coefficients are dimensionless. This is standard
finite differences and mass-weighted velocity Verlet, not a new integrator.
The deformation f_eta is a spatial diffeomorphism; it does not change winding,
spectral genus, spatial dimension or microscopic spin/charge normalization.
"""
from __future__ import annotations
import csv, time,io,os
from pathlib import Path
import numpy as np

PI=np.pi

def sech(z):
    return 1/np.cosh(np.clip(z,-710,710))

def mapping(x,eta=.5,w=3.):
    """f_eta and f'_eta, valid for eta > -1 and w > 0."""
    x=np.asarray(x)
    return x+eta*w*np.tanh(x/w),1+eta*sech(x/w)**2

def exact_kink(x,t=0.,eta=.5,w=3.,v=.35,x0=-24.):
    y,J=mapping(x,eta,w);y0=float(mapping(x0,eta,w)[0]);gamma=1/np.sqrt(1-v*v)
    z=gamma*(y-v*t-y0)
    u=4*np.arctan(np.exp(np.clip(z,-710,710)))
    p=-2*v*gamma*sech(z)
    return u,p

def coefficients(x,eta=.5,w=3.,G=0.,lam=1.,family='pullback'):
    """Node rho,K and analytic-midpoint positive edge A.

    pullback: rho=J, A=1/J, K=J[1+G sech²(f_eta/3)].
    fixed_anisotropy: K=J_eta, rho=R_lambda=1+lambda eta sech²(x/w),
      A=1/R_lambda. lambda=0 is an anisotropy barrier, lambda=1 exact matching.
    """
    y,J=mapping(x,eta,w);mid=.5*(x[:-1]+x[1:])
    if family=='pullback':
        rho=J;K=J*(1+G*sech(y/3.)**2);A=1/mapping(mid,eta,w)[1]
    elif family=='fixed_anisotropy':
        rho=1+lam*eta*sech(x/w)**2;K=J;A=1/(1+lam*eta*sech(mid/w)**2)
    else:raise ValueError('Unknown coefficient family.')
    return rho,A,K

def hamiltonian(u,p,rho,A,K,h):
    # Endpoint velocities are zero and endpoint potentials vanish.
    return float(h*np.sum(.5*rho[1:-1]*p[1:-1]**2+K[1:-1]*(1-np.cos(u[1:-1])))+
        .5*np.sum(A*np.diff(u)**2)/h)

def acceleration(u,rho,A,K,h):
    z=np.zeros_like(u)
    edgeflux=A*np.diff(u)/h
    z[1:-1]=((edgeflux[1:]-edgeflux[:-1])/h-K[1:-1]*np.sin(u[1:-1]))/rho[1:-1]
    return z

def center_velocity(u,p,x,h):
    ids=np.flatnonzero((u[:-1]<=PI)&(u[1:]>PI))
    if len(ids)!=1:return np.nan,np.nan
    i=int(ids[0]);r=(PI-u[i])/(u[i+1]-u[i]);X=x[i]+r*h
    return float(X),float(-(p[i]*(1-r)+p[i+1]*r)/((u[i+1]-u[i])/h))

def solve(eta=.5,w=3.,G=0.,lam=1.,family='pullback',h=.1,dt=.02,v=.35,
          x0=-24.,L=320.,T=240.,detector=10.,stop_on_exit=True,
          save_fields=False,exact_audit=False,record_interval=.4,initial_y0=None,tag=''):
    start=time.perf_counter()
    values=[eta,w,G,lam,h,dt,v,x0,L,T,detector,record_interval]
    if not np.all(np.isfinite(values)):raise ValueError('Parameters must be finite.')
    if eta<=-1 or min(w,h,dt,L,T,record_interval)<=0 or G<0 or not 0<=lam<=1 or abs(v)>=1:
        raise ValueError('Need eta>-1, positive scales, G>=0, 0<=lambda<=1 and |v|<1.')
    if abs(x0)>=L/2 or abs(detector)>=L/2:raise ValueError('Initial center and detector must be interior.')
    x=np.linspace(-L/2,L/2,int(round(L/h))+1);h=float(x[1]-x[0]);steps=int(round(T/dt));dt=T/steps
    rho,A,K=coefficients(x,eta,w,G,lam,family)
    if min(np.min(rho),np.min(A),np.min(K))<=0:raise ValueError('Nonpositive coefficient.')
    omega_bound_sq=np.max(2*(A[1:]+A[:-1])/(rho[1:-1]*h*h)+abs(K[1:-1])/rho[1:-1])
    stability=float(dt*np.sqrt(omega_bound_sq))
    if stability>=2:raise ValueError('Conservative Verlet stability bound violated.')
    map_eta=eta if family=='pullback' else lam*eta
    y,J=mapping(x,map_eta,w);y0=float(mapping(x0,map_eta,w)[0]) if initial_y0 is None else float(initial_y0)
    gamma=1/np.sqrt(1-v*v)
    # Same physical initial center and virtual incident speed in every preset;
    # the weak initial tail changes with f_lambda as required by the pullback.
    z=gamma*(y-y0)
    u=4*np.arctan(np.exp(np.clip(z,-710,710)));p=-2*v*gamma*sech(z)
    u[0]=0.;u[-1]=2*PI;p[0]=p[-1]=0.;u0=u.copy();p0=p.copy()
    H0=hamiltonian(u,p,rho,A,K,h);acc=acceleration(u,rho,A,K,h)
    # Physical cut interpolates u and p linearly; its flux identity remains the
    # exact kinematic relation P=-(u_d(t)-u_d(0))/(2 pi).
    idet=min(len(x)-2,max(0,int(np.searchsorted(x,detector)-1)));rr=(detector-x[idet])/h
    cutvalue=lambda q:float(q[idet]*(1-rr)+q[idet+1]*rr)
    u0d=cutvalue(u);jprev=-cutvalue(p)/(2*PI);jint=0.;maxflux=0.;maxbalance=0.;maxH=0.
    max_field=0.;max_p=0.;max_center_y=0.;max_exact_v=0.;records=[];fields=[]
    stride=max(1,int(round(record_interval/dt)));state='unresolved'
    X=vel=np.nan;Q0=(u[-1]-u[0])/(2*PI)
    def record(t,H,P,Qright):
        records.append(dict(t=float(t),energy=H,relative_energy_defect=(H-H0)/H0,
            total_charge=float((u[-1]-u[0])/(2*PI)),detector_current=-cutvalue(p)/(2*PI),
            detector_charge=P,integrated_current=jint,charge_right=Qright,
            center=X,center_y=float(mapping(X,map_eta,w)[0]),center_velocity=vel))
        if save_fields:fields.append(u.copy())
    for n in range(steps+1):
        t=n*dt;H=hamiltonian(u,p,rho,A,K,h);P=-(cutvalue(u)-u0d)/(2*PI)
        maxH=max(maxH,abs(H-H0));maxflux=max(maxflux,abs(jint-P))
        Qright=(u[-1]-cutvalue(u))/(2*PI);Qright0=(u0[-1]-u0d)/(2*PI)
        maxbalance=max(maxbalance,abs(Qright-Qright0-jint));X,vel=center_velocity(u,p,x,h)
        if exact_audit:
            zz=gamma*(y-v*t-y0);ue=4*np.arctan(np.exp(np.clip(zz,-710,710)));pe=-2*v*gamma*sech(zz)
            max_field=max(max_field,float(np.max(abs(u-ue))));max_p=max(max_p,float(np.max(abs(p-pe))))
            max_center_y=max(max_center_y,abs(float(mapping(X,map_eta,w)[0])-y0-v*t))
            max_exact_v=max(max_exact_v,abs(vel-v/float(mapping(X,map_eta,w)[1])))
        if n%stride==0 or n==steps:record(t,H,P,Qright)
        if n>0:
            if X>16 and vel>0 and P>.995:state='transmitted'
            elif X<-16 and vel<0 and abs(P)<.005:state='reflected'
            else:state='unresolved'
            if stop_on_exit and state!='unresolved':break
        if n==steps:break
        ph=p+.5*dt*acc;unew=u+dt*ph;anew=acceleration(unew,rho,A,K,h);pnew=ph+.5*dt*anew
        jnew=-cutvalue(pnew)/(2*PI);jint+=.5*dt*(jprev+jnew);jprev=jnew
        u,p,acc=unew,pnew,anew
    if records[-1]['t']!=t:record(t,H,P,Qright)
    outcome_raw=state;reasons=[];limits={'energy_relative':1e-3,'flux_absolute':1e-3,'charge_absolute':1e-14}
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(p)) or not np.all(np.isfinite([H,H0,maxH,maxflux,X,vel])):
        reasons.append('nonfinite_or_nonunique_center')
    if maxH/H0>limits['energy_relative']:reasons.append('energy_defect_exceeds_limit')
    if maxflux>limits['flux_absolute']:reasons.append('flux_defect_exceeds_limit')
    if abs((u[-1]-u[0])/(2*PI)-Q0)>limits['charge_absolute']:reasons.append('winding_drift_exceeds_limit')
    if reasons:state='rejected_numerics'
    summary=dict(tag=tag,eta=eta,w=w,G=G,lam=lam,family=family,h=h,dt=dt,v=v,x0=x0,
        initial_y0=y0,L=L,T=T,detector_x=detector,detector_y=float(mapping(detector,map_eta,w)[0]),
        outcome=state,outcome_raw=outcome_raw,numerics_accepted=not reasons,rejection_reason=';'.join(reasons),
        acceptance_limits=limits,verlet_stability_number=stability,t_final=t,center_final=X,
        center_y_final=float(mapping(X,map_eta,w)[0]),velocity_final=vel,detector_charge=P,
        integrated_detector_current=jint,max_flux_defect=maxflux,max_local_balance_defect=maxbalance,
        total_charge_initial=Q0,total_charge_final=(u[-1]-u[0])/(2*PI),energy_initial=H0,energy_final=H,
        max_energy_relative_defect=maxH/H0,continuum_incident_energy=8*gamma,
        exact_max_field_error=max_field,exact_max_velocity_field_error=max_p,
        exact_max_center_y_error=max_center_y,exact_max_center_speed_error=max_exact_v,
        steps=n,force_evaluations=n+1,grid_points=len(x),runtime_seconds=time.perf_counter()-start)
    arrays=dict(x=x,t=np.array([r['t'] for r in records]),u=np.array(fields),u0=u0,p0=p0,uf=u,pf=p,
        rho=rho,A_edge=A,K=K)
    return summary,records,arrays

def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    atomic_write(path,f.getvalue().encode('utf-8'))

def atomic_write(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def crossing_time(records,coordinate):
    for a,b in zip(records[:-1],records[1:]):
        if a['center']<=coordinate<b['center']:
            return a['t']+(coordinate-a['center'])*(b['t']-a['t'])/(b['center']-a['center'])
    return np.nan

def local_lambda_bracket(lo,hi,tol=.001,h=.1,dt=.02,T=240.,max_iterations=15):
    rows=[]
    for lam in [lo,hi]:
        s,_,_=solve(eta=.25,lam=lam,family='fixed_anisotropy',h=h,dt=dt,T=T);rows.append(s)
    if rows[0]['outcome']!='reflected' or rows[1]['outcome']!='transmitted':
        raise ValueError('Require accepted reflected low lambda and transmitted high lambda.')
    reason='width_tolerance';iterations=0
    while hi-lo>tol and iterations<max_iterations:
        mid=.5*(lo+hi);s,_,_=solve(eta=.25,lam=mid,family='fixed_anisotropy',h=h,dt=dt,T=T)
        rows.append(s);iterations+=1
        print('LAMBDA_BISECT',mid,s['outcome'],s['t_final'],flush=True)
        if s['outcome']=='reflected':lo=mid
        elif s['outcome']=='transmitted':hi=mid
        else:reason='numerical_rejection' if s['outcome']=='rejected_numerics' else 'unresolved_midpoint';break
    if hi-lo>tol and iterations>=max_iterations and reason=='width_tolerance':reason='iteration_budget'
    return dict(lambda_reflected=lo,lambda_transmitted=hi,bracket_width=hi-lo,tolerance=tol,
        h=h,dt=dt,T=T,stop_reason=reason,iterations=iterations,PDE_calls=len(rows),
        force_evaluations=sum(s['force_evaluations'] for s in rows),
        note='Finite-time local sampled bracket; no global monotonicity certificate.'),rows
