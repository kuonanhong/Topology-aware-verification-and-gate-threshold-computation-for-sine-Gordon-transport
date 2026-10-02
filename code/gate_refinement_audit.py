#!/usr/bin/env python3
"""Additional predictor, independent time integration, and h/dt audits.
The static BVP assumes a centered symmetric saddle; its energy criterion neglects
radiation in the actual scattering orbit. DOP853 shares the spatial operator but
not the Verlet time step. No solver priority or device performance is claimed.
"""
from pathlib import Path
import csv,json,time,hashlib
import numpy as np
import scipy
from scipy.integrate import solve_bvp,solve_ivp,simpson
from scipy.optimize import brentq
from gate_threshold import solve_gate,local_bracket,write_csv,predict_threshold,sech
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';PI=np.pi

def centered_saddle(G,w=3.,R=35.,tol=1e-8):
 x=np.linspace(0,R,401);u=4*np.arctan(np.exp(x));du=2/np.cosh(x)
 def rhs(x,y):return np.vstack((y[1],(1+G/np.cosh(x/w)**2)*np.sin(y[0])))
 sol=solve_bvp(rhs,lambda l,r:np.array([l[0]-PI,r[0]-2*PI]),x,np.vstack((u,du)),tol=tol,max_nodes=10000)
 xx=np.linspace(0,R,8193);f,p=sol.sol(xx);H=2*simpson(.5*p*p+(1+G/np.cosh(xx/w)**2)*(1-np.cos(f)),x=xx)
 return dict(G=G,R=R,tolerance=tol,energy=float(H),success=bool(sol.success),nodes=len(sol.x),max_rms_residual=float(np.max(sol.rms_residuals)))

def independent_dop853(G,dx=.1,T=130.,L=320.,v=.35,x0=-24.):
 """No production RHS or initial-profile function used in this integrator."""
 start=time.perf_counter();x=np.linspace(-L/2,L/2,round(L/dx)+1);dx=x[1]-x[0];n=len(x)-2;z=(x-x0)/np.sqrt(1-v*v)
 full=4*np.arctan(np.exp(z));pi=-2*v/np.sqrt(1-v*v)/np.cosh(z);a=1+G/np.cosh(x/3)**2
 y0=np.concatenate((full[1:-1],pi[1:-1]))
 def rhs(t,y):
  q=np.empty(n+2);q[0]=0.;q[-1]=2*PI;q[1:-1]=y[:n]
  f=(q[:-2]-2*q[1:-1]+q[2:])/dx**2-a[1:-1]*np.sin(q[1:-1])
  return np.concatenate((y[n:],f))
 sol=solve_ivp(rhs,(0,T),y0,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.15)
 uf=np.r_[0.,sol.y[:n,-1],2*PI];pf=np.r_[0.,sol.y[n:,-1],0.];det=round((10+L/2)/dx)
 return dict(G=G,dx=float(dx),dt_max=.15,T=T,rtol=2e-10,atol=2e-12,success=bool(sol.success),nfev=sol.nfev,steps=len(sol.t)-1,detector_charge=float(-(uf[det]-full[det])/(2*PI)),runtime_seconds=time.perf_counter()-start),uf,pf

def main():
 start=time.perf_counter();B=[]
 for G in [0,.1,.1459667997782244,.15,.2,.25]:B.append(centered_saddle(G))
 Ein=8/np.sqrt(1-.35**2);crit=brentq(lambda G:centered_saddle(G)['energy']-Ein,.12,.18,xtol=2e-10)
 cs=centered_saddle(crit);cs2=centered_saddle(crit,R=45,tol=2e-9)
 write_csv(OUT/'gate_saddle_energy.csv',B)
 refs=[];runs=[]
 for h,dt in [(.2,.02),(.1,.02),(.05,.02),(.1,.04),(.1,.01)]:
  b,rr=local_bracket(.149609375,.1498046875,tol=2e-5,dx=h,dt=dt);refs.append(b);runs+=rr
 write_csv(OUT/'gate_fine_refinement.csv',refs);write_csv(OUT/'gate_fine_refinement_runs.csv',runs)
 # Temporal convergence at a fixed spatial mesh against independently refined h_time.
 fields={};temporal=[]
 for dt in [.04,.02,.01,.005]:
  s,r,f=solve_gate(G=.1,dx=.1,dt=dt,tmax=130.,stop_on_exit=False);fields[dt]=f['uf'];temporal.append(s)
 final=fields[.005]
 for r in temporal:r['field_max_error_vs_dt0005']=float(np.max(abs(fields[r['dt']]-final)))
 write_csv(OUT/'gate_time_refinement.csv',temporal)
 independent=[]
 for G in [.1,.25]:
  r,uf,pf=independent_dop853(G);s,_,f=solve_gate(G=G,dx=.1,dt=.02,tmax=130.,stop_on_exit=False)
  r['verlet_dt']=.02;r['verlet_detector_charge']=s['detector_charge'];r['detector_difference']=abs(s['detector_charge']-r['detector_charge']);r['field_max_difference']=float(np.max(abs(uf-f['uf'])));independent.append(r)
 write_csv(OUT/'gate_independent_time_check.csv',independent)
 # Exercise the actual acceptance paths: a timestep outside the declared
 # stability bound, a stable-but-inaccurate run, and an exhausted search.
 cfl_rejected=False
 try:solve_gate(dx=.1,dt=.2,tmax=1.)
 except ValueError:cfl_rejected=True
 poor,_,_=solve_gate(kind='breather',dx=.8,dt=.7,tmax=14.,stop_on_exit=False)
 budget,_=local_bracket(.125,.15,tol=.0002,max_iterations=0)
 contract=dict(cfl_rejected=cfl_rejected,
   inaccurate_run_outcome=poor['outcome'],inaccurate_run_raw_outcome=poor['outcome_raw'],
   inaccurate_run_rejection_reason=poor['numerical_rejection_reason'],
   inaccurate_run_energy_defect=poor['max_energy_relative_defect'],
   iteration_budget_stop=budget['stop_reason'],
   all_passed=bool(cfl_rejected and poor['outcome']=='rejected_numerics' and budget['stop_reason']=='iteration_budget'))
 pred=predict_threshold();best=next(r for r in refs if r['dx']==.05);mid=(best['G_transmitted']+best['G_reflected'])/2
 data=dict(centered_saddle_predictor=cs,saddle_refinement=cs2,incident_continuum_energy=Ein,
  saddle_predictor_G=crit,rigid_predictor_G=pred['G_predict'],saddle_energy_refinement_change=abs(cs['energy']-cs2['energy']),
  fine_brackets=refs,finest_spatial_bracket=best,
  rigid_predictor_error_against_finest_bracket=[pred['G_predict']-best['G_reflected'],pred['G_predict']-best['G_transmitted']],
  saddle_predictor_error_against_finest_bracket=[crit-best['G_reflected'],crit-best['G_transmitted']],
  independent_time_cases=independent,temporal_convergence=temporal,acceptance_contract=contract,
  notes=['The stationary-saddle energy is a conservative predictor, not an exact scattering threshold: incident discreteness and radiative loss matter.',
  'DOP853 is an independent temporal scheme on the same finite-difference spatial model.',
  'Fine brackets remain finite-time outcome intervals; unresolved points are retained, never forced into T or R.'],
  all_passed=bool(cs['success'] and cs2['success'] and all(r['success'] for r in independent) and max(r['field_max_difference'] for r in independent)<3e-3 and contract['all_passed']),
  runtime_seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'gate_extended_audit.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2));assert data['all_passed']
if __name__=='__main__':main()
