#!/usr/bin/env python3
"""Execute all transformation-gate experiments and their numerical audits."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,json,platform,time,io
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp,quad
from scipy.optimize import brentq
from transformation_gate import (solve,mapping,sech,coefficients,exact_kink,acceleration,
    write_csv,crossing_time,local_lambda_bracket,atomic_write)

ROOT=Path(__file__).resolve().parents[1]

def analytic_and_discrete_residual(eta,h,t=70.,v=.35,w=3.,x0=-24.):
    x=np.linspace(-160,160,round(320/h)+1);h=float(x[1]-x[0]);y,J=mapping(x,eta,w)
    gamma=1/np.sqrt(1-v*v);y0=float(mapping(x0,eta,w)[0]);z=gamma*(y-v*t-y0)
    u=4*np.arctan(np.exp(np.clip(z,-710,710)));Uyy=-2*gamma*gamma*sech(z)*np.tanh(z)
    Utt=v*v*Uyy
    continuum=J*(Utt-Uyy+np.sin(u))
    rho,A,K=coefficients(x,eta,w)
    discrete=rho*(Utt-acceleration(u,rho,A,K,h))
    return dict(eta=eta,h=h,t=t,continuum_residual_max=float(np.max(abs(continuum[1:-1]))),
        discrete_residual_max=float(np.max(abs(discrete[1:-1]))),
        discrete_residual_l2=float(np.sqrt(h*np.sum(discrete[1:-1]**2))))

def independent_dop853(eta=.5,h=.2,T=80.,v=.35,w=3.,x0=-24.):
    """Separate temporal integrator; same finite-difference spatial model."""
    start=time.perf_counter();x=np.linspace(-160,160,round(320/h)+1);h=float(x[1]-x[0])
    y=x+eta*w*np.tanh(x/w);R=1+eta/np.cosh(x/w)**2
    mid=.5*(x[:-1]+x[1:]);edge=1/(1+eta/np.cosh(mid/w)**2)
    gamma=1/np.sqrt(1-v*v);y0=x0+eta*w*np.tanh(x0/w);z=gamma*(y-y0)
    u0=4*np.arctan(np.exp(z));p0=-2*gamma*v/np.cosh(z);n=len(x)-2
    def rhs(t,state):
        q=np.r_[0.,state[:n],2*np.pi]
        force=(edge[1:]*(q[2:]-q[1:-1])-edge[:-1]*(q[1:-1]-q[:-2]))/h**2
        return np.r_[state[n:],(force-R[1:-1]*np.sin(q[1:-1]))/R[1:-1]]
    sol=solve_ivp(rhs,(0,T),np.r_[u0[1:-1],p0[1:-1]],method='DOP853',rtol=2e-10,atol=2e-12,max_step=.15)
    q=np.r_[0.,sol.y[:n,-1],2*np.pi];p=np.r_[0.,sol.y[n:,-1],0.]
    s,r,f=solve(eta=eta,h=h,dt=.01,T=T,stop_on_exit=False)
    det=10.;P=-(np.interp(det,x,q)-np.interp(det,x,u0))/(2*np.pi)
    return dict(eta=eta,h=h,T=T,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.15,
        success=bool(sol.success),nfev=sol.nfev,steps=len(sol.t)-1,verlet_dt=.01,
        max_field_difference=float(np.max(abs(q-f['uf']))),max_velocity_field_difference=float(np.max(abs(p-f['pf']))),
        detector_charge_difference=abs(P-s['detector_charge']),runtime_seconds=time.perf_counter()-start)

def save_run(out,name,summary,trace,arrays):
    write_csv(out/(name+'_trace.csv'),trace)
    buf=io.BytesIO();np.savez_compressed(buf,**arrays)
    atomic_write(out/(name+'_fields.npz'),buf.getvalue())
    atomic_write(out/(name+'_summary.json'),json.dumps(summary,indent=2).encode('utf-8'))

def mapped_comparison(eta,G,h,T=140.):
    dt=.02
    sm,rm,fm=solve(eta=eta,G=G,h=h,dt=dt,T=T,stop_on_exit=False,save_fields=True,tag='mapped')
    y0=float(mapping(-24,eta,3)[0]);yd=float(mapping(10,eta,3)[0]);L=320+2*eta*3
    sv,rv,fv=solve(eta=0.,G=G,h=h,dt=dt,T=T,L=L,detector=yd,x0=y0,
        initial_y0=y0,stop_on_exit=False,save_fields=True,tag='virtual_reference')
    yp=mapping(fm['x'],eta,3)[0]
    field_errors=[float(np.max(abs(a-np.interp(yp,fv['x'],b)))) for a,b in zip(fm['u'],fv['u'])]
    current_errors=[abs(a['detector_current']-b['detector_current']) for a,b in zip(rm,rv)]
    cut_errors=[abs(a['detector_charge']-b['detector_charge']) for a,b in zip(rm,rv)]
    center_errors=[abs(a['center_y']-b['center']) for a,b in zip(rm,rv)]
    comp=dict(eta=eta,G=G,h=h,dt=dt,T=T,initial_y0=y0,physical_detector=10.,virtual_detector=yd,
        mapped_outcome=sm['outcome'],virtual_outcome=sv['outcome'],
        max_field_difference_over_trace=max(field_errors),final_field_difference=field_errors[-1],
        max_detector_current_difference=max(current_errors),max_detector_charge_difference=max(cut_errors),
        max_center_y_difference=max(center_errors),mapped_energy_defect=sm['max_energy_relative_defect'],
        virtual_energy_defect=sv['max_energy_relative_defect'],
        note='Spatial schemes are not exactly coordinate-conjugate at finite h; virtual fields are linearly interpolated onto f_eta(x).')
    trace=[dict(t=a['t'],mapped_center_x=a['center'],mapped_center_y=a['center_y'],virtual_center_y=b['center'],
        mapped_current=a['detector_current'],virtual_current=b['detector_current'],
        mapped_detector_charge=a['detector_charge'],virtual_detector_charge=b['detector_charge'],
        field_max_difference=err) for a,b,err in zip(rm,rv,field_errors)]
    return comp,trace,sm,rm,fm,sv,rv,fv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results');args=ap.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True);(ROOT/'audit').mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();allruns=[]
    def trial_barrier(lam):
        def integrand(x):
            fl=x+lam*.25*3*np.tanh(x/3.)
            return 2*.25*(1-lam)*sech(x/3.)**2*sech(fl)**2
        return quad(integrand,-80,80,epsabs=2e-12,epsrel=2e-12)[0]
    incident_kinetic=8*(1/np.sqrt(1-.35**2)-1)
    predicted_lambda=brentq(lambda lam:trial_barrier(lam)-incident_kinetic,0.,1.,xtol=2e-12)
    predictor=dict(lambda_predict=predicted_lambda,trial_barrier_at_root=trial_barrier(predicted_lambda),
        incident_kinetic_energy=incident_kinetic,eta=.25,w=3.,v=.35,
        approximation='Static trial profile u=4 atan exp(f_lambda(x)); no profile relaxation/radiation. This predictor is compared with, not used to seed, the fixed five-point screen.')
    write_csv(out/'fixed_anisotropy_trial_energy.csv',[dict(lam=lam,trial_barrier_energy=trial_barrier(lam),
        incident_kinetic_energy=incident_kinetic) for lam in np.linspace(0,1,41)])
    residuals=[analytic_and_discrete_residual(eta,h) for eta in [.25,.5] for h in [.2,.1,.05]]
    write_csv(out/'matched_residuals.csv',residuals)
    spatial=[];temporal=[];timeref={}
    for eta in [.25,.5]:
        for h in [.2,.1,.05]:
            s,_,_=solve(eta=eta,h=h,dt=.02,T=80.,stop_on_exit=False,exact_audit=True,tag='matched_space')
            spatial.append(s);allruns.append(s)
            print('MATCHED_SPACE',eta,h,s['exact_max_field_error'],s['max_energy_relative_defect'],flush=True)
    write_csv(out/'matched_spatial_convergence.csv',spatial)
    for dt in [.04,.02,.01,.005]:
        s,_,f=solve(eta=.5,h=.1,dt=dt,T=80.,stop_on_exit=False,exact_audit=True,tag='matched_time')
        temporal.append(s);allruns.append(s);timeref[dt]=f['uf']
    for s in temporal:s['final_field_difference_vs_dt0005']=float(np.max(abs(timeref[s['dt']]-timeref[.005])))
    write_csv(out/'matched_temporal_convergence.csv',temporal)
    delays=[]
    for eta in [.25,.5]:
        s,r,f=solve(eta=eta,h=.05,dt=.01,save_fields=True,record_interval=.2,
            exact_audit=True,tag=f'matched_eta{eta:g}')
        left=crossing_time(r,-12.);right=crossing_time(r,12.)
        expected=2*eta*3*np.tanh(12/3)/.35;measured=right-left-24/.35
        delay=dict(eta=eta,w=3.,v=.35,section_left=-12.,section_right=12.,
            crossing_left=left,crossing_right=right,measured_extra_time=measured,
            exact_finite_section_delay=expected,asymptotic_delay=2*eta*3/.35,
            finite_section_delay_error=measured-expected,h=s['h'],dt=s['dt'],outcome=s['outcome'])
        delays.append(delay);allruns.append(s);save_run(out,f'matched_eta{eta:g}',s,r,f)
        print('DELAY',eta,measured,expected,flush=True)
    write_csv(out/'matched_delay.csv',delays)
    comparisons=[]
    for G in [.1,.25]:
        for h in [.1,.05]:
            c,tr,sm,rm,fm,sv,rv,fv=mapped_comparison(.5,G,h)
            comparisons.append(c);allruns.extend([sm,sv]);tag=f'mapped_G{G:g}_h{h:g}'
            write_csv(out/(tag+'_comparison_trace.csv'),tr)
            if h==.1:
                save_run(out,tag,sm,rm,fm);save_run(out,f'virtual_G{G:g}',sv,rv,fv)
            print('MAPPED',G,h,c['mapped_outcome'],c['virtual_outcome'],c['max_field_difference_over_trace'],flush=True)
    write_csv(out/'mapped_gate_comparison.csv',comparisons)
    screen=[]
    for lam in [0.,.25,.5,.75,1.]:
        s,r,f=solve(eta=.25,lam=lam,family='fixed_anisotropy',save_fields=lam in [0.,1.],tag=f'lambda{lam:g}')
        screen.append(s);allruns.append(s)
        if lam in [0.,1.]:save_run(out,f'fixedK_lambda{lam:g}',s,r,f)
        print('FIXED_K_SCREEN',lam,s['outcome'],s['detector_charge'],flush=True)
    write_csv(out/'fixed_anisotropy_screen.csv',screen)
    windows=[(a['lam'],b['lam']) for a,b in zip(screen[:-1],screen[1:])
        if a['outcome']=='reflected' and b['outcome']=='transmitted']
    brackets=[];bruns=[]
    if len(windows)==1:
        for h,dt in [(.1,.02),(.05,.02),(.1,.01)]:
            b,rr=local_lambda_bracket(*windows[0],tol=.001,h=h,dt=dt)
            brackets.append(b);bruns.extend(rr);allruns.extend(rr)
    write_csv(out/'fixed_anisotropy_brackets.csv',brackets);write_csv(out/'fixed_anisotropy_bracket_runs.csv',bruns)
    x=np.linspace(-15,15,601);coef=[]
    for xx in x:
        yy,J=mapping(xx,.25,3.)
        row=dict(x=xx,f_eta=float(yy),K_fixed=float(J))
        for lam in [0.,.25,.5,.75,1.]:
            R=1+lam*.25*sech(xx/3.)**2
            row.update({f'rho_lambda{lam:g}':float(R),f'A_lambda{lam:g}':float(1/R),f'K_over_rho_lambda{lam:g}':float(J/R)})
        coef.append(row)
    write_csv(out/'fixed_anisotropy_coefficients.csv',coef)
    independent=independent_dop853();write_csv(out/'matched_independent_time_check.csv',[independent])
    # Exercise the declared rejection path and preserve the distinction between
    # stability, accuracy and a finite-time undecided physical trajectory.
    cfl=False
    try:solve(h=.1,dt=.2,T=1.)
    except ValueError:cfl=True
    short,_,_=solve(eta=.25,lam=.375,family='fixed_anisotropy',T=20.)
    contracts=dict(invalid_cfl_rejected=cfl,short_observation_outcome=short['outcome'],
        short_window_unresolved=short['outcome']=='unresolved')
    write_csv(out/'all_run_diagnostics.csv',allruns)
    checks=dict(analytic_residual=max(r['continuum_residual_max'] for r in residuals)<1e-12,
        residual_spatial_convergence=all(residuals[i]['discrete_residual_max']>residuals[i+1]['discrete_residual_max']
            for i in [0,1,3,4]),
        field_spatial_convergence=all(spatial[i]['exact_max_field_error']>spatial[i+1]['exact_max_field_error'] for i in [0,1,3,4]),
        temporal_convergence=temporal[0]['final_field_difference_vs_dt0005']>temporal[1]['final_field_difference_vs_dt0005']>
            temporal[2]['final_field_difference_vs_dt0005']>0,
        delay=all(abs(r['finite_section_delay_error'])<.05 for r in delays),
        mapped_outcomes=all(c['mapped_outcome']==c['virtual_outcome'] for c in comparisons),
        mapped_field_convergence=comparisons[0]['max_field_difference_over_trace']>comparisons[1]['max_field_difference_over_trace'] and
            comparisons[2]['max_field_difference_over_trace']>comparisons[3]['max_field_difference_over_trace'],
        fixed_K_endpoints=screen[0]['outcome']=='reflected' and screen[-1]['outcome']=='transmitted',
        screen_one_R_to_T_window=len(windows)==1,
        local_brackets=bool(brackets) and all(b['bracket_width']<=b['tolerance'] and b['stop_reason']=='width_tolerance' for b in brackets),
        all_run_acceptance=all(s['numerics_accepted'] for s in allruns),
        independent_time=independent['success'] and independent['max_field_difference']<1e-3,
        acceptance_contract=contracts['invalid_cfl_rejected'] and contracts['short_window_unresolved'])
    checks={k:bool(v) for k,v in checks.items()}
    summary=dict(model='Static pullback SG and coordinated coefficient presets',
        transformation='f_eta(x)=x+eta w tanh(x/w), J_eta=1+eta sech^2(x/w)',
        exact_pullback='rho=J_eta, A=1/J_eta, K=J_eta [1+G sech^2(f_eta(x)/3)]',
        fixed_anisotropy_path='K=J_eta, rho=1+lambda eta sech^2(x/w), A=1/rho; eta=.25,w=3',
        initial_state='Same physical center x0=-24 and virtual incident speed v=.35; initial profile uses each setting coordinate map.',
        domain=[-160.,160.],boundary=[0.,2*np.pi],physical_detector=10.,
        spatial_convergence=spatial,temporal_convergence=temporal,residuals=residuals,delay_cases=delays,
        mapped_gate_cases=comparisons,fixed_K_screen=screen,local_lambda_brackets=brackets,
        fixed_K_trial_energy_predictor=predictor,
        independent_time=independent,acceptance_contract=contracts,verification=checks,all_passed=all(checks.values()),
        max_energy_relative_defect=max(s['max_energy_relative_defect'] for s in allruns),
        max_cut_flux_defect=max(s['max_flux_defect'] for s in allruns),
        max_winding_drift=max(abs(s['total_charge_final']-s['total_charge_initial']) for s in allruns),
        limitations=['Every coefficient setting is static during a trajectory; dynamic programming work/bandwidth is not modeled.',
            'Coordinated rho and A tuning is a proposed material-control target, not an experimentally established VCMA-only knob.',
            'The exact mapped continuum G threshold is invariant, but finite grids and interpolation introduce numerical differences.',
            'Local finite-time lambda brackets do not prove global monotonicity or exclude narrow resonance windows.',
            'Topological cut current is not a calibrated electrical or microscopic spin current.',
            'This is a one-dimensional classical model, with no electrical gain or quantum information performance claim.'],
        runtime_seconds=time.perf_counter()-start,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    atomic_write(out/'transformation_gate_summary.json',json.dumps(summary,indent=2).encode('utf-8'))
    receipt=dict(script_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
        results_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()},
        platform=platform.platform(),runtime_seconds=summary['runtime_seconds'],all_passed=summary['all_passed'])
    atomic_write(ROOT/'audit'/'execution_receipt.json',json.dumps(receipt,indent=2).encode('utf-8'))
    print(json.dumps(summary,indent=2))
    if not summary['all_passed']:raise SystemExit('Transformation-gate verification failed; inspect results.')

if __name__=='__main__':main()
