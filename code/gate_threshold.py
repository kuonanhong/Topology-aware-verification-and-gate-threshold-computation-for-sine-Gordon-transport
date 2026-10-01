#!/usr/bin/env python3
"""Audited SG gate-threshold workflow (classical field, dimensionless units).

The solver is standard second-order finite differences and velocity Verlet.
The workflow, not a new integrator: independent kink-energy predictor, sampled
screen for multiple transition windows, ternary outcomes, local bisection,
independent h/dt refinement, and energy/charge-flux diagnostics. No electrical
transistor, quantum spin-current, or universal threshold claim is made.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PI=np.pi

def sech(x): return 1/np.cosh(np.clip(x,-710,710))
def kink(x,t,v=.35,x0=-24.):
    z=(x-x0-v*t)/np.sqrt(1-v*v)
    u=4*np.arctan(np.exp(np.clip(z,-710,710)))
    p=-2*v/np.sqrt(1-v*v)*sech(z)
    return u,p

def breather(x,t,omega=.8):
    kap=np.sqrt(1-omega*omega)
    b=kap/omega*np.sin(omega*t)*sech(kap*x)
    u=4*np.arctan(b)
    p=4*kap*np.cos(omega*t)*sech(kap*x)/(1+b*b)
    return u,p

def energy(u,p,a,dx):
    return dx*(.5*np.sum(p[1:-1]**2)+np.sum(a[1:-1]*(1-np.cos(u[1:-1]))))+.5*np.sum(np.diff(u)**2)/dx

def center_velocity(u,p,x,dx):
    ids=np.flatnonzero((u[:-1]<=PI)&(u[1:]>PI))
    if len(ids)!=1:return np.nan,np.nan
    i=int(ids[0]);r=(PI-u[i])/(u[i+1]-u[i]);X=x[i]+r*dx
    pp=p[i]*(1-r)+p[i+1]*r
    return float(X),float(-pp/((u[i+1]-u[i])/dx))

def solve_gate(G=0.,width=3.,dx=.1,dt=.04,v=.35,x0=-24.,L=320.,tmax=240.,
               kind='kink',stop_on_exit=True,save_fields=False,tag=''):
    begin=time.perf_counter()
    if not np.all(np.isfinite([G,width,dx,dt,v,x0,L,tmax])):
        raise ValueError('All input parameters must be finite.')
    if G<0 or min(width,dx,dt,L,tmax)<=0 or abs(v)>=1 or abs(x0)>=L/2 or L<=32:
        raise ValueError('This workflow requires a nonnegative barrier, positive scales, |v|<1, and an interior initial kink.')
    if kind not in ['kink','breather']:
        raise ValueError('Unknown initial-state kind.')
    nc=int(round(L/dx)); x=np.linspace(-L/2,L/2,nc+1);dx=float(x[1]-x[0])
    steps=int(round(tmax/dt));dt=tmax/steps
    a=1+G*sech(x/width)**2
    stability_number=float(dt*np.sqrt(4/dx**2+np.max(abs(a))))
    if stability_number>=2:
        raise ValueError('The conservative Verlet stability bound dt*sqrt(4/h²+max|a|)<2 is violated.')
    u,p=(kink(x,0,v,x0) if kind=='kink' else breather(x,0))
    u[0]=0.;u[-1]=2*PI if kind=='kink' else 0.;p[0]=p[-1]=0.
    uinit=u.copy(); E0=energy(u,p,a,dx);Eerr=0.;Q0=(u[-1]-u[0])/(2*PI)
    detector=10.;ic=int(round((detector+L/2)/dx));detector=float(x[ic])
    integrated_current=0.;prev_j=-p[ic]/(2*PI);max_flux_defect=0.;max_local_balance_defect=0.
    records=[];fields=[];stride=max(1,int(round(.8/dt)));state='unresolved';X=vel=np.nan
    # The discrete derivative is the negative gradient of this same Hamiltonian.
    def accel(z):
        b=np.zeros_like(z);b[1:-1]=(z[2:]-2*z[1:-1]+z[:-2])/dx**2-a[1:-1]*np.sin(z[1:-1]);return b
    acc=accel(u)
    max_phase_error=0.;max_velocity_error=0.
    for n in range(steps+1):
        t=n*dt
        E=energy(u,p,a,dx);Eerr=max(Eerr,abs(E-E0));cut=-(u[ic]-uinit[ic])/(2*PI)
        flux_defect=abs(integrated_current-cut);max_flux_defect=max(max_flux_defect,flux_defect)
        qright=(u[-1]-u[ic])/(2*PI);qright0=(uinit[-1]-uinit[ic])/(2*PI)
        max_local_balance_defect=max(max_local_balance_defect,abs(qright-qright0-integrated_current))
        if kind=='kink':X,vel=center_velocity(u,p,x,dx)
        if G==0:
            ue,pe=(kink(x,t,v,x0) if kind=='kink' else breather(x,t))
            max_phase_error=max(max_phase_error,float(np.max(abs(u-ue))))
            max_velocity_error=max(max_velocity_error,float(np.max(abs(p-pe))))
        if n%stride==0 or n==steps:
            records.append(dict(t=t,energy=E,relative_energy_defect=(E-E0)/E0,
                total_charge=(u[-1]-u[0])/(2*PI),detector_current=-p[ic]/(2*PI),
                detector_charge=cut,integrated_current=integrated_current,charge_right=qright,
                center=X,center_velocity=vel))
            if save_fields:fields.append(u.copy())
        if kind=='kink' and stop_on_exit and n>0:
            if X>16. and vel>0 and cut>.995:state='transmitted';break
            if X<-16. and vel<0 and abs(cut)<.005:state='reflected';break
        if n==steps:break
        ph=p+.5*dt*acc;unew=u+dt*ph;anew=accel(unew);pnew=ph+.5*dt*anew
        jj=-pnew[ic]/(2*PI);integrated_current+=.5*dt*(prev_j+jj);prev_j=jj
        u,p,acc=unew,pnew,anew
    if records[-1]['t']!=t:
        records.append(dict(t=t,energy=E,relative_energy_defect=(E-E0)/E0,total_charge=(u[-1]-u[0])/(2*PI),
            detector_current=-p[ic]/(2*PI),detector_charge=cut,integrated_current=integrated_current,
            charge_right=qright,center=X,center_velocity=vel))
        if save_fields:fields.append(u.copy())
    if kind=='breather':state='neutral_control'
    # Numerical acceptance is part of the search decision, not merely a report
    # written after a physical label has already updated the bracket.
    outcome_raw=state
    acceptance_limits=dict(energy_relative=1e-3,flux_absolute=1e-3,charge_absolute=1e-14)
    reasons=[]
    charge_final=(u[-1]-u[0])/(2*PI)
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(p)) or not np.all(np.isfinite([E,E0,Eerr,max_flux_defect])):
        reasons.append('nonfinite_field_or_diagnostic')
    if kind=='kink' and not np.all(np.isfinite([X,vel])):
        reasons.append('nonfinite_or_nonunique_kink_center')
    if Eerr/E0>acceptance_limits['energy_relative']:
        reasons.append('energy_defect_exceeds_limit')
    if max_flux_defect>acceptance_limits['flux_absolute']:
        reasons.append('flux_defect_exceeds_limit')
    if abs(charge_final-Q0)>acceptance_limits['charge_absolute']:
        reasons.append('winding_drift_exceeds_limit')
    accepted=not reasons
    if not accepted:state='rejected_numerics'
    summary=dict(tag=tag,G=G,width=width,dx=dx,dt=dt,v=v,x0=x0,L=L,tmax=tmax,kind=kind,
        outcome=state,outcome_raw=outcome_raw,numerics_accepted=accepted,
        numerical_rejection_reason=';'.join(reasons),
        acceptance_energy_relative=acceptance_limits['energy_relative'],
        acceptance_flux_absolute=acceptance_limits['flux_absolute'],
        acceptance_charge_absolute=acceptance_limits['charge_absolute'],
        verlet_stability_number=stability_number,
        t_final=t,center_final=X,velocity_final=vel,detector_x=detector,
        detector_charge=cut,integrated_detector_current=integrated_current,
        max_flux_defect=max_flux_defect,max_local_balance_defect=max_local_balance_defect,
        total_charge_initial=Q0,total_charge_final=charge_final,
        energy_initial=E0,energy_final=E,max_energy_relative_defect=Eerr/E0,
        free_exact_max_phase_error=max_phase_error,free_exact_max_velocity_error=max_velocity_error,
        steps=n,force_evaluations=n+1,grid_points=len(x),runtime_seconds=time.perf_counter()-begin)
    fields_out=dict(x=x,t=np.array([z['t'] for z in records]),u=np.array(fields),u0=uinit,uf=u)
    return summary,records,fields_out

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def predict_threshold(v=.35,width=3.):
    B,err=quad(lambda x:2*sech(x/width)**2*sech(x)**2,-80,80,epsabs=2e-12,epsrel=2e-12)
    return dict(v=v,width=width,kink_rest_energy=8.,incoming_energy=8/np.sqrt(1-v*v),
        barrier_per_unit_G=B,quadrature_error_estimate=err,G_predict=(8/np.sqrt(1-v*v)-8)/B,
        approximation='Undeformed static-width kink at barrier top; ignores profile relaxation and radiation.')

def scan(Gs,**kw):
    rows=[]
    for G in Gs:
        r,_,_=solve_gate(G=float(G),**kw);rows.append(r)
        print('SCAN',G,r['outcome'],r['center_final'],r['t_final'],flush=True)
    return rows

def local_bracket(lo,hi,tol=.0005,dx=.1,dt=.04,tmax=240.,max_iterations=20):
    rows=[]
    for G in [lo,hi]:
        r,_,_=solve_gate(G=G,dx=dx,dt=dt,tmax=tmax);rows.append(r)
    if rows[0]['outcome']!='transmitted' or rows[1]['outcome']!='reflected':
        raise ValueError('A local bracket requires observed T on low G and R on high G')
    iterations=0;reason='width_tolerance'
    while hi-lo>tol and iterations<max_iterations:
        G=.5*(lo+hi);r,_,_=solve_gate(G=G,dx=dx,dt=dt,tmax=tmax);rows.append(r);iterations+=1
        print('BISECT',dx,dt,G,r['outcome'],r['t_final'],flush=True)
        if r['outcome']=='transmitted':lo=G
        elif r['outcome']=='reflected':hi=G
        else:reason='numerical_rejection' if r['outcome']=='rejected_numerics' else 'unresolved_midpoint';break
    if hi-lo>tol and iterations>=max_iterations and reason=='width_tolerance':
        reason='iteration_budget'
    return dict(G_transmitted=lo,G_reflected=hi,bracket_width=hi-lo,tolerance=tol,
        iterations=iterations,PDE_calls=len(rows),stop_reason=reason,dx=dx,dt=dt,tmax=tmax,
        runtime_seconds=sum(r['runtime_seconds'] for r in rows),force_evaluations=sum(r['force_evaluations'] for r in rows),
        note='Local sampled transition bracket; not a proof of a unique global threshold.'),rows

def plots(out):
    with open(out/'gate_screen.csv') as f:screen=list(csv.DictReader(f))
    with open(out/'gate_refinement.csv') as f:ref=list(csv.DictReader(f))
    with open(out/'gate_controls.csv') as f:ctrl=list(csv.DictReader(f))
    summ=json.loads((out/'gate_summary.json').read_text());pred=summ['predictor'];br=summ['production_bracket']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','font.family':'DejaVu Sans'})
    colors={'transmitted':'#167d8d','reflected':'#c4552d','unresolved':'#777777'}
    fig,ax=plt.subplots(1,2,figsize=(9.8,3.5),constrained_layout=True)
    for state in colors:
        rr=[r for r in screen if r['outcome']==state]
        if rr:ax[0].plot([float(r['G']) for r in rr],[float(r['detector_charge']) for r in rr],'o',label=state,color=colors[state])
    ax[0].axvline(pred['G_predict'],ls='--',color='0.3',label='Rigid-kink predictor')
    ax[0].axvspan(br['G_transmitted'],br['G_reflected'],color='#e5ad31',alpha=.7,label='PDE local bracket')
    ax[0].set(xlabel='Gate amplitude G',ylabel='Charge crossing detector',title='(a) Sampled gate response');ax[0].legend(fontsize=7)
    for i,r in enumerate(ref):
        lo=float(r['G_transmitted']);hi=float(r['G_reflected']);ax[1].errorbar((lo+hi)/2,i,xerr=(hi-lo)/2,fmt='o',capsize=4)
    ax[1].set(yticks=range(len(ref)),yticklabels=[f"h={float(r['dx']):g}, dt={float(r['dt']):g}" for r in ref],xlabel='Local threshold bracket',title='(b) Independent space/time refinement')
    for ext in ['pdf','png']:fig.savefig(out/f'gate_threshold.{ext}',dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(10,6.8),constrained_layout=True)
    for name,c in [('on','#167d8d'),('off','#c4552d')]:
        with open(out/f'gate_{name}_trace.csv') as f:rr=list(csv.DictReader(f))
        tt=np.array([float(z['t']) for z in rr]);label='T: G=0.10' if name=='on' else 'R: G=0.25'
        ax[0,0].plot(tt,[float(z['center']) for z in rr],color=c,label=label)
        ax[0,1].plot(tt,[float(z['detector_charge']) for z in rr],color=c,label=label)
        ax[1,0].plot(tt,[float(z['relative_energy_defect']) for z in rr],color=c,label=label)
        ax[1,1].plot(tt,[float(z['total_charge']) for z in rr],color=c,label=label)
    ax[0,0].axhline(0,color='0.7',lw=.6);ax[0,0].set(xlabel='Time',ylabel='Kink center',title='(a) Reflection and transmission')
    ax[0,1].set(xlabel='Time',ylabel='Integrated topological current',title='(b) Detector response at x=10')
    ax[1,0].set(xlabel='Time',ylabel='(H(t)-H(0))/H(0)',title='(c) Discrete energy audit')
    ax[1,1].set(xlabel='Time',ylabel='Total winding Q',ylim=(-.1,1.15),title='(d) Both outcomes preserve Q=1')
    with open(out/'gate_breather_trace.csv') as f:rr=list(csv.DictReader(f))
    ax[1,1].plot([float(z['t']) for z in rr],[float(z['total_charge']) for z in rr],ls='--',c='0.4',label='Breather control Q=0')
    for a in ax.flat:a.legend(fontsize=7)
    for ext in ['pdf','png']:fig.savefig(out/f'gate_diagnostics.{ext}',dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    for a,name in zip(ax,['on','off']):
        z=np.load(out/f'gate_{name}_fields.npz');mask=abs(z['x'])<45
        mesh=a.pcolormesh(z['x'][mask],z['t'],z['u'][:,mask]/(2*PI),shading='auto',cmap='viridis',vmin=0,vmax=1)
        a.axvline(0,color='w',ls='--',lw=.8);a.axvline(10,color='w',ls=':',lw=.8)
        a.set(xlabel='Position x',ylabel='Time',title='Transmitted: G=0.10' if name=='on' else 'Reflected: G=0.25')
    fig.colorbar(mesh,ax=ax,label='u/(2π)')
    for ext in ['pdf','png']:fig.savefig(out/f'gate_spacetime.{ext}',dpi=200)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results');ap.add_argument('--pilot',action='store_true');args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();predictor=predict_threshold();print('PREDICT',predictor,flush=True)
    if args.pilot:
        rows=scan(np.linspace(0,.3,13));write_csv(out/'gate_pilot.csv',rows);return
    screen=scan(np.linspace(0,.3,13));write_csv(out/'gate_screen.csv',screen)
    transitions=[(screen[i]['G'],screen[i+1]['G']) for i in range(len(screen)-1) if screen[i]['outcome']=='transmitted' and screen[i+1]['outcome']=='reflected']
    if len(transitions)!=1:raise RuntimeError('Report multiple or absent local windows before refining.')
    lo,hi=transitions[0];prod,prows=local_bracket(lo,hi,tol=.0002)
    write_csv(out/'gate_bisection.csv',prows)
    refinements=[prod];allref=prows[:]
    for h,dt in [(.2,.02),(.1,.02),(.05,.02),(.1,.01)]:
        b,rr=local_bracket(lo,hi,tol=.0002,dx=h,dt=dt);refinements.append(b);allref+=rr
    write_csv(out/'gate_refinement.csv',refinements);write_csv(out/'gate_refinement_runs.csv',allref)
    controls=[]
    for tag,G in [('on',.1),('off',.25)]:
        s,r,f=solve_gate(G=G,save_fields=True,tag=tag);controls.append(s);write_csv(out/f'gate_{tag}_trace.csv',r);np.savez_compressed(out/f'gate_{tag}_fields.npz',**f)
    for h,dt in [(.2,.02),(.1,.02),(.05,.02),(.1,.04),(.1,.01)]:
        s,_,_=solve_gate(G=0.,dx=h,dt=dt,tmax=80.,stop_on_exit=False,tag='free_kink');controls.append(s)
    s,r,f=solve_gate(kind='breather',G=0.,dx=.05,dt=.02,tmax=80.,stop_on_exit=False,save_fields=True,tag='breather');controls.append(s);write_csv(out/'gate_breather_trace.csv',r);np.savez_compressed(out/'gate_breather_fields.npz',**f)
    # Finite observation window must leave sufficiently slow trajectories undecided.
    s,_,_=solve_gate(G=.5*(prod['G_transmitted']+prod['G_reflected']),tmax=95.,tag='short_window');controls.append(s)
    # Same center-region outcome on a wider domain tests finite-domain sensitivity.
    s,_,_=solve_gate(G=.1,L=400.,tag='wider_domain');controls.append(s)
    write_csv(out/'gate_controls.csv',controls)
    # Fair baseline: dense uniform sweep over the same pre-screened local interval,
    # at the same requested width. No global monotonicity certificate is assumed.
    ninterval=int(np.ceil((hi-lo)/prod['bracket_width']));dense=scan(np.linspace(lo,hi,ninterval+1));write_csv(out/'gate_uniform_baseline.csv',dense)
    cost=dict(initial_interval=[lo,hi],screen_PDE_calls=len(screen),
        bisection_calls=len(prows),uniform_calls=len(dense),bisection_force_evaluations=sum(r['force_evaluations'] for r in prows),
        uniform_force_evaluations=sum(r['force_evaluations'] for r in dense),
        bisection_runtime_seconds=sum(r['runtime_seconds'] for r in prows),uniform_runtime_seconds=sum(r['runtime_seconds'] for r in dense),
        cost_scope='After the same coarse screen. This compares sampling strategies, not time integrators; an accurate prior bracket benefits both.')
    on=next(r for r in controls if r['tag']=='on');off=next(r for r in controls if r['tag']=='off');short=next(r for r in controls if r['tag']=='short_window')
    gates=dict(coarse_single_T_to_R_transition=len(transitions)==1,production_width=prod['bracket_width']<=.0002,
        on_transmitted=on['outcome']=='transmitted',off_reflected=off['outcome']=='reflected',
        short_window_unresolved=short['outcome']=='unresolved',
        energy=max(r['max_energy_relative_defect'] for r in allref+controls+screen)<1e-3,
        topological_charge=all(abs(r['total_charge_final']-r['total_charge_initial'])<1e-14 for r in allref+controls+screen),
        flux=max(r['max_flux_defect'] for r in allref+controls+screen)<1e-3)
    gates={key:bool(value) for key,value in gates.items()}
    summary=dict(model='Inertial classical SG with a static anisotropy barrier',equation='u_tt-u_xx+[1+G sech^2(x/3)] sin u=0',
        boundary='Fixed u(-160)=0,u(160)=2pi; breather control has 0 at both endpoints.',
        normalization='Background wave speed and inverse gap are 1; background kink rest energy 8.',
        initial_state='Continuum Lorentz-contracted kink, v=0.35, center=-24; endpoint values imposed.',
        topological_current='rho=u_x/(2pi), j=-u_t/(2pi). Not a calibrated electrical or microscopic spin current.',
        predictor=predictor,production_bracket=prod,refinement_brackets=refinements,cost=cost,
        on=on,off=off,short_window=short,verification=gates,all_passed=all(gates.values()),
        limitations=['Static preset gate; gate-switching energy and bandwidth unmodeled.','No damping, reservoirs, thermal noise, disorder, charge contacts or experimental calibration.',
        'A finite parameter screen does not prove global monotonicity or exclude narrow resonance windows.',
        'Dirichlet endpoints enforce total winding; a separate detector-current quadrature tests local continuity numerically.',
        'The standard integrator is not new; algorithmic novelty/priority of the workflow is not established.'],
        runtime_seconds=time.perf_counter()-start,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'gate_summary.json').write_text(json.dumps(summary,indent=2));plots(out);print(json.dumps(summary,indent=2));
    if not summary['all_passed']:raise SystemExit('Gate verification failed; inspect before reuse.')
if __name__=='__main__':main()
