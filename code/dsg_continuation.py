#!/usr/bin/env python3
"""Classical geometric transport in a phase-driven double sine-Gordon ring.

This script independently generates every dsg_* result and figure.  All variables
are dimensionless. It does not model a quantum many-body spin chain.  The
Landen transformation is used only as a baseline representation identity,
not as the cause of a physical time evolution.

Run: python code/dsg_continuation.py --output results
Dependencies: numpy, scipy, matplotlib.  --quick runs a reduced smoke study.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import simpson, cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ellipk, ellipe, ellipj
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TWOPI = 2*np.pi


def potential(u, eta, chi):
    return 1-np.cos(u)+eta*(1-np.cos(2*u+chi))


def force(u, eta, chi):
    return np.sin(u)+2*eta*np.sin(2*u+chi)


@dataclass
class Branch:
    eta: float
    chi: float
    L: float
    angle: np.ndarray
    x: np.ndarray
    speed: np.ndarray
    phase_derivative_x: np.ndarray
    pressure: float
    stress: float
    vacuum: float
    action: float
    compliance: float
    connection: float
    length_error: float
    connection_endpoint_error: float

    def field(self, points):
        """Lifted monotone field with q(0)=pi and q(x+L)=q(x)+2pi."""
        z=np.asarray(points)
        y=np.mod(z,self.L)
        cycles=np.floor(z/self.L)
        # Enforce the already quadrature-verified endpoint to exactly L.
        xx=self.x.copy();xx[-1]=self.L
        return CubicSpline(xx,self.angle)(y)+TWOPI*cycles

    def record(self):
        return dict(eta=self.eta,chi=self.chi,L=self.L,
                    pressure_excess=self.pressure,stress_C=self.stress,
                    potential_minimum=self.vacuum,action_A=self.action,
                    compliance_B=self.compliance,
                    longwave_coefficient=self.L**2/(self.action*self.compliance),
                    compression_modulus=self.L/self.compliance,
                    connection=self.connection,length_error=self.length_error,
                    connection_endpoint_error=self.connection_endpoint_error)


def static_branch(eta,chi,L,angle_nodes=8193):
    """Solve L=int du/sqrt(2(V+C)) on one lifted turn by quadrature.

    P=C+min(V)>0 is found by a scalar monotone root solve.  The fixed phase
    q(0)=pi is a coordinate choice, not a physical boundary pinning condition.
    The phase derivative is analytic under the quadrature:
      C_chi=-int V_chi/s^3 / int 1/s^3,
      x_chi=-int_pi^u (V_chi+C_chi)/s^3,
      connection=-int s*x_chi / int s.
    """
    angle=np.linspace(np.pi,3*np.pi,angle_nodes)
    val=potential(angle,eta,chi)
    idx=int(np.argmin(val)); da=angle[1]-angle[0]
    minimum=minimize_scalar(lambda z:potential(z,eta,chi),
               bounds=(angle[idx]-da,angle[idx]+da),method='bounded',
               options={'xatol':1e-14})
    vmin=float(minimum.fun)
    shifted=np.maximum(val-vmin,0.0)
    def length(P):
        return simpson(1/np.sqrt(2*(shifted+P)),x=angle)
    lower=1e-12
    # Uniform-angle quadrature is reliable for the declared finite study, not
    # an arbitrary separatrix limit. Never silently fabricate a bracket.
    if length(lower)<=L:
        raise ValueError('Unresolved narrow vacuum: increase angle_nodes or reduce L.')
    upper=max(1.,2*(TWOPI/L)**2)
    while length(upper)>L: upper*=4
    P=brentq(lambda p:length(p)-L,lower,upper,xtol=2e-14,rtol=2e-14)
    speed=np.sqrt(2*(shifted+P))
    x=cumulative_simpson(1/speed,x=angle,initial=0)
    B=float(simpson(speed**-3,x=angle))
    vchi=eta*np.sin(2*angle+chi)
    Cchi=-simpson(vchi/speed**3,x=angle)/B
    xchi=-cumulative_simpson((vchi+Cchi)/speed**3,x=angle,initial=0)
    A=float(simpson(speed,x=angle))
    connection=float(-simpson(speed*xchi,x=angle)/A)
    return Branch(eta,chi,L,angle,x,speed,xchi,float(P),float(P-vmin),vmin,
                  A,B,connection,float(x[-1]-L),float(xchi[-1]))


def sg_analytic(L):
    k=brentq(lambda k:2*k*ellipk(k*k)-L,1e-8,1-1e-12,xtol=1e-14)
    K=ellipk(k*k);E=ellipe(k*k)
    return dict(k=k,pressure=2*(1/k**2-1),action=8*E/k,
                compliance=k**3*E/(2*(1-k*k)),L=2*k*K)


def hessian_spectrum(branch,N=128):
    x=np.arange(N)*branch.L/N
    u=branch.field(x)
    w=u-TWOPI*x/branch.L
    k=TWOPI*np.fft.fftfreq(N,d=branch.L/N)
    lap=np.fft.ifft(-(k*k)[:,None]*np.fft.fft(np.eye(N),axis=0),axis=0).real
    matrix=-lap+np.diag(np.cos(u)+4*branch.eta*np.cos(2*u+branch.chi))
    eigenvalues,eigenvectors=eigh(matrix,subset_by_index=(0,5))
    qx=TWOPI/branch.L+np.fft.ifft(1j*k*np.fft.fft(w)).real
    qx/=np.linalg.norm(qx)
    overlap=abs(float(np.dot(qx,eigenvectors[:,0])))
    translation_residual=float(np.linalg.norm(matrix@qx))
    return dict(eta=branch.eta,chi=branch.chi,L=branch.L,N=N,
                gap=float(eigenvalues[1]),
                translation_overlap=overlap,
                translation_residual=translation_residual,
                **{f'lambda{i}':float(v) for i,v in enumerate(eigenvalues)})


def ramp(t,T,orientation=1):
    z=float(np.clip(t/T,0,1))
    chi=orientation*TWOPI*(3*z*z-2*z*z*z)
    rate=orientation*TWOPI*6*z*(1-z)/T if 0<t<T else 0.
    return chi,rate


def run_pde(eta=.2,L=8.,N=128,T=100.,dt=.05,relax=50.,orientation=1,
            angle_nodes=8193,save_fields=False):
    """Fourier ETDRK4 integration of u_t=u_xx-V_u with one spatial winding.

    Write u=2*pi*x/L+w, w periodic. ETDRK4's four evaluations use the actual
    forcing times (t,t+h/2,t+h/2,t+h). The zero Fourier mode is not suppressed:
    suppressing it would artificially remove the transported phase.
    """
    nsteps=int(round((T+relax)/dt))
    if abs(nsteps*dt-(T+relax))>1e-10:
        raise ValueError('T+relax must be a multiple of dt')
    x=np.arange(N)*L/N;base=TWOPI*x/L
    modes=TWOPI*np.fft.fftfreq(N,d=L/N)
    lin=-modes*modes
    e=np.exp(dt*lin);e2=np.exp(dt*lin/2)
    roots=np.exp(1j*np.pi*(np.arange(1,33)-.5)/32)
    lr=dt*lin[:,None]+roots[None,:]
    q=dt*np.real(np.mean(np.expm1(lr/2)/lr,axis=1))
    f1=dt*np.real(np.mean((-4-lr+np.exp(lr)*(4-3*lr+lr**2))/lr**3,axis=1))
    f2=dt*np.real(np.mean((2+lr+np.exp(lr)*(-2+lr))/lr**3,axis=1))
    f3=dt*np.real(np.mean((-4-3*lr-lr**2+np.exp(lr)*(4-lr))/lr**3,axis=1))
    branch=static_branch(eta,0,L,angle_nodes)
    u0=branch.field(x); v=np.fft.fft(u0-base)
    stride=max(1,nsteps//1000)
    dissipation=np.empty(nsteps+1);work=np.empty(nsteps+1)
    times=np.linspace(0,T+relax,nsteps+1)
    records=[];snapshots=[]
    projection_max=0.;initial_energy=None;cycle_charge=np.nan
    def nonlinear(z,t):
        u=base+np.fft.ifft(z).real
        chi,_=ramp(t,T,orientation)
        return np.fft.fft(-force(u,eta,chi))
    def diagnose(z,t,index):
        nonlocal projection_max,initial_energy,cycle_charge
        u=base+np.fft.ifft(z).real
        ux=TWOPI/L+np.fft.ifft(1j*modes*z).real
        chi,chidot=ramp(t,T,orientation)
        ut=np.fft.ifft(lin*z).real-force(u,eta,chi)
        E=L*np.mean(.5*ux*ux+potential(u,eta,chi))
        D=L*np.mean(ut*ut)
        W=L*chidot*np.mean(eta*np.sin(2*u+chi))
        projection=L*np.mean(ux*ut)
        projection_max=max(projection_max,abs(float(projection)))
        charge=-(float(np.mean(u-u0)))/TWOPI
        dissipation[index]=D;work[index]=W
        if initial_energy is None:initial_energy=E
        if abs(t-T)<dt/4:cycle_charge=charge
        if index%stride==0 or index==nsteps:
            records.append(dict(t=t,chi=chi,chi_rate=chidot,energy=E,
                dissipation=D,input_power=W,charge_average=charge,
                charge_cut0=-(u[0]-u0[0])/TWOPI,
                winding=L*np.mean(ux)/TWOPI,translation_projection=projection))
            if save_fields:snapshots.append(u.copy())
        return u,E,charge
    diagnose(v,0.,0)
    for i in range(nsteps):
        t=i*dt
        nv=nonlinear(v,t)
        a=e2*v+q*nv;na=nonlinear(a,t+dt/2)
        b=e2*v+q*na;nb=nonlinear(b,t+dt/2)
        c=e2*a+q*(2*nb-nv);nc=nonlinear(c,t+dt)
        v=e*v+f1*nv+2*f2*(na+nb)+f3*nc
        uf,Ef,charge=diagnose(v,(i+1)*dt,i+1)
    total_diss=float(simpson(dissipation,dx=dt))
    total_work=float(simpson(work,dx=dt))
    balance=float(Ef-initial_energy+total_diss-total_work)
    X=L*charge
    reference=branch.field(x-X)
    shape_error=float(np.sqrt(np.mean((uf-reference)**2)))
    local=-(uf-u0)/TWOPI
    summary=dict(eta=eta,L=L,N=N,T=T,dt=dt,relax=relax,orientation=orientation,
        charge_average=float(charge),charge_at_cycle_end=float(cycle_charge),
        displacement_from_average=float(X),charge_cut0=float(local[0]),
        charge_cut_min=float(np.min(local)),charge_cut_max=float(np.max(local)),
        relaxed_shape_rms_error=shape_error,energy_initial=float(initial_energy),
        energy_final=float(Ef),dissipated_energy=total_diss,input_work=total_work,
        energy_balance_residual=balance,
        energy_balance_relative=abs(balance)/max(1.,abs(total_diss),abs(total_work)),
        translation_projection_max=projection_max,
        time_steps=nsteps)
    return summary,records,dict(x=x,t=np.array([r['t'] for r in records]),
             u=np.asarray(snapshots) if save_fields else np.empty((0,N)),u0=u0,uf=uf)


def write_csv(path,records):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def build_figures(out,static,geometric,dynamics,refinement,trace,fields):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.bbox':'tight','figure.dpi':150})
    col=['#156082','#E97132','#196B24','#A02B93']
    fig,ax=plt.subplots(2,2,figsize=(10.8,7.5))
    for j,eta in enumerate([0.,.1,.2,.3]):
        rows=[r for r in static if np.isclose(r['eta'],eta)]
        ax[0,0].plot(np.array([r['chi'] for r in rows])/np.pi,
                     [r['connection'] for r in rows],label=f'$\\eta={eta:g}$',c=col[j])
    ax[0,0].set(xlabel='$\\chi/\\pi$',ylabel='Connection $\\mathcal{A}(\\chi)$',
                title='(a) Quasistatic translation connection')
    ax[0,0].legend(frameon=False,ncol=2)
    ax[0,1].plot([r['eta'] for r in geometric],[r['charge_average'] for r in geometric],
                 'o-',c=col[0],label='Quadrature prediction')
    ax[0,1].set(xlabel='Second-harmonic amplitude $\\eta$',
               ylabel='$\\overline{Q}_{\\mathrm{ad}}=\\Delta X/L$',
               title='(b) Classical transport; $L=8$')
    rows=[r for r in dynamics if r['eta']==.2 and r['orientation']==1]
    ad=next(r['charge_average'] for r in geometric if r['eta']==.2)
    ax[1,0].plot([r['T'] for r in rows],[r['charge_average'] for r in rows],'o-',c=col[1],label='PDE + endpoint relaxation')
    ax[1,0].axhline(ad,c=col[0],ls='--',label='Adiabatic prediction')
    ax[1,0].set(xlabel='Cycle duration $T$',ylabel='Average transported charge',
               title='(c) Finite-rate approach')
    ax[1,0].legend(frameon=False,fontsize=8)
    T=np.array([r['T'] for r in rows]);err=np.abs(np.array([r['charge_average'] for r in rows])-ad)
    ax[1,1].loglog(T,err,'o-',c=col[1],label='Measured error')
    ax[1,1].loglog(T,err[-2]*(T[-2]/T)**2,'--',c='0.4',label='$T^{-2}$ guide')
    ax[1,1].set(xlabel='Cycle duration $T$',ylabel='$|\\overline{Q}(T)-\\overline{Q}_{ad}|$',
               title='(d) Rate error for the smooth ramp')
    ax[1,1].legend(frameon=False)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(out/f'dsg_geometric_transport.{ext}')
    plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(10.8,7.2))
    ta=np.array([r['t'] for r in trace]); qa=np.array([r['charge_average'] for r in trace])
    ax[0,0].plot(ta,qa,c=col[0],label='Spatial average')
    ax[0,0].plot(ta,[r['charge_cut0'] for r in trace],c=col[1],label='One spatial cut')
    ax[0,0].set(xlabel='Time',ylabel='Integrated current',title='(a) A local cut is not the spatial mean')
    ax[0,0].legend(frameon=False,fontsize=8)
    ax[0,1].plot(ta,[r['input_power'] for r in trace],c=col[1],label='Drive power')
    ax[0,1].plot(ta,[r['dissipation'] for r in trace],c=col[0],label='Dissipation')
    ax[0,1].set(xlabel='Time',ylabel='Power',title='(b) Explicit forcing and dissipation')
    ax[0,1].legend(frameon=False,fontsize=8)
    mesh=ax[1,0].pcolormesh(fields['x'],fields['t'],fields['u']-fields['u0'][None,:],shading='auto',cmap='RdBu_r')
    fig.colorbar(mesh,ax=ax[1,0],label='$u(x,t)-u(x,0)$')
    ax[1,0].set(xlabel='Position $x$',ylabel='Time',title='(c) Actual ETDRK4 trajectory')
    for eta in [0.,.2,.3]:
        rows=[r for r in static if np.isclose(r['eta'],eta)]
        ax[1,1].plot(np.array([r['chi'] for r in rows])/np.pi,[r['longwave_coefficient'] for r in rows],label=f'$\\eta={eta:g}$')
    ax[1,1].set(xlabel='$\\chi/\\pi$',ylabel='$L^2/(AB)$',title='(d) Static long-wavelength coefficient')
    ax[1,1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(out/f'dsg_diagnostics.{ext}')
    plt.close(fig)


def summarize(out,static,geo,limits,spectra,georef,dynamics,refinement,runtime_seconds=None):
    ad=next(r['charge_average'] for r in geo if r['eta']==.2)
    plus=next(r for r in dynamics if r['eta']==.2 and r['T']==200. and r['orientation']==1)
    reverse=next(r for r in dynamics if r['orientation']==-1)
    zero=next(r for r in dynamics if r['eta']==0.)
    checks=dict(sg_limit_max_pressure_relative=max(r['pressure_relative_error'] for r in limits),
       sg_limit_max_action_relative=max(r['action_relative_error'] for r in limits),
       sg_limit_max_compliance_relative=max(r['compliance_relative_error'] for r in limits),
       sg_limit_max_field_error=max(r['field_max_error'] for r in limits),
       hessian_min_lambda0=min(r['lambda0'] for r in spectra),
       hessian_max_translation_residual=max(r['translation_residual'] for r in spectra),
       hessian_min_translation_overlap=min(r['translation_overlap'] for r in spectra),
       hessian_min_positive_gap=min(r['gap'] for r in spectra),
       geometric_finest_change=abs(georef[-1]['charge_average']-georef[-2]['charge_average']),
       eta_zero_transported_charge=zero['charge_average'],
       cycle_orientation_antisymmetry_error=abs(plus['charge_average']+reverse['charge_average']),
       max_main_energy_balance_relative=max(r['energy_balance_relative'] for r in dynamics),
       max_main_translation_projection=max(r['translation_projection_max'] for r in dynamics),
       max_main_relaxed_shape_rms=max(r['relaxed_shape_rms_error'] for r in dynamics),
       adiabatic_charge_eta02_L8=ad)
    assertions=dict(sg_pressure=checks['sg_limit_max_pressure_relative']<1e-8,
       sg_action=checks['sg_limit_max_action_relative']<1e-9,
       sg_field=checks['sg_limit_max_field_error']<1e-7,
       stationary_stability=checks['hessian_min_lambda0']>-1e-7,
       zero_control=abs(checks['eta_zero_transported_charge'])<1e-8,
       orientation=checks['cycle_orientation_antisymmetry_error']<1e-8,
       energy_balance=checks['max_main_energy_balance_relative']<2e-6,
       geometric_refinement=checks['geometric_finest_change']<1e-7)
    assertions={key:bool(value) for key,value in assertions.items()}
    metadata=dict(model='Overdamped classical phase-driven double sine-Gordon ring',
       potential='1-cos(u)+eta*(1-cos(2*u+chi))',boundary='u(x+L)=u(x)+2pi',
       phase_gauge='q(0;chi)=pi',ramp='chi=orientation*2pi*(3s^2-2s^3), s=t/T',
       current='rho=u_x/(2pi), j=-u_t/(2pi)',
       transport='Q_average=-(mean(u_final)-mean(u_initial))/(2pi)=DeltaX/L after shape relaxation',
       note='Classical topological charge transport is not quantized quantum spin pumping. Static quadrature and ETDRK4 are independently evaluated.',
       runtime_seconds=runtime_seconds,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       static_cases=len(static),dynamic_cases=len(dynamics),refinement_cases=len(refinement),
       weak_amplitude_cases=5,weak_amplitude_chi_points=129,
       checks=checks,assertions=assertions,all_passed=all(assertions.values()))
    (out/'dsg_verification.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)
    if not metadata['all_passed']:raise SystemExit('A declared DSG verification gate failed; inspect output before reuse.')


def weak_amplitude_study(out):
    """Evenness control and small-amplitude coefficient; not a quantum invariant."""
    rows=[]
    for eta in [.005,.01,.02,.04,-.01]:
        phase=np.linspace(0,TWOPI,129)
        conn=[static_branch(eta,c,8.,4097).connection for c in phase]
        charge=float(simpson(conn,x=phase)/8.)
        rows.append(dict(eta=eta,L=8.,angle_nodes=4097,chi_points=129,
                         charge_average=charge,charge_over_eta_squared=charge/eta**2))
    write_csv(out/'dsg_weak_amplitude.csv',rows)
    return rows


def length_sweep_study(out,include_pde=True):
    """Finite-density study at fixed eta; both displacement and mean charge."""
    rows=[];pd=[];branch_records=[]
    cases=[(4.,8193,257),(6.,8193,257),(8.,8193,257),(10.,8193,257),
           (12.,8193,257),(12.,16385,513)]
    for length,nangle,nphase in cases:
        ch=np.linspace(0,TWOPI,nphase)
        branches=[static_branch(.2,c,length,nangle) for c in ch]
        branch_records.extend(dict(**b.record(),angle_nodes=nangle,chi_points=nphase) for b in branches)
        displacement=float(simpson([b.connection for b in branches],x=ch))
        rows.append(dict(eta=.2,L=length,winding_density=1/length,
               angle_nodes=nangle,chi_points=nphase,
               displacement=displacement,charge_average=displacement/length,
               minimum_excess_pressure=min(b.pressure for b in branches),
               maximum_length_error=max(abs(b.length_error) for b in branches)))
        print('LENGTH',rows[-1],flush=True)
    if include_pde:
        for length,step in [(4.,.05),(12.,.05),(4.,.025),(12.,.025)]:
            result,_,_=run_pde(L=length,T=400.,N=128,dt=step,angle_nodes=16385)
            pred=next(r for r in reversed(rows) if r['L']==length)['charge_average']
            result['adiabatic_charge_prediction']=pred
            result['finite_rate_charge_error']=result['charge_average']-pred
            pd.append(result);print('LENGTH PDE',result,flush=True)
        write_csv(out/'dsg_length_pde.csv',pd)
    write_csv(out/'dsg_length_sweep.csv',rows)
    write_csv(out/'dsg_length_branches.csv',branch_records)
    plot_length_study(out,rows,pd)
    if pd:verify_length_study(out,rows,pd)
    return rows,pd


def verify_length_study(out,rows,pd):
    checks=dict(L12_quadrature_charge_change=abs(rows[-1]['charge_average']-rows[-2]['charge_average']),
        maximum_length_error=max(r['maximum_length_error'] for r in rows),
        maximum_energy_residual=max(r['energy_balance_relative'] for r in pd),
        maximum_relaxed_shape_error=max(r['relaxed_shape_rms_error'] for r in pd),
        time_step_charge_changes={str(L):abs(
            next(r['charge_average'] for r in pd if r['L']==L and r['dt']==.05)-
            next(r['charge_average'] for r in pd if r['L']==L and r['dt']==.025)) for L in [4,12]})
    result=dict(checks=checks,main_lengths=[4,6,8,10,12],quadrature_refinement_cases=1,
        actual_PDE_cases=4,all_passed=checks['L12_quadrature_charge_change']<1e-8 and
        checks['maximum_energy_residual']<1e-7 and checks['maximum_relaxed_shape_error']<1e-7)
    (out/'dsg_length_verification.json').write_text(json.dumps(result,indent=2))
    if not result['all_passed']:raise ValueError('Length-study verification gate failed')
    return result


def plot_length_study(out,rows,pd):
    plotrows=rows[:5]
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.8))
    ll=[r['L'] for r in plotrows]
    axes[0].plot(ll,[r['displacement'] for r in plotrows],'o-',label='Adiabatic quadrature')
    axes[1].plot(ll,[r['charge_average'] for r in plotrows],'o-',label='Adiabatic quadrature')
    if pd:
        pd_fine=[r for r in pd if r['dt']==.025]
        axes[0].plot([r['L'] for r in pd_fine],[r['displacement_from_average'] for r in pd_fine],
                     'x',ms=9,label='PDE, T=400, h=0.025')
        axes[1].plot([r['L'] for r in pd_fine],[r['charge_average'] for r in pd_fine],
                     'x',ms=9,label='PDE, T=400, h=0.025')
    axes[0].set(xlabel='Cell length L',ylabel='Translation per cycle, Delta X',
                title='(a) Absolute profile displacement')
    axes[1].set(xlabel='Cell length L',ylabel='Average transported charge, Delta X / L',
                title='(b) Winding density changes transport')
    for ax in axes:
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'dsg_length_dependence.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results');parser.add_argument('--quick',action='store_true');parser.add_argument('--summarize-only',action='store_true');parser.add_argument('--length-only',action='store_true');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);start=time.time()
    if args.length_only:
        length_sweep_study(out)
        return
    if args.summarize_only:
        def read(name):
            with open(out/name) as f:
                return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        summarize(out,read('dsg_static_branches.csv'),read('dsg_geometric_prediction.csv'),
            read('dsg_sg_limit.csv'),read('dsg_hessian.csv'),read('dsg_geometric_refinement.csv'),
            read('dsg_pde_cycles.csv'),read('dsg_pde_refinement.csv'))
        return

    nchi=65 if args.quick else 257
    etas=[0.,.1,.2,.3] if args.quick else [0.,.05,.1,.15,.2,.25,.3]
    static=[];geo=[]
    for eta in etas:
        rows=[]
        for chi in np.linspace(0,TWOPI,nchi):
            row=static_branch(eta,chi,8.).record();rows.append(row);static.append(row)
        geo.append(dict(eta=eta,L=8.,chi_points=nchi,angle_nodes=8193,
            displacement=float(simpson([r['connection'] for r in rows],x=[r['chi'] for r in rows])),
            charge_average=float(simpson([r['connection'] for r in rows],x=[r['chi'] for r in rows])/8)))
        print('GEOMETRIC',geo[-1],flush=True)
    write_csv(out/'dsg_static_branches.csv',static);write_csv(out/'dsg_geometric_prediction.csv',geo)
    limits=[]
    for L in [4.,6.,8.,10.]:
        b=static_branch(0.,0.,L);a=sg_analytic(L)
        x=np.linspace(0,L,501)
        # SciPy returns the continuously lifted amplitude for real arguments.
        amp=ellipj(x/a['k'],a['k']**2)[3]
        exact=np.pi+2*amp
        limits.append(dict(L=L,k=a['k'],pressure_relative_error=abs(b.pressure/a['pressure']-1),
              action_relative_error=abs(b.action/a['action']-1),
              compliance_relative_error=abs(b.compliance/a['compliance']-1),
              field_max_error=float(np.max(abs(b.field(x)-exact)))))
    write_csv(out/'dsg_sg_limit.csv',limits)
    spectra=[]
    for eta in [0.,.2,.3]:
        for chi in np.linspace(0,TWOPI,9):
            spectra.append(hessian_spectrum(static_branch(eta,chi,8.),128))
    write_csv(out/'dsg_hessian.csv',spectra)
    georef=[]
    for nodes,nc in ([(4097,129),(8193,257)] if args.quick else [(2049,65),(4097,129),(8193,257),(16385,513)]):
        ch=np.linspace(0,TWOPI,nc)
        values=[static_branch(.2,c,8.,nodes).connection for c in ch]
        georef.append(dict(angle_nodes=nodes,chi_points=nc,charge_average=float(simpson(values,x=ch)/8)))
    write_csv(out/'dsg_geometric_refinement.csv',georef)
    weak_amplitude_study(out)
    if not args.quick:length_sweep_study(out)
    dynamics=[];trace=[];fields=None
    durations=[50.,200.] if args.quick else [25.,50.,100.,200.,400.,800.]
    for T in durations:
        summary,records,data=run_pde(T=T,save_fields=(T==200.))
        dynamics.append(summary)
        if T==200.:trace=records;fields=data
        print('PDE',summary,flush=True)
    for eta,orient in [(0.,1),(.2,-1),(.1,1),(.3,1)]:
        summary,_,_=run_pde(eta=eta,T=200.,orientation=orient)
        dynamics.append(summary);print('CONTROL',summary,flush=True)
    write_csv(out/'dsg_pde_cycles.csv',dynamics)
    write_csv(out/'dsg_pde_trace.csv',trace)
    np.savez_compressed(out/'dsg_pde_fields.npz',**fields)
    refinement=[]
    for N,dt in ([(64,.05),(128,.05),(128,.025)] if args.quick else [(32,.05),(64,.05),(128,.05),(256,.05),(128,.2),(128,.1),(128,.025)]):
        summary,_,_=run_pde(N=N,dt=dt,T=100.)
        refinement.append(summary);print('REFINEMENT',summary,flush=True)
    write_csv(out/'dsg_pde_refinement.csv',refinement)
    build_figures(out,static,geo,dynamics,refinement,trace,fields)
    summarize(out,static,geo,limits,spectra,georef,dynamics,refinement,time.time()-start)

if __name__=='__main__':main()
