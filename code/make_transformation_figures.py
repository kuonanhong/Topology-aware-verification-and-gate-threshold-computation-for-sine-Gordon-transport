#!/usr/bin/env python3
"""Rebuild four scientific figures exclusively from executed suite data."""
from pathlib import Path
import argparse,csv,json,hashlib,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformation_gate import atomic_write

ROOT=Path(__file__).resolve().parents[1]
C=['#167d8d','#c4552d','#5264a2','#6c7b3b','#95549b']

def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        rr=list(csv.DictReader(f))
    for r in rr:
        for k,v in r.items():
            try:r[k]=float(v)
            except (ValueError,TypeError):pass
    return rr

def col(rr,key):return np.array([r[key] for r in rr])

def save(fig,out,name):
    for ext in ['pdf','png']:
        buf=io.BytesIO();fig.savefig(buf,format=ext,dpi=220)
        atomic_write(out/f'{name}.{ext}',buf.getvalue())
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=ROOT/'results')
    ap.add_argument('--out',type=Path,default=ROOT/'results');args=ap.parse_args();r=args.results;out=args.out;out.mkdir(parents=True,exist_ok=True)
    s=json.loads((r/'transformation_gate_summary.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
    # Design and exact solution residual: analytic formulas sampled on grids;
    # the dynamical data are separate and do not masquerade as measurements.
    fig,ax=plt.subplots(1,3,figsize=(10.8,3.3),constrained_layout=True)
    x=np.linspace(-12,12,501);eta=.5;w=3.;J=1+eta/np.cosh(x/w)**2
    ax[0].plot(x,J,c=C[0],label=r'$\rho=K=J$');ax[0].plot(x,1/J,c=C[1],label=r'$A=1/J$')
    ax[0].set(xlabel='Physical position x',ylabel='Dimensionless coefficient',title='(a) Exact matching, eta=0.5');ax[0].legend(fontsize=8)
    for eta,c in zip([0.,.25,.5],C):
        ax[1].plot(x,x+eta*w*np.tanh(x/w),c=c,label=f'eta={eta:g}')
    ax[1].set(xlabel='Physical position x',ylabel='Virtual position f(x)',title='(b) Monotone coordinate map');ax[1].legend(fontsize=8)
    rr=rows(r/'matched_residuals.csv')
    for eta,c in zip([.25,.5],C):
        z=[a for a in rr if a['eta']==eta]
        ax[2].loglog(col(z,'h'),col(z,'discrete_residual_max'),'o-',c=c,label=f'FD eta={eta:g}')
    ax[2].set(xlabel='Physical mesh spacing h',ylabel='Max residual',title='(c) Exact-profile spatial residual')
    ax[2].text(.03,.06,f"Analytic residual < {max(a['continuum_residual_max'] for a in rr):.1e}",
        transform=ax[2].transAxes,fontsize=8);ax[2].legend(fontsize=8)
    save(fig,out,'transformation_design')
    fig,ax=plt.subplots(2,2,figsize=(9.8,6.6),constrained_layout=True)
    rr=rows(r/'matched_spatial_convergence.csv')
    for eta,c in zip([.25,.5],C):
        z=[a for a in rr if a['eta']==eta]
        ax[0,0].loglog(col(z,'h'),col(z,'exact_max_field_error'),'o-',c=c,label=f'eta={eta:g}')
    ax[0,0].set(xlabel='Mesh spacing h (dt=0.02)',ylabel='Max |u - u_exact|, t<=80',title='(a) Matched solution convergence');ax[0,0].legend(fontsize=8)
    rr=rows(r/'matched_temporal_convergence.csv')[:-1]
    ax[0,1].loglog(col(rr,'dt'),col(rr,'final_field_difference_vs_dt0005'),'o-',c=C[0])
    ax[0,1].set(xlabel='Time step dt (h=0.1)',ylabel='Final field error vs dt=0.005',title='(b) Temporal refinement, eta=0.5')
    dd=rows(r/'matched_delay.csv');xx=np.arange(len(dd));bw=.24
    ax[1,0].bar(xx-bw/2,col(dd,'measured_extra_time'),bw,color=C[0],label='Executed transit delay')
    ax[1,0].bar(xx+bw/2,col(dd,'exact_finite_section_delay'),bw,color=C[1],label='Exact finite-section delay')
    ax[1,0].set(xticks=xx,xticklabels=[f"eta={a['eta']:g}" for a in dd],ylabel='Extra time across x=-12 to +12',title='(c) Delay at h=0.05, dt=0.01')
    ax[1,0].legend(fontsize=8)
    for eta,c in zip([.25,.5],C):
        tr=rows(r/f'matched_eta{eta:g}_trace.csv')
        ax[1,1].plot(col(tr,'t'),col(tr,'center'),c=c,label=f'eta={eta:g}: executed')
        xgrid=np.linspace(-160,160,20001);fgrid=xgrid+eta*3*np.tanh(xgrid/3);y0=-24+eta*3*np.tanh(-8)
        xe=np.interp(.35*col(tr,'t')+y0,fgrid,xgrid)
        ax[1,1].plot(col(tr,'t'),xe,'--',c=c,lw=1,label=f'eta={eta:g}: exact')
    ax[1,1].set(xlabel='Time',ylabel='Physical kink center',title='(d) Matched kink slows locally');ax[1,1].legend(fontsize=7,ncol=2)
    save(fig,out,'matched_convergence_delay')
    fig,ax=plt.subplots(2,2,figsize=(10,6.7),constrained_layout=True)
    for G,c in zip([.1,.25],C):
        tr=rows(r/f'mapped_G{G:g}_h0.1_comparison_trace.csv')
        ax[0,0].plot(col(tr,'t'),col(tr,'mapped_center_y'),c=c,label=f'G={G:g}: mapped')
        ax[0,0].plot(col(tr,'t'),col(tr,'virtual_center_y'),'--',c=c,label=f'G={G:g}: virtual')
        ax[0,1].plot(col(tr,'t'),col(tr,'mapped_detector_charge'),c=c,label=f'G={G:g}: mapped')
        ax[0,1].plot(col(tr,'t'),col(tr,'virtual_detector_charge'),'--',c=c,label=f'G={G:g}: virtual')
        ax[1,0].plot(col(tr,'t'),col(tr,'mapped_current'),c=c,label=f'G={G:g}: mapped')
        ax[1,0].plot(col(tr,'t'),col(tr,'virtual_current'),'--',c=c,label=f'G={G:g}: virtual')
    ax[0,0].set(xlabel='Time',ylabel='Virtual kink center',title='(a) Same virtual scattering, eta=0.5')
    ax[0,1].set(xlabel='Time',ylabel='Charge passing the same mapped cut',title='(b) Cut-flux comparison')
    ax[1,0].set(xlabel='Time',ylabel='Topological cut current',title='(c) Current comparison at mapped cuts')
    cc=rows(r/'mapped_gate_comparison.csv')
    for G,c in zip([.1,.25],C):
        z=[a for a in cc if a['G']==G]
        ax[1,1].loglog(col(z,'h'),col(z,'max_field_difference_over_trace'),'o-',c=c,label=f'G={G:g}')
    ax[1,1].set(xlabel='Physical / virtual mesh spacing h',ylabel='Max mapped field discrepancy',title='(d) Finite-grid mismatch converges')
    for a in ax.flat:a.legend(fontsize=7,ncol=2)
    save(fig,out,'mapped_virtual_comparison')
    fig,ax=plt.subplots(2,3,figsize=(11.2,7),constrained_layout=True)
    rr=rows(r/'fixed_anisotropy_coefficients.csv')
    ax[0,0].plot(col(rr,'x'),col(rr,'K_fixed'),c='0.25',lw=2,label='K fixed')
    ax[0,0].plot(col(rr,'x'),col(rr,'rho_lambda0'),'--',c=C[1],label='rho, lambda=0')
    ax[0,0].plot(col(rr,'x'),col(rr,'rho_lambda1'),':',c=C[0],lw=2,label='rho, lambda=1')
    ax[0,0].plot(col(rr,'x'),col(rr,'A_lambda1'),c=C[2],label='A, lambda=1')
    ax[0,0].set(xlabel='Position x',ylabel='Dimensionless coefficient',title='(a) Fixed anisotropy, co-control');ax[0,0].legend(fontsize=7)
    for lam,c,label in [(0.,C[1],'lambda=0: reflected'),(1.,C[0],'lambda=1: transmitted')]:
        tr=rows(r/f'fixedK_lambda{lam:g}_trace.csv')
        ax[0,1].plot(col(tr,'t'),col(tr,'detector_charge'),c=c,label=label)
        ax[1,1].plot(col(tr,'t'),col(tr,'relative_energy_defect'),c=c,label=label)
        ax[1,2].plot(col(tr,'t'),col(tr,'total_charge'),c=c,label=label)
    ax[0,1].set(xlabel='Time',ylabel='Charge passing x=10',title='(b) Reflected / transmitted presets');ax[0,1].legend(fontsize=7)
    ss=rows(r/'fixed_anisotropy_screen.csv')
    for a in ss:
        ax[0,2].scatter(a['lam'],a['detector_charge'],c=C[0] if a['outcome']=='transmitted' else C[1],s=40)
    pred=s['fixed_K_trial_energy_predictor']['lambda_predict']
    ax[0,2].axvline(pred,c='0.3',ls='--',label=f'Trial predictor {pred:.4f}')
    ax[0,2].set(xlabel='Co-control setting lambda',ylabel='Final cut charge',title='(c) Five-point response screen');ax[0,2].legend(fontsize=7)
    bb=rows(r/'fixed_anisotropy_brackets.csv')
    for i,a in enumerate(bb):
        lo=a['lambda_reflected'];hi=a['lambda_transmitted']
        ax[1,0].errorbar(.5*(lo+hi),i,xerr=.5*(hi-lo),fmt='o',capsize=4,c=C[i])
    ax[1,0].set(yticks=np.arange(len(bb)),yticklabels=[f"h={a['h']:g}, dt={a['dt']:g}" for a in bb],
        xlabel='Local finite-time lambda bracket',title='(d) Space / time audit')
    ax[1,1].set(xlabel='Time',ylabel='(H(t)-H(0))/H(0)',title='(e) Static-preset energy audit')
    ax[1,2].set(xlabel='Time',ylabel='Total winding Q',ylim=(.9,1.1),title='(f) Both presets conserve Q=1')
    save(fig,out,'coordinated_fixedK_gate')
    # Small optional comparison gives the cheap predictor an inspectable curve.
    rr=rows(r/'fixed_anisotropy_trial_energy.csv');fig,ax=plt.subplots(figsize=(5.7,3.4),constrained_layout=True)
    ax.plot(col(rr,'lam'),col(rr,'trial_barrier_energy'),c=C[0],label='Static trial barrier W(lambda)')
    ax.axhline(rr[0]['incident_kinetic_energy'],c=C[1],ls='--',label='Incident kinetic energy')
    ax.axvline(pred,c='0.3',ls=':',label=f'Trial root {pred:.6f}')
    for b in bb[:1]:ax.axvspan(b['lambda_reflected'],b['lambda_transmitted'],color='#e5ad31',alpha=.6,label='Executed local bracket')
    ax.set(xlabel='Co-control setting lambda',ylabel='Energy',title='Static predictor and scattering differ');ax.legend(fontsize=8)
    save(fig,out,'coordinated_trial_energy')
    receipt=dict(figure_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figures_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.pdf'))},
        source_summary_sha256=hashlib.sha256((r/'transformation_gate_summary.json').read_bytes()).hexdigest())
    (ROOT/'audit').mkdir(parents=True,exist_ok=True)
    atomic_write(ROOT/'audit'/'figure_receipt.json',json.dumps(receipt,indent=2).encode('utf-8'))
    # Figure generation is a separate executed stage. Refresh only its source
    # hash and figure artifact hashes, preserving the simulation provenance.
    ep=ROOT/'audit'/'execution_receipt.json'
    if ep.exists():
        er=json.loads(ep.read_text());er['script_sha256'][Path(__file__).name]=receipt['figure_script_sha256']
        for p in sorted(out.iterdir()):
            if p.is_file() and p.suffix in ['.pdf','.png']:
                er['results_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        er['figure_step']='Figures regenerated after simulation; figure source/artifact hashes refreshed by the executed figure driver.'
        atomic_write(ep,json.dumps(er,indent=2).encode('utf-8'))
    print('Generated 5 PDF/PNG figures from executed data.')

if __name__=='__main__':main()
