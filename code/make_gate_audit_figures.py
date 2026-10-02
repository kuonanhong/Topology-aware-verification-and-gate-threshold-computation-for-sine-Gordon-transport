#!/usr/bin/env python3
"""Render the additional audit tables. No physics is recomputed here."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def read(name):
 with (OUT/name).open() as f:return list(csv.DictReader(f))

def main():
 d=json.loads((OUT/'gate_extended_audit.json').read_text());ctrl=read('gate_controls.csv');temp=read('gate_time_refinement.csv');saddle=read('gate_saddle_energy.csv');fine=read('gate_fine_refinement.csv')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
 fig,ax=plt.subplots(2,2,figsize=(10.3,7.4),constrained_layout=True)
 G=np.array([float(r['G']) for r in saddle]);H=np.array([float(r['energy']) for r in saddle]);B=json.loads((OUT/'gate_summary.json').read_text())['predictor']['barrier_per_unit_G']
 ax[0,0].plot(G,8+B*G,'--',c='0.5',label='Rigid static-width kink')
 ax[0,0].plot(G,H,'o-',c='#167d8d',label='Relaxed centered BVP')
 ax[0,0].axhline(d['incident_continuum_energy'],c='#c4552d',ls=':',label='Incoming continuum energy')
 ax[0,0].set(xlabel='Gate amplitude G',ylabel='Static configuration energy',title='(a) Energy predictors');ax[0,0].legend(fontsize=7)
 for i,r in enumerate(fine):
  lo=float(r['G_transmitted']);hi=float(r['G_reflected']);ax[0,1].errorbar((lo+hi)/2,i,xerr=(hi-lo)/2,fmt='o',capsize=4,c='#167d8d')
 ax[0,1].axvline(d['saddle_predictor_G'],c='#c4552d',ls='--',label='Saddle-energy predictor')
 ax[0,1].ticklabel_format(axis='x',style='plain',useOffset=False);ax[0,1].tick_params(axis='x',labelsize=8)
 ax[0,1].set(yticks=range(len(fine)),yticklabels=[f"h={float(r['dx']):g}, dt={float(r['dt']):g}" for r in fine],xlabel='Gate amplitude G',title='(b) Refined local threshold brackets');ax[0,1].legend(fontsize=7)
 free=[r for r in ctrl if r['tag']=='free_kink' and abs(float(r['dt'])-.02)<1e-12];h=np.array([float(r['dx']) for r in free]);err=np.array([float(r['free_exact_max_phase_error']) for r in free])
 ax[1,0].loglog(h,err,'o-',c='#167d8d',label='Max field error, t≤80')
 ax[1,0].loglog(h,err[-1]*(h/h[-1])**2,'--',c='0.5',label='h² guide')
 ax[1,0].set(xlabel='Spatial spacing h (dt=0.02)',ylabel='Error against exact free kink',title='(c) Continuum spatial convergence');ax[1,0].legend(fontsize=7)
 tt=np.array([float(r['dt']) for r in temp[:-1]]);ee=np.array([float(r['field_max_error_vs_dt0005']) for r in temp[:-1]]);guide=(tt**2-.005**2);guide*=ee[0]/guide[0]
 ax[1,1].loglog(tt,ee,'o-',c='#c4552d',label='Max field difference')
 ax[1,1].loglog(tt,guide,'--',c='0.5',label='(dt²−0.005²) guide')
 ax[1,1].set(xlabel='Time step dt (h=0.1)',ylabel='Difference from dt=0.005 at t=130',title='(d) Independent temporal refinement');ax[1,1].legend(fontsize=7)
 for ext in ['png','pdf']:fig.savefig(OUT/f'gate_predictors_convergence.{ext}',dpi=200)
 plt.close(fig)
if __name__=='__main__':main()
