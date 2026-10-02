#!/usr/bin/env python3
"""Repeat-cell check for Q=m; phase number and space dimension are unchanged.
This imports the original branch generator; it is an extension check, NOT an
independent reproduction of the original quadrature or time integrator.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.integrate import simpson
from dsg_continuation import static_branch,force,ramp,run_pde
ROOT=Path(__file__).resolve().parents[1]

def run(m=2,L=8.,N=256,T=400.,dt=.025,relax=50.,eta=.2):
 x=np.arange(N)*L/N;base=2*np.pi*m*x/L;k=2*np.pi*np.fft.fftfreq(N,d=L/N);lin=-k*k
 b=static_branch(eta,0,L/m,16385);u0=b.field(x);z=np.fft.fft(u0-base)
 e=np.exp(dt*lin);e2=np.exp(dt*lin/2);roots=np.exp(1j*np.pi*(np.arange(1,33)-.5)/32);lr=dt*lin[:,None]+roots
 q=dt*np.real(np.mean(np.expm1(lr/2)/lr,axis=1));f1=dt*np.real(np.mean((-4-lr+np.exp(lr)*(4-3*lr+lr*lr))/lr**3,axis=1));f2=dt*np.real(np.mean((2+lr+np.exp(lr)*(-2+lr))/lr**3,axis=1));f3=dt*np.real(np.mean((-4-3*lr-lr*lr+np.exp(lr)*(4-lr))/lr**3,axis=1))
 def nl(z,t):return np.fft.fft(-force(base+np.fft.ifft(z).real,eta,ramp(t,T)[0]))
 for i in range(round((T+relax)/dt)):
  t=i*dt;nz=nl(z,t);a=e2*z+q*nz;na=nl(a,t+dt/2);bb=e2*z+q*na;nb=nl(bb,t+dt/2);c=e2*a+q*(2*nb-nz);nc=nl(c,t+dt);z=e*z+f1*nz+2*f2*(na+nb)+f3*nc
 uf=base+np.fft.ifft(z).real;ux=2*np.pi*m/L+np.fft.ifft(1j*k*z).real
 return dict(m=m,L=L,cell_length=L/m,N=N,T=T,dt=dt,relax=relax,charge_average=float(-np.mean(uf-u0)/(2*np.pi)),winding=float(L*np.mean(ux)/(2*np.pi))),uf,u0

def main():
 start=time.perf_counter();one,_,a=run_pde(L=4.,N=128,T=400.,dt=.025,angle_nodes=16385);two,uf,u0=run();err=float(np.max(abs((uf-u0)[:128]-(a['uf']-a['u0']))));charge_error=abs(two['charge_average']-one['charge_average'])
 data=dict(relation='For a monotone repeated m-winding static family at length L, DeltaX_m(L)=DeltaX_1(L/m), Qbar_m(L)=Qbar_1(L/m).',
  one_winding_at_L4=one,two_winding_at_L8=two,field_increment_max_difference=err,averaged_charge_difference=charge_error,
  interpretation='Equal averaged transported charge is not equal total winding: total Q=1 versus Q=2. This is a repeated-cell construction, not a time evolution changing Q.',
  all_passed=bool(err<1e-10 and charge_error<1e-12),runtime_seconds=time.perf_counter()-start)
 (ROOT/'results'/'ring_multisector_check.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2));assert data['all_passed']
if __name__=='__main__':main()
