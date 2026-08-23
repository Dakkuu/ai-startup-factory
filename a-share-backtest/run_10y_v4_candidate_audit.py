from __future__ import annotations
from pathlib import Path
import argparse, numpy as np, pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_alpha2f_v2 as sim

SEED=20260823
SPECS={
 'linear_robust':({'name':'linear_robust','kind':'linear','liq':.55,'skew':.80,'w':.60},dict(n=10,hold=120,entry=.10,keep=.30)),
 'linear_growth':({'name':'linear_growth','kind':'linear','liq':.55,'skew':.80,'w':.60},dict(n=10,hold=120,entry=.05,keep=.20)),
 'nonlinear_robust':({'name':'nonlinear_robust','kind':'risk_meanmax','liq':.70,'skew':.90,'requires':['dsemi60','max20']},dict(n=20,hold=80,entry=.10,keep=.30)),
}

def noisy(q,sigma,rng):
    x=q.copy(); v=x.rank_test.to_numpy(dtype=float,copy=True); m=np.isfinite(v)
    v[m]=np.clip(v[m]+rng.normal(0,float(sigma),int(m.sum())),0,1); x['rank_test']=v
    x.loc[m,'rank_test']=x.loc[m].groupby('signal_date').rank_test.rank(pct=True,method='average'); return x

def random_rank(q,rng):
    x=q.copy(); v=x.rank_test.to_numpy(dtype=float,copy=True); m=np.isfinite(v); z=np.full(len(v),np.nan); z[m]=rng.random(int(m.sum())); x['rank_test']=z; return x

def main(name):
    out=Path(f'results_v4_candidate_audit_{name}'); out.mkdir(exist_ok=True)
    spec,cfg=SPECS[name]; p,cal,members,ua,market_code,bm=mo.build_panel(out,need_fwd=False); q=mo.rerank(p,spec); rng=np.random.default_rng(SEED+sum(map(ord,name)))
    st,eq,tr,tm=mo.full_run(q,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm); st['candidate']=name
    pd.DataFrame([st]).to_csv(out/'phase0.csv',index=False); sim.annual_returns(eq).to_csv(out/'phase0_annual.csv',index=False); pd.DataFrame([sim.robustness(eq,tr)]).to_csv(out/'tail.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        x,_,_,_=mo.full_run(q,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm,cost=cm); x['cost_mult_test']=cm; costs.append(x)
    pd.DataFrame(costs).to_csv(out/'costs.csv',index=False)
    caps=[]
    for cash in (1e7,5e7,1e8):
      for vp in (.01,.05):
        x,_,_,_=mo.full_run(q,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm,initial_cash=cash,vol_part=vp); x.update(cash_test=cash,vp_test=vp); caps.append(x)
    pd.DataFrame(caps).to_csv(out/'capacity.csv',index=False)
    phases=[]; eqs=[]; step=max(1,round(cfg['hold']/5))
    for ph in range(step):
        x,e,_,_=mo.full_run(q,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm,phase=ph); x['phase']=ph; phases.append(x); eqs.append(e)
    ph=pd.DataFrame(phases); ph.to_csv(out/'phases.csv',index=False)
    ee=fa.phase_ensemble(eqs); es=fa.perf_eq(ee,bm); es.update(candidate=name,phase_count=step,train_2016_2021_return=fa.period_return(ee,mo.START,mo.TRAIN_END),pseudo_oos_2022_2026_return=fa.period_return(ee,mo.PSEUDO_START,mo.END)); pd.DataFrame([es]).to_csv(out/'phase_ensemble.csv',index=False); fa.annual(ee).to_csv(out/'phase_ensemble_annual.csv',index=False)
    ec=[]
    for cm in (4.,8.):
        xs=[]
        for phv in range(step):
            _,e,_,_=mo.full_run(q,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm,phase=phv,cost=cm); xs.append(e)
        ecm=fa.phase_ensemble(xs); r=fa.perf_eq(ecm,bm); r['cost_mult_test']=cm; ec.append(r)
    pd.DataFrame(ec).to_csv(out/'phase_ensemble_costs.csv',index=False)
    delays=[]
    for d in (1,3,5):
        x,_,_,_=fa.run_delayed(q,cfg,d,cal,members,bm); x['delay_sessions']=d; delays.append(x)
    pd.DataFrame(delays).to_csv(out/'delays.csv',index=False)
    nr=[]
    for sig in (.02,.05,.10):
      for rep in range(12):
        x,_,_,_=mo.full_run(noisy(q,sig,rng),cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm); x.update(noise_sigma=sig,rep=rep); nr.append(x)
    pd.DataFrame(nr).to_csv(out/'rank_noise.csv',index=False)
    codes=np.array(sorted(q.code.unique())); dr=[]
    for rep in range(20):
        drop=set(rng.choice(codes,size=int(.20*len(codes)),replace=False).tolist()); x,_,_,_=mo.full_run(q[~q.code.isin(drop)],cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm); x.update(deleted_share=.20,rep=rep); dr.append(x)
    pd.DataFrame(dr).to_csv(out/'random_delete20.csv',index=False)
    pl=[]
    for rep in range(50):
        x,_,_,_=mo.full_run(random_rank(q,rng),cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm); x['rep']=rep; pl.append(x)
    pdf=pd.DataFrame(pl); pdf.to_csv(out/'placebo.csv',index=False)
    gates={'candidate':name,'phase0_total':st['total_return'],'phase_ensemble_total':es['total_return'],'all_phases_positive':int((ph.total_return>0).all()),'all_pseudo_oos_phases_positive':int((ph.pseudo_oos_2022_2026_return>0).all()),'noise10_median':float(pd.DataFrame(nr).query('noise_sigma==0.10').total_return.median()),'delete20_min':float(pd.DataFrame(dr).total_return.min()),'placebo_p':float((pdf.total_return>=st['total_return']).mean()),'beats_old_baseline_phase0':int(st['total_return']>1.749407),'beats_old_baseline_phase_ensemble':int(es['total_return']>1.749407)}
    pd.DataFrame([gates]).to_csv(out/'gates.csv',index=False)
    bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    pd.DataFrame([{**ua,'market_factor':market_code,'candidate':name,'spec':str(spec),'config':str(cfg),'selection_lock':'fixed from V4 2016-2021-only selection before this audit','signal_universe':'T-only signal-pure','volume_unit_shares':100,'timing_violations':bad,'seed':SEED}]).to_csv(out/'audit.csv',index=False)
    print('PHASE0',pd.DataFrame([st]).to_string(index=False),flush=True); print('ENSEMBLE',pd.DataFrame([es]).to_string(index=False),flush=True); print('GATES',pd.DataFrame([gates]).to_string(index=False),flush=True)
    if bad: raise RuntimeError('timing violation')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('candidate',choices=tuple(SPECS)); a=ap.parse_args(); main(a.candidate)
