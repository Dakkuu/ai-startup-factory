from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_alpha2f_v2 as sim

OUT=Path('results_geff55_frozen_audit'); OUT.mkdir(exist_ok=True); SEED=20260823
SPEC=next(s for s in mega.specs_twostage() if s['name']=='g_eff_55')
CFG=dict(hold=90,n=15,entry=.10,keep=.30)


def noisy(q,sigma,rng):
    x=q.copy(); v=x.rank_test.to_numpy(dtype=float,copy=True); m=np.isfinite(v); v[m]=np.clip(v[m]+rng.normal(0,float(sigma),int(m.sum())),0,1); x['rank_test']=v; x.loc[m,'rank_test']=x.loc[m].groupby('signal_date').rank_test.rank(pct=True,method='average'); return x

def randomq(q,rng):
    x=q.copy(); v=x.rank_test.to_numpy(dtype=float,copy=True); m=np.isfinite(v); z=np.full(len(v),np.nan); z[m]=rng.random(int(m.sum())); x['rank_test']=z; return x

def annual(eq): return mega.annual(eq)

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); q=mega.make_rank(p,SPEC); rng=np.random.default_rng(SEED)
    st,eq,tr,tm=mo.full_run(q,CFG['hold'],CFG['n'],CFG['entry'],CFG['keep'],cal,members,bm); pd.DataFrame([st]).to_csv(OUT/'phase0.csv',index=False); annual(eq).to_csv(OUT/'annual.csv',index=False); pd.DataFrame([sim.robustness(eq,tr)]).to_csv(OUT/'tail.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        x,_,_,_=mo.full_run(q,CFG['hold'],CFG['n'],CFG['entry'],CFG['keep'],cal,members,bm,cost=cm); x['cost_mult_test']=cm; costs.append(x)
    pd.DataFrame(costs).to_csv(OUT/'costs.csv',index=False)
    caps=[]
    for cash in (1e7,5e7,1e8,2e8):
      for vp in (.01,.05):
        x,_,_,_=mo.full_run(q,CFG['hold'],CFG['n'],CFG['entry'],CFG['keep'],cal,members,bm,initial_cash=cash,vol_part=vp); x.update(cash_test=cash,vp_test=vp); caps.append(x)
    pd.DataFrame(caps).to_csv(OUT/'capacity.csv',index=False)
    phases=[]; eqs=[]; step=18
    for ph in range(step):
        x,e,_,_=mo.full_run(q,CFG['hold'],CFG['n'],CFG['entry'],CFG['keep'],cal,members,bm,phase=ph); x['phase']=ph; phases.append(x); eqs.append(e)
    ph=pd.DataFrame(phases); ph.to_csv(OUT/'phases.csv',index=False); ee=fa.phase_ensemble(eqs); es=fa.perf_eq(ee,bm); es.update(phase_count=step,train_2016_2021_return=fa.period_return(ee,mo.START,mo.TRAIN_END),pseudo_oos_2022_2026_return=fa.period_return(ee,mo.PSEUDO_START,mo.END)); pd.DataFrame([es]).to_csv(OUT/'phase_ensemble.csv',index=False); annual(ee).to_csv(OUT/'phase_ensemble_annual.csv',index=False)
    ec=[]
    for cm in (2.,4.,8.):
        ex=[]
        for phv in range(step): _,e,_,_=mo.full_run(q,CFG['hold'],CFG['n'],CFG['entry'],CFG['keep'],cal,members,bm,phase=phv,cost=cm); ex.append(e)
        z=fa.phase_ensemble(ex); r=fa.perf_eq(z,bm); r['cost_mult_test']=cm; ec.append(r)
    pd.DataFrame(ec).to_csv(OUT/'phase_ensemble_costs.csv',index=False)
    delays=[]
    for d in (1,3,5):
        x,_,_,_=fa.run_delayed(q,CFG,d,cal,members,bm); x['delay_sessions']=d; delays.append(x)
    pd.DataFrame(delays).to_csv(OUT/'delays.csv',index=False)
    nr=[]
    for sig in (.02,.05,.10):
      for rep in range(20):
        x,_,_,_=mo.full_run(noisy(q,sig,rng),CFG['hold'],CFG['n'],CFG['entry'],CFG['keep'],cal,members,bm); x.update(noise_sigma=sig,rep=rep); nr.append(x)
    pd.DataFrame(nr).to_csv(OUT/'rank_noise.csv',index=False)
    codes=np.array(sorted(q.code.unique())); dr=[]
    for rep in range(30):
        drop=set(rng.choice(codes,size=int(.20*len(codes)),replace=False).tolist()); x,_,_,_=mo.full_run(q[~q.code.isin(drop)],CFG['hold'],CFG['n'],CFG['entry'],CFG['keep'],cal,members,bm); x.update(rep=rep,deleted_share=.20); dr.append(x)
    pd.DataFrame(dr).to_csv(OUT/'random_delete20.csv',index=False)
    pl=[]
    for rep in range(100):
        x,_,_,_=mo.full_run(randomq(q,rng),CFG['hold'],CFG['n'],CFG['entry'],CFG['keep'],cal,members,bm); x['rep']=rep; pl.append(x)
    pdf=pd.DataFrame(pl); pdf.to_csv(OUT/'placebo.csv',index=False)
    gates={'phase0_total':st['total_return'],'phase0_pseudo':st['pseudo_oos_2022_2026_return'],'phase_ensemble_total':es['total_return'],'phase_ensemble_pseudo':es['pseudo_oos_2022_2026_return'],'all_phases_positive':int((ph.total_return>0).all()),'all_phase_pseudo_positive':int((ph.pseudo_oos_2022_2026_return>0).all()),'phase_min':ph.total_return.min(),'phase_median':ph.total_return.median(),'phase_pseudo_min':ph.pseudo_oos_2022_2026_return.min(),'phase_pseudo_median':ph.pseudo_oos_2022_2026_return.median(),'noise10_median':pd.DataFrame(nr).query('noise_sigma==0.10').total_return.median(),'delete20_min':pd.DataFrame(dr).total_return.min(),'placebo_p':float((pdf.total_return>=st['total_return']).mean()),'target500_phase0':int(st['total_return']>=5.0),'beats_old_baseline_phase0':int(st['total_return']>1.749407),'beats_old_baseline_phase_ensemble':int(es['total_return']>1.749407)}
    pd.DataFrame([gates]).to_csv(OUT/'gates.csv',index=False)
    bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0; pd.DataFrame([{**ua,'market_factor':market_code,'candidate':'g_eff_55','spec':str(SPEC),'config':str(CFG),'selection_lock':'formula and config frozen from MegaOpt train-only selection before this audit','signal_universe':'T-only signal-pure','volume_unit_shares':100,'timing_violations':bad,'seed':SEED}]).to_csv(OUT/'audit.csv',index=False)
    print('PHASE0');print(pd.DataFrame([st]).to_string(index=False),flush=True);print('ENSEMBLE');print(pd.DataFrame([es]).to_string(index=False),flush=True);print('GATES');print(pd.DataFrame([gates]).to_string(index=False),flush=True)
    if bad:raise RuntimeError('timing violation')
if __name__=='__main__':main()
