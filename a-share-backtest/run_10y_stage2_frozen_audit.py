from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

import run_10y_alpha_stage2 as s2
import run_10y_alpha_composites_qv as comp
import run_10y_max_audit as ma
import run_10y_alpha2f_v2 as sim

OUT=Path('results_stage2_frozen_audit'); OUT.mkdir(exist_ok=True)
SEED=20260823
N=8; HOLD=40; ENTRY=.10; KEEP=.30


def run(rq,cal,members,bm,phase=0,cost=1.,cash=1e6,vp=.05):
    st,eq,tr,tm=ma.run_q(rq,HOLD,phase,cal,members,bm,n=N,entry=ENTRY,keep=KEEP,cost=cost,initial_cash=cash,vol_part=vp)
    st['phase']=phase; st['cost_mult']=cost; st['initial_cash']=cash; st['volume_participation']=vp
    st['train_return']=sim.period_return(eq,'2016-07-29','2021-12-31')
    st['pseudo_oos_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
    return st,eq,tr,tm


def main():
    p,cal,members,ua,market_code,bm=s2.build_panel(OUT)
    p=comp.add_composite_scores(p)
    rq=comp.ranked(p,'anti_lottery_momentum')

    base,eq,tr,tm=run(rq,cal,members,bm); pd.DataFrame([base]).to_csv(OUT/'baseline.csv',index=False)
    sim.annual_returns(eq).to_csv(OUT/'annual.csv',index=False)
    pd.DataFrame([sim.robustness(eq,tr)]).to_csv(OUT/'robust.csv',index=False)

    phases=[]
    for ph in range(8):
        st,_,_,_=run(rq,cal,members,bm,phase=ph); phases.append(st)
    pd.DataFrame(phases).to_csv(OUT/'phase_offsets.csv',index=False)

    costs=[]
    for c in (2.,4.,8.):
        st,_,_,_=run(rq,cal,members,bm,cost=c); costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'costs.csv',index=False)

    caps=[]
    for cash in (1e6,5e6,1e7,5e7,1e8):
        for vp in (.01,.05):
            st,_,_,_=run(rq,cal,members,bm,cash=cash,vp=vp); caps.append(st)
    pd.DataFrame(caps).to_csv(OUT/'capacity.csv',index=False)

    rng=np.random.default_rng(SEED); codes=np.array(sorted(rq.code.unique())); dels=[]
    for k in range(20):
        drop=set(rng.choice(codes,size=max(1,int(.20*len(codes))),replace=False).tolist())
        st,_,_,_=run(rq[~rq.code.isin(drop)],cal,members,bm); st['seed']=k; dels.append(st)
    pd.DataFrame(dels).to_csv(OUT/'random_delete20.csv',index=False)

    noise=[]; finite=np.isfinite(rq.rank_test.to_numpy(float))
    for sigma in (.02,.05,.10):
        for k in range(10):
            x=rq.copy(); vals=x.rank_test.to_numpy(float,copy=True)
            vals[finite]=np.clip(vals[finite]+rng.normal(0,sigma,finite.sum()),0,1); x['rank_test']=vals
            x.loc[finite,'rank_test']=x.loc[finite].groupby('signal_date').rank_test.rank(pct=True,method='average')
            st,_,_,_=run(x,cal,members,bm); st.update({'sigma':sigma,'seed':k}); noise.append(st)
    pd.DataFrame(noise).to_csv(OUT/'rank_noise.csv',index=False)

    placebo=[]
    for k in range(100):
        x=rq.copy(); vals=np.full(len(x),np.nan); vals[finite]=rng.random(finite.sum()); x['rank_test']=vals
        st,_,_,_=run(x,cal,members,bm); st['seed']=k; placebo.append(st)
    pl=pd.DataFrame(placebo); pl.to_csv(OUT/'placebo.csv',index=False)

    gates={
      'timing_zero':int(len(tm)==0 or (pd.to_datetime(tm.signal_date)<pd.to_datetime(tm.trade_date)).all()),
      'all_8_phases_positive':int((pd.DataFrame(phases).total_return>0).all()),
      'all_8_phases_pseudo_oos_positive':int((pd.DataFrame(phases).pseudo_oos_return>0).all()),
      'cost8_positive':int(pd.DataFrame(costs).iloc[-1].total_return>0),
      'delete20_all_positive':int((pd.DataFrame(dels).total_return>0).all()),
      'noise10_median_positive':int(pd.DataFrame(noise).query('sigma==0.10').total_return.median()>0),
      'capacity_10m_1pct_positive':int(pd.DataFrame(caps).query('initial_cash==1e7 and volume_participation==0.01').total_return.iloc[0]>0),
      'placebo_return_p_le_5pct':int((pl.total_return>=base['total_return']).mean()<=.05),
      'placebo_sharpe_p_le_5pct':int((pl.sharpe>=base['sharpe']).mean()<=.05),
    }
    g=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); g.to_csv(OUT/'gates.csv',index=False)
    verdict={**ua,'market_factor':market_code,'frozen_rule':'anti_lottery_momentum; N8; hold40; entry10%; keep30%; phase0; next-open','base_total_return':base['total_return'],'base_cagr':base['cagr'],'base_mdd':base['max_drawdown'],'base_sharpe':base['sharpe'],'placebo_return_p':float((pl.total_return>=base['total_return']).mean()),'placebo_sharpe_p':float((pl.sharpe>=base['sharpe']).mean()),'gates_passed':int(g['pass'].sum()),'gates_total':len(g),'hard_pass':int(g['pass'].all()),'signal_universe':'signal-pure T-only','volume_source_unit_shares':100}
    pd.DataFrame([verdict]).to_csv(OUT/'verdict.csv',index=False)
    print('=== VERDICT ==='); print(pd.DataFrame([verdict]).to_string(index=False),flush=True)
    print('=== GATES ==='); print(g.to_string(index=False),flush=True)
    print('=== PHASE ==='); print(pd.DataFrame(phases)[['phase','total_return','cagr','max_drawdown','sharpe','pseudo_oos_return']].to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(costs)[['cost_mult','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True)
    print('=== CAPACITY ==='); print(pd.DataFrame(caps)[['initial_cash','volume_participation','total_return','cagr','max_drawdown']].to_string(index=False),flush=True)

if __name__=='__main__': main()
