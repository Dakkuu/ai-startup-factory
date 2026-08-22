from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import run_10y_alpha_discovery_qv as qv
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_factor_mine2 as mine2
import run_10y_max_audit as ma

OUT=Path('results_alpha_adaptive_qv'); OUT.mkdir(exist_ok=True)
qv.sp.OUT=OUT

FACTOR_H={n:h for n,h,_ in qv.MANIFEST}
VARIANTS=(
 ('top1_52',52,1,'topk'),
 ('top3_52',52,3,'topk'),
 ('posweight_52',52,0,'posweight'),
 ('top3_104',104,3,'topk'),
 ('posweight_104',104,0,'posweight'),
)
MIN_IC_OBS=20


def add_good_ranks(p,factors):
    q=p.copy()
    for f in factors:
        out='g_'+f; q[out]=np.nan
        m=(q.liq_rank_pct<=qv.LIQ_KEEP)&np.isfinite(q[f])
        # oriented factors: higher raw factor is better; higher good-rank is better.
        q.loc[m,out]=q.loc[m].groupby('signal_date')[f].rank(pct=True,method='average',ascending=True)
    return q


def historical_ic(p,factors,horizon):
    label=f'fwd{horizon}'; rows=[]
    for d,g in p[p.liq_rank_pct<=qv.LIQ_KEEP].groupby('signal_date',sort=True):
        for f in factors:
            z=g[[f,label]].dropna()
            if len(z)<300: continue
            ic=z[f].corr(z[label],method='spearman')
            if np.isfinite(ic): rows.append({'signal_date':pd.Timestamp(d),'factor':f,'ic':float(ic)})
    z=pd.DataFrame(rows)
    return z.pivot(index='signal_date',columns='factor',values='ic').sort_index() if len(z) else pd.DataFrame(columns=factors)


def dynamic_rank(p,factors,horizon,window,topk,mode,variant):
    q=add_good_ranks(p,factors); q['rank_test']=np.nan
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); pos={d:i for i,d in enumerate(dates)}
    ic=historical_ic(q,factors,horizon).reindex(dates)
    lag=max(2,horizon//5+1)  # signal grid is 5 sessions; +1 ensures the forward label has fully matured.
    weight_rows=[]
    for d in dates:
        j=pos[d]; end=max(0,j-lag); start=max(0,end-window)
        hist=ic.iloc[start:end]
        mu=hist.mean(skipna=True); n=hist.count()
        valid=(n>=MIN_IC_OBS)&np.isfinite(mu)
        w=pd.Series(0.0,index=factors,dtype=float)
        if valid.any():
            m=mu.where(valid)
            if mode=='topk':
                chosen=m.sort_values(ascending=False).head(topk)
                chosen=chosen[chosen>0]
                if len(chosen): w.loc[chosen.index]=1.0/len(chosen)
            else:
                positive=m.clip(lower=0).fillna(0)
                if positive.sum()>0: w=positive/positive.sum()
        # Causal warm-up/fallback is fixed equal weight, not chosen from future outcomes.
        if w.sum()<=0: w.loc[:]=1.0/len(factors)
        g=q[q.signal_date==d]
        cols=['g_'+f for f in factors]; X=g[cols].to_numpy(float); ww=w.to_numpy(float)
        ok=np.isfinite(X); den=(ok*ww).sum(axis=1); num=np.nansum(X*ww,axis=1); score=np.divide(num,den,out=np.full(len(g),np.nan),where=den>0)
        idx=g.index; finite=np.isfinite(score)
        if finite.any():
            s=pd.Series(score[finite],index=idx[finite]); q.loc[s.index,'rank_test']=s.rank(pct=True,method='average',ascending=False)
        for f in factors: weight_rows.append({'signal_date':d,'variant':variant,'horizon':horizon,'factor':f,'weight':float(w[f]),'trailing_ic':float(mu.get(f,np.nan)) if len(mu) else np.nan,'ic_observations':int(n.get(f,0)) if len(n) else 0,'maturity_lag_signal_steps':lag})
    return q,pd.DataFrame(weight_rows)


def exact(q,h,cal,members,bm,cost=1.0):
    st,eq,tr,tm=ma.run_q(q,h,0,cal,members,bm,n=20,entry=.10,keep=.30,cost=cost)
    st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31'); st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
    return st,eq,tr,tm


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=mine2.add_extra(p,cal,market_close); p=qv.add_qv_fields(p,cal); p=qv.attach_oriented_existing(p)
    bm=market_close.loc[sim.START:sim.END].dropna()
    all_rows=[]; all_weights=[]; cache={}
    for h in (20,60):
        factors=[n for n,hh,_ in qv.MANIFEST if hh==h]
        for variant,window,topk,mode in VARIANTS:
            print('ADAPTIVE',h,variant,'factors',len(factors),flush=True)
            rq,w=dynamic_rank(p,factors,h,window,topk,mode,variant); all_weights.append(w)
            st,eq,tr,tm=exact(rq,h,cal,members,bm); st.update({'variant':variant,'horizon':h,'ic_window_signal_dates':window,'mode':mode,'topk':topk,'factor_count':len(factors)}); all_rows.append(st); cache[(variant,h)]=(rq,eq,tr,tm)
    grid=pd.DataFrame(all_rows).sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]); grid.to_csv(OUT/'adaptive_grid.csv',index=False)
    pd.concat(all_weights,ignore_index=True).to_csv(OUT/'adaptive_weights.csv',index=False)

    w=grid.iloc[0]; key=(str(w.variant),int(w.horizon)); rq,eq,tr,tm=cache[key]
    pd.DataFrame([w]).to_csv(OUT/'winner.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=exact(rq,key[1],cal,members,bm,cm); st.update({'variant':key[0],'horizon':key[1]}); costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_costs.csv',index=False)
    a=sim.annual_returns(eq); a['variant']=key[0]; a['horizon']=key[1]; a.to_csv(OUT/'winner_annual.csv',index=False)
    rr=sim.robustness(eq,tr); rr.update({'variant':key[0],'horizon':key[1]}); pd.DataFrame([rr]).to_csv(OUT/'winner_robust.csv',index=False)

    allt=pd.concat([x[3].assign(variant=k[0],horizon=k[1]) for k,x in cache.items() if len(x[3])],ignore_index=True)
    bad=int((pd.to_datetime(allt.signal_date)>=pd.to_datetime(allt.trade_date)).sum()) if len(allt) else 0
    hits=int((grid.total_return>=5.0).sum())
    audit={**ua,'market_factor':market_code,'research_round':'causal trailing-IC adaptive factor rotation','variants':len(VARIANTS)*2,'target_500_hits':hits,'selection':'reported winner chosen on 2016-2021 only; adaptive weights at each date use only fully matured historical labels','maturity':'20d uses >=5 signal-step lag; 60d uses >=13 signal-step lag','warmup':'fixed equal weights until minimum historical IC observations exist','signal_universe':'signal-pure T-only','volume_source_unit_shares':100,'timing_violations':bad}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad: raise RuntimeError('timing violation')
    print('=== ADAPTIVE GRID ==='); print(grid.to_string(index=False),flush=True)
    print('=== COSTS ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)

if __name__=='__main__': main()
