from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_ivol_trendfilter as tf

OUT=Path('results_mom_ivol_twostage'); OUT.mkdir(exist_ok=True)
VARIANTS=('momtop50_ivol','momtop30_ivol','momtop20_ivol','ivollow50_mom','ivollow30_mom','ivollow20_mom','equalblend')
N_GRID=(10,20)
HOLD_GRID=(20,60,120)

def add_ranks(p):
    q=p.copy()
    liq=(q.liq_rank_pct<=sim.LIQ_KEEP_PCT)
    q['mom120_pct']=np.nan; q['ivol60_base_pct']=np.nan
    q.loc[liq & q.mom120.notna(),'mom120_pct']=q.loc[liq & q.mom120.notna()].groupby('signal_date').mom120.rank(pct=True,ascending=False,method='average')
    q.loc[liq & q.ivol60.notna(),'ivol60_base_pct']=q.loc[liq & q.ivol60.notna()].groupby('signal_date').ivol60.rank(pct=True,ascending=True,method='average')
    return q

def rerank(p,variant):
    q=p.copy(); q['rank_test']=np.nan
    liq=(q.liq_rank_pct<=sim.LIQ_KEEP_PCT)
    if variant.startswith('momtop'):
        frac=int(variant[6:8])/100
        m=liq & q.mom120_pct.notna() & (q.mom120_pct<=frac) & q.ivol60.notna()
        q.loc[m,'rank_test']=q.loc[m].groupby('signal_date').ivol60.rank(pct=True,ascending=True,method='average')
    elif variant.startswith('ivollow'):
        frac=int(variant[7:9])/100
        m=liq & q.ivol60_base_pct.notna() & (q.ivol60_base_pct<=frac) & q.mom120.notna()
        q.loc[m,'rank_test']=q.loc[m].groupby('signal_date').mom120.rank(pct=True,ascending=False,method='average')
    elif variant=='equalblend':
        m=liq & q.mom120_pct.notna() & q.ivol60_base_pct.notna()
        score=0.5*q.loc[m,'mom120_pct']+0.5*q.loc[m,'ivol60_base_pct']
        q.loc[m,'rank_test']=score.groupby(q.loc[m,'signal_date']).rank(pct=True,ascending=True,method='average')
    else: raise ValueError(variant)
    return q

def subset(q,hold):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,hold//5); chosen=set(dates[::step])
    z=q[q.signal_date.isin(chosen)][['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy()
    z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')

def run(q,hold,n,cal,members,bm,cost=1.0):
    old=sim.N_HOLD; sim.N_HOLD=n
    try:
        z=subset(q,hold); eq,tr,tm,to=sim.simulate(z,'ivol',cal,members,cost,daily_mtm=True)
        st=sim.perf(eq,tr,to,bm); st['hold_days']=hold; st['n_hold']=n; st['cost_mult']=cost
        st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31')
        st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
        return st,eq,tr,tm
    finally: sim.N_HOLD=old

def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=tf.add_trend(p,cal); p=add_ranks(p)
    bm=market_close.loc[sim.START:sim.END].dropna(); br=float(bm.iloc[-1]/bm.iloc[0]-1)
    rows=[]; cache={}
    for v in VARIANTS:
        q=rerank(p,v)
        for n in N_GRID:
            for h in HOLD_GRID:
                print('RUN',v,'N',n,'H',h,flush=True)
                st,eq,tr,tm=run(q,h,n,cal,members,bm,1.0); st['variant']=v;st['benchmark_return']=br;st['excess']=st['total_return']-br
                rows.append(st);cache[(v,n,h)]=(eq,tr,tm,q)
    grid=pd.DataFrame(rows);grid.to_csv(OUT/'grid.csv',index=False)
    # pick by training only, but 2022-26 is labelled pseudo-OOS because research iteration has already viewed it.
    w=grid.sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]).iloc[0]
    key=(str(w.variant),int(w.n_hold),int(w.hold_days)); print('TRAIN WINNER',key,flush=True)
    eq,tr,tm,q=cache[key]
    cost=[]
    for cm in (2.,4.,8.):
        st,_,_,_=run(q,key[2],key[1],cal,members,bm,cm);st['variant']=key[0];cost.append(st)
    pd.DataFrame(cost).to_csv(OUT/'winner_cost.csv',index=False)
    rob=sim.robustness(eq,tr);rob.update({'variant':key[0],'n_hold':key[1],'hold_days':key[2]});pd.DataFrame([rob]).to_csv(OUT/'winner_robust.csv',index=False)
    ann=sim.annual_returns(eq);ann['variant']=key[0];ann['n_hold']=key[1];ann['hold_days']=key[2];ann.to_csv(OUT/'winner_annual.csv',index=False)
    stab=grid.groupby('variant').agg(median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),median_train=('train_2016_2021_return','median'),median_pseudo_oos=('pseudo_oos_2022_2026_return','median'),positive_pseudo_oos=('pseudo_oos_2022_2026_return',lambda x:int((x>0).sum())),median_mdd=('max_drawdown','median')).reset_index();stab.to_csv(OUT/'stability.csv',index=False)
    allt=pd.concat([tm.assign(v=k[0],n=k[1],h=k[2]) for k,(eq,tr,tm,q) in cache.items() if len(tm)],ignore_index=True);bad=int((pd.to_datetime(allt.signal_date)>=pd.to_datetime(allt.trade_date)).sum()) if len(allt) else 0
    audit={**ua,'market_factor':market_code,'variants':'|'.join(VARIANTS),'n_grid':'|'.join(map(str,N_GRID)),'hold_grid':'|'.join(map(str,HOLD_GRID)),'selection':'training-period ranking only; 2022-2026 reported as pseudo-OOS due iterative research','timing_violations':bad}
    if bad: raise RuntimeError('timing')
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    print('=== GRID ===');print(grid.sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== STABILITY ===');print(stab.to_string(index=False),flush=True)
    print('=== COST ===');print(pd.DataFrame(cost).to_string(index=False),flush=True)
    print('=== ROBUST ===');print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== ANNUAL ===');print(ann.to_string(index=False),flush=True)
    print('=== AUDIT ===');print(pd.DataFrame([audit]).to_string(index=False),flush=True)
if __name__=='__main__':main()
