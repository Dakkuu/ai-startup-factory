from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_combo_rotation as combo

OUT=Path('results_ivol_trendfilter'); OUT.mkdir(exist_ok=True)
FILTERS=('none','mom20_pos','mom60_pos','mom120_pos','ma60_pos','ma120_pos','mom60_120_pos')
HOLD_GRID=(20,60,120)
N_GRID=(20,30)

def add_trend(panel,cal):
    p=panel.copy()
    for c in ['mom20','mom60','mom120','ma60gap','ma120gap']: p[c]=np.nan
    groups=p.groupby('code').groups
    for i,(code,idx) in enumerate(groups.items(),1):
        s=base.qb.read_bin(code,'close',cal)
        if s.empty: continue
        ds=pd.DatetimeIndex(p.loc[idx,'signal_date'])
        fac={
          'mom20':s.pct_change(20,fill_method=None),
          'mom60':s.pct_change(60,fill_method=None),
          'mom120':s.pct_change(120,fill_method=None),
          'ma60gap':s/s.rolling(60,min_periods=50).mean()-1,
          'ma120gap':s/s.rolling(120,min_periods=100).mean()-1,
        }
        for c,x in fac.items(): p.loc[idx,c]=x.reindex(ds).to_numpy(dtype=float)
        if i%1000==0: print('trend histories',i,'/',len(groups),flush=True)
    return p

def mask_for(p,name):
    if name=='none': return np.ones(len(p),dtype=bool)
    if name=='mom20_pos': return (p.mom20>0).to_numpy()
    if name=='mom60_pos': return (p.mom60>0).to_numpy()
    if name=='mom120_pos': return (p.mom120>0).to_numpy()
    if name=='ma60_pos': return (p.ma60gap>0).to_numpy()
    if name=='ma120_pos': return (p.ma120gap>0).to_numpy()
    if name=='mom60_120_pos': return ((p.mom60>0)&(p.mom120>0)).to_numpy()
    raise ValueError(name)

def rerank(p,name):
    q=p.copy(); q['rank_test']=np.nan
    liq=(q.liq_rank_pct<=sim.LIQ_KEEP_PCT).to_numpy()
    m=liq & mask_for(q,name) & np.isfinite(q.ivol60.to_numpy())
    q.loc[m,'rank_test']=q.loc[m].groupby('signal_date').ivol60.rank(pct=True,method='average',ascending=True)
    return q

def subset(q,hold):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=hold//5; chosen=set(dates[::step])
    z=q[q.signal_date.isin(chosen)][['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy()
    z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')

def run(q,hold,n,cal,members,bm,cost=1.0):
    old=sim.N_HOLD; sim.N_HOLD=n
    try:
        z=subset(q,hold); eq,tr,tm,to=sim.simulate(z,'ivol',cal,members,cost,daily_mtm=True)
        st=sim.perf(eq,tr,to,bm); st['hold_days']=hold; st['n_hold']=n; st['cost_mult']=cost
        st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31')
        st['sealed_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
        return st,eq,tr,tm
    finally: sim.N_HOLD=old

def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=add_trend(p,cal)
    bm=market_close.loc[sim.START:sim.END].dropna(); br=float(bm.iloc[-1]/bm.iloc[0]-1)
    rows=[]; cache={}
    for f in FILTERS:
        q=rerank(p,f)
        for n in N_GRID:
            for h in HOLD_GRID:
                print('RUN',f,'N',n,'H',h,flush=True)
                st,eq,tr,tm=run(q,h,n,cal,members,bm,1.0)
                st['filter']=f; st['benchmark_return']=br; st['excess']=st.total_return-br
                rows.append(st); cache[(f,n,h)]=(eq,tr,tm,q)
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'grid.csv',index=False)
    # choose using training period only
    w=grid.sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]).iloc[0]
    key=(str(w['filter']),int(w.n_hold),int(w.hold_days)); print('TRAIN WINNER',key,flush=True)
    eq,tr,tm,q=cache[key]
    cost=[]
    for cm in (2.,4.,8.):
        st,_,_,_=run(q,key[2],key[1],cal,members,bm,cm); st['filter']=key[0]; cost.append(st)
    pd.DataFrame(cost).to_csv(OUT/'winner_cost.csv',index=False)
    rob=sim.robustness(eq,tr); rob.update({'filter':key[0],'n_hold':key[1],'hold_days':key[2]}); pd.DataFrame([rob]).to_csv(OUT/'winner_robust.csv',index=False)
    ann=sim.annual_returns(eq); ann['filter']=key[0]; ann['n_hold']=key[1]; ann['hold_days']=key[2]; ann.to_csv(OUT/'winner_annual.csv',index=False)
    stab=grid.groupby('filter').agg(median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),median_train=('train_2016_2021_return','median'),median_sealed=('sealed_2022_2026_return','median'),positive_sealed=('sealed_2022_2026_return',lambda x:int((x>0).sum())),median_mdd=('max_drawdown','median')).reset_index(); stab.to_csv(OUT/'stability.csv',index=False)
    allt=pd.concat([tm.assign(f=k[0],n=k[1],h=k[2]) for k,(eq,tr,tm,q) in cache.items() if len(tm)],ignore_index=True)
    bad=int((pd.to_datetime(allt.signal_date)>=pd.to_datetime(allt.trade_date)).sum()) if len(allt) else 0
    audit={**ua,'market_factor':market_code,'filters':'|'.join(FILTERS),'n_grid':'|'.join(map(str,N_GRID)),'hold_grid':'|'.join(map(str,HOLD_GRID)),'selection':'train 2016-2021 only; sealed 2022-2026 untouched','timing_violations':bad}
    if bad: raise RuntimeError('timing')
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    print('=== GRID ==='); print(grid.sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== STABILITY ==='); print(stab.to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(cost).to_string(index=False),flush=True)
    print('=== ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== ANNUAL ==='); print(ann.to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
if __name__=='__main__': main()
