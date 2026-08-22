from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq

OUT=Path('results_skewfilter_surface'); OUT.mkdir(exist_ok=True)
SKEW_WINDOWS=(40,60,80)
KEEP_SKEW=(0.70,0.80,0.90)
IVOL_WEIGHTS=(0.60,2/3,0.70)
HOLD_GRID=(60,120)
N_HOLD=20
ANCHOR=(60,0.80,2/3,120)

def add_skews(panel,cal,market_close):
    p=panel.copy()
    for w in SKEW_WINDOWS: p[f'skew{w}']=np.nan
    mret=market_close.reindex(cal[(cal>=sim.WARM)&(cal<=sim.END)]).pct_change(fill_method=None)
    mmu=mret.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).mean().shift(1)
    mvar=mret.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).var().shift(1)
    groups=p.groupby('code').groups
    for i,(code,idx) in enumerate(groups.items(),1):
        c=base.qb.read_bin(code,'close',cal).loc[sim.WARM:sim.END]
        if c.empty: continue
        r=c.pct_change(fill_method=None); m=mret.reindex(c.index)
        smu=r.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).mean().shift(1)
        cov=r.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).cov(m).shift(1)
        beta=cov/mvar.reindex(c.index); alpha=smu-beta*mmu.reindex(c.index); resid=r-alpha-beta*m
        ds=pd.DatetimeIndex(p.loc[idx,'signal_date'])
        for w in SKEW_WINDOWS:
            s=resid.rolling(w,min_periods=max(30,int(w*.8))).skew()
            p.loc[idx,f'skew{w}']=s.reindex(ds).to_numpy(float)
        if i%1000==0: print('skew histories',i,'/',len(groups),flush=True)
    return p

def rerank(p,sw,keep,wiv):
    q=p.copy(); q['rank_test']=np.nan
    m=(q.liq_rank_pct<=sim.LIQ_KEEP_PCT)&np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q[f'skew{sw}'])
    sp=q.loc[m].groupby('signal_date')[f'skew{sw}'].rank(pct=True,method='average',ascending=True)
    ok=pd.Series(False,index=q.index); ok.loc[sp.index]=sp<=keep; m=m&ok
    iv=q.loc[m].groupby('signal_date').ivol60.rank(pct=True,method='average',ascending=True)
    ef=q.loc[m].groupby('signal_date').eff120.rank(pct=True,method='average',ascending=False)
    raw=wiv*iv+(1-wiv)*ef
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q

def subset(q,hold):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(hold/5)); chosen=set(dates[::step])
    z=q[q.signal_date.isin(chosen)][['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def run(q,hold,cal,members,bm,cost=1.0,daily=True,n=N_HOLD):
    old=sim.N_HOLD; sim.N_HOLD=n
    try:
        z=subset(q,hold); eq,tr,tm,to=sim.simulate(z,'ivol',cal,members,cost,daily_mtm=daily)
        st=sim.perf(eq,tr,to,bm if daily else None); st.update({'hold_days':hold,'n_hold':n,'cost_mult':cost})
        st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31'); st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
        return st,eq,tr,tm
    finally: sim.N_HOLD=old

def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=add_skews(p,cal,market_close)
    bm=market_close.loc[sim.START:sim.END].dropna(); br=float(bm.iloc[-1]/bm.iloc[0]-1)
    rows=[]
    for sw in SKEW_WINDOWS:
      for keep in KEEP_SKEW:
       for wiv in IVOL_WEIGHTS:
        q=rerank(p,sw,keep,wiv)
        for h in HOLD_GRID:
            print('RUN skew',sw,'keep',keep,'wiv',wiv,'h',h,flush=True)
            st,_,_,_=run(q,h,cal,members,bm,1.0,False); st.update({'skew_window':sw,'skew_keep_pct':keep,'ivol_weight':wiv,'benchmark_return':br,'excess':st['total_return']-br}); rows.append(st)
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'grid.csv',index=False)
    surface=grid.groupby(['skew_window','skew_keep_pct','ivol_weight']).agg(median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),median_train=('train_2016_2021_return','median'),median_pseudo_oos=('pseudo_oos_2022_2026_return','median'),min_pseudo_oos=('pseudo_oos_2022_2026_return','min')).reset_index(); surface.to_csv(OUT/'surface.csv',index=False)
    # Exact anchor fixed before this surface run.
    sw,keep,wiv,h=ANCHOR; qa=rerank(p,sw,keep,wiv); st,eq,tr,tm=run(qa,h,cal,members,bm,1.0,True); st.update({'skew_window':sw,'skew_keep_pct':keep,'ivol_weight':wiv}); pd.DataFrame([st]).to_csv(OUT/'anchor_summary.csv',index=False)
    ann=sim.annual_returns(eq); ann.to_csv(OUT/'anchor_annual.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        x,_,_,_=run(qa,h,cal,members,bm,cm,True); x.update({'skew_window':sw,'skew_keep_pct':keep,'ivol_weight':wiv}); costs.append(x)
    pd.DataFrame(costs).to_csv(OUT/'anchor_cost.csv',index=False)
    rob=sim.robustness(eq,tr); pd.DataFrame([rob]).to_csv(OUT/'anchor_robust.csv',index=False)
    # Construction check at the already fixed central factor parameters.
    cr=[]
    for n in (15,20,30):
      for hh in (60,120):
        x,_,_,_=run(qa,hh,cal,members,bm,1.0,False,n=n); x.update({'skew_window':sw,'skew_keep_pct':keep,'ivol_weight':wiv}); cr.append(x)
    pd.DataFrame(cr).to_csv(OUT/'construction.csv',index=False)
    bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    audit={**ua,'market_factor':market_code,'skew_windows':'40|60|80','skew_keep_pct':'0.70|0.80|0.90','ivol_weights':'0.60|0.6667|0.70','hold_grid':'60|120','anchor':'skew60 keep80%, ivol weight 2/3, hold120 fixed before this run','timing_violations':bad}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad: raise RuntimeError('timing violation')
    print('=== TOP GRID ==='); print(grid.sort_values('total_return',ascending=False).head(30).to_string(index=False),flush=True)
    print('=== SURFACE ==='); print(surface.sort_values('median_return',ascending=False).to_string(index=False),flush=True)
    print('=== ANCHOR ==='); print(pd.DataFrame([st]).to_string(index=False),flush=True)
    print('=== ANNUAL ==='); print(ann.to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== CONSTRUCTION ==='); print(pd.DataFrame(cr).sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
if __name__=='__main__': main()
