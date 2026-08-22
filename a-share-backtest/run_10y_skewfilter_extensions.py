from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_skewfilter_hard as hard

OUT=Path('results_skewfilter_extensions'); OUT.mkdir(exist_ok=True)
# Anchor A discovered before this run: skew40, keep 80%, IVOL weight .60, 60d rotation, N=20.
ANCHOR=('skew40_base',40,.80,.60,60,20)
VARIANTS=('skew40_base','price3','max20_keep80','eff_keep80','price3_max20')
HOLDS=(40,60,80)
NS=(15,20,25)

def add_signal_fields(panel,cal,market_close):
    p=panel.copy(); p['raw_price']=np.nan; p['max20x']=np.nan; p['mom120x']=np.nan
    mret=market_close.reindex(cal[(cal>=sim.WARM)&(cal<=sim.END)]).pct_change(fill_method=None)
    mmu=mret.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).mean().shift(1)
    mvar=mret.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).var().shift(1)
    groups=p.groupby('code').groups
    for i,(code,idx) in enumerate(groups.items(),1):
        c=base.qb.read_bin(code,'close',cal).loc[sim.WARM:sim.END]; f=base.qb.read_bin(code,'factor',cal).loc[sim.WARM:sim.END]
        if c.empty: continue
        f=f.replace(0,np.nan).reindex(c.index).fillna(1.0); rawp=c/f
        r=c.pct_change(fill_method=None); m=mret.reindex(c.index)
        smu=r.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).mean().shift(1)
        cov=r.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).cov(m).shift(1)
        beta=cov/mvar.reindex(c.index); alpha=smu-beta*mmu.reindex(c.index); resid=r-alpha-beta*m
        mx=resid.rolling(20,min_periods=16).max(); mom=c/c.shift(120)-1
        ds=pd.DatetimeIndex(p.loc[idx,'signal_date'])
        p.loc[idx,'raw_price']=rawp.reindex(ds).to_numpy(float); p.loc[idx,'max20x']=mx.reindex(ds).to_numpy(float); p.loc[idx,'mom120x']=mom.reindex(ds).to_numpy(float)
        if i%1000==0: print('extension fields',i,'/',len(groups),flush=True)
    return p

def base_mask_and_raw(q):
    m=(q.liq_rank_pct<=sim.LIQ_KEEP_PCT)&np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q.skew40)
    sp=q.loc[m].groupby('signal_date').skew40.rank(pct=True,method='average',ascending=True)
    ok=pd.Series(False,index=q.index); ok.loc[sp.index]=sp<=.80; m=m&ok
    return m

def rerank(p,name):
    q=p.copy(); q['rank_test']=np.nan; m=base_mask_and_raw(q)
    if name in ('price3','price3_max20'):
        m=m & np.isfinite(q.raw_price) & (q.raw_price>=3.0)
    if name in ('max20_keep80','price3_max20'):
        mm=m & np.isfinite(q.max20x)
        xp=q.loc[mm].groupby('signal_date').max20x.rank(pct=True,method='average',ascending=True)
        ok=pd.Series(False,index=q.index); ok.loc[xp.index]=xp<=.80; m=mm&ok
    if name=='eff_keep80':
        mm=m & np.isfinite(q.eff120)
        ep=q.loc[mm].groupby('signal_date').eff120.rank(pct=True,method='average',ascending=False)
        ok=pd.Series(False,index=q.index); ok.loc[ep.index]=ep<=.80; m=mm&ok
    iv=q.loc[m].groupby('signal_date').ivol60.rank(pct=True,method='average',ascending=True)
    ef=q.loc[m].groupby('signal_date').eff120.rank(pct=True,method='average',ascending=False)
    raw=.60*iv+.40*ef
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q

def subset(q,hold):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(hold/5)); chosen=set(dates[::step])
    z=q[q.signal_date.isin(chosen)][['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def run(q,hold,n,cal,members,bm,cost=1.0):
    old=hard.N_HOLD; hard.N_HOLD=n
    try:
        z=subset(q,hold); eq,tr,tm,to=hard.hard_simulate(z,cal,members,cost); st=sim.perf(eq,tr,to,bm)
        st.update({'hold_days':hold,'n_hold':n,'cost_mult':cost,'positions_max':int(eq.positions.max()),'positions_median':float(eq.positions.median())})
        st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31'); st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
        return st,eq,tr,tm
    finally: hard.N_HOLD=old

def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=add_signal_fields(p,cal,market_close)
    bm=market_close.loc[sim.START:sim.END].dropna(); rows=[]; cache={}
    # All extension variants at fixed discovered anchor construction, plus hold neighborhood.
    for name in VARIANTS:
        q=rerank(p,name)
        for h in HOLDS:
            print('EXT',name,'hold',h,flush=True); st,eq,tr,tm=run(q,h,20,cal,members,bm); st['variant']=name; rows.append(st); cache[(name,h,20)]=(q,eq,tr,tm)
    # Concentration neighborhood only for fixed base signal at h=60.
    qb=rerank(p,'skew40_base')
    for n in (15,25):
        st,eq,tr,tm=run(qb,60,n,cal,members,bm); st['variant']='skew40_base'; rows.append(st); cache[('skew40_base',60,n)]=(qb,eq,tr,tm)
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'grid.csv',index=False)
    # Exact winner-A anchor diagnostics, fixed before this run.
    q,eq,tr,tm=cache[('skew40_base',60,20)]
    ann=sim.annual_returns(eq); ann.to_csv(OUT/'anchor_annual.csv',index=False)
    rob=sim.robustness(eq,tr); pd.DataFrame([rob]).to_csv(OUT/'anchor_robust.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=run(q,60,20,cal,members,bm,cm); st['variant']='skew40_base'; costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'anchor_cost.csv',index=False)
    # Summary by extension: don't pick on pseudo-OOS.
    stab=grid.groupby('variant').agg(median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),median_train=('train_2016_2021_return','median'),median_pseudo_oos=('pseudo_oos_2022_2026_return','median'),min_pseudo_oos=('pseudo_oos_2022_2026_return','min'),median_mdd=('max_drawdown','median')).reset_index(); stab.to_csv(OUT/'stability.csv',index=False)
    alltm=pd.concat([x[3] for x in cache.values() if len(x[3])],ignore_index=True); bad=int((pd.to_datetime(alltm.signal_date)>=pd.to_datetime(alltm.trade_date)).sum()) if len(alltm) else 0
    audit={**ua,'market_factor':market_code,'anchor':'skew40 keep80%, score=.60 lowIVOL + .40 efficiency, hold60, N20 fixed before run','extensions':'|'.join(VARIANTS),'hold_grid':'40|60|80','n_grid':'15|20|25','execution':'deterministic hard cap; trapped positions occupy slots','timing_violations':bad,'all_positions_within_target':int(all(r['positions_max']<=r['n_hold'] for r in rows))}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad or not audit['all_positions_within_target']: raise RuntimeError('audit failure')
    print('=== GRID ==='); print(grid.sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== STABILITY ==='); print(stab.sort_values('median_return',ascending=False).to_string(index=False),flush=True)
    print('=== ANCHOR ANNUAL ==='); print(ann.to_string(index=False),flush=True)
    print('=== ANCHOR COST ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== ANCHOR ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
if __name__=='__main__': main()
