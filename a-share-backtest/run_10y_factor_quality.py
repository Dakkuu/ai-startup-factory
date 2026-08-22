from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4

OUT=Path('results_factor_quality'); OUT.mkdir(exist_ok=True)
VARIANTS=('ivol','eff_only','quietmom_only','ivol_eff_pos','ivol_eff_top50','blend_ivol_eff','blend_ivol_quiet')
HOLD_GRID=(60,120)
N_HOLD=20

def add_factors(panel,cal):
    p=panel.copy()
    p['eff120']=np.nan; p['quietmom160']=np.nan
    groups=p.groupby('code').groups
    for i,(code,idx) in enumerate(groups.items(),1):
        c=base.qb.read_bin(code,'close',cal); h=base.qb.read_bin(code,'high',cal); l=base.qb.read_bin(code,'low',cal)
        if c.empty or h.empty or l.empty: continue
        df=pd.concat([c.rename('c'),h.rename('h'),l.rename('l')],axis=1).dropna()
        r=df.c.pct_change(fill_method=None)
        mom120=df.c/df.c.shift(120)-1
        path=r.abs().rolling(120,min_periods=100).sum()
        eff=mom120/path.replace(0,np.nan)
        amp=df.h/df.l-1
        q70=amp.rolling(160,min_periods=140).quantile(.70)
        quiet_r=r.where(amp<=q70,0.0)
        qm=quiet_r.rolling(160,min_periods=140).sum()
        ds=pd.DatetimeIndex(p.loc[idx,'signal_date'])
        p.loc[idx,'eff120']=eff.reindex(ds).to_numpy(float)
        p.loc[idx,'quietmom160']=qm.reindex(ds).to_numpy(float)
        if i%1000==0: print('quality histories',i,'/',len(groups),flush=True)
    return p

def rerank(p,name):
    q=p.copy(); q['rank_test']=np.nan
    liq=(q.liq_rank_pct<=sim.LIQ_KEEP_PCT)
    valid=liq & np.isfinite(q.ivol60)
    if name=='ivol':
        m=valid; asc=True; col='ivol60'
        q.loc[m,'rank_test']=q.loc[m].groupby('signal_date')[col].rank(pct=True,method='average',ascending=asc)
    elif name=='eff_only':
        m=valid & np.isfinite(q.eff120)
        q.loc[m,'rank_test']=q.loc[m].groupby('signal_date').eff120.rank(pct=True,method='average',ascending=False)
    elif name=='quietmom_only':
        m=valid & np.isfinite(q.quietmom160)
        q.loc[m,'rank_test']=q.loc[m].groupby('signal_date').quietmom160.rank(pct=True,method='average',ascending=False)
    elif name in ('ivol_eff_pos','ivol_eff_top50'):
        m=valid & np.isfinite(q.eff120)
        if name=='ivol_eff_pos': m=m & (q.eff120>0)
        else:
            ep=q.loc[valid & np.isfinite(q.eff120)].groupby('signal_date').eff120.rank(pct=True,method='average',ascending=False)
            ok=pd.Series(False,index=q.index); ok.loc[ep.index]=ep<=.50; m=m & ok
        q.loc[m,'rank_test']=q.loc[m].groupby('signal_date').ivol60.rank(pct=True,method='average',ascending=True)
    elif name in ('blend_ivol_eff','blend_ivol_quiet'):
        col='eff120' if name=='blend_ivol_eff' else 'quietmom160'
        m=valid & np.isfinite(q[col])
        a=q.loc[m].groupby('signal_date').ivol60.rank(pct=True,method='average',ascending=True)
        b=q.loc[m].groupby('signal_date')[col].rank(pct=True,method='average',ascending=False)
        raw=(a+b)/2
        q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    else: raise ValueError(name)
    return q

def subset(q,hold):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=hold//5; chosen=set(dates[::step])
    z=q[q.signal_date.isin(chosen)][['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy()
    z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')

def run(q,hold,cal,members,bm,cost=1.0):
    old=sim.N_HOLD; sim.N_HOLD=N_HOLD
    try:
        z=subset(q,hold); eq,tr,tm,to=sim.simulate(z,'ivol',cal,members,cost,daily_mtm=True)
        st=sim.perf(eq,tr,to,bm); st['hold_days']=hold; st['n_hold']=N_HOLD; st['cost_mult']=cost
        st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31')
        st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
        return st,eq,tr,tm
    finally: sim.N_HOLD=old

def annual(eq):
    a=sim.annual_returns(eq); return {int(r.year):float(r['return']) for _,r in a.iterrows()}

def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=add_factors(p,cal)
    bm=market_close.loc[sim.START:sim.END].dropna(); br=float(bm.iloc[-1]/bm.iloc[0]-1)
    rows=[]; annual_rows=[]; cache={}
    for v in VARIANTS:
        q=rerank(p,v)
        for h in HOLD_GRID:
            print('RUN',v,'H',h,flush=True)
            st,eq,tr,tm=run(q,h,cal,members,bm,1.0); st['variant']=v; st['benchmark_return']=br; st['excess']=st['total_return']-br
            rows.append(st); cache[(v,h)]=(q,eq,tr,tm)
            for y,r in annual(eq).items(): annual_rows.append({'variant':v,'hold_days':h,'year':y,'return':r})
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'grid.csv',index=False); pd.DataFrame(annual_rows).to_csv(OUT/'annual_all.csv',index=False)
    # winner chosen only on 2016-2021 return; 2022-2026 remains pseudo-OOS because this is iterative research
    w=grid.sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]).iloc[0]
    key=(str(w.variant),int(w.hold_days)); q,eq,tr,tm=cache[key]
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=run(q,key[1],cal,members,bm,cm); st['variant']=key[0]; costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_cost.csv',index=False)
    rob=sim.robustness(eq,tr); rob.update({'variant':key[0],'hold_days':key[1]}); pd.DataFrame([rob]).to_csv(OUT/'winner_robust.csv',index=False)
    stab=grid.groupby('variant').agg(median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),median_train=('train_2016_2021_return','median'),median_pseudo_oos=('pseudo_oos_2022_2026_return','median'),positive_pseudo_oos=('pseudo_oos_2022_2026_return',lambda x:int((x>0).sum())),median_mdd=('max_drawdown','median')).reset_index(); stab.to_csv(OUT/'stability.csv',index=False)
    alltm=pd.concat([x[3].assign(variant=k[0],h=k[1]) for k,x in cache.items() if len(x[3])],ignore_index=True)
    bad=int((pd.to_datetime(alltm.signal_date)>=pd.to_datetime(alltm.trade_date)).sum()) if len(alltm) else 0
    audit={**ua,'market_factor':market_code,'variants':'|'.join(VARIANTS),'hold_grid':'|'.join(map(str,HOLD_GRID)),'n_hold':N_HOLD,'selection':'train 2016-2021 only; 2022-2026 pseudo-OOS due iterative research','timing_violations':bad,'quietmom_definition':'160d rolling sum of returns on days whose amplitude <= causal rolling 160d 70th percentile'}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad: raise RuntimeError('timing violation')
    print('=== GRID ==='); print(grid.sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== STABILITY ==='); print(stab.to_string(index=False),flush=True)
    print('=== ANNUAL ==='); print(pd.DataFrame(annual_rows).to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
if __name__=='__main__': main()
