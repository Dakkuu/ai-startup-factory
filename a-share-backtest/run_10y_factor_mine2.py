from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq

OUT=Path('results_factor_mine2'); OUT.mkdir(exist_ok=True)
HOLD_GRID=(60,120)
N_HOLD=20
VARIANTS=(
    'anchor','ivol_max','ivol_skew','ivol_dsemi','ivol_beta',
    'anchor_max','anchor_skew','anchor_capture',
    'anchor_max_filter','anchor_skew_filter',
    'anchor_mktmom','switch_eff_mktmom'
)

def add_extra(panel,cal,market_close):
    p=panel.copy()
    for c in ['max20','skew60','dsemi60','beta252','capture120','mom120']:
        p[c]=np.nan
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
        beta=cov/mvar.reindex(c.index); alpha=smu-beta*mmu.reindex(c.index)
        resid=r-alpha-beta*m
        max20=resid.rolling(20,min_periods=16).max()
        skew60=resid.rolling(60,min_periods=48).skew()
        neg=np.minimum(resid,0.0)
        dsemi=np.sqrt((neg*neg).rolling(60,min_periods=48).mean())
        mom120=c/c.shift(120)-1
        up_r=r.where(m>0); up_m=m.where(m>0)
        dn_r=r.where(m<0); dn_m=m.where(m<0)
        up=up_r.rolling(120,min_periods=40).mean()/up_m.rolling(120,min_periods=40).mean()
        dn=dn_r.rolling(120,min_periods=40).mean()/dn_m.rolling(120,min_periods=40).mean()
        capture=up-dn
        ds=pd.DatetimeIndex(p.loc[idx,'signal_date'])
        p.loc[idx,'max20']=max20.reindex(ds).to_numpy(float)
        p.loc[idx,'skew60']=skew60.reindex(ds).to_numpy(float)
        p.loc[idx,'dsemi60']=dsemi.reindex(ds).to_numpy(float)
        p.loc[idx,'beta252']=beta.reindex(ds).to_numpy(float)
        p.loc[idx,'capture120']=capture.reindex(ds).to_numpy(float)
        p.loc[idx,'mom120']=mom120.reindex(ds).to_numpy(float)
        if i%1000==0: print('extra histories',i,'/',len(groups),flush=True)
    return p

def rank_series(q,mask,col,ascending):
    return q.loc[mask].groupby('signal_date')[col].rank(pct=True,method='average',ascending=ascending)

def rerank(p,name,market_close):
    q=p.copy(); q['rank_test']=np.nan
    liq=(q.liq_rank_pct<=sim.LIQ_KEEP_PCT)
    valid=liq & np.isfinite(q.ivol60) & np.isfinite(q.eff120)
    iv=rank_series(q,valid,'ivol60',True)
    ef=rank_series(q,valid,'eff120',False)
    anchor=(2*iv+ef)/3
    m=valid
    if name=='anchor': raw=anchor
    elif name=='ivol_max':
        m=valid & np.isfinite(q.max20); iv=rank_series(q,m,'ivol60',True); x=rank_series(q,m,'max20',True); raw=(2*iv+x)/3
    elif name=='ivol_skew':
        m=valid & np.isfinite(q.skew60); iv=rank_series(q,m,'ivol60',True); x=rank_series(q,m,'skew60',True); raw=(2*iv+x)/3
    elif name=='ivol_dsemi':
        m=valid & np.isfinite(q.dsemi60); iv=rank_series(q,m,'ivol60',True); x=rank_series(q,m,'dsemi60',True); raw=(2*iv+x)/3
    elif name=='ivol_beta':
        m=valid & np.isfinite(q.beta252); iv=rank_series(q,m,'ivol60',True); x=rank_series(q,m,'beta252',False); raw=(2*iv+x)/3
    elif name=='anchor_max':
        m=valid & np.isfinite(q.max20); iv=rank_series(q,m,'ivol60',True); ef=rank_series(q,m,'eff120',False); x=rank_series(q,m,'max20',True); raw=.5*iv+.25*ef+.25*x
    elif name=='anchor_skew':
        m=valid & np.isfinite(q.skew60); iv=rank_series(q,m,'ivol60',True); ef=rank_series(q,m,'eff120',False); x=rank_series(q,m,'skew60',True); raw=.5*iv+.25*ef+.25*x
    elif name=='anchor_capture':
        m=valid & np.isfinite(q.capture120); iv=rank_series(q,m,'ivol60',True); ef=rank_series(q,m,'eff120',False); x=rank_series(q,m,'capture120',False); raw=.5*iv+.25*ef+.25*x
    elif name in ('anchor_max_filter','anchor_skew_filter'):
        col='max20' if name=='anchor_max_filter' else 'skew60'
        mm=valid & np.isfinite(q[col]); xp=rank_series(q,mm,col,True)
        ok=pd.Series(False,index=q.index); ok.loc[xp.index]=xp<=.80
        m=mm & ok
        iv=rank_series(q,m,'ivol60',True); ef=rank_series(q,m,'eff120',False); raw=(2*iv+ef)/3
    elif name in ('anchor_mktmom','switch_eff_mktmom'):
        sigdates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
        mmom=(market_close/market_close.shift(120)-1).reindex(sigdates)
        gate_map=(mmom>0).to_dict(); gate=q.signal_date.map(gate_map).fillna(False).astype(bool)
        if name=='anchor_mktmom':
            m=valid & gate
            iv=rank_series(q,m,'ivol60',True); ef=rank_series(q,m,'eff120',False); raw=(2*iv+ef)/3
        else:
            # Strong market: 50/50 IVOL + efficiency. Weak market: pure IVOL.
            m=valid
            iv=rank_series(q,m,'ivol60',True); ef=rank_series(q,m,'eff120',False)
            raw=iv.copy(); gm=gate.loc[m]
            raw.loc[gm.index[gm.to_numpy()]]=(.5*iv+.5*ef).loc[gm.index[gm.to_numpy()]]
    else: raise ValueError(name)
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q

def subset(q,hold):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(hold/5)); chosen=set(dates[::step])
    z=q[q.signal_date.isin(chosen)][['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy()
    z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def run(q,hold,cal,members,bm,cost=1.0):
    old=sim.N_HOLD; sim.N_HOLD=N_HOLD
    try:
        z=subset(q,hold); eq,tr,tm,to=sim.simulate(z,'ivol',cal,members,cost,daily_mtm=True)
        st=sim.perf(eq,tr,to,bm); st['hold_days']=hold; st['n_hold']=N_HOLD; st['cost_mult']=cost
        st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31')
        st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
        return st,eq,tr,tm
    finally: sim.N_HOLD=old

def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=add_extra(p,cal,market_close)
    bm=market_close.loc[sim.START:sim.END].dropna(); br=float(bm.iloc[-1]/bm.iloc[0]-1)
    rows=[]; annual=[]; cache={}
    for name in VARIANTS:
        q=rerank(p,name,market_close)
        for h in HOLD_GRID:
            print('RUN',name,'H',h,flush=True)
            st,eq,tr,tm=run(q,h,cal,members,bm); st.update({'variant':name,'benchmark_return':br,'excess':st['total_return']-br}); rows.append(st); cache[(name,h)]=(q,eq,tr,tm)
            a=sim.annual_returns(eq); a['variant']=name; a['hold_days']=h; annual.append(a)
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'grid.csv',index=False); pd.concat(annual,ignore_index=True).to_csv(OUT/'annual_all.csv',index=False)
    # Exploratory winner: choose on 2016-2021 only, never on 2022-2026.
    w=grid.sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]).iloc[0]
    key=(str(w.variant),int(w.hold_days)); q,eq,tr,tm=cache[key]
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=run(q,key[1],cal,members,bm,cm); st.update({'variant':key[0]}); costs.append(st)
    rob=sim.robustness(eq,tr); rob.update({'variant':key[0],'hold_days':key[1]})
    pd.DataFrame(costs).to_csv(OUT/'winner_cost.csv',index=False); pd.DataFrame([rob]).to_csv(OUT/'winner_robust.csv',index=False)
    wa=sim.annual_returns(eq); wa['variant']=key[0]; wa['hold_days']=key[1]; wa.to_csv(OUT/'winner_annual.csv',index=False)
    stab=grid.groupby('variant').agg(median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),median_train=('train_2016_2021_return','median'),median_pseudo_oos=('pseudo_oos_2022_2026_return','median'),min_pseudo_oos=('pseudo_oos_2022_2026_return','min'),median_mdd=('max_drawdown','median')).reset_index(); stab.to_csv(OUT/'stability.csv',index=False)
    alltm=pd.concat([x[3].assign(variant=k[0],h=k[1]) for k,x in cache.items() if len(x[3])],ignore_index=True)
    bad=int((pd.to_datetime(alltm.signal_date)>=pd.to_datetime(alltm.trade_date)).sum()) if len(alltm) else 0
    audit={**ua,'market_factor':market_code,'variants':'|'.join(VARIANTS),'hold_grid':'60|120','n_hold':N_HOLD,'selection':'exploratory; winner chosen only on 2016-2021; 2022-2026 pseudo-OOS','timing_violations':bad}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad: raise RuntimeError('timing violation')
    print('=== GRID ==='); print(grid.sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== STABILITY ==='); print(stab.to_string(index=False),flush=True)
    print('=== WINNER COST ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== WINNER ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== WINNER ANNUAL ==='); print(wa.to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
if __name__=='__main__': main()
