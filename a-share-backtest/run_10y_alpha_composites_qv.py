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

OUT=Path('results_alpha_composites_qv'); OUT.mkdir(exist_ok=True)
qv.sp.OUT=OUT

# Fixed before this run. Larger component values are always oriented as better.
COMPOSITES=[
 ('defensive_lottery',60,('low_ivol','anti_max','anti_skew','low_downside')),
 ('lowrisk_trend',60,('low_ivol','efficiency','residual_momentum')),
 ('allweather_core',60,('low_ivol','efficiency','residual_momentum','anti_max','anti_skew')),
 ('trend_quality',60,('efficiency','residual_momentum','near_52w_high','skip_momentum_63_5','skip_momentum_126_20')),
 ('trend_compression',60,('efficiency','near_52w_high','vol_compression','range_compression')),
 ('quiet_information',60,('quiet_trend','near_52w_high','residual_momentum','relative_volume')),
 ('lowrisk_capture',60,('low_ivol','low_beta','capture_asymmetry','low_downside')),
 ('anti_lottery_momentum',60,('low_ivol','anti_max','anti_skew','residual_momentum','efficiency')),
 ('reversal_liquidity',20,('short_reversal_5','short_reversal_20','oversold_volume','gap_reversal','intraday_reversal')),
 ('reversal_activity',20,('short_reversal_5','oversold_volume','relative_volume','gap_reversal')),
 ('compression_break',20,('range_compression','vol_compression','relative_volume','near_52w_high')),
 ('bar_reversal',20,('intraday_reversal','gap_reversal','short_reversal_5','short_reversal_20')),
]
NS=(10,15,20)
BUFFERS=((.05,.20),(.10,.30))


def add_composite_scores(p):
    q=p.copy()
    for name,h,cols in COMPOSITES:
        valid=(q.liq_rank_pct<=qv.LIQ_KEEP)
        for c in cols: valid &= np.isfinite(q[c])
        score=pd.Series(np.nan,index=q.index,dtype=float)
        if valid.any():
            mats=[]
            for c in cols:
                # best observation receives percentile near 1.0
                rr=q.loc[valid].groupby('signal_date')[c].rank(pct=True,method='average',ascending=True)
                mats.append(rr.rename(c))
            score.loc[valid]=pd.concat(mats,axis=1).mean(axis=1)
        q[name]=score
    return q


def ranked(q,name):
    z=q.copy(); z['rank_test']=np.nan
    m=(z.liq_rank_pct<=qv.LIQ_KEEP)&np.isfinite(z[name])
    z.loc[m,'rank_test']=z.loc[m].groupby('signal_date')[name].rank(pct=True,method='average',ascending=False)
    return z


def exact(q,h,n,e,k,cal,members,bm,cost=1.0):
    st,eq,tr,tm=ma.run_q(q,h,0,cal,members,bm,n=n,entry=e,keep=k,cost=cost)
    st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31')
    st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
    return st,eq,tr,tm


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=mine2.add_extra(p,cal,market_close); p=qv.add_qv_fields(p,cal); p=qv.attach_oriented_existing(p); p=add_composite_scores(p)
    bm=market_close.loc[sim.START:sim.END].dropna()

    pd.DataFrame([{'composite':n,'horizon':h,'components':'|'.join(c),'weights':'equal cross-sectional percentile ranks','selection':'pre-registered before run'} for n,h,c in COMPOSITES]).to_csv(OUT/'composite_manifest.csv',index=False)
    screens=[]; decs=[]; aics=[]
    for name,h,cols in COMPOSITES:
        print('COMPOSITE SCREEN',name,h,flush=True)
        r,d,a=qv.factor_diag(p,name,h); r['components']='|'.join(cols); screens.append(r)
        if len(d): decs.append(d)
        if len(a): aics.append(a)
    screen=pd.DataFrame(screens).sort_values(['screen_pass','ic_t','mean_ic'],ascending=[False,False,False]); screen.to_csv(OUT/'composite_screen.csv',index=False)
    if decs: pd.concat(decs,ignore_index=True).to_csv(OUT/'composite_deciles.csv',index=False)
    if aics: pd.concat(aics,ignore_index=True).to_csv(OUT/'composite_annual_ic.csv',index=False)

    # Construction discovery is allowed only after a training-period factor screen.
    passed=screen[screen.screen_pass==1]
    finalists=passed.head(6) if len(passed) else screen[(screen.mean_ic>0)&np.isfinite(screen.ic_t)].head(4)
    rows=[]; cache={}
    for r in finalists.itertuples(index=False):
        name=str(r.factor); h=int(r.horizon); rq=ranked(p,name)
        for n in NS:
            for e,k in BUFFERS:
                print('COMPOSITE EXACT',name,h,n,e,k,flush=True)
                st,eq,tr,tm=exact(rq,h,n,e,k,cal,members,bm); st.update({'composite':name,'horizon':h,'n_hold':n,'entry_pct':e,'keep_pct':k,'screen_pass':int(r.screen_pass),'ic_t_train':float(r.ic_t),'mean_ic_train':float(r.mean_ic)}); rows.append(st); cache[(name,h,n,e,k)]=(rq,eq,tr,tm)
    grid=pd.DataFrame(rows)
    if len(grid): grid=grid.sort_values(['screen_pass','train_2016_2021_return','max_drawdown'],ascending=[False,False,False])
    grid.to_csv(OUT/'construction_grid.csv',index=False)

    costs=[]; annual=[]; robust=[]; winner=None
    eligible=grid[grid.screen_pass==1] if len(grid) else pd.DataFrame()
    if len(eligible):
        w=eligible.sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]).iloc[0]
        winner=(str(w.composite),int(w.horizon),int(w.n_hold),float(w.entry_pct),float(w.keep_pct)); rq,eq,tr,tm=cache[winner]
        pd.DataFrame([w]).to_csv(OUT/'winner.csv',index=False)
        for cm in (2.,4.,8.):
            st,_,_,_=exact(rq,winner[1],winner[2],winner[3],winner[4],cal,members,bm,cm); st.update({'composite':winner[0]}); costs.append(st)
        a=sim.annual_returns(eq); a['composite']=winner[0]; annual.append(a)
        rr=sim.robustness(eq,tr); rr['composite']=winner[0]; robust.append(rr)
    if costs: pd.DataFrame(costs).to_csv(OUT/'winner_costs.csv',index=False)
    if annual: pd.concat(annual,ignore_index=True).to_csv(OUT/'winner_annual.csv',index=False)
    if robust: pd.DataFrame(robust).to_csv(OUT/'winner_robust.csv',index=False)

    allt=pd.concat([x[3].assign(composite=k[0]) for k,x in cache.items() if len(x[3])],ignore_index=True) if cache else pd.DataFrame()
    bad=int((pd.to_datetime(allt.signal_date)>=pd.to_datetime(allt.trade_date)).sum()) if len(allt) else 0
    hits=int((grid.total_return>=5.0).sum()) if len(grid) else 0
    audit={**ua,'market_factor':market_code,'research_round':'pre-registered QV composite families','composites':len(COMPOSITES),'screen_pass_count':int(screen.screen_pass.sum()),'construction_points':len(grid),'target_500_hits':hits,'selection':'IC screen and construction selection use 2016-2021 only; 2022-2026 pseudo-OOS','signal_universe':'signal-pure T-only; T+1 quote cannot alter T ranking','volume_source_unit_shares':100,'timing_violations':bad}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad: raise RuntimeError('timing violation')
    print('=== COMPOSITE SCREEN ==='); print(screen.to_string(index=False),flush=True)
    print('=== CONSTRUCTION TOP ==='); print(grid.head(40).to_string(index=False) if len(grid) else 'NONE',flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)

if __name__=='__main__': main()
