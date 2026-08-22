from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import run_10y_alpha_discovery_qv as qv
import run_10y_alpha_composites_qv as comp
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_factor_mine2 as mine2
import run_10y_max_audit as ma

OUT=Path('results_alpha_regime_qv'); OUT.mkdir(exist_ok=True)
qv.sp.OUT=OUT
VARIANTS=('trend_defensive','trend_lowrisk','vol_switch','two_by_two','trend_cash')
NS=(10,20)


def market_states(market_close,dates):
    c=market_close.sort_index(); r=c.pct_change(fill_method=None); mom120=c/c.shift(120)-1; ma120=c/c.rolling(120,min_periods=100).mean()-1
    vol20=r.rolling(20,min_periods=16).std(); vol_med=vol20.rolling(252,min_periods=126).median()
    z=pd.DataFrame({'signal_date':dates}); z['bull']=mom120.reindex(dates).to_numpy(float)>0; z['above_ma']=ma120.reindex(dates).to_numpy(float)>0; z['highvol']=(vol20.reindex(dates)>vol_med.reindex(dates)).fillna(False).to_numpy(bool)
    return z.set_index('signal_date')


def regime_rank(p,states,variant):
    q=p.copy(); q['rank_test']=np.nan
    for d,g in q.groupby('signal_date',sort=True):
        if d not in states.index: continue
        s=states.loc[d]; bull=bool(s.bull); hv=bool(s.highvol); above=bool(s.above_ma)
        if variant=='trend_defensive': source='trend_quality' if bull else 'defensive_lottery'
        elif variant=='trend_lowrisk': source='trend_quality' if bull else 'lowrisk_capture'
        elif variant=='vol_switch': source='defensive_lottery' if hv else 'trend_compression'
        elif variant=='two_by_two':
            if bull and not hv: source='trend_compression'
            elif bull and hv: source='lowrisk_trend'
            elif (not bull) and hv: source='lowrisk_capture'
            else: source='allweather_core'
        elif variant=='trend_cash':
            if not above: continue
            source='trend_quality'
        else: raise ValueError(variant)
        m=(g.liq_rank_pct<=qv.LIQ_KEEP)&np.isfinite(g[source])
        if not m.any(): continue
        idx=g.index[m]; q.loc[idx,'rank_test']=g.loc[idx,source].rank(pct=True,method='average',ascending=False)
    return q


def exact(q,n,cal,members,bm,cost=1.0):
    st,eq,tr,tm=ma.run_q(q,60,0,cal,members,bm,n=n,entry=.10,keep=.30,cost=cost)
    st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31'); st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
    return st,eq,tr,tm


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=mine2.add_extra(p,cal,market_close); p=qv.add_qv_fields(p,cal); p=qv.attach_oriented_existing(p); p=comp.add_composite_scores(p)
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(p.signal_date.unique()))); states=market_states(market_close,dates); states.reset_index().to_csv(OUT/'market_states.csv',index=False)
    bm=market_close.loc[sim.START:sim.END].dropna(); rows=[]; cache={}
    for variant in VARIANTS:
        rq=regime_rank(p,states,variant)
        for n in NS:
            print('REGIME',variant,'N',n,flush=True); st,eq,tr,tm=exact(rq,n,cal,members,bm); st.update({'variant':variant,'n_hold':n,'hold_days':60}); rows.append(st); cache[(variant,n)]=(rq,eq,tr,tm)
    grid=pd.DataFrame(rows).sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]); grid.to_csv(OUT/'regime_grid.csv',index=False)
    w=grid.iloc[0]; key=(str(w.variant),int(w.n_hold)); rq,eq,tr,tm=cache[key]; pd.DataFrame([w]).to_csv(OUT/'winner.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=exact(rq,key[1],cal,members,bm,cm); st.update({'variant':key[0],'n_hold':key[1]}); costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_costs.csv',index=False)
    a=sim.annual_returns(eq); a['variant']=key[0]; a['n_hold']=key[1]; a.to_csv(OUT/'winner_annual.csv',index=False)
    rr=sim.robustness(eq,tr); rr.update({'variant':key[0],'n_hold':key[1]}); pd.DataFrame([rr]).to_csv(OUT/'winner_robust.csv',index=False)
    allt=pd.concat([x[3].assign(variant=k[0],n=k[1]) for k,x in cache.items() if len(x[3])],ignore_index=True); bad=int((pd.to_datetime(allt.signal_date)>=pd.to_datetime(allt.trade_date)).sum()) if len(allt) else 0
    audit={**ua,'market_factor':market_code,'research_round':'causal market-regime conditional alpha','variants':'|'.join(VARIANTS),'n_grid':'10|20','market_state_inputs':'T-close 120d momentum, 120d MA gap, 20d vol versus trailing-252 median','selection':'variant/N winner chosen on 2016-2021 only; 2022-2026 pseudo-OOS','signal_universe':'signal-pure T-only','volume_source_unit_shares':100,'target_500_hits':int((grid.total_return>=5.0).sum()),'timing_violations':bad}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad: raise RuntimeError('timing violation')
    print('=== REGIME GRID ==='); print(grid.to_string(index=False),flush=True); print('=== COSTS ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True); print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)

if __name__=='__main__': main()
