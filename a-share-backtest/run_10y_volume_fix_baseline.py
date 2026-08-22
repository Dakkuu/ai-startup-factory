from __future__ import annotations
from pathlib import Path
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_grand_opt as grand
import run_10y_balanced_exact as be
import run_10y_max_audit as ma
import run_execution_units_fixed as exfix

OUT=Path('results_volume_unit_fix'); OUT.mkdir(exist_ok=True)


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members)
    bm=market_close.loc[sim.START:sim.END].dropna(); q=be.anchor_weighted(p,'liq70',.60)

    # Legacy exact run, before unit patch. Same frozen signal and portfolio rules.
    old_st,old_eq,old_tr,old_tm=ma.run_q(q,60,0,cal,members,bm,n=20,entry=.10,keep=.30)
    old_st.update({'execution_volume_unit':'legacy_lots_misread_as_shares'})

    exfix.install()
    unit=exfix.unit_audit(); pd.DataFrame([unit]).to_csv(OUT/'unit_conversion.csv',index=False)
    if not unit['unit_conversion_ok']: raise RuntimeError('volume conversion unit audit failed')

    # Corrected exact daily-MTM run. No strategy parameter changed.
    new_st,new_eq,new_tr,new_tm=ma.run_q(q,60,0,cal,members,bm,n=20,entry=.10,keep=.30)
    new_st.update({'execution_volume_unit':'tushare_lots_x100_to_shares'})
    pd.DataFrame([old_st,new_st]).to_csv(OUT/'old_vs_corrected.csv',index=False)
    sim.annual_returns(new_eq).to_csv(OUT/'corrected_annual.csv',index=False)
    pd.DataFrame([sim.robustness(new_eq,new_tr)]).to_csv(OUT/'corrected_tail.csv',index=False)

    costs=[]
    z=ma.subset_phase(q,60,0)
    for cm in (1.,2.,4.,8.):
        st,_,_,_=ma.run_panel(z,cal,members,bm,n=20,entry=.10,keep=.30,cost=cm)
        st['cost_mult_test']=cm; costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'corrected_costs.csv',index=False)

    caps=[]
    for cash in (1e6,5e6,1e7,5e7,1e8):
        for vp in (.01,.05):
            st,_,_,_=ma.run_panel(z,cal,members,bm,n=20,entry=.10,keep=.30,initial_cash=cash,vol_part=vp)
            st.update({'cash_test':cash,'vp_test':vp}); caps.append(st)
    pd.DataFrame(caps).to_csv(OUT/'corrected_capacity.csv',index=False)

    timing_bad=int((pd.to_datetime(new_tm.signal_date)>=pd.to_datetime(new_tm.trade_date)).sum()) if len(new_tm) else 0
    verdict={**ua,'market_factor':market_code,'frozen_rule':'liq top70%; remove highest 20% skew40; score=.60 low-IVOL60 + .40 efficiency120; N20; hold60; entry10 keep30; next-open','unit_fix_only':1,'timing_violations':timing_bad,'legacy_total_return':old_st['total_return'],'corrected_total_return':new_st['total_return'],'legacy_cagr':old_st['cagr'],'corrected_cagr':new_st['cagr'],'legacy_mdd':old_st['max_drawdown'],'corrected_mdd':new_st['max_drawdown'],'delta_total_return':new_st['total_return']-old_st['total_return'],'positions_max':new_st['positions_max']}
    pd.DataFrame([verdict]).to_csv(OUT/'verdict.csv',index=False)
    print('=== VOLUME UNIT FIX VERDICT ==='); print(pd.DataFrame([verdict]).to_string(index=False),flush=True)
    print('=== OLD VS CORRECTED ==='); print(pd.DataFrame([old_st,new_st]).to_string(index=False),flush=True)
    print('=== COSTS ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== CAPACITY ==='); print(pd.DataFrame(caps).to_string(index=False),flush=True)
    if timing_bad or new_st['positions_max']>20: raise RuntimeError('corrected baseline execution audit failed')

if __name__=='__main__': main()
