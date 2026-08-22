from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_grand_opt as grand
import run_10y_balanced_exact as be
import run_10y_max_audit as ma

OUT=Path('results_backward_holdout'); OUT.mkdir(exist_ok=True)
START=pd.Timestamp('2007-01-04')
END=pd.Timestamp('2016-07-28')
WARM=pd.Timestamp('2005-01-01')

# Frozen before observing this period's performance.
RULE='liq top70%; remove highest 20% skew40; score=.60 low-IVOL60 rank + .40 efficiency120 rank; N20; 60d; entry10 keep30; next-open'

def main():
    sim.START=START; sim.END=END; sim.WARM=WARM
    base.START=START; base.END=END; base.WARM=WARM; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base()
    market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close)
    p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members)
    bm=market_close.loc[START:END].dropna(); q=be.anchor_weighted(p,'liq70',.60)
    st,eq,tr,tm=ma.run_q(q,60,0,cal,members,bm,n=20,entry=.10,keep=.30)
    pd.DataFrame([st]).to_csv(OUT/'summary.csv',index=False)
    sim.annual_returns(eq).to_csv(OUT/'annual.csv',index=False)

    phases=[]
    for ph in range(12):
        x,_,_,_=ma.run_q(q,60,ph,cal,members,bm,n=20,entry=.10,keep=.30)
        phases.append({**x,'phase':ph})
    ph=pd.DataFrame(phases); ph.to_csv(OUT/'phase_offsets.csv',index=False)

    blocks=[]
    for name,a,b in [('2007_2009','2007-01-04','2009-12-31'),('2010_2012','2010-01-01','2012-12-31'),('2013_2016','2013-01-01','2016-07-28')]:
        z=grand.period_metrics(eq,a,b); z['block']=name; blocks.append(z)
    bl=pd.DataFrame(blocks); bl.to_csv(OUT/'blocks.csv',index=False)

    costs=[]
    for cm in (2.,4.,8.):
        z=ma.subset_phase(q,60,0); x,_,_,_=ma.run_panel(z,cal,members,bm,n=20,entry=.10,keep=.30,cost=cm)
        costs.append({**x,'cost_mult_test':cm})
    co=pd.DataFrame(costs); co.to_csv(OUT/'costs.csv',index=False)

    delays=[]
    for d in (1,3,5):
        z=ma.delay_panel(q,d,cal,members); x,_,_,_=ma.run_panel(z,cal,members,bm,n=20,entry=.10,keep=.30)
        delays.append({**x,'delay_sessions':d})
    de=pd.DataFrame(delays); de.to_csv(OUT/'delays.csv',index=False)

    bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    gates={
      'timing_zero':int(bad==0),
      'total_return_positive':int(st['total_return']>0),
      'cagr_ge_5pct':int(st['cagr']>=.05),
      'mdd_better_than_minus40pct':int(st['max_drawdown']>-0.40),
      'sharpe_ge_0_4':int(st['sharpe']>=.40),
      'all_12_phases_positive':int((ph.total_return>0).all()),
      'phase_median_cagr_ge_5pct':int(ph.cagr.median()>=.05),
      'all_3_blocks_positive':int((bl['return']>0).all()),
      'cost4_positive':int(float(co.loc[co.cost_mult_test==4,'total_return'].iloc[0])>0),
      'delay3_positive':int(float(de.loc[de.delay_sessions==3,'total_return'].iloc[0])>0),
    }
    gd=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); gd.to_csv(OUT/'gates.csv',index=False)
    verdict={**ua,'market_factor':market_code,'holdout_start':str(START.date()),'holdout_end':str(END.date()),'frozen_rule':RULE,'total_return':st['total_return'],'cagr':st['cagr'],'mdd':st['max_drawdown'],'sharpe':st['sharpe'],'gates_passed':int(gd['pass'].sum()),'gates_total':len(gd),'backward_holdout_hard_pass':int(gd['pass'].all())}
    pd.DataFrame([verdict]).to_csv(OUT/'verdict.csv',index=False)
    print('=== BACKWARD HOLDOUT VERDICT ==='); print(pd.DataFrame([verdict]).to_string(index=False),flush=True)
    print('=== GATES ==='); print(gd.to_string(index=False),flush=True)
    print('=== PHASES ==='); print(ph[['phase','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True)
    print('=== BLOCKS ==='); print(bl.to_string(index=False),flush=True)
    print('=== COSTS ==='); print(co[['cost_mult_test','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True)
    print('=== DELAYS ==='); print(de[['delay_sessions','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True)

if __name__=='__main__': main()
