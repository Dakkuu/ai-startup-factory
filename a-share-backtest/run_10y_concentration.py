from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_combo_rotation as combo

OUT=Path('results_concentration'); OUT.mkdir(exist_ok=True)
N_GRID=(5,10,15,20,30)
HOLD_GRID=(20,60,120)
VARIANTS=('ivol60','ivol_pricefloor3')

def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base()
    market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close)
    p=combo.add_raw_price(p,cal)
    p=combo.add_rank_columns(p)
    bm=market_close.loc[sim.START:sim.END].dropna(); br=float(bm.iloc[-1]/bm.iloc[0]-1)

    rows=[]; cache={}
    orig_n=sim.N_HOLD
    try:
        for variant in VARIANTS:
            for n in N_GRID:
                sim.N_HOLD=n
                for hold in HOLD_GRID:
                    print('RUN',variant,'N',n,'HOLD',hold,flush=True)
                    st,eq,tr,tm=combo.run_one(p,variant,hold,cal,members,bm,1.0,True)
                    st['n_hold']=n; st['posthoc']=variant!='ivol60'; st['benchmark_return']=br; st['excess']=st['total_return']-br
                    rows.append(st); cache[(variant,n,hold)]=(eq,tr,tm)
    finally:
        sim.N_HOLD=orig_n
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'grid.csv',index=False)

    # Strict selection: only pre-registered pure Low-IVOL, training window only.
    core=grid[~grid.posthoc].copy()
    w=core.sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False]).iloc[0]
    winner=(str(w.variant),int(w.n_hold),int(w.hold_days))
    print('TRAIN-ONLY WINNER',winner,flush=True)

    # Cost stress for strict winner.
    cost=[]; sim.N_HOLD=winner[1]
    try:
        for cm in (2.,4.,8.):
            st,_,_,_=combo.run_one(p,winner[0],winner[2],cal,members,bm,cm,True)
            st['n_hold']=winner[1]; cost.append(st)
    finally:
        sim.N_HOLD=orig_n
    pd.DataFrame(cost).to_csv(OUT/'winner_cost.csv',index=False)

    eq,tr,tm=cache[winner]
    rob=sim.robustness(eq,tr); rob['variant']=winner[0]; rob['n_hold']=winner[1]; rob['hold_days']=winner[2]
    pd.DataFrame([rob]).to_csv(OUT/'winner_robust.csv',index=False)
    ann=sim.annual_returns(eq); ann['variant']=winner[0]; ann['n_hold']=winner[1]; ann['hold_days']=winner[2]; ann.to_csv(OUT/'winner_annual.csv',index=False)

    stab=grid.groupby(['variant','posthoc','n_hold']).agg(
        median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),
        median_train=('train_2016_2021_return','median'),median_sealed=('sealed_2022_2026_return','median'),
        positive_sealed=('sealed_2022_2026_return',lambda x:int((x>0).sum())),median_mdd=('max_drawdown','median')
    ).reset_index(); stab.to_csv(OUT/'stability.csv',index=False)

    allt=pd.concat([tm.assign(test_variant=k[0],test_n=k[1],test_hold=k[2]) for k,(eq,tr,tm) in cache.items() if len(tm)],ignore_index=True)
    bad=int((pd.to_datetime(allt.signal_date)>=pd.to_datetime(allt.trade_date)).sum()) if len(allt) else 0
    audit={**ua,'market_factor':market_code,'benchmark_return':br,'n_grid':'|'.join(map(str,N_GRID)),'hold_grid':'|'.join(map(str,HOLD_GRID)),
           'strict_variant':'ivol60 only','posthoc_variant':'ivol_pricefloor3','selection':'highest 2016-2021 return among pure ivol grid only; sealed 2022-2026 untouched','timing_violations':bad}
    if bad: raise RuntimeError(f'timing violations {bad}')
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)

    print('=== GRID ==='); print(grid.sort_values(['posthoc','total_return'],ascending=[True,False]).to_string(index=False),flush=True)
    print('=== STABILITY ==='); print(stab.to_string(index=False),flush=True)
    print('=== WINNER COST ==='); print(pd.DataFrame(cost).to_string(index=False),flush=True)
    print('=== WINNER ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== WINNER ANNUAL ==='); print(ann.to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)

if __name__=='__main__': main()
