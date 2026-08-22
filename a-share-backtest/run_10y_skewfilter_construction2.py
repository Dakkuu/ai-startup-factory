from __future__ import annotations
from pathlib import Path
import pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_skewfilter_extensions as ext

OUT=Path('results_skewfilter_construction2'); OUT.mkdir(exist_ok=True)
NS=(10,15,20)
HOLDS=(50,60,70)

def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=ext.add_signal_fields(p,cal,market_close)
    q=ext.rerank(p,'skew40_base'); bm=market_close.loc[sim.START:sim.END].dropna(); rows=[]; annual=[]
    for n in NS:
      for h in HOLDS:
        print('RUN N',n,'H',h,flush=True); st,eq,tr,tm=ext.run(q,h,n,cal,members,bm); st.update({'variant':'skew40_base'}); rows.append(st)
        a=sim.annual_returns(eq); a['n_hold']=n; a['hold_days']=h; annual.append(a)
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'grid.csv',index=False); pd.concat(annual,ignore_index=True).to_csv(OUT/'annual_all.csv',index=False)
    audit={**ua,'market_factor':market_code,'signal':'skew40 keep80%; score=.60 low-IVOL + .40 efficiency','n_grid':'10|15|20','hold_grid':'50|60|70','execution':'deterministic hard cap; trapped positions occupy slots','all_positions_within_target':int(all(r['positions_max']<=r['n_hold'] for r in rows))}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    print('=== GRID ==='); print(grid.sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
if __name__=='__main__': main()
