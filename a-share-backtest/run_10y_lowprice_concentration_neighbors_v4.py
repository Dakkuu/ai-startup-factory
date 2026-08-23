from __future__ import annotations
from pathlib import Path
import pandas as pd

import run_10y_lowprice_concentration_v4 as cv4
import run_10y_baseline_maxopt_v3 as mo
import run_10y_lowprice_signalpure_v1 as lp
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_lowprice_concentration_neighbors_v4'); OUT.mkdir(exist_ok=True)
WEIGHTS={'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12}
LIQ=.55; FLOOR=2.0; HOLD=60; ENTRY=.15; KEEP=.40; NS=(4,6,8)


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    p=lp.attach_price(p,cal); p=strict.attach_gap_flags(p,cal,'board'); q=lp.rank_signal(p,WEIGHTS,LIQ,FLOOR)
    rows=[]; anns=[]; phases=[]
    for n in NS:
        eq,sts,tms=cv4.phase_run(q,HOLD,n,ENTRY,KEEP,cal,members,bm,mo.START,mo.END,1e6,1.)
        s=fa.perf_eq(eq,bm); s['train_2016_2021_return']=fa.period_return(eq,mo.START,mo.TRAIN_END); s['pseudo_oos_2022_2026_return']=fa.period_return(eq,mo.PSEUDO_START,mo.END); s.update(n_hold=n,hold=HOLD,entry=ENTRY,keep=KEEP,total_cash=1e6,phase_count=len(sts)); rows.append(s)
        a=fa.annual(eq); a['n_hold']=n; anns.append(a)
        for ph,st in enumerate(sts):
            phases.append({**st,'n_hold_test':n,'phase':ph})
    pd.DataFrame(rows).to_csv(OUT/'neighbors_full.csv',index=False)
    pd.concat(anns,ignore_index=True).to_csv(OUT/'neighbors_annual.csv',index=False)
    pd.DataFrame(phases).to_csv(OUT/'neighbors_phase.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'experiment':'frozen construction-neighbor validation','weights':str(WEIGHTS),'liq':LIQ,'floor':FLOOR,'hold':HOLD,'entry':ENTRY,'keep':KEEP,'n_values':'4|6|8','purpose':'test whether N4 winner is a single-point construction overfit','selection':'no winner selected here; all three are reported','warning':'2022-2026 is reused pseudo-OOS and not untouched OOS','executor':'hard_v3; exact RMB1m all-phase split; board-limit block; 100-share lots; no replacement'}]).to_csv(OUT/'audit.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':main()
