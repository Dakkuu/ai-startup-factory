from __future__ import annotations
from pathlib import Path
import pandas as pd

import run_10y_lowprice_strict_validation_v2 as sv
import run_10y_maxopt_v3_frozen_audit as fa

# Pre-registered anti-overfit family ensemble from the H60/N4-N6-N8 training plateau.
sv.WEIGHTS={'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12}
sv.CFG={'liq':.55,'floor':2.0,'hold':60,'n':4,'entry':.15,'keep':.40}
sv.NPHASE=12
NS=(4,6,8)


def run_family(q,cal,members,bm,total_cash=1e6,cost=1.,vp=.05):
    phases=list(range(sv.NPHASE)); pieces=len(NS)*len(phases); per=float(total_cash)/pieces
    eqs=[]; trs=[]; initials=[]
    oldn=sv.CFG['n']
    try:
        for n in NS:
            sv.CFG['n']=int(n)
            for ph in phases:
                st,e,tr,tm=sv.run_phase(q,ph,cal,members,bm,per,cost,vp)
                eqs.append(e); initials.append(per)
                if len(tr): trs.append(tr.assign(n_family=n,phase=ph))
    finally:
        sv.CFG['n']=oldn
    eq=sv.combine_abs(eqs,initials)
    s=sv.summarize(eq,bm,trs)
    s.update(total_initial_cash=total_cash,subaccounts=pieces,cash_per_subaccount=per,cost_mult_test=cost,volume_participation=vp,family_ns='4|6|8',hold=60,entry=.15,keep=.40)
    return s,eq,trs


def main():
    out=Path('results_lowprice_plateau_ensemble_v5'); out.mkdir(exist_ok=True)
    p,q,cal,members,ua,market_code,bm=sv.build(out)
    s,eq,trs=run_family(q,cal,members,bm,1e6,1.,.05)
    pd.DataFrame([s]).to_csv(out/'frozen_full_validation.csv',index=False)
    fa.annual(eq).to_csv(out/'annual.csv',index=False)
    pd.DataFrame([sv.tail(eq,trs)]).to_csv(out/'tail.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        x,xe,xtr=run_family(q,cal,members,bm,1e6,cm,.05); costs.append(x)
    pd.DataFrame(costs).to_csv(out/'cost_stress.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'candidate':'LowPrice-H60-N4N6N8-Plateau-Ensemble-V5','weights':str(sv.WEIGHTS),'liq':.55,'floor':2.0,'hold':60,'entry':.15,'keep':.40,'family_ns':'4|6|8 equal capital','selection_period':'2016-07-29..2021-12-31','selection_validation_accesses':0,'reason':'N4/N6/N8 were a pre-observed train-only exact-split performance plateau; equal family averaging fixed before neighbor validation','implementation':'RMB1m split equally across 3 N-configs and 12 phases = 36 real subaccounts; 100-share lots; hard_v3'}]).to_csv(out/'audit.csv',index=False)
    print(pd.DataFrame([s]).to_string(index=False),flush=True)

if __name__=='__main__': main()
