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

OUT=Path('results_lowprice_construction_bag_v4'); OUT.mkdir(exist_ok=True)
# Predeclared training plateau discovered without 2022-2026: same frozen signal,
# H60/E15/K40, neighboring N=4/6/8. This is a robustness ensemble, not a retune.
WEIGHTS={'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12}
LIQ=.55; FLOOR=2.0; HOLD=60; ENTRY=.15; KEEP=.40; NS=(4,6,8); NPHASE=12


def combine_books(eqs,initials,start):
    return cv4.combine_abs(eqs,initials,start)


def run_book(q,n,cal,members,bm,start,end,cash,cost=1.):
    eq,sts,tms=cv4.phase_run(q,HOLD,n,ENTRY,KEEP,cal,members,bm,start,end,total_cash=cash,cost=cost)
    return eq,sts,tms


def summarize(eq,bm):
    s=fa.perf_eq(eq,bm)
    s['train_2016_2021_return']=fa.period_return(eq,mo.START,mo.TRAIN_END)
    s['pseudo_oos_2022_2026_return']=fa.period_return(eq,mo.PSEUDO_START,mo.END)
    return s


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    p=lp.attach_price(p,cal); p=strict.attach_gap_flags(p,cal,'board')
    q=lp.rank_signal(p,WEIGHTS,LIQ,FLOOR)

    book_cash=1e6/len(NS); books=[]; rows=[]
    for n in NS:
        eq,sts,tms=run_book(q,n,cal,members,bm,mo.START,mo.END,book_cash,1.)
        books.append(eq); st=summarize(eq,bm); st.update(n_hold=n,book_cash=book_cash); rows.append(st)
    bag=combine_books(books,[book_cash]*len(NS),mo.START); bsum=summarize(bag,bm)
    bsum.update({'construction':'equal_cash_N4_N6_N8','hold':HOLD,'entry':ENTRY,'keep':KEEP,'total_cash':1e6,'books':len(NS),'phases_per_book':NPHASE,'effective_sleeves':len(NS)*NPHASE})
    pd.DataFrame(rows).to_csv(OUT/'individual_books.csv',index=False)
    pd.DataFrame([bsum]).to_csv(OUT/'bag_full.csv',index=False)
    fa.annual(bag).to_csv(OUT/'bag_annual.csv',index=False)

    costs=[]
    for cm in (2.,4.,8.):
        ceqs=[]
        for n in NS:
            eq,_,_=run_book(q,n,cal,members,bm,mo.START,mo.END,book_cash,cm); ceqs.append(eq)
        cbag=combine_books(ceqs,[book_cash]*len(NS),mo.START); cs=summarize(cbag,bm); cs['cost_mult']=cm; costs.append(cs)
    pd.DataFrame(costs).to_csv(OUT/'bag_costs.csv',index=False)

    pd.DataFrame([{**ua,'market_factor':market_code,'experiment':'construction plateau bagging','weights':str(WEIGHTS),'liq':LIQ,'floor':FLOOR,'hold':HOLD,'entry':ENTRY,'keep':KEEP,'n_books':'4|6|8','selection_basis':'all three belonged to H60 training robustness plateau; no N6/N8 full-period validation consulted before registration','warning':'ensemble architecture itself was proposed after seeing N4 frozen full result; treat 2022-2026 as exploratory pseudo-OOS, not clean OOS','cash_model':'RMB1m total split equally across 3 books, each book split across 12 phase sleeves; 100-share lots preserved','executor':'hard_v3'}]).to_csv(OUT/'audit.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('BAG',pd.DataFrame([bsum]).to_string(index=False),flush=True)
    print('COSTS',pd.DataFrame(costs).to_string(index=False),flush=True)

if __name__=='__main__': main()
