from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np

import run_10y_lowprice_concentration_v4 as cv
import run_10y_baseline_maxopt_v3 as mo
import run_10y_lowprice_signalpure_v1 as lp
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_sleeve_concentration_frontier_v5'); OUT.mkdir(exist_ok=True)
WEIGHTS={'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12}
LIQ=.55; FLOOR=2.0; HOLD=60; ENTRY=.15; KEEP=.40; NS=(2,3,4,6)
START=mo.START; TRAIN_END=mo.TRAIN_END; PSEUDO=mo.PSEUDO_START; END=mo.END


def metrics(eq, sts):
    cgs=np.array([s['cagr'] for s in sts],float); rets=np.array([s['total_return'] for s in sts],float)
    ec=cv.eq_cagr(eq); er=cv.eq_return(eq); em=cv.eq_mdd(eq)
    h1=cv.eq_cagr(eq,START,'2018-12-31'); h2=cv.eq_cagr(eq,'2019-01-01',TRAIN_END)
    score=float(ec+.50*np.min(cgs)-.25*np.std(cgs)+.10*em)
    hard=int((rets>0).all() and h1>0 and h2>0 and em>-0.35)
    return {'train_return':er,'train_cagr':ec,'train_mdd':em,'min_phase_return':rets.min(),'median_phase_return':np.median(rets),'min_phase_cagr':cgs.min(),'median_phase_cagr':np.median(cgs),'std_phase_cagr':cgs.std(),'half1_cagr':h1,'half2_cagr':h2,'robust_score':score,'hard_pass':hard}


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    p=lp.attach_price(p,cal); p=strict.attach_gap_flags(p,cal,'board')
    q=lp.rank_signal(p,WEIGHTS,LIQ,FLOOR)
    train=[]
    for n in NS:
        eq,sts,tms=cv.phase_run(q,HOLD,n,ENTRY,KEEP,cal,members,bm,START,TRAIN_END,1e6,1.)
        row={'n_hold':n,**metrics(eq,sts)}; train.append(row); print('TRAIN',row,flush=True)
    t=pd.DataFrame(train); t.to_csv(OUT/'train_frontier.csv',index=False)
    z=t[t.hard_pass==1].copy(); z=z if len(z) else t.copy()
    win=z.sort_values(['robust_score','train_cagr'],ascending=[False,False]).iloc[0].to_dict()
    pd.DataFrame([win]).to_csv(OUT/'train_only_winner.csv',index=False)

    n=int(win['n_hold'])
    eq,sts,tms=cv.phase_run(q,HOLD,n,ENTRY,KEEP,cal,members,bm,START,END,1e6,1.)
    full=fa.perf_eq(eq,bm); full['train_2016_2021_return']=fa.period_return(eq,START,TRAIN_END); full['pseudo_oos_2022_2026_return']=fa.period_return(eq,PSEUDO,END)
    full.update(n_hold=n,hold=HOLD,entry=ENTRY,keep=KEEP,weights=str(WEIGHTS),liq=LIQ,floor=FLOOR,selection_validation_accesses=0,selection_rule='N in 2/3/4/6 only; all train phases positive; both train blocks positive; exact-split train MDD>-35%; maximize robust score')
    pd.DataFrame([full]).to_csv(OUT/'frozen_full_validation.csv',index=False); fa.annual(eq).to_csv(OUT/'annual.csv',index=False)
    ph=[]
    for i,s in enumerate(sts): ph.append({**s,'phase':i})
    pd.DataFrame(ph).to_csv(OUT/'full_phase_diagnostics.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        ceq,csts,ctm=cv.phase_run(q,HOLD,n,ENTRY,KEEP,cal,members,bm,START,END,1e6,cm)
        cs=fa.perf_eq(ceq,bm); cs['cost_mult']=cm; cs['pseudo_return']=fa.period_return(ceq,PSEUDO,END); costs.append(cs)
    pd.DataFrame(costs).to_csv(OUT/'cost_stress.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'experiment':'Sleeve concentration frontier V5','ns':'2|3|4|6','fixed_signal':str(WEIGHTS),'fixed_hold':HOLD,'fixed_buffer':'15/40','selection_period':'2016-07-29..2021-12-31','validation_2022_2026_accesses_before_freeze':0,'research_note':'This frontier was motivated after observing the N4 validation result, so pseudo-OOS remains exploratory rather than clean untouched OOS.'}]).to_csv(OUT/'audit.csv',index=False)
    print('WINNER',pd.DataFrame([win]).to_string(index=False),flush=True); print('FULL',pd.DataFrame([full]).to_string(index=False),flush=True)

if __name__=='__main__': main()
