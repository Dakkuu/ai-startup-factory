from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

import run_10y_lowprice_strict_validation_v2 as sv
import run_10y_maxopt_v3_frozen_audit as fa

# Frozen from 2016-2021 train-only concentration/horizon V4 search.
sv.WEIGHTS={'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12}
sv.CFG={'liq':.55,'floor':2.0,'hold':60,'n':4,'entry':.15,'keep':.40}
sv.NPHASE=12


def core(out,p,q,cal,members,ua,market_code,bm):
    scr=sv.causal_scramble(p)
    pd.DataFrame([scr]).to_csv(out/'causal_exec_scramble.csv',index=False)
    diag=[]
    for ph in range(sv.NPHASE):
        st,e,tr,tm=sv.run_phase(q,ph,cal,members,bm,1e6)
        st['phase']=ph
        st['train_2016_2021_return']=fa.period_return(e,sv.mo.START,sv.mo.TRAIN_END)
        st['pseudo_oos_2022_2026_return']=fa.period_return(e,sv.mo.PSEUDO_START,sv.mo.END)
        diag.append(st)
    pd.DataFrame(diag).to_csv(out/'phase_diagnostic_1m_each.csv',index=False)

    rows=[]
    saved=None
    for cash in (1e6,1e7):
        s,e,tr,tm,sts=sv.run_ensemble(q,list(range(sv.NPHASE)),cal,members,bm,cash)
        s['implementation']='all12_exact_split_cash'
        rows.append(s)
        if cash==1e6: saved=(e,tr)
    pd.DataFrame(rows).to_csv(out/'implementations.csv',index=False)
    e,tr=saved
    fa.annual(e).to_csv(out/'annual_exact_1m_all12.csv',index=False)
    pd.DataFrame([sv.tail(e,tr)]).to_csv(out/'tail_exact_1m_all12.csv',index=False)

    costs=[]
    for cm in (2.,4.,8.):
        s,e,tr,tm,sts=sv.run_ensemble(q,list(range(sv.NPHASE)),cal,members,bm,1e6,cost=cm)
        costs.append(s)
    pd.DataFrame(costs).to_csv(out/'costs_exact_1m_all12.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'candidate':'LowPrice-N4-H60-Frozen-V5','weights':str(sv.WEIGHTS),'config':str(sv.CFG),'selection_period':'2016-07-29..2021-12-31','selection_validation_accesses':0,'selection_rule':'all phases positive + both train halves positive + exact-split MDD>-50%; max train exact-split CAGR among predeclared concentration/horizon grid','executor':'hard_v3; T-close -> later open; 100-share lots; board-limit block; no replacement','causal_exec_scramble_pass':scr['pass']}]).to_csv(out/'audit.csv',index=False)


def main(mode):
    out=Path(f'results_lowprice_n4_frozen_v5_{mode}'); out.mkdir(exist_ok=True)
    p,q,cal,members,ua,market_code,bm=sv.build(out)
    if mode=='core': core(out,p,q,cal,members,ua,market_code,bm)
    elif mode=='capacity': sv.capacity(out,q,cal,members,bm)
    elif mode=='delay': sv.delay(out,q,cal,members,bm)
    elif mode=='noise': sv.noise(out,q,cal,members,bm)
    elif mode=='delete': sv.deletion(out,q,cal,members,bm)
    elif mode=='placebo': sv.placebo(out,q,cal,members,bm)
    else: raise ValueError(mode)
    pd.DataFrame([{'candidate':'LowPrice-N4-H60-Frozen-V5','weights':str(sv.WEIGHTS),'config':str(sv.CFG),'nphase':sv.NPHASE,'selection_period':'2016-07-29..2021-12-31','validation_2022_2026_used_in_selection':0,'parameter_lock':1}]).to_csv(out/'frozen_provenance.csv',index=False)
    print('DONE',mode,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('core','capacity','delay','noise','delete','placebo')); a=ap.parse_args(); main(a.mode)
