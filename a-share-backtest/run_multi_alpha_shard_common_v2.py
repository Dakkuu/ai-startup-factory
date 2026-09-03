from __future__ import annotations
import numpy as np
import pandas as pd
import run_10y_max_audit as ma
import run_multi_alpha_system_v1 as v1


def run_candidate_v2(q,h,n,e,k,cal,members,bm,cost=1.0,long_mode=False):
    phases=v1.phase_list(h,long_mode)
    eqs=[]; trade_counts=[]
    for ph in phases:
        st,eq,tr,tm=ma.run_panel(v1.subset(q,h,ph),cal,members,bm,n=n,entry=e,keep=k,cost=cost)
        eqs.append(eq); trade_counts.append(len(tr))
    eq=v1.fixed_mix(eqs,[1/len(eqs)]*len(eqs))
    s=v1.series_from_eq(eq)
    full=v1.perf_series(s); train=v1.perf_series(s,v1.START,v1.TRAIN_END); pseudo=v1.perf_series(s,v1.PSEUDO,v1.END)
    return eq,{**full,
        'train_cagr':train['cagr'],'train_mdd':train['max_drawdown'],'train_sharpe':train['sharpe'],'train_calmar':train['calmar'],
        'pseudo_cagr':pseudo['cagr'],'pseudo_mdd':pseudo['max_drawdown'],'pseudo_sharpe':pseudo['sharpe'],
        'phase_count':len(phases),'trade_count_mean':float(np.mean(trade_counts)) if trade_counts else np.nan}


def normalize_equity(eq):
    x=eq[['trade_date','equity']].copy(); x['trade_date']=pd.to_datetime(x.trade_date)
    x=x.sort_values('trade_date').drop_duplicates('trade_date',keep='last')
    if len(x): x['equity']=x.equity.astype(float)/float(x.equity.iloc[0])*1_000_000.0
    return x
