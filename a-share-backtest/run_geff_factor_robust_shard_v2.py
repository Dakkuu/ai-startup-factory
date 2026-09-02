from __future__ import annotations
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

import run_geff_factor_robust_opt_v1 as v1
import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()
HORIZONS=v1.HORIZONS; N=v1.N; ENTRY=v1.ENTRY; KEEP=v1.KEEP; HALF1=v1.HALF1; HALF2=v1.HALF2; PSEUDO=v1.PSEUDO
CANDS=list(v1.CANDIDATES.items())

def subset_exact(q,h,phase=0):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(h/5)); chosen=set(dates[phase::step])
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')

def run(q,h,ph,cal,members,bm):
    st,eq,tr,tm=ma.run_panel(subset_exact(q,h,ph),cal,members,bm,n=N,entry=ENTRY,keep=KEEP,cost=1.0)
    st['half1_cagr']=v1.period_cagr(eq,*HALF1); st['half2_cagr']=v1.period_cagr(eq,*HALF2); st['pseudo_cagr']=v1.period_cagr(eq,*PSEUDO); return st

def main():
    shard=int(sys.argv[1]); nshards=4; OUT=Path(f'results_geff_factor_shard_v2_{shard}'); OUT.mkdir(exist_ok=True)
    selected=[(n,w) for i,(n,w) in enumerate(CANDS) if i%nshards==shard]
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    rows=[]
    for name,w in selected:
        print('CAND',name,flush=True); q=mega.make_rank(p,{'name':name,'kind':'gate','g':{'ef':.55},'w':w})
        for h in HORIZONS:
            for ph in range(max(1,round(h/5))):
                st=run(q,h,ph,cal,members,bm); st.update(candidate=name,H=h,phase=ph,weights=json.dumps(w,sort_keys=True)); rows.append(st)
    d=pd.DataFrame(rows); detail,summary=v1.aggregate(rows)
    detail.to_csv(OUT/'all_phase_detail.csv',index=False); summary.to_csv(OUT/'summary.csv',index=False)
    meta={'shard':shard,'candidates':[x[0] for x in selected],'implementation':'source-identical statistics_v2 subset + run_panel','market_factor':market_code,'universe_audit':ua}; (OUT/'metadata.json').write_text(json.dumps(meta,indent=2,default=str))
    print(summary.to_string(index=False),flush=True)
if __name__=='__main__': main()
