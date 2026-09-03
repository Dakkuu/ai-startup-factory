from __future__ import annotations
from pathlib import Path
import json,sys
import pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_geff55_strict_audit_v2 as strict
import run_multi_alpha_system_v1 as v1
from run_multi_alpha_shard_common_v2 import run_candidate_v2,normalize_equity
import run_10y_hard_executor_v3 as hv3
hv3.patch()

def main():
    fam=str(sys.argv[1]);
    if fam not in v1.SHORT_FAMILIES: raise ValueError(fam)
    OUT=Path(f'results_multi_alpha_short_{fam}_v2'); OUT.mkdir(exist_ok=True)
    p,cal,members,ua,mc,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board'); p=v1.add_short_features(p,cal); q=v1.make_short_q(p,fam)
    rows=[]; cache={}
    for h in v1.SHORT_H:
      for n in v1.SHORT_N:
       for e,k in v1.SHORT_BUF:
        print('RUN',fam,h,n,e,k,flush=True); eq,st=run_candidate_v2(q,h,n,e,k,cal,members,bm,1.0,False); key=f'{fam}|h{h}|n{n}|e{e}|k{k}'; rows.append({**st,'family':fam,'H':h,'N':n,'entry':e,'keep':k,'key':key}); cache[key]=(eq,(h,n,e,k))
    d=pd.DataFrame(rows); ok=d[(d.train_cagr>0)&(d.train_mdd>-0.45)].copy();
    if len(ok)==0: ok=d.copy()
    win=ok.sort_values(['train_calmar','train_sharpe','trade_count_mean'],ascending=[False,False,True]).iloc[0]; key=str(win.key); h,n,e,k=cache[key][1]
    d.to_csv(OUT/f'short_{fam}_grid.csv',index=False); normalize_equity(cache[key][0]).to_csv(OUT/f'short_{fam}_equity_cost1.csv',index=False)
    stress=[{**win.to_dict(),'cost_mult':1.0}]
    for cm in (2.0,4.0):
      eq,st=run_candidate_v2(q,h,n,e,k,cal,members,bm,cm,False); normalize_equity(eq).to_csv(OUT/f'short_{fam}_equity_cost{int(cm)}.csv',index=False); stress.append({**st,'family':fam,'H':h,'N':n,'entry':e,'keep':k,'key':key,'cost_mult':cm})
    pd.DataFrame(stress).to_csv(OUT/f'short_{fam}_winner_metrics.csv',index=False)
    meta={'family':fam,'winner_key':key,'selection':'2016-2021 train Calmar then Sharpe then lower trade count; 2022-2026 validation only','market_factor':mc,'universe_audit':ua}; (OUT/f'short_{fam}_meta.json').write_text(json.dumps(meta,indent=2,default=str))
    print('WINNER',key,flush=True); print(pd.DataFrame(stress).to_string(index=False),flush=True)
if __name__=='__main__': main()
