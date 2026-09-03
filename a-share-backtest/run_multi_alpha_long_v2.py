from __future__ import annotations
from pathlib import Path
import glob,json
import pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_geff55_strict_audit_v2 as strict
import run_geff_fundamental_integrated_v3 as iv3
import run_multi_alpha_system_v1 as v1
from run_multi_alpha_shard_common_v2 import run_candidate_v2,normalize_equity
import run_10y_hard_executor_v3 as hv3
hv3.patch()

def locate(p):
    h=glob.glob(p,recursive=True)
    if not h: raise FileNotFoundError(p)
    return h[0]

def main():
    OUT=Path('results_multi_alpha_long_v2'); OUT.mkdir(exist_ok=True)
    p,cal,members,ua,mc,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False); sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False)
    iv3.verify_attach(p,va,'value'); iv3.verify_attach(p,sa,'3stmt'); p2,z=iv3.fund_ranks(p,va,sa); p2=v1.add_long_ranks(p2,z)
    rows=[]; cache={}; qs={}
    for fam in v1.LONG_FAMILIES:
      q=v1.make_long_q(p2,fam); qs[fam]=q
      for h in v1.LONG_H:
       for n in v1.LONG_N:
        e,k=v1.LONG_BUF; print('RUN LONG',fam,h,n,flush=True); eq,st=run_candidate_v2(q,h,n,e,k,cal,members,bm,1.0,True); key=f'{fam}|h{h}|n{n}|e{e}|k{k}'; rows.append({**st,'family':fam,'H':h,'N':n,'entry':e,'keep':k,'key':key}); cache[key]=(eq,(fam,h,n,e,k))
    d=pd.DataFrame(rows); ok=d[(d.train_cagr>0)&(d.train_mdd>-0.45)].copy();
    if len(ok)==0: ok=d.copy()
    win=ok.sort_values(['train_calmar','train_sharpe'],ascending=False).iloc[0]; key=str(win.key); fam,h,n,e,k=cache[key][1]
    d.to_csv(OUT/'long_grid.csv',index=False); normalize_equity(cache[key][0]).to_csv(OUT/'long_equity_cost1.csv',index=False)
    stress=[{**win.to_dict(),'cost_mult':1.0}]
    for cm in (2.0,4.0):
      eq,st=run_candidate_v2(qs[fam],h,n,e,k,cal,members,bm,cm,True); normalize_equity(eq).to_csv(OUT/f'long_equity_cost{int(cm)}.csv',index=False); stress.append({**st,'family':fam,'H':h,'N':n,'entry':e,'keep':k,'key':key,'cost_mult':cm})
    pd.DataFrame(stress).to_csv(OUT/'long_winner_metrics.csv',index=False)
    meta={'winner_key':key,'selection':'2016-2021 train Calmar then Sharpe; 2022-2026 validation only','market_factor':mc,'universe_audit':ua}; (OUT/'long_meta.json').write_text(json.dumps(meta,indent=2,default=str)); print('LONG WINNER',key,flush=True); print(pd.DataFrame(stress).to_string(index=False),flush=True)
if __name__=='__main__': main()
