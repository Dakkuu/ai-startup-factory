from __future__ import annotations
from pathlib import Path
import glob,json
import pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_geff55_strict_audit_v2 as strict
import run_geff_fundamental_integrated_v3 as iv3
import run_multi_alpha_system_v1 as v1
from run_multi_alpha_shard_common_v2 import normalize_equity
import run_10y_hard_executor_v3 as hv3
hv3.patch()

def locate(p):
    h=glob.glob(p,recursive=True)
    if not h: raise FileNotFoundError(p)
    return h[0]

def main():
    OUT=Path('results_multi_alpha_medium_v2'); OUT.mkdir(exist_ok=True)
    p,cal,members,ua,mc,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False); sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False)
    iv3.verify_attach(p,va,'value'); iv3.verify_attach(p,sa,'3stmt'); p2,z=iv3.fund_ranks(p,va,sa)
    rows=[]
    for cm in (1.0,2.0,4.0):
        print('RUN MEDIUM COST',cm,flush=True); eq=v1.medium_equity(p2,cal,members,bm,cm); eq=normalize_equity(eq); eq.to_csv(OUT/f'medium_equity_cost{int(cm)}.csv',index=False)
        s=v1.series_from_eq(eq); f=v1.perf_series(s); tr=v1.perf_series(s,v1.START,v1.TRAIN_END); ps=v1.perf_series(s,v1.PSEUDO,v1.END)
        rows.append({'cost_mult':cm,**f,'train_cagr':tr['cagr'],'train_mdd':tr['max_drawdown'],'train_sharpe':tr['sharpe'],'train_calmar':tr['calmar'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe']})
    pd.DataFrame(rows).to_csv(OUT/'medium_metrics.csv',index=False)
    pd.DataFrame({'trade_date':bm.index,'benchmark_close':bm.to_numpy(float)}).to_csv(OUT/'benchmark.csv',index=False)
    meta={'strategy':'mom_cfo10_qv10 H60; 75% N10 + 25% N5 rank tilt; phases 0/4/8 equal','fixed_before_multi_alpha_search':True,'market_factor':mc,'universe_audit':ua}; (OUT/'medium_meta.json').write_text(json.dumps(meta,indent=2,default=str)); print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__': main()
