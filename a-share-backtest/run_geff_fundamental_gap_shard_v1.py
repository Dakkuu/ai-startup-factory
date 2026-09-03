from __future__ import annotations
from pathlib import Path
import glob,json,sys
import pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_geff55_strict_audit_v2 as strict
import run_geff_fundamental_integrated_v3 as iv3
import run_geff_fundamental_structure_opt_v1 as so
import run_10y_hard_executor_v3 as hv3
hv3.patch()
def locate(p):
 h=glob.glob(p,recursive=True)
 if not h:raise FileNotFoundError(p)
 return h[0]
def main():
 shard=int(sys.argv[1]); gn,gc=list(so.GAPS.items())[shard];out=Path(f'results_geff_fund_gap_shard_{shard}');out.mkdir(exist_ok=True);p,cal,members,ua,mc,bm=mo.build_panel(out,need_fwd=False);p=strict.attach_gap_flags(p,cal,'board');va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False);sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False);iv3.verify_attach(p,va,'value');iv3.verify_attach(p,sa,'3stmt');p2,_=iv3.fund_ranks(p,va,sa);q=iv3.build_candidates(p2)['mom_cfo10_qv10'];rows=[]
 for h in so.HORIZONS:
  for ph in range(max(1,round(h/5))):
   st,eq,tr,tm=so.run(q,h,ph,.10,.30,cal,members,bm,gc,1.0);st.update(config='base_e10k30',gap=gn,gap_cap=gc,H=h,phase=ph);rows.append(st)
 d=pd.DataFrame(rows);s=so.aggregate(d,['config','gap','gap_cap']);d.to_csv(out/'all_phase.csv',index=False);s.to_csv(out/'summary.csv',index=False);print(s.to_string(index=False),flush=True)
if __name__=='__main__':main()
