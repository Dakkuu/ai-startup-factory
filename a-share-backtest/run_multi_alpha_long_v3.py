from __future__ import annotations
from pathlib import Path
import glob,json
import numpy as np
import pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_geff55_strict_audit_v2 as strict
import run_geff_fundamental_integrated_v3 as iv3
import run_multi_alpha_system_v1 as v1
from run_multi_alpha_shard_common_v2 import run_candidate_v2,normalize_equity
import run_10y_hard_executor_v3 as hv3
hv3.patch()
OUT=Path('results_multi_alpha_long_v3'); OUT.mkdir(exist_ok=True)
ENSEMBLES={'L_all3_equal':{'value':1/3,'quality':1/3,'value_quality':1/3},'L_value_dominant_equal':{'value':.5,'value_quality':.5},'L_value70_vq30':{'value':.7,'value_quality':.3}}

def locate(p):
    h=glob.glob(p,recursive=True)
    if not h: raise FileNotFoundError(p)
    return h[0]

def main():
    old=pd.read_csv(locate('long_v2_input/**/long_grid.csv'))
    reps={}
    for fam,g in old.groupby('family'):
        r=g.sort_values(['train_calmar','train_sharpe'],ascending=False).iloc[0]; reps[fam]={'H':int(r.H),'N':int(r.N),'entry':float(r.entry),'keep':float(r.keep),'key':str(r.key),'train_calmar_v2':float(r.train_calmar)}
    pd.DataFrame([{'family':k,**v} for k,v in reps.items()]).to_csv(OUT/'family_representatives.csv',index=False)
    p,cal,members,ua,mc,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False); sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False)
    iv3.verify_attach(p,va,'value'); iv3.verify_attach(p,sa,'3stmt'); p2,z=iv3.fund_ranks(p,va,sa); p2=v1.add_long_ranks(p2,z)
    fam_eq={1:{},2:{},4:{}}; fam_metrics=[]
    for fam,cfg in reps.items():
        q=v1.make_long_q(p2,fam)
        for cm in (1.0,2.0,4.0):
            print('FAMILY',fam,'COST',cm,flush=True); eq,st=run_candidate_v2(q,cfg['H'],cfg['N'],cfg['entry'],cfg['keep'],cal,members,bm,cm,True); fam_eq[int(cm)][fam]=normalize_equity(eq); fam_metrics.append({'family':fam,'cost_mult':cm,**cfg,**st})
    pd.DataFrame(fam_metrics).to_csv(OUT/'family_rep_metrics.csv',index=False)
    rows=[]; cache={}
    for cm in (1,2,4):
      for name,wmap in ENSEMBLES.items():
        names=list(wmap); ws=[wmap[x] for x in names]; eq=normalize_equity(v1.fixed_mix([fam_eq[cm][x] for x in names],ws)); cache[(name,cm)]=eq
        s=v1.series_from_eq(eq); f=v1.perf_series(s); tr=v1.perf_series(s,v1.START,v1.TRAIN_END); ps=v1.perf_series(s,v1.PSEUDO,v1.END); rows.append({'ensemble':name,'cost_mult':float(cm),**f,'train_cagr':tr['cagr'],'train_mdd':tr['max_drawdown'],'train_sharpe':tr['sharpe'],'train_calmar':tr['calmar'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe']})
    d=pd.DataFrame(rows); d.to_csv(OUT/'long_ensemble_metrics.csv',index=False); d1=d[d.cost_mult==1].sort_values(['train_calmar','train_sharpe'],ascending=False); win=str(d1.iloc[0].ensemble)
    for cm in (1,2,4): cache[(win,cm)].to_csv(OUT/f'long_v3_selected_equity_cost{cm}.csv',index=False)
    (OUT/'long_v3_meta.json').write_text(json.dumps({'selected_train_only':win,'ensemble_specs':ENSEMBLES,'family_representatives':reps,'post_diagnostic':True,'market_factor':mc,'universe_audit':ua},indent=2,default=str)); print('SELECTED',win,flush=True); print(d.to_string(index=False),flush=True)
if __name__=='__main__': main()
