from __future__ import annotations
from pathlib import Path
import glob,json
import numpy as np
import pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_maxopt_v3_frozen_audit as fa
import run_geff_fundamental_integrated_v3 as iv3
import run_geff_fundamental_ranktilt_v1 as rt
import run_geff_fundamental_structure_opt_v1 as so
import run_10y_hard_executor_v3 as hv3
hv3.patch()
OUT=Path('results_geff_fundamental_ranktilt_ensemble_v1');OUT.mkdir(exist_ok=True)
HORIZONS=(60,75,90);ENTRY=.10;KEEP=.30;MIX={'N10':[(10,1.0)],'RT25':[(5,.25),(10,.75)]}
STRUCT={60:{'3s':[0,4,8],'5s':[0,2,5,7,10],'all':list(range(12))},75:{'3s':[0,5,10],'5s':[0,3,6,9,12],'all':list(range(15))},90:{'3s':[0,6,12],'5s':[0,4,7,11,14],'all':list(range(18))}}
def locate(p):
 h=glob.glob(p,recursive=True)
 if not h:raise FileNotFoundError(p)
 return h[0]
def build():
 p,cal,members,ua,mc,bm=mo.build_panel(OUT,need_fwd=False);p=strict.attach_gap_flags(p,cal,'board');va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False);sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False);iv3.verify_attach(p,va,'value');iv3.verify_attach(p,sa,'3stmt');p2,_=iv3.fund_ranks(p,va,sa);q=iv3.build_candidates(p2)['mom_cfo10_qv10'];return q,cal,members,ua,mc,bm
def subset(q,h,ph):
 ds=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())));step=max(1,round(h/5));chosen=set(ds[ph::step]);cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns];z=q[q.signal_date.isin(chosen)][cols].copy();z['ivol60_pct']=z.rank_test;return z.drop(columns='rank_test')
def mixeq(cache,h,ph,spec):return rt.weighted_mix([cache[(n,h,ph)] for n,w in spec],[w for n,w in spec])
def stats(eq):
 s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index();s=s[~s.index.duplicated(keep='last')];r=s.pct_change().dropna();days=max(1,(s.index[-1]-s.index[0]).days);return {'cagr':float((s.iloc[-1]/s.iloc[0])**(365.25/days)-1),'mdd':float((s/s.cummax()-1).min()),'sharpe':float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if len(r)>2 and r.std(ddof=1)>0 else np.nan}
def main():
 q,cal,members,ua,mc,bm=build();rows=[];daily=[]
 for cost in (1.0,2.0):
  cache={}
  for h in HORIZONS:
   for ph in range(max(1,round(h/5))):
    z=subset(q,h,ph)
    for n in (5,10):
     st,eq,tr,tm=ma.run_panel(z,cal,members,bm,n=n,entry=ENTRY,keep=KEEP,cost=cost);cache[(n,h,ph)]=eq
  # build each H structure
  built={}
  for mixname,spec in MIX.items():
   phasecurve={(h,ph):mixeq(cache,h,ph,spec) for h in HORIZONS for ph in range(max(1,round(h/5)))}
   for h in HORIZONS:
    for sname,ids in STRUCT[h].items():
     eq=fa.phase_ensemble([phasecurve[(h,i)] for i in ids]);built[(mixname,h,sname)]=eq;st=stats(eq);trn=so.slice_stats(eq,so.START,so.TRAIN_END);ps=so.slice_stats(eq,so.PSEUDO,so.END);rows.append({**st,'mix':mixname,'cost_mult':cost,'structure':f'H{h}_{sname}','H':h,'sleeves':len(ids),'train_cagr':trn['cagr'],'train_mdd':trn['mdd'],'train_calmar':trn['cagr']/abs(trn['mdd']),'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['mdd']})
   # practical cross-H 3x3 and 5x3; all-phase crossH
   for sname in ('3s','5s','all'):
    eq=fa.phase_ensemble([built[(mixname,h,sname)] for h in HORIZONS]);st=stats(eq);trn=so.slice_stats(eq,so.START,so.TRAIN_END);ps=so.slice_stats(eq,so.PSEUDO,so.END);label=f'crossH_{sname}';rows.append({**st,'mix':mixname,'cost_mult':cost,'structure':label,'H':'60|75|90','sleeves':sum(len(STRUCT[h][sname]) for h in HORIZONS),'train_cagr':trn['cagr'],'train_mdd':trn['mdd'],'train_calmar':trn['cagr']/abs(trn['mdd']),'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['mdd']});
    if cost==1.0 and mixname=='RT25' and sname in ('3s','5s','all'):
      x=eq[['trade_date','equity']].copy();x['label']=label;daily.append(x)
 d=pd.DataFrame(rows).sort_values(['cost_mult','train_calmar','train_cagr'],ascending=[True,False,False]);d.to_csv(OUT/'ensemble_summary.csv',index=False);pd.concat(daily,ignore_index=True).to_csv(OUT/'rt25_crossH_daily.csv',index=False);(OUT/'metadata.json').write_text(json.dumps({'rank_tilt':'75% N10 + 25% N5 independent capital sleeves','structures':STRUCT,'selection':'training metrics are reported; pseudo validation never used to form phase ids','phase_ids':'evenly spaced predeclared structural vintages, not optimized','market_factor':mc,'universe_audit':ua},indent=2,default=str));print(d.to_string(index=False),flush=True)
if __name__=='__main__':main()
