from __future__ import annotations
from pathlib import Path
import glob,json
import numpy as np
import pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_geff_fundamental_integrated_v3 as iv3
import run_geff_fundamental_ranktilt_v1 as rt
import run_10y_hard_executor_v3 as hv3
hv3.patch()
OUT=Path('results_geff_fundamental_ranktilt_robust_v2'); OUT.mkdir(exist_ok=True)
HORIZONS=(45,60,75,90,105); ENTRY=.10; KEEP=.30
MIXES={'N10':[(10,1.0)],'N5_25_N10_75':[(5,.25),(10,.75)],'N5_25_N12_75':[(5,.25),(12,.75)],'N5_50_N10_50':[(5,.50),(10,.50)]}
START=pd.Timestamp('2016-08-02'); TRAIN_END=pd.Timestamp('2021-12-31'); PSEUDO=pd.Timestamp('2022-01-01'); END=pd.Timestamp('2026-07-29'); HALF1_END=pd.Timestamp('2019-12-31'); HALF2_START=pd.Timestamp('2020-01-01')
def locate(p):
 h=glob.glob(p,recursive=True)
 if not h:raise FileNotFoundError(p)
 return h[0]
def build():
 p,cal,members,ua,mc,bm=mo.build_panel(OUT,need_fwd=False);p=strict.attach_gap_flags(p,cal,'board');va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False);sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False);iv3.verify_attach(p,va,'value');iv3.verify_attach(p,sa,'3stmt');p2,_=iv3.fund_ranks(p,va,sa);q=iv3.build_candidates(p2)['mom_cfo10_qv10'];return q,cal,members,ua,mc,bm
def subset(q,h,ph):
 ds=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())));step=max(1,round(h/5));chosen=set(ds[ph::step]);cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns];z=q[q.signal_date.isin(chosen)][cols].copy();z['ivol60_pct']=z.rank_test;return z.drop(columns='rank_test')
def slice_stats(eq,a,b):
 z=eq.copy();z['trade_date']=pd.to_datetime(z.trade_date);z=z[(z.trade_date>=pd.Timestamp(a))&(z.trade_date<=pd.Timestamp(b))]
 if len(z)<20:return {'cagr':np.nan,'mdd':np.nan}
 s=z.set_index('trade_date').equity.astype(float).sort_index();s=s[~s.index.duplicated(keep='last')];days=max(1,(s.index[-1]-s.index[0]).days);return {'cagr':float((s.iloc[-1]/s.iloc[0])**(365.25/days)-1),'mdd':float((s/s.cummax()-1).min())}
def perf(eq):
 s=eq.set_index('trade_date').equity.astype(float).sort_index();s=s[~s.index.duplicated(keep='last')];r=s.pct_change().dropna();days=max(1,(s.index[-1]-s.index[0]).days);return {'cagr':float((s.iloc[-1]/s.iloc[0])**(365.25/days)-1),'max_drawdown':float((s/s.cummax()-1).min()),'sharpe':float(r.mean()/r.std(ddof=1)*np.sqrt(252)) if len(r)>2 and r.std(ddof=1)>0 else np.nan}
def mixeq(eqs,weights):return rt.weighted_mix(eqs,weights)
def run_cost(q,cal,members,bm,cost):
 rows=[];cache={}
 for h in HORIZONS:
  for ph in range(max(1,round(h/5))):
   z=subset(q,h,ph)
   for n in (5,10,12):
    st,eq,tr,tm=ma.run_panel(z,cal,members,bm,n=n,entry=ENTRY,keep=KEEP,cost=cost);cache[(n,h,ph)]=eq
   for name,spec in MIXES.items():
    eq=mixeq([cache[(n,h,ph)] for n,w in spec],[w for n,w in spec]);st=perf(eq);trn=slice_stats(eq,START,TRAIN_END);a=slice_stats(eq,START,HALF1_END);b=slice_stats(eq,HALF2_START,TRAIN_END);ps=slice_stats(eq,PSEUDO,END);st.update(mix=name,H=h,phase=ph,cost_mult=cost,train_cagr=trn['cagr'],train_mdd=trn['mdd'],half1_cagr=a['cagr'],half2_cagr=b['cagr'],pseudo_cagr=ps['cagr'],pseudo_mdd=ps['mdd']);rows.append(st)
 return pd.DataFrame(rows)
def summary(d):
 rows=[]
 for (name,cost),g in d.groupby(['mix','cost_mult']):
  r={'mix':name,'cost_mult':cost,'runs':len(g),'full_cagr_median':g.cagr.median(),'full_cagr_p25':g.cagr.quantile(.25),'full_cagr_min':g.cagr.min(),'mdd_median':g.max_drawdown.median(),'mdd_worst':g.max_drawdown.min(),'sharpe_median':g.sharpe.median(),'train_cagr_median':g.train_cagr.median(),'train_cagr_p25':g.train_cagr.quantile(.25),'train_mdd_median':g.train_mdd.median(),'train_maximin':min(g.half1_cagr.median(),g.half2_cagr.median()),'train_calmar_robust':g.train_cagr.quantile(.25)/abs(g.train_mdd.median()),'pseudo_cagr_median':g.pseudo_cagr.median(),'pseudo_cagr_p25':g.pseudo_cagr.quantile(.25),'pseudo_mdd_median':g.pseudo_mdd.median()}
  for h in HORIZONS:r[f'h{h}_median']=g[g.H==h].cagr.median()
  rows.append(r)
 return pd.DataFrame(rows).sort_values(['cost_mult','train_calmar_robust','train_maximin'],ascending=[True,False,False])
def main():
 q,cal,members,ua,mc,bm=build();d1=run_cost(q,cal,members,bm,1.0);d2=run_cost(q,cal,members,bm,2.0);d=pd.concat([d1,d2],ignore_index=True);s=summary(d);d.to_csv(OUT/'all_phase.csv',index=False);s.to_csv(OUT/'summary.csv',index=False);(OUT/'metadata.json').write_text(json.dumps({'selection':'2016-2021 robust metrics; 2022-2026 validation only','horizons':HORIZONS,'mixes':MIXES,'market_factor':mc,'universe_audit':ua},indent=2,default=str));print(s.to_string(index=False),flush=True)
if __name__=='__main__':main()
