from __future__ import annotations
from pathlib import Path
import glob,json
import numpy as np
import pandas as pd
import run_10y_era_backtest as base
import run_10y_baseline_maxopt_v3 as mo
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_geff_fundamental_integrated_v3 as iv3
import run_geff_fundamental_structure_opt_v1 as so
import run_10y_hard_executor_v3 as hv3
hv3.patch()
OUT=Path('results_geff_fundamental_distress_filter_v1');OUT.mkdir(exist_ok=True)
HORIZONS=(60,75,90);N=10;ENTRY=.10;KEEP=.30
FILTERS={'none':(None,False),'px2':(2.0,False),'px3':(3.0,False),'px5':(5.0,False),'stproxy60':(None,True),'px3_stproxy60':(3.0,True)}
def locate(p):
 h=glob.glob(p,recursive=True)
 if not h:raise FileNotFoundError(p)
 return h[0]
def is_main(code):
 c=str(code).upper();return not (c.startswith('SZ300') or c.startswith('SZ301') or c.startswith('SH688') or c.startswith('BJ') or c.startswith('8') or c.startswith('4'))
def attach_distress(p,cal):
 sigdates=pd.DatetimeIndex(sorted(pd.to_datetime(p.signal_date.unique()))); rec=[]
 for i,code in enumerate(sorted(p.code.unique()),1):
  close=base.qb.read_bin(code,'close',cal); factor=base.qb.read_bin(code,'factor',cal); op=base.qb.read_bin(code,'open',cal); hi=base.qb.read_bin(code,'high',cal); lo=base.qb.read_bin(code,'low',cal)
  if not len(close):continue
  z=pd.concat({'close':close,'factor':factor,'open':op,'high':hi,'low':lo},axis=1).sort_index(); f=z.factor.replace(0,np.nan); rawc=z.close/f; rawo=z.open/f; rawh=z.high/f; rawl=z.low/f; prev=rawc.shift(1)
  one=(rawh-rawl).abs()<=1e-10
  ret=rawo/prev-1
  five=one & ret.abs().between(.045,.055) if is_main(code) else pd.Series(False,index=z.index)
  # Signal close may use today's fully observed one-price behavior; no future data.
  stp=five.astype(int).rolling(60,min_periods=1).max().astype(bool)
  rr=pd.DataFrame({'signal_date':sigdates,'code':code,'raw_price':rawc.reindex(sigdates).to_numpy(),'stproxy60':stp.reindex(sigdates).fillna(False).to_numpy(bool)})
  rec.append(rr)
  if i%500==0:print('DISTRESS',i,flush=True)
 return p.merge(pd.concat(rec,ignore_index=True),on=['signal_date','code'],how='left',validate='one_to_one')
def subset(q,h,ph,px,st):
 ds=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())));step=max(1,round(h/5));chosen=set(ds[ph::step]);cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy','raw_price','stproxy60'] if c in q.columns];z=q[q.signal_date.isin(chosen)][cols].copy()
 # Distress filter only blocks NEW BUY eligibility by setting rank unusable; incumbents outside entry universe will naturally fail keep and be sold at review. This is a deliberate conservative filter.
 bad=pd.Series(False,index=z.index)
 if px is not None:bad|=pd.to_numeric(z.raw_price,errors='coerce')<float(px)
 if st:bad|=z.stproxy60.fillna(False).astype(bool)
 z.loc[bad,'rank_test']=np.nan
 z['ivol60_pct']=z.rank_test;return z.drop(columns='rank_test')
def main():
 p,cal,members,ua,mc,bm=mo.build_panel(OUT,need_fwd=False);p=strict.attach_gap_flags(p,cal,'board');va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False);sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False);iv3.verify_attach(p,va,'value');iv3.verify_attach(p,sa,'3stmt');p2,_=iv3.fund_ranks(p,va,sa);q=iv3.build_candidates(p2)['mom_cfo10_qv10'];q=attach_distress(q,cal);rows=[]
 for name,(px,stf) in FILTERS.items():
  for h in HORIZONS:
   for ph in range(max(1,round(h/5))):
    stt,eq,tr,tm=ma.run_panel(subset(q,h,ph,px,stf),cal,members,bm,n=N,entry=ENTRY,keep=KEEP,cost=1.0);trn=so.slice_stats(eq,so.START,so.TRAIN_END);a=so.slice_stats(eq,so.START,so.HALF1_END);b=so.slice_stats(eq,so.HALF2_START,so.TRAIN_END);ps=so.slice_stats(eq,so.PSEUDO,so.END);stt.update(filter=name,price_floor=px,stproxy=stf,H=h,phase=ph,train_cagr=trn['cagr'],train_mdd=trn['mdd'],half1_cagr=a['cagr'],half2_cagr=b['cagr'],pseudo_cagr=ps['cagr'],pseudo_mdd=ps['mdd']);rows.append(stt)
 d=pd.DataFrame(rows);s=so.aggregate(d,['filter','price_floor','stproxy']).sort_values(['train_calmar_robust','train_maximin'],ascending=False);d.to_csv(OUT/'all_phase.csv',index=False);s.to_csv(OUT/'summary.csv',index=False);audit=q.groupby('signal_date').agg(rows=('code','size'),stproxy=('stproxy60','sum'),px2=('raw_price',lambda x:(x<2).sum()),px3=('raw_price',lambda x:(x<3).sum()),px5=('raw_price',lambda x:(x<5).sum())).reset_index();audit.to_csv(OUT/'distress_coverage.csv',index=False);(OUT/'metadata.json').write_text(json.dumps({'filters':FILTERS,'selection':'2016-2021 train metrics only; 2022-2026 validation','stproxy_definition':'main-board one-price raw open move absolute 4.5%-5.5% observed within trailing 60 sessions, causal proxy not exact ST history','market_factor':mc,'universe_audit':ua},indent=2,default=str));print(s.to_string(index=False),flush=True)
if __name__=='__main__':main()
