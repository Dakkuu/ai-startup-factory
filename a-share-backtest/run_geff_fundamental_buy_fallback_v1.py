from __future__ import annotations
from pathlib import Path
import glob,json
import numpy as np
import pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_baseline_maxopt_v3 as mo
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_geff_fundamental_integrated_v3 as iv3
import run_geff_fundamental_structure_opt_v1 as so
import run_10y_skewfilter_hard as old
import run_10y_hard_executor_v2 as hv2
import run_10y_hard_executor_v3 as hv3
hv3.patch()
OUT=Path('results_geff_fundamental_buy_fallback_v1');OUT.mkdir(exist_ok=True)
HORIZONS=(60,75,90);N=10;ENTRY=.10;KEEP=.30

def locate(p):
 h=glob.glob(p,recursive=True)
 if not h:raise FileNotFoundError(p)
 return h[0]
def build():
 p,cal,members,ua,mc,bm=mo.build_panel(OUT,need_fwd=False);p=strict.attach_gap_flags(p,cal,'board');va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False);sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False);iv3.verify_attach(p,va,'value');iv3.verify_attach(p,sa,'3stmt');p2,_=iv3.fund_ranks(p,va,sa);q=iv3.build_candidates(p2)['mom_cfo10_qv10'];return q,cal,members,ua,mc,bm
def subset(q,h,ph):
 ds=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())));step=max(1,round(h/5));chosen=set(ds[ph::step]);cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns];z=q[q.signal_date.isin(chosen)][cols].copy();z['ivol60_pct']=z.rank_test;return z.drop(columns='rank_test')
def ranked_lists(g,current):
 x=g[np.isfinite(g.ivol60_pct)].sort_values(['ivol60_pct','liq20','code'],ascending=[True,False,True]).copy()
 keep=[c for c in x.loc[x.ivol60_pct<=KEEP,'code'].tolist() if c in current][:N]
 entrants=[c for c in x.loc[x.ivol60_pct<=ENTRY,'code'].tolist() if c not in current and c not in keep]
 return keep,entrants

def hard_fallback(panel,cal,members,cost_mult=1.0):
 by={d:g.set_index('code',drop=False) for d,g in panel.groupby('signal_date')};dates=sorted(by);cash=sim.INITIAL_CASH;pos={};equity=[];trades=[];timing=[];turnover=0.;member_end=members.groupby('code').end.max().to_dict();close_cache={};blocked_buys=0;fallback_buys=0
 def close_series(c):
  if c not in close_cache:close_cache[c]=base.qb.read_bin(c,'close',cal).loc[sim.START:sim.END]
  return close_cache[c]
 trade_cal=cal[(cal>=sim.START)&(cal<=sim.END)];slip=sim.SLIPPAGE*cost_mult
 for j,d in enumerate(dates):
  g=by[d];td=pd.Timestamp(g.trade_date.iloc[0]);keep,entrants=ranked_lists(g.reset_index(drop=True),set(pos));keep_set=set(keep)
  for c,pp in list(pos.items()):
   if c in g.index and np.isfinite(g.loc[c].exec_open):pp.last_price=float(g.loc[c].exec_open)
   elif pd.Timestamp(member_end.get(c,sim.END))<td:
    oldp=pos.pop(c);trades.append({'code':c,'entry_date':oldp.entry_date,'exit_date':td,'net_pnl':-oldp.entry_cost,'net_return':-1.,'exit_reason':'membership_end_writeoff'})
  nav_open=cash+sum(pp.units*pp.last_price for pp in pos.values())
  for c in sorted(list(pos)):
   if c in keep_set or c not in g.index:continue
   r=g.loc[c]
   if not hv3.row_sell_allowed(r):continue
   locked=abs(float(r.exec_high)-float(r.exec_low))<1e-12 and abs(float(r.exec_open)-float(r.exec_high))<1e-12
   if locked:continue
   px=float(r.exec_open)*(1-slip);gross=pos[c].units*px;fee=sim.fee(gross,'sell',td,cost_mult);oldp=pos.pop(c);cash+=gross-fee;turnover+=gross;trades.append({'code':c,'entry_date':oldp.entry_date,'exit_date':td,'net_pnl':gross-fee-oldp.entry_cost,'net_return':(gross-fee)/oldp.entry_cost-1,'exit_reason':'rank_exit'});timing.append({'signal_date':pd.Timestamp(d),'trade_date':td,'side':'sell','code':c})
  per=nav_open*.99/N;attempt=0
  for c in entrants:
   if len(pos)>=N:break
   attempt+=1
   if c in pos or c not in g.index:continue
   r=g.loc[c]
   if not hv3.row_buy_allowed(r):blocked_buys+=1;continue
   locked=abs(float(r.exec_high)-float(r.exec_low))<1e-12 and abs(float(r.exec_open)-float(r.exec_high))<1e-12
   if locked or not np.isfinite(r.exec_factor) or float(r.exec_factor)<=0:blocked_buys+=1;continue
   factor=float(r.exec_factor);adjpx=float(r.exec_open)*(1+slip);rawpx=adjpx/factor
   if not np.isfinite(rawpx) or rawpx<=0:blocked_buys+=1;continue
   maxraw=hv2.max_participation_shares(float(r.exec_volume),factor,sim.VOLUME_PARTICIPATION);shares=int(min(per,cash*.98)//(rawpx*100))*100;shares=min(shares,maxraw)
   if shares<=0:blocked_buys+=1;continue
   units=shares/factor;gross=units*adjpx;fee=sim.fee(gross,'buy',td,cost_mult);total=gross+fee
   if total>cash:continue
   cash-=total;pos[c]=sim.Pos(units,total,td,float(r.exec_open));turnover+=gross;fallback_buys+=int(attempt>(N-len(keep)));timing.append({'signal_date':pd.Timestamp(d),'trade_date':td,'side':'buy','code':c,'candidate_attempt':attempt})
  next_td=pd.Timestamp(by[dates[j+1]].trade_date.iloc[0]) if j+1<len(dates) else sim.END+pd.Timedelta(days=1);seg=trade_cal[(trade_cal>=td)&(trade_cal<next_td)]
  for day in seg:
   for c,pp in pos.items():
    px=close_series(c).get(day,np.nan)
    if np.isfinite(px) and px>0:pp.last_price=float(px)
   nav=cash+sum(pp.units*pp.last_price for pp in pos.values());equity.append({'signal_date':pd.Timestamp(d),'trade_date':pd.Timestamp(day),'equity':nav,'cash':cash,'positions':len(pos)})
 e=pd.DataFrame(equity).drop_duplicates('trade_date',keep='last').sort_values('trade_date');t=pd.DataFrame(trades);tm=pd.DataFrame(timing);return e,t,tm,turnover,blocked_buys,fallback_buys

def run_fallback(z,cal,members,bm,cost=1.0):
 oldstate=(old.N_HOLD,sim.ENTRY_PCT,sim.KEEP_PCT);old.N_HOLD=N;sim.ENTRY_PCT=ENTRY;sim.KEEP_PCT=KEEP
 try:
  eq,tr,tm,to,bb,fb=hard_fallback(z,cal,members,cost);st=sim.perf(eq,tr,to,bm);st.update(blocked_buy_attempts=bb,fallback_buys=fb,positions_median=float(eq.positions.median()),cash_share_median=float((eq.cash/eq.equity).median()));return st,eq,tr,tm
 finally:old.N_HOLD,sim.ENTRY_PCT,sim.KEEP_PCT=oldstate
def main():
 q,cal,members,ua,mc,bm=build();rows=[]
 for mode in ('baseline','fallback'):
  for h in HORIZONS:
   for ph in range(max(1,round(h/5))):
    z=subset(q,h,ph)
    if mode=='baseline':st,eq,tr,tm=ma.run_panel(z,cal,members,bm,n=N,entry=ENTRY,keep=KEEP,cost=1.0);st.update(blocked_buy_attempts=np.nan,fallback_buys=0,positions_median=float(eq.positions.median()),cash_share_median=float((eq.cash/eq.equity).median()))
    else:st,eq,tr,tm=run_fallback(z,cal,members,bm,1.0)
    trn=so.slice_stats(eq,so.START,so.TRAIN_END);a=so.slice_stats(eq,so.START,so.HALF1_END);b=so.slice_stats(eq,so.HALF2_START,so.TRAIN_END);ps=so.slice_stats(eq,so.PSEUDO,so.END);st.update(mode=mode,H=h,phase=ph,train_cagr=trn['cagr'],train_mdd=trn['mdd'],half1_cagr=a['cagr'],half2_cagr=b['cagr'],pseudo_cagr=ps['cagr'],pseudo_mdd=ps['mdd']);rows.append(st)
 d=pd.DataFrame(rows);s=so.aggregate(d,['mode']);extra=d.groupby('mode').agg(positions_median=('positions_median','median'),cash_share_median=('cash_share_median','median'),blocked_buy_attempts=('blocked_buy_attempts','sum'),fallback_buys=('fallback_buys','sum')).reset_index();s=s.merge(extra,on='mode');d.to_csv(OUT/'all_phase.csv',index=False);s.to_csv(OUT/'summary.csv',index=False);(OUT/'metadata.json').write_text(json.dumps({'mechanism':'within same top10% entry pool, continue to next ranked entrant after execution failure; no stale-signal extension beyond entry threshold','selection':'mechanism test; no parameter grid','market_factor':mc,'universe_audit':ua},indent=2,default=str));print(s.to_string(index=False),flush=True)
if __name__=='__main__':main()
