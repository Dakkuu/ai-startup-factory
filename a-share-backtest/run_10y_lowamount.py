from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
START=pd.Timestamp('2016-07-29');END=pd.Timestamp('2026-07-29');WARM=pd.Timestamp('2015-01-01');OUT=Path('results_lowamount');OUT.mkdir(exist_ok=True)

def active(mm,ds):
 o=np.zeros(len(ds),bool);v=~pd.isna(ds)
 for r in mm.itertuples(index=False):o|=v&(ds>=r.start)&(ds<=r.end)
 return o

def build(cal,members):
 tc=cal[(cal>=START)&(cal<=END)];sd=pd.DatetimeIndex(tc[::5]);at=cal[cal<=END];ed=[]
 for s in sd:
  k=at.searchsorted(s,side='right');ed.append(at[k] if k<len(at) else pd.NaT)
 ed=pd.DatetimeIndex(ed);fs=[];codes=sorted(members.code.unique())
 for i,c in enumerate(codes,1):
  mm=members[members.code==c];cols={}
  for f in ['open','high','low','close','volume','factor']:
   x=base.qb.read_bin(c,f,cal)
   if not x.empty:cols[f]=x
  if not all(f in cols for f in ['open','high','low','close','volume']):continue
  z=pd.concat(cols,axis=1).loc[WARM:END].copy()
  if z.empty:continue
  if 'factor' not in z:z['factor']=1.;z['factor']=z.factor.replace(0,np.nan).fillna(1.)
  cnt=z.close.notna().rolling(120).sum();amt=(z.close.abs()*z.volume.abs()).rolling(20,min_periods=16).mean();sig=pd.DataFrame({'cnt':cnt.reindex(sd).to_numpy(),'amt20':amt.reindex(sd).to_numpy()});ex=z.reindex(ed).reset_index(drop=True)
  ok=active(mm,sd)&active(mm,ed)&(~pd.isna(ed))&np.asarray(sig.cnt>=120)&np.isfinite(sig[['amt20']].to_numpy()).all(1)&np.isfinite(ex[['open','high','low','volume']].to_numpy()).all(1)
  if not ok.any():continue
  ix=np.flatnonzero(ok);fs.append(pd.DataFrame({'signal_date':sd[ix],'trade_date':ed[ix],'code':c,'amt20':sig.amt20.to_numpy()[ix].astype(float),'exec_open':ex.open.to_numpy()[ix].astype(float),'exec_high':ex.high.to_numpy()[ix].astype(float),'exec_low':ex.low.to_numpy()[ix].astype(float),'exec_volume':ex.volume.to_numpy()[ix].astype(float),'exec_factor':ex.factor.to_numpy()[ix].astype(float)}))
  if i%1000==0:print('histories',i,'/',len(codes),flush=True)
 p=pd.concat(fs,ignore_index=True);p['full_amt_pct']=p.groupby('signal_date').amt20.rank(pct=True,ascending=True,method='average')
 # Exclude the bottom 10% by traded amount; rank the remaining stocks from least to most traded.
 p['ivol60_pct']=np.nan;m=p.full_amt_pct>=.10;p.loc[m,'ivol60_pct']=p.loc[m].groupby('signal_date').amt20.rank(pct=True,ascending=True,method='average')
 gs=p[m].groupby('signal_date').size();print('PANEL',p.shape,'dates',p.signal_date.nunique(),'eligible median',gs.median(),flush=True)
 if p.signal_date.nunique()<480 or gs.median()<2000:raise RuntimeError('bad universe')
 return p

def run(p,cal,members,bm,cost=1.,daily=True):
 e,t,tm,to=sim.simulate(p,'ivol',cal,members,cost,daily_mtm=daily);s=sim.perf(e,t,to,bm if daily else None);s['cost_mult']=cost;s['train']=sim.period_return(e,'2016-07-29','2021-12-31');s['sealed']=sim.period_return(e,'2022-01-01','2026-07-29');return s,e,t,tm

def main():
 base.START=START;base.WARM=WARM;base.END=END;base.OUT=OUT;sim.START=START;sim.WARM=WARM;sim.END=END
 cal,members,ua=base.load_base();p=build(cal,members);bm=base.qb.read_bin('SH000300','close',cal).loc[START:END].dropna();br=bm.iloc[-1]/bm.iloc[0]-1
 s,e,t,tm=run(p,cal,members,bm);s['benchmark_return']=br;s['excess']=s['total_return']-br
 stress=[]
 for c in [2.,4.]:q,_,_,_=run(p,cal,members,bm,c);stress.append(q)
 # neighboring universe floors are diagnostics only, not used to select the core 10% floor
 floors=[];orig=p.ivol60_pct.copy()
 for floor in [.05,.10,.20]:
  q=p.copy();q['ivol60_pct']=np.nan;m=q.full_amt_pct>=floor;q.loc[m,'ivol60_pct']=q.loc[m].groupby('signal_date').amt20.rank(pct=True,ascending=True,method='average');z,_,_,_=run(q,cal,members,bm,1.,False);z['excluded_bottom_amt_pct']=floor;floors.append(z)
 r=e.equity.pct_change().dropna();x=r.copy();x.loc[x.nlargest(5).index]=0;pnl=t.net_pnl.sum();b5=t.nlargest(min(5,len(t)),'net_pnl').net_pnl.sum();rob={'base_return':s['total_return'],'without_best5_days':(1+x).prod()-1,'pnl_without_best5_trades':pnl-b5}
 ar=sim.annual_returns(e);bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum());audit={**ua,'signal_dates':p.signal_date.nunique(),'timing_violations':bad,'factor':'20d mean close*volume; exclude lowest 10% amount; buy lowest remaining decile','selection':'fully pre-specified; floors 5/10/20 diagnostics only'}
 if bad:raise RuntimeError('timing')
 pd.DataFrame([s]).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(stress).to_csv(OUT/'cost.csv',index=False);pd.DataFrame(floors).to_csv(OUT/'floor_grid.csv',index=False);pd.DataFrame([rob]).to_csv(OUT/'robust.csv',index=False);ar.to_csv(OUT/'annual.csv',index=False);pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
 print('=== SUMMARY ===');print(pd.DataFrame([s]).to_string(index=False),flush=True);print('=== COST ===');print(pd.DataFrame(stress).to_string(index=False),flush=True);print('=== FLOORS ===');print(pd.DataFrame(floors).to_string(index=False),flush=True);print('=== ROBUST ===');print(pd.DataFrame([rob]).to_string(index=False),flush=True);print('=== ANNUAL ===');print(ar.to_string(index=False),flush=True)
if __name__=='__main__':main()
