from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd,math
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
START=pd.Timestamp('2016-07-29');END=pd.Timestamp('2026-07-29');WARM=pd.Timestamp('2014-01-01');OUT=Path('results_trading2');OUT.mkdir(exist_ok=True);FS=('relamt','amihud')
def active(mm,ds):
 o=np.zeros(len(ds),bool);v=~pd.isna(ds)
 for r in mm.itertuples(index=False):o|=v&(ds>=r.start)&(ds<=r.end)
 return o
def build(cal,members):
 tc=cal[(cal>=START)&(cal<=END)];sd=pd.DatetimeIndex(tc[::5]);at=cal[cal<=END];ed=[]
 for s in sd:
  k=at.searchsorted(s,side='right');ed.append(at[k] if k<len(at) else pd.NaT)
 ed=pd.DatetimeIndex(ed);frames=[];codes=sorted(members.code.unique())
 for i,c in enumerate(codes,1):
  mm=members[members.code==c];cols={}
  for f in ['open','high','low','close','volume','factor']:
   x=base.qb.read_bin(c,f,cal)
   if not x.empty:cols[f]=x
  if not all(f in cols for f in ['open','high','low','close','volume']):continue
  z=pd.concat(cols,axis=1).loc[WARM:END].copy()
  if z.empty:continue
  if 'factor' not in z:z['factor']=1.
  z['factor']=z.factor.replace(0,np.nan).fillna(1.);r=z.close.pct_change(fill_method=None);amt=z.close.abs()*z.volume.abs();a20=amt.rolling(20,min_periods=16).mean();a252=amt.rolling(252,min_periods=200).mean();cnt=z.close.notna().rolling(120).sum();fac={'relamt':a20/a252,'amihud':(r.abs()/amt.replace(0,np.nan)).rolling(20,min_periods=16).mean()}
  sig={'cnt':cnt.reindex(sd).to_numpy(),'liq20':a20.reindex(sd).to_numpy()}
  for k,x in fac.items():sig[k]=x.reindex(sd).to_numpy()
  sig=pd.DataFrame(sig);ex=z.reindex(ed).reset_index(drop=True);ok=active(mm,sd)&active(mm,ed)&(~pd.isna(ed))&np.asarray(sig.cnt>=120)&np.isfinite(sig[['liq20']].to_numpy()).all(1)&np.isfinite(ex[['open','high','low','volume']].to_numpy()).all(1)
  if not ok.any():continue
  ix=np.flatnonzero(ok);rec={'signal_date':sd[ix],'trade_date':ed[ix],'code':c,'liq20':sig.liq20.to_numpy()[ix].astype(float),'exec_open':ex.open.to_numpy()[ix].astype(float),'exec_high':ex.high.to_numpy()[ix].astype(float),'exec_low':ex.low.to_numpy()[ix].astype(float),'exec_volume':ex.volume.to_numpy()[ix].astype(float),'exec_factor':ex.factor.to_numpy()[ix].astype(float)}
  for f in FS:rec[f]=sig[f].to_numpy()[ix].astype(float)
  frames.append(pd.DataFrame(rec));
  if i%1000==0:print('hist',i,'/',len(codes),flush=True)
 p=pd.concat(frames,ignore_index=True);p['liq_pct']=p.groupby('signal_date').liq20.rank(pct=True,ascending=False,method='average');liq=p.liq_pct<=.80
 p['relamt_pct']=np.nan;m=liq&p.relamt.notna();p.loc[m,'relamt_pct']=p.loc[m].groupby('signal_date').relamt.rank(pct=True,ascending=True,method='average')
 p['amihud_pct']=np.nan;m=liq&p.amihud.notna();p.loc[m,'amihud_pct']=p.loc[m].groupby('signal_date').amihud.rank(pct=True,ascending=False,method='average')
 gs=p[liq].groupby('signal_date').size();print('PANEL',p.shape,p.signal_date.nunique(),gs.median(),flush=True);return p
def run(p,f,cal,members,bm,cost=1,daily=True):
 q=p.copy();q['ivol60_pct']=q[f+'_pct'];e,t,tm,to=sim.simulate(q,'ivol',cal,members,cost,daily_mtm=daily);s=sim.perf(e,t,to,bm if daily else None);s.update(factor=f,cost_mult=cost,train=sim.period_return(e,'2016-07-29','2021-12-31'),sealed=sim.period_return(e,'2022-01-01','2026-07-29'));return s,e,t,tm
def main():
 base.START=START;base.WARM=WARM;base.END=END;base.OUT=OUT;sim.START=START;sim.WARM=WARM;sim.END=END;cal,members,ua=base.load_base();p=build(cal,members);bm=base.qb.read_bin('SH000300','close',cal).loc[START:END].dropna();br=bm.iloc[-1]/bm.iloc[0]-1;rows=[];store={}
 for f in FS:
  print('RUN',f,flush=True);s,e,t,tm=run(p,f,cal,members,bm);s['benchmark']=br;s['excess']=s['total_return']-br;rows.append(s);store[f]=(e,t,tm)
 sm=pd.DataFrame(rows);winner=sm.sort_values('train',ascending=False).iloc[0].factor;stress=[]
 for c in [2.,4.]:s,_,_,_=run(p,winner,cal,members,bm,c);stress.append(s)
 rob=[]
 for f,(e,t,tm) in store.items():
  r=e.equity.pct_change().dropna();x=r.copy();x.loc[x.nlargest(5).index]=0;z={'factor':f,'without_best5_days':(1+x).prod()-1,'pnl_without_best5_trades':t.net_pnl.sum()-t.nlargest(min(5,len(t)),'net_pnl').net_pnl.sum()};rob.append(z)
 bad=sum(int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) for _,_,tm in store.values());audit={**ua,'signal_dates':p.signal_date.nunique(),'winner_train_only':winner,'timing_violations':bad,'factors':'relamt20/252 low; Amihud20 high; top80% liquidity eligibility'}
 if bad:raise RuntimeError('timing')
 sm.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(stress).to_csv(OUT/'cost.csv',index=False);pd.DataFrame(rob).to_csv(OUT/'robust.csv',index=False);pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
 print('=== SUMMARY ===');print(sm.to_string(index=False),flush=True);print('WINNER',winner,flush=True);print('=== COST ===');print(pd.DataFrame(stress).to_string(index=False),flush=True);print('=== ROBUST ===');print(pd.DataFrame(rob).to_string(index=False),flush=True)
if __name__=='__main__':main()
