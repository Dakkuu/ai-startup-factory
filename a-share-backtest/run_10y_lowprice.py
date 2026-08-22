from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
START=pd.Timestamp('2016-07-29'); END=pd.Timestamp('2026-07-29'); WARM=pd.Timestamp('2015-01-01')
OUT=Path('results_lowprice'); OUT.mkdir(exist_ok=True)

def active(mm,ds):
    out=np.zeros(len(ds),dtype=bool); valid=~pd.isna(ds)
    for r in mm.itertuples(index=False): out |= valid&(ds>=r.start)&(ds<=r.end)
    return out

def build(cal,members):
    tc=cal[(cal>=START)&(cal<=END)]; sd=pd.DatetimeIndex(tc[::5]); at=cal[cal<=END]
    ed=[]
    for s in sd:
        k=at.searchsorted(s,side='right'); ed.append(at[k] if k<len(at) else pd.NaT)
    ed=pd.DatetimeIndex(ed); frames=[]; codes=sorted(members.code.unique())
    for i,c in enumerate(codes,1):
        mm=members[members.code==c]; cols={}
        for f in ['open','high','low','close','volume','factor']:
            x=base.qb.read_bin(c,f,cal)
            if not x.empty: cols[f]=x
        if not all(f in cols for f in ['open','high','low','close','volume','factor']): continue
        z=pd.concat(cols,axis=1).loc[WARM:END].copy()
        if z.empty: continue
        z['factor']=z.factor.replace(0,np.nan)
        raw_close=z.close/z.factor
        cnt=z.close.notna().rolling(120).sum(); liq20=(z.close.abs()*z.volume.abs()).rolling(20,min_periods=16).mean()
        sig=pd.DataFrame({'cnt':cnt.reindex(sd).to_numpy(),'liq20':liq20.reindex(sd).to_numpy(),'raw_price':raw_close.reindex(sd).to_numpy()})
        ex=z.reindex(ed).reset_index(drop=True)
        ok=active(mm,sd)&active(mm,ed)&(~pd.isna(ed))&np.asarray(sig.cnt>=120)
        ok &= np.isfinite(sig[['liq20','raw_price']].to_numpy()).all(axis=1)&(sig.raw_price.to_numpy()>0)
        ok &= np.isfinite(ex[['open','high','low','volume','factor']].to_numpy()).all(axis=1)&(ex.factor.to_numpy()>0)
        if not ok.any(): continue
        ix=np.flatnonzero(ok)
        frames.append(pd.DataFrame({'signal_date':sd[ix],'trade_date':ed[ix],'code':c,
          'liq20':sig.liq20.to_numpy()[ix].astype(float),'raw_price':sig.raw_price.to_numpy()[ix].astype(float),
          'exec_open':ex.open.to_numpy()[ix].astype(float),'exec_high':ex.high.to_numpy()[ix].astype(float),
          'exec_low':ex.low.to_numpy()[ix].astype(float),'exec_volume':ex.volume.to_numpy()[ix].astype(float),
          'exec_factor':ex.factor.to_numpy()[ix].astype(float)}))
        if i%1000==0: print('histories',i,'/',len(codes),flush=True)
    p=pd.concat(frames,ignore_index=True)
    if not (pd.to_datetime(p.signal_date)<pd.to_datetime(p.trade_date)).all(): raise RuntimeError('timing')
    p['liq_pct']=p.groupby('signal_date').liq20.rank(pct=True,ascending=False,method='average')
    elig=p.liq_pct<=.80
    p['ivol60_pct']=np.nan
    p.loc[elig,'ivol60_pct']=p.loc[elig].groupby('signal_date').raw_price.rank(pct=True,ascending=True,method='average')
    gs=p[elig].groupby('signal_date').size(); print('PANEL',p.shape,'dates',p.signal_date.nunique(),'median eligible',gs.median(),flush=True)
    if p.signal_date.nunique()<480 or gs.median()<1500: raise RuntimeError('bad universe')
    return p

def run(p,cal,members,bm,cost=1.0,daily=True):
    e,t,tm,to=sim.simulate(p,'ivol',cal,members,cost,daily_mtm=daily)
    s=sim.perf(e,t,to,bm if daily else None); s['cost_mult']=cost
    s['train_2016_2021']=sim.period_return(e,'2016-07-29','2021-12-31'); s['sealed_2022_2026']=sim.period_return(e,'2022-01-01','2026-07-29')
    return s,e,t,tm

def rerank_with_floor(p,floor):
    q=p.copy(); q['ivol60_pct']=np.nan; m=(q.liq_pct<=.80)&(q.raw_price>=floor)
    q.loc[m,'ivol60_pct']=q.loc[m].groupby('signal_date').raw_price.rank(pct=True,ascending=True,method='average'); return q

def main():
    base.START=START;base.WARM=WARM;base.END=END;base.OUT=OUT;sim.START=START;sim.WARM=WARM;sim.END=END
    cal,members,ua=base.load_base(); p=build(cal,members)
    bm=base.qb.read_bin('SH000300','close',cal).loc[START:END].dropna(); br=float(bm.iloc[-1]/bm.iloc[0]-1)
    s,e,t,tm=run(p,cal,members,bm); s['benchmark_return']=br; s['excess']=s['total_return']-br
    stress=[]
    for c in [2.,4.]: z,_,_,_=run(p,cal,members,bm,c); stress.append(z)
    floors=[]
    for f in [1.,2.,3.,5.]:
        q=rerank_with_floor(p,f); z,_,_,_=run(q,cal,members,bm,1.,False); z['min_raw_price']=f; floors.append(z)
    r=e.equity.pct_change().dropna(); x=r.copy(); x.loc[x.nlargest(5).index]=0
    rob={'base_return':s['total_return'],'without_best5_days':float((1+x).prod()-1),
         'completed_pnl':float(t.net_pnl.sum()),'pnl_without_best5_trades':float(t.net_pnl.sum()-t.nlargest(min(5,len(t)),'net_pnl').net_pnl.sum())}
    ar=sim.annual_returns(e); bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum())
    audit={**ua,'signal_dates':p.signal_date.nunique(),'timing_violations':bad,
           'factor':'raw nominal close = qlib adjusted close / factor; lower is better',
           'portfolio':'top80pct liquidity; 30 equal sleeves; enter lowest10pct; retain through lowest30pct; 5d rebalance; next open',
           'selection':'hypothesis fixed before 2016-2026 run; price floors are robustness diagnostics only'}
    if bad: raise RuntimeError('timing')
    pd.DataFrame([s]).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(stress).to_csv(OUT/'cost.csv',index=False)
    pd.DataFrame(floors).to_csv(OUT/'floor_robustness.csv',index=False);pd.DataFrame([rob]).to_csv(OUT/'robust.csv',index=False);ar.to_csv(OUT/'annual.csv',index=False);pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    print('=== SUMMARY ===');print(pd.DataFrame([s]).to_string(index=False),flush=True)
    print('=== COST ===');print(pd.DataFrame(stress).to_string(index=False),flush=True)
    print('=== PRICE FLOORS ===');print(pd.DataFrame(floors).to_string(index=False),flush=True)
    print('=== ROBUST ===');print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== ANNUAL ===');print(ar.to_string(index=False),flush=True)
if __name__=='__main__': main()
