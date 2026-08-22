from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim

START=pd.Timestamp('2016-07-29'); END=pd.Timestamp('2026-07-29'); WARM=pd.Timestamp('2014-01-01')
OUT=Path('results_simple3b'); OUT.mkdir(exist_ok=True)
FACTORS=('lowvol60','mom12_1','high52')

def active_mask(mm,dates):
    out=np.zeros(len(dates),dtype=bool); valid=~pd.isna(dates)
    for r in mm.itertuples(index=False): out |= valid&(dates>=r.start)&(dates<=r.end)
    return out

def build_panel(cal,members):
    tc=cal[(cal>=START)&(cal<=END)]; sd=pd.DatetimeIndex(tc[::5]); at=cal[cal<=END]
    ed=[]
    for s in sd:
        k=at.searchsorted(s,side='right'); ed.append(at[k] if k<len(at) else pd.NaT)
    ed=pd.DatetimeIndex(ed); frames=[]; codes=sorted(members.code.unique())
    for i,code in enumerate(codes,1):
        mm=members[members.code==code]; cols={}
        for f in ['open','high','low','close','volume','factor']:
            x=base.qb.read_bin(code,f,cal)
            if not x.empty: cols[f]=x
        if not all(f in cols for f in ['open','high','low','close','volume']): continue
        z=pd.concat(cols,axis=1).loc[WARM:END].copy()
        if z.empty: continue
        if 'factor' not in z:z['factor']=1.0
        z['factor']=z.factor.replace(0,np.nan).fillna(1.0)
        r=z.close.pct_change(fill_method=None); cnt=z.close.notna().rolling(120).sum(); liq=(z.close.abs()*z.volume.abs()).rolling(20).mean()
        fac={
          'lowvol60':r.rolling(60,min_periods=48).std(),
          'mom12_1':-(z.close.shift(20)/z.close.shift(252)-1.0),
          'high52':1.0-z.close/z.close.rolling(252,min_periods=200).max(),
        }
        sig={'count120':cnt.reindex(sd).to_numpy(),'liq20':liq.reindex(sd).to_numpy()}
        for k,x in fac.items():sig[k]=x.reindex(sd).to_numpy()
        sig=pd.DataFrame(sig); ex=z.reindex(ed).reset_index(drop=True)
        valid=active_mask(mm,sd)&active_mask(mm,ed)&(~pd.isna(ed)); valid &= np.asarray(sig.count120>=120)
        valid &= np.isfinite(sig[['liq20']].to_numpy()).all(axis=1); valid &= np.isfinite(ex[['open','high','low','volume']].to_numpy()).all(axis=1)
        if not valid.any():continue
        ix=np.flatnonzero(valid); rec={'signal_date':sd[ix],'trade_date':ed[ix],'code':code,'liq20':sig.liq20.to_numpy()[ix].astype(float),'exec_open':ex.open.to_numpy()[ix].astype(float),'exec_high':ex.high.to_numpy()[ix].astype(float),'exec_low':ex.low.to_numpy()[ix].astype(float),'exec_volume':ex.volume.to_numpy()[ix].astype(float),'exec_factor':ex.factor.to_numpy()[ix].astype(float)}
        for f in FACTORS:rec[f]=sig[f].to_numpy()[ix].astype(float)
        frames.append(pd.DataFrame(rec))
        if i%1000==0:print('histories',i,'/',len(codes),flush=True)
    p=pd.concat(frames,ignore_index=True)
    if not (pd.to_datetime(p.signal_date)<pd.to_datetime(p.trade_date)).all():raise RuntimeError('timing')
    p['liq_rank_pct']=p.groupby('signal_date').liq20.rank(pct=True,ascending=False,method='average'); liquid=p.liq_rank_pct<=.80
    for f in FACTORS:
        p[f+'_pct']=np.nan; m=liquid&p[f].notna(); p.loc[m,f+'_pct']=p.loc[m].groupby('signal_date')[f].rank(pct=True,ascending=True,method='average')
    gs=p[liquid].groupby('signal_date').size(); print('PANEL',p.shape,'dates',p.signal_date.nunique(),'median',gs.median(),flush=True)
    if p.signal_date.nunique()<480 or gs.median()<1500:raise RuntimeError('bad universe')
    return p

def run(panel,f,cal,members,bm,cost=1.0,daily=True):
    q=panel.copy(); q['ivol60_pct']=q[f+'_pct']; e,t,tm,to=sim.simulate(q,'ivol',cal,members,cost,daily_mtm=daily); s=sim.perf(e,t,to,bm if daily else None); s.update(factor=f,cost_mult=cost,train_2016_2021=sim.period_return(e,'2016-07-29','2021-12-31'),sealed_2022_2026=sim.period_return(e,'2022-01-01','2026-07-29')); return s,e,t,tm

def ic(panel):
    p=panel.sort_values(['code','trade_date']).copy(); p['fwd5']=p.groupby('code').exec_open.shift(-1)/p.exec_open-1; rows=[]
    for d,g in p.groupby('signal_date'):
        z=g.dropna(subset=['fwd5']);
        if len(z)<500:continue
        for f in FACTORS:
            zz=z.dropna(subset=[f]); rows.append({'date':d,'factor':f,'ic':(-zz[f]).corr(zz.fwd5,method='spearman')})
    q=pd.DataFrame(rows); out=[]
    for f,g in q.groupby('factor'):
        x=g.ic.dropna();out.append({'factor':f,'mean_ic':x.mean(),'icir':x.mean()/x.std()*np.sqrt(52),'positive_ic_rate':(x>0).mean(),'n':len(x)})
    return pd.DataFrame(out)

def robust(eq,tr):
    r=eq.equity.pct_change().dropna(); x=r.copy(); x.loc[x.nlargest(5).index]=0; pnl=tr.net_pnl.sum() if len(tr) else 0; b=tr.nlargest(min(5,len(tr)),'net_pnl').net_pnl.sum() if len(tr) else 0
    return {'base':eq.equity.iloc[-1]/1e6-1,'without_best5_days':(1+x).prod()-1,'pnl_without_best5_trades':pnl-b}

def main():
    base.START=START;base.WARM=WARM;base.END=END;base.OUT=OUT;sim.START=START;sim.WARM=WARM;sim.END=END
    cal,members,ua=base.load_base();p=build_panel(cal,members);bm=base.qb.read_bin('SH000300','close',cal).loc[START:END].dropna();br=bm.iloc[-1]/bm.iloc[0]-1
    rows=[];store={}
    for f in FACTORS:
        print('RUN',f,flush=True);s,e,t,tm=run(p,f,cal,members,bm);s['benchmark_return']=br;s['excess']=s['total_return']-br;rows.append(s);store[f]=(e,t,tm)
    sm=pd.DataFrame(rows);winner=sm.sort_values('train_2016_2021',ascending=False).iloc[0].factor;print('TRAIN WINNER',winner,flush=True)
    stress=[]
    for c in [2.,4.]:s,_,_,_=run(p,winner,cal,members,bm,c);stress.append(s)
    ics=ic(p);rob=[];annual=[]
    for f,(e,t,tm) in store.items():q=robust(e,t);q['factor']=f;rob.append(q);a=sim.annual_returns(e);a['factor']=f;annual.append(a)
    alltm=pd.concat([tm for _,_,tm in store.values()]);audit={**ua,'panel_rows':len(p),'signal_dates':p.signal_date.nunique(),'benchmark_return':br,'winner_selected_on':'2016-2021 only','timing_violations':int((pd.to_datetime(alltm.signal_date)>=pd.to_datetime(alltm.trade_date)).sum())}
    if audit['timing_violations']:raise RuntimeError('timing')
    sm.to_csv(OUT/'summary.csv',index=False);ics.to_csv(OUT/'ic.csv',index=False);pd.DataFrame(stress).to_csv(OUT/'cost.csv',index=False);pd.DataFrame(rob).to_csv(OUT/'robust.csv',index=False);pd.concat(annual).to_csv(OUT/'annual.csv',index=False);pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    print('=== SUMMARY ===');print(sm.to_string(index=False),flush=True);print('=== IC ===');print(ics.to_string(index=False),flush=True);print('=== COST ===');print(pd.DataFrame(stress).to_string(index=False),flush=True);print('=== ROBUST ===');print(pd.DataFrame(rob).to_string(index=False),flush=True);print('=== ANNUAL ===');print(pd.concat(annual).pivot(index='year',columns='factor',values='return').to_string(),flush=True)
if __name__=='__main__':main()
