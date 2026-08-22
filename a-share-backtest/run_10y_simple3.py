from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim

START=pd.Timestamp('2016-07-29'); END=pd.Timestamp('2026-07-29'); WARM=pd.Timestamp('2015-01-01')
OUT=Path('results_simple3'); OUT.mkdir(exist_ok=True)
FACTORS=('lowmax20','reversal20','lowrange20')

def active_mask(mm, dates):
    out=np.zeros(len(dates),dtype=bool); valid=~pd.isna(dates)
    for r in mm.itertuples(index=False): out |= valid & (dates>=r.start) & (dates<=r.end)
    return out

def build_panel(cal,members):
    trade_cal=cal[(cal>=START)&(cal<=END)]
    signal_dates=pd.DatetimeIndex(trade_cal[::5]); alltrade=cal[cal<=END]
    exec_dates=[]
    for s in signal_dates:
        k=alltrade.searchsorted(s,side='right'); exec_dates.append(alltrade[k] if k<len(alltrade) else pd.NaT)
    exec_dates=pd.DatetimeIndex(exec_dates)
    frames=[]; codes=sorted(members.code.unique())
    for i,code in enumerate(codes,1):
        mm=members[members.code==code]; cols={}
        for f in ['open','high','low','close','volume','factor']:
            s=base.qb.read_bin(code,f,cal)
            if not s.empty: cols[f]=s
        if not all(f in cols for f in ['open','high','low','close','volume']): continue
        z=pd.concat(cols,axis=1).loc[WARM:END].copy()
        if z.empty: continue
        if 'factor' not in z: z['factor']=1.0
        z['factor']=z.factor.replace(0,np.nan).fillna(1.0)
        r=z.close.pct_change(fill_method=None)
        count120=z.close.notna().rolling(120).sum()
        liq20=(z.close.abs()*z.volume.abs()).rolling(20).mean()
        fac={
            'lowmax20':r.rolling(20,min_periods=16).max(),
            'reversal20':z.close/z.close.shift(20)-1.0,
            'lowrange20':((z.high-z.low).abs()/z.close.abs().replace(0,np.nan)).rolling(20,min_periods=16).mean(),
        }
        sig={'count120':count120.reindex(signal_dates).to_numpy(),'liq20':liq20.reindex(signal_dates).to_numpy()}
        for k,s in fac.items(): sig[k]=s.reindex(signal_dates).to_numpy()
        sig=pd.DataFrame(sig); ex=z.reindex(exec_dates).reset_index(drop=True)
        valid=active_mask(mm,signal_dates)&active_mask(mm,exec_dates)&(~pd.isna(exec_dates))
        valid &= np.asarray(sig.count120>=120)
        valid &= np.isfinite(sig[['liq20']].to_numpy()).all(axis=1)
        valid &= np.isfinite(ex[['open','high','low','volume']].to_numpy()).all(axis=1)
        if not valid.any(): continue
        idx=np.flatnonzero(valid)
        rec={'signal_date':signal_dates[idx],'trade_date':exec_dates[idx],'code':code,
             'liq20':sig.liq20.to_numpy()[idx].astype(float),
             'exec_open':ex.open.to_numpy()[idx].astype(float),'exec_high':ex.high.to_numpy()[idx].astype(float),
             'exec_low':ex.low.to_numpy()[idx].astype(float),'exec_volume':ex.volume.to_numpy()[idx].astype(float),
             'exec_factor':ex.factor.to_numpy()[idx].astype(float)}
        for f in FACTORS: rec[f]=sig[f].to_numpy()[idx].astype(float)
        frames.append(pd.DataFrame(rec))
        if i%1000==0: print('histories',i,'/',len(codes),flush=True)
    p=pd.concat(frames,ignore_index=True)
    if not (pd.to_datetime(p.signal_date)<pd.to_datetime(p.trade_date)).all(): raise RuntimeError('timing')
    p['liq_rank_pct']=p.groupby('signal_date').liq20.rank(pct=True,ascending=False,method='average')
    liq=p.liq_rank_pct<=0.80
    for f in FACTORS:
        p[f+'_pct']=np.nan; m=liq&p[f].notna()
        p.loc[m,f+'_pct']=p.loc[m].groupby('signal_date')[f].rank(pct=True,ascending=True,method='average')
    gs=p[liq].groupby('signal_date').size()
    print('PANEL',p.shape,'dates',p.signal_date.nunique(),'liquid median',gs.median(),flush=True)
    if p.signal_date.nunique()<480 or gs.median()<1500: raise RuntimeError('bad universe')
    return p

def run_factor(panel,factor,cal,members,bm,cost=1.0,daily=True):
    q=panel.copy(); q['ivol60_pct']=q[factor+'_pct']
    e,t,tm,to=sim.simulate(q,'ivol',cal,members,cost,daily_mtm=daily)
    s=sim.perf(e,t,to,bm if daily else None); s['factor']=factor; s['cost_mult']=cost
    s['train_2016_2021']=sim.period_return(e,'2016-07-29','2021-12-31')
    s['sealed_2022_2026']=sim.period_return(e,'2022-01-01','2026-07-29')
    return s,e,t,tm

def ic_stats(panel):
    p=panel.sort_values(['code','trade_date']).copy(); p['fwd5']=p.groupby('code').exec_open.shift(-1)/p.exec_open-1
    raw=[]
    for d,g in p.groupby('signal_date'):
        z=g.dropna(subset=['fwd5'])
        if len(z)<500: continue
        for f in FACTORS:
            zz=z.dropna(subset=[f])
            raw.append({'signal_date':d,'factor':f,'ic':(-zz[f]).corr(zz.fwd5,method='spearman')})
    q=pd.DataFrame(raw); out=[]
    for f,g in q.groupby('factor'):
        x=g.ic.dropna(); out.append({'factor':f,'mean_ic':x.mean(),'icir':x.mean()/x.std()*np.sqrt(52),'positive_ic_rate':(x>0).mean(),'n':len(x)})
    return q,pd.DataFrame(out)

def robustness(eq,tr):
    r=eq.equity.pct_change().dropna(); x=r.copy(); x.loc[x.nlargest(5).index]=0
    k=max(1,int(math.ceil(len(r)*.01))); y=r.copy(); y.loc[y.nlargest(k).index]=0
    pnl=tr.net_pnl.sum() if len(tr) else 0; b5=tr.nlargest(min(5,len(tr)),'net_pnl').net_pnl.sum() if len(tr) else 0
    return {'base_return':eq.equity.iloc[-1]/1_000_000-1,'without_best5_days':(1+x).prod()-1,'without_best1pct_days':(1+y).prod()-1,'pnl_without_best5_trades':pnl-b5}

def main():
    base.START=START; base.WARM=WARM; base.END=END; base.OUT=OUT
    sim.START=START; sim.WARM=WARM; sim.END=END
    cal,members,ua=base.load_base(); panel=build_panel(cal,members); panel.to_pickle(OUT/'panel.pkl')
    bm=base.qb.read_bin('SH000300','close',cal).loc[START:END].dropna(); bmret=float(bm.iloc[-1]/bm.iloc[0]-1)
    rows=[]; store={}
    for f in FACTORS:
        print('RUN',f,flush=True); s,e,t,tm=run_factor(panel,f,cal,members,bm); s['benchmark_return']=bmret; s['excess']=s['total_return']-bmret; rows.append(s); store[f]=(e,t,tm)
    sm=pd.DataFrame(rows); sm.to_csv(OUT/'summary.csv',index=False)
    winner=sm.sort_values('train_2016_2021',ascending=False).iloc[0].factor
    print('TRAIN WINNER',winner,flush=True)
    stress=[]
    for c in [2.0,4.0]:
        s,_,_,_=run_factor(panel,winner,cal,members,bm,cost=c); stress.append(s)
    pd.DataFrame(stress).to_csv(OUT/'cost_stress.csv',index=False)
    icts,ics=ic_stats(panel); icts.to_csv(OUT/'ic_timeseries.csv',index=False); ics.to_csv(OUT/'ic_summary.csv',index=False)
    rob=[]
    for f in FACTORS:
        e,t,_=store[f]; q=robustness(e,t); q['factor']=f; rob.append(q)
    pd.DataFrame(rob).to_csv(OUT/'robustness.csv',index=False)
    annual=[]
    for f,(e,t,tm) in store.items():
        ar=sim.annual_returns(e); ar['factor']=f; annual.append(ar)
    pd.concat(annual).to_csv(OUT/'annual.csv',index=False)
    alltm=pd.concat([tm.assign(factor=f) for f,(_,_,tm) in store.items()])
    audit={**ua,'panel_rows':len(panel),'signal_dates':panel.signal_date.nunique(),'benchmark':'SH000300','benchmark_return':bmret,'winner_selected_on':'2016-2021 total return only','sealed_period':'2022-01-01..2026-07-29','timing_violations':int((pd.to_datetime(alltm.signal_date)>=pd.to_datetime(alltm.trade_date)).sum()),'rule':'single factor; top80% liquidity; 30 equal sleeves; enter top10 keep top30; 5d rebalance; next-open execution'}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if audit['timing_violations']: raise RuntimeError('timing audit')
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
    print('=== SUMMARY ==='); print(sm.to_string(index=False),flush=True)
    print('=== IC ==='); print(ics.to_string(index=False),flush=True)
    print('=== WINNER COST ==='); print(pd.DataFrame(stress).to_string(index=False),flush=True)
    print('=== ROBUSTNESS ==='); print(pd.DataFrame(rob).to_string(index=False),flush=True)
    print('=== ANNUAL ==='); print(pd.concat(annual).pivot(index='year',columns='factor',values='return').to_string(),flush=True)

if __name__=='__main__': main()
