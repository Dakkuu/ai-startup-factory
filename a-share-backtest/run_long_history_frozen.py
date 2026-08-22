from __future__ import annotations
from pathlib import Path
import math, re
import numpy as np
import pandas as pd

import run_backtest_qlib as qb
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_skewfilter_hard as hard

OUT=Path('results_long_history'); OUT.mkdir(exist_ok=True)
REQUEST_START=pd.Timestamp('2006-01-04'); WARM=pd.Timestamp('2005-01-01'); END=pd.Timestamp('2026-07-29')
MIN_CROSS=300; MIN_RATIO=.20; STABLE_SIGNALS=12
STOCK_RE=base.STOCK_RE


def load_universe():
    qb.RELEASE_TAG='2026-07-29'; qb.ROOT=Path('qlib_data'); qb.download_and_extract()
    cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(qb.ROOT/'calendars'/'day.txt',header=None)[0]))
    m=pd.read_csv(qb.ROOT/'instruments'/'all.txt',sep='\t',header=None,names=['code','start','end'],usecols=[0,1,2])
    m['code']=m.code.astype(str).str.upper(); m['start']=pd.to_datetime(m.start); m['end']=pd.to_datetime(m.end)
    m=m[m.code.str.match(STOCK_RE)].copy(); m=m[(m.end>=WARM)&(m.start<=END)]
    return cal,m


def active_mask(mm,dates):
    out=np.zeros(len(dates),dtype=bool); valid=~pd.isna(dates)
    for r in mm.itertuples(index=False): out |= valid&(dates>=r.start)&(dates<=r.end)
    return out


def build_minimal(cal,members,market_close):
    sig=cal[(cal>=REQUEST_START)&(cal<=END)][::5]; alltrade=cal[cal<=END]
    ex=[]
    for s in sig:
        k=alltrade.searchsorted(s,side='right'); ex.append(alltrade[k] if k<len(alltrade) else pd.NaT)
    ex=pd.DatetimeIndex(ex)
    bmret=market_close.reindex(cal[(cal>=WARM)&(cal<=END)]).pct_change(fill_method=None)
    bmmu=bmret.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).mean().shift(1)
    bmvar=bmret.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).var().shift(1)
    frames=[]; codes=sorted(members.code.unique())
    for i,code in enumerate(codes,1):
        mm=members[members.code==code]; cols={}
        for f in ['open','high','low','close','volume','factor']:
            s=qb.read_bin(code,f,cal)
            if not s.empty: cols[f]=s
        if not all(f in cols for f in ['open','high','low','close','volume']): continue
        z=pd.concat(cols,axis=1).loc[WARM:END].copy()
        if z.empty: continue
        if 'factor' not in z: z['factor']=1.0
        z['factor']=z.factor.replace(0,np.nan).fillna(1.0)
        r=z.close.pct_change(fill_method=None); cnt=z.close.notna().rolling(sim.MIN_LIST_DAYS).sum(); liq=(z.close.abs()*z.volume.abs()).rolling(20).mean()
        mr=bmret.reindex(z.index); smu=r.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).mean().shift(1); cov=r.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).cov(mr).shift(1)
        beta=cov/bmvar.reindex(z.index); alpha=smu-beta*bmmu.reindex(z.index); resid=r-alpha-beta*mr
        iv=resid.rolling(60,min_periods=48).std()
        ss=pd.DataFrame({'count120':cnt.reindex(sig).to_numpy(),'liq20':liq.reindex(sig).to_numpy(),'ivol60':iv.reindex(sig).to_numpy()}); ee=z.reindex(ex).reset_index(drop=True)
        valid=active_mask(mm,sig)&active_mask(mm,ex)&(~pd.isna(ex))&(ss.count120.to_numpy()>=sim.MIN_LIST_DAYS)
        valid&=np.isfinite(ss[['liq20','ivol60']].to_numpy()).all(axis=1)&np.isfinite(ee[['open','high','low','volume']].to_numpy()).all(axis=1)
        if not valid.any(): continue
        k=np.flatnonzero(valid); frames.append(pd.DataFrame({'signal_date':sig[k],'trade_date':ex[k],'code':code,'liq20':ss.liq20.to_numpy()[k],'ivol60':ss.ivol60.to_numpy()[k],'exec_open':ee.open.to_numpy()[k],'exec_high':ee.high.to_numpy()[k],'exec_low':ee.low.to_numpy()[k],'exec_volume':ee.volume.to_numpy()[k],'exec_factor':ee.factor.to_numpy()[k]}))
        if i%750==0: print('LONG BUILD',i,'/',len(codes),flush=True)
    p=pd.concat(frames,ignore_index=True)
    if not (pd.to_datetime(p.signal_date)<pd.to_datetime(p.trade_date)).all(): raise RuntimeError('long panel timing')
    p['liq_rank_pct']=p.groupby('signal_date').liq20.rank(pct=True,method='average',ascending=False)
    return p


def reliable_start(p,members):
    baseok=np.isfinite(p.ivol60)&np.isfinite(p.eff120)&np.isfinite(p.skew40)&(p.liq_rank_pct<=.70)
    t=p.loc[baseok,['signal_date','skew40']].copy(); t['skrank']=t.groupby('signal_date').skew40.rank(pct=True,method='average',ascending=True); elig=t[t.skrank<=.80].groupby('signal_date').size()
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(p.signal_date.unique())))
    active=pd.Series({d:int(((members.start<=d)&(members.end>=d)).sum()) for d in dates})
    df=pd.DataFrame({'eligible':elig.reindex(dates).fillna(0).astype(int),'active':active.reindex(dates).astype(int)},index=dates); df['ratio']=df.eligible/df.active.replace(0,np.nan); df['good']=(df.eligible>=MIN_CROSS)&(df.ratio>=MIN_RATIO)
    start=None
    for i in range(0,max(0,len(df)-STABLE_SIGNALS+1)):
        if bool(df.good.iloc[i:i+STABLE_SIGNALS].all()): start=df.index[i]; break
    df.to_csv(OUT/'eligibility_timeline.csv')
    if start is None: raise RuntimeError('no reliable long-history start')
    return pd.Timestamp(start),df


def make_rank(p,start):
    q=p[p.signal_date>=start].copy(); q['rank_test']=np.nan
    m=np.isfinite(q.ivol60)&np.isfinite(q.eff120)&np.isfinite(q.skew40)&(q.liq_rank_pct<=.70)
    sk=q.loc[m].groupby('signal_date').skew40.rank(pct=True,method='average',ascending=True); ok=pd.Series(False,index=q.index); ok.loc[sk.index]=sk<=.80; m=m&ok
    iv=q.loc[m].groupby('signal_date').ivol60.rank(pct=True,method='average',ascending=True); ef=q.loc[m].groupby('signal_date').eff120.rank(pct=True,method='average',ascending=False); raw=.60*iv+.40*ef
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q


def subset(q,phase=0):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[phase::12])
    z=q[q.signal_date.isin(chosen)][['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')


def hist_fee(gross,side,d,mult):
    d=pd.Timestamp(d); commission=max(5.0,gross*.00025)
    transfer=gross*(.00002 if d<pd.Timestamp('2022-04-29') else .00001)
    stamp=0.0
    if d<pd.Timestamp('2007-05-30'): stamp=gross*.001
    elif d<pd.Timestamp('2008-04-24'): stamp=gross*.003
    elif d<pd.Timestamp('2008-09-19'): stamp=gross*.001
    elif d<pd.Timestamp('2023-08-28'): stamp=gross*.001 if side=='sell' else 0.0
    else: stamp=gross*.0005 if side=='sell' else 0.0
    return mult*(commission+transfer+stamp)


def run(z,cal,members,bm,start,cost=1.0):
    old=(sim.START,sim.END,sim.WARM,hard.N_HOLD,sim.ENTRY_PCT,sim.KEEP_PCT,sim.fee)
    sim.START=start; sim.END=END; sim.WARM=WARM; hard.N_HOLD=20; sim.ENTRY_PCT=.10; sim.KEEP_PCT=.30; sim.fee=hist_fee
    try:
        eq,tr,tm,to=hard.hard_simulate(z,cal,members,cost); st=sim.perf(eq,tr,to,bm); st['positions_max']=int(eq.positions.max()); st['positions_median']=float(eq.positions.median()); return st,eq,tr,tm
    finally:
        sim.START,sim.END,sim.WARM,hard.N_HOLD,sim.ENTRY_PCT,sim.KEEP_PCT,sim.fee=old


def main():
    sim.START=REQUEST_START; sim.END=END; sim.WARM=WARM; base.START=REQUEST_START; base.END=END; base.WARM=WARM; base.OUT=OUT; v4.OUT=OUT
    cal,members=load_universe(); market_code,market_close,_=v4.pick_market(cal)
    p=build_minimal(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close)
    start,timeline=reliable_start(p,members); q=make_rank(p,start); bm=market_close.loc[start:END].dropna(); z=subset(q,0)
    st,eq,tr,tm=run(z,cal,members,bm,start,1.0); years=(END-start).days/365.25
    sim.annual_returns(eq).to_csv(OUT/'annual.csv',index=False)
    phases=[]
    for ph in range(12):
        x,_,_,_=run(subset(q,ph),cal,members,bm,start,1.0); phases.append({**x,'phase':ph})
    pd.DataFrame(phases).to_csv(OUT/'phase_offsets.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        x,_,_,_=run(z,cal,members,bm,start,cm); costs.append({**x,'cost_mult_test':cm})
    pd.DataFrame(costs).to_csv(OUT/'costs.csv',index=False)
    blocks=[]; cursor=start
    while cursor<END:
        e=min(cursor+pd.DateOffset(years=5),END); s=eq[(pd.to_datetime(eq.trade_date)>=cursor)&(pd.to_datetime(eq.trade_date)<=e)]
        if len(s)>=20:
            nav=s.equity.astype(float); total=float(nav.iloc[-1]/nav.iloc[0]-1); dd=float((nav/nav.cummax()-1).min()); blocks.append({'start':str(cursor.date()),'end':str(pd.Timestamp(e).date()),'return':total,'mdd':dd})
        cursor=pd.Timestamp(e)
    pd.DataFrame(blocks).to_csv(OUT/'five_year_blocks.csv',index=False)
    audit={'requested_start':str(REQUEST_START.date()),'reliable_start':str(start.date()),'end':str(END.date()),'actual_years':years,'market_factor':market_code,'union_members':int(members.code.nunique()),'eligibility_threshold':MIN_CROSS,'eligibility_ratio_threshold':MIN_RATIO,'stable_signals_required':STABLE_SIGNALS,'total_return':st['total_return'],'cagr':st['cagr'],'mdd':st['max_drawdown'],'sharpe':st['sharpe'],'timing_violations':int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0,'all_phases_positive':int((pd.DataFrame(phases).total_return>0).all()),'cost4_positive':int(float(pd.DataFrame(costs).loc[pd.DataFrame(costs).cost_mult_test==4,'total_return'].iloc[0])>0),'note':'PIT ST not yet applied before 2015; historical stamp duty schedule included; commission stressed separately'}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if audit['timing_violations'] or st['positions_max']>20: raise RuntimeError('long-history execution audit')
    print('=== LONG AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
    print('=== LONG PHASES ==='); print(pd.DataFrame(phases)[['phase','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True)
    print('=== LONG COSTS ==='); print(pd.DataFrame(costs)[['cost_mult_test','total_return','cagr','max_drawdown','sharpe']].to_string(index=False),flush=True)
    print('=== LONG ANNUAL ==='); print(sim.annual_returns(eq).to_string(index=False),flush=True)

if __name__=='__main__': main()
