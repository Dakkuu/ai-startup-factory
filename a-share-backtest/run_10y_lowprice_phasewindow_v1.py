from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_lowprice_signalpure_v1 as lp
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_max_audit as ma
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_lowprice_phasewindow_v1'); OUT.mkdir(exist_ok=True)
W={'price':.25,'iv':.20,'ef':.20,'rmom':.22,'tstat':.13}; LIQ=.55; FLOOR=2.; H=90; N=8; E=.10; K=.30; PC=18
START=mo.START; TRAIN_END=mo.TRAIN_END; PSEUDO=mo.PSEUDO_START; END=mo.END
BASECOLS=strict.BASECOLS+['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy']

def subset(q,ph):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[int(ph)::PC]); z=q[q.signal_date.isin(chosen)][[c for c in BASECOLS if c in q.columns]].copy(); z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')
def run(q,ph,cal,members,bm,cash,train=False,cost=1.):
    kw=dict(n=N,entry=E,keep=K,initial_cash=float(cash),cost=float(cost))
    if train: kw.update(start=START,end=TRAIN_END)
    return ma.run_panel(subset(q,ph),cal,members,bm,**kw)
def combine(eqs,initials,start=START):
    start=pd.Timestamp(start); idx={start}; ss=[]
    for e,init in zip(eqs,initials):
        s=e.set_index(pd.to_datetime(e.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]; s=pd.concat([pd.Series({start:float(init)}),s]); s=s[~s.index.duplicated(keep='last')].sort_index(); ss.append(s); idx.update(s.index)
    idx=pd.DatetimeIndex(sorted(idx)); arr=[s.reindex(idx).ffill().fillna(float(init)) for s,init in zip(ss,initials)]; t=pd.concat(arr,axis=1).sum(axis=1); return pd.DataFrame({'trade_date':idx,'equity':t.to_numpy(float)})
def perf(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); r=s.pct_change().dropna(); y=max((s.index[-1]-s.index[0]).days/365.25,1e-9); return {'total_return':float(s.iloc[-1]/s.iloc[0]-1),'cagr':float((s.iloc[-1]/s.iloc[0])**(1/y)-1),'max_drawdown':float((s/s.cummax()-1).min()),'sharpe':float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else np.nan}
def period(eq,a,b):
    z=eq[(pd.to_datetime(eq.trade_date)>=pd.Timestamp(a))&(pd.to_datetime(eq.trade_date)<=pd.Timestamp(b))]; return float(z.equity.iloc[-1]/z.equity.iloc[0]-1) if len(z)>1 else np.nan

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=lp.attach_price(p,cal); p=strict.attach_gap_flags(p,cal,'board'); q=lp.rank_signal(p,W,LIQ,FLOOR)
    phrows=[]
    for ph in range(PC):
        st,e,tr,tm=run(q,ph,cal,members,bm,1e6,train=True); phrows.append({'phase':ph,**st,'half1':period(e,START,mo.HALF1_END),'half2':period(e,mo.HALF2_START,TRAIN_END)})
    phdf=pd.DataFrame(phrows); phdf.to_csv(OUT/'train_phases.csv',index=False)
    wins=[]
    for k in range(2,10):
        cand=[]
        for st in range(PC):
            ix=[(st+i)%PC for i in range(k)]; g=phdf[phdf.phase.isin(ix)]
            # Selection uses training period only. Penalize phase dispersion and require both train subperiods.
            a=float(g.half1.mean()); b=float(g.half2.mean()); tr=float(g.total_return.mean()); sd=float(g.cagr.std(ddof=0)); score=min(a,b)+.25*tr-.35*sd
            cand.append({'k':k,'start_phase':st,'phases':'|'.join(map(str,ix)),'train_half1_mean':a,'train_half2_mean':b,'train_return_mean':tr,'train_cagr_std':sd,'train_score':score})
        z=pd.DataFrame(cand).sort_values(['train_score','train_return_mean'],ascending=False); z.to_csv(OUT/f'train_windows_k{k}.csv',index=False); wins.append(z.iloc[0])
    wdf=pd.DataFrame(wins).sort_values('k'); wdf.to_csv(OUT/'train_selected_windows.csv',index=False)
    full=[]; costs=[]
    for r in wdf.itertuples(index=False):
        ix=[int(x) for x in str(r.phases).split('|')]; per=1e6/len(ix); eqs=[]
        for ph in ix:
            st,e,tr,tm=run(q,ph,cal,members,bm,per,train=False); eqs.append(e)
        eq=combine(eqs,[per]*len(ix)); s=perf(eq); s.update(k=r.k,start_phase=r.start_phase,phases=r.phases,train_score=r.train_score,train_2016_2021_return=period(eq,START,TRAIN_END),pseudo_oos_2022_2026_return=period(eq,PSEUDO,END),total_cash=1e6,cash_per_sleeve=per); full.append(s)
        for cm in (2.,4.,8.):
            eqs2=[]
            for ph in ix:
                _,e,_,_=run(q,ph,cal,members,bm,per,train=False,cost=cm); eqs2.append(e)
            ee=combine(eqs2,[per]*len(ix)); ss=perf(ee); ss.update(k=r.k,phases=r.phases,cost_mult_test=cm); costs.append(ss)
    pd.DataFrame(full).to_csv(OUT/'exact_full_windows.csv',index=False); pd.DataFrame(costs).to_csv(OUT/'costs.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'stock_signal':'frozen lowprice trend','window_selection':'2016-2021 only, contiguous cyclic phase windows k=2..9; score uses two train subperiods + phase dispersion penalty','important_caveat':'phase-window family discovered after repeated inspection of 2016-2026 path; results are exploratory, not clean untouched OOS','execution':'hard_v3 board-limit; total 1m split across selected sleeves; 100-share lots; blocked execution no replacement'}]).to_csv(OUT/'audit.csv',index=False)
    print('TRAIN WINDOWS',wdf.to_string(index=False),flush=True); print('FULL EXACT',pd.DataFrame(full).to_string(index=False),flush=True)
if __name__=='__main__': main()
