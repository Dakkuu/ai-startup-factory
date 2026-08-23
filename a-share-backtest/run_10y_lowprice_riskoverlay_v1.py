from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_lowprice_signalpure_v1 as lp
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_max_audit as ma
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_lowprice_riskoverlay_v1'); OUT.mkdir(exist_ok=True)
START=mo.START; TRAIN_END=mo.TRAIN_END; END=mo.END; PSEUDO=mo.PSEUDO_START
W={'price':.25,'iv':.20,'ef':.20,'rmom':.22,'tstat':.13}; LIQ=.55; FLOOR=2.; H=90; N=8; E=.10; K=.30; PC=18
BASECOLS=strict.BASECOLS+['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy']

RULES=[
 ('always',{}),
 ('ma200_m05',{'kind':'ma','x':-.05}),('ma200_000',{'kind':'ma','x':0.}),('ma200_p03',{'kind':'ma','x':.03}),
 ('mom120_m10',{'kind':'mom120','x':-.10}),('mom120_m05',{'kind':'mom120','x':-.05}),('mom120_000',{'kind':'mom120','x':0.}),
 ('dd252_m20',{'kind':'dd','x':-.20}),('dd252_m15',{'kind':'dd','x':-.15}),('dd252_m10',{'kind':'dd','x':-.10}),
 ('breadth035',{'kind':'breadth','x':.35}),('breadth045',{'kind':'breadth','x':.45}),('breadth055',{'kind':'breadth','x':.55}),
 ('ma0_or_b45',{'kind':'or','a':('ma',0.),'b':('breadth',.45)}),
 ('mam05_or_b45',{'kind':'or','a':('ma',-.05),'b':('breadth',.45)}),
 ('ma0_or_mom0',{'kind':'or','a':('ma',0.),'b':('mom120',0.)}),
 ('mam05_and_b35',{'kind':'and','a':('ma',-.05),'b':('breadth',.35)}),
 ('ma0_and_b35',{'kind':'and','a':('ma',0.),'b':('breadth',.35)}),
 ('dd15_or_b45',{'kind':'or','a':('dd',-.15),'b':('breadth',.45)}),
 ('dd20_and_b35',{'kind':'and','a':('dd',-.20),'b':('breadth',.35)}),
]

def features(p,bm):
    s=bm.astype(float).sort_index(); r=s.pct_change(); f=pd.DataFrame(index=s.index)
    f['ma']=s/s.rolling(200,min_periods=120).mean()-1
    f['mom60']=s/s.shift(60)-1; f['mom120']=s/s.shift(120)-1
    f['dd']=s/s.rolling(252,min_periods=120).max()-1
    sig=pd.DatetimeIndex(sorted(pd.to_datetime(p.signal_date.unique())))
    f=f.reindex(sig).ffill()
    x=p[['signal_date','mom120','liq_rank_pct']].copy(); m=(x.liq_rank_pct<=.70)&np.isfinite(x.mom120)
    br=x[m].groupby('signal_date').mom120.apply(lambda z: float((z>0).mean())).rename('breadth')
    f['breadth']=br.reindex(f.index)
    return f

def atom(f,a):
    k,x=a
    return f[k]>float(x)
def risk_on(f,spec):
    kind=spec.get('kind','always')
    if kind=='always': return pd.Series(True,index=f.index)
    if kind in ('ma','mom120','dd','breadth'): return f[kind]>float(spec['x'])
    if kind=='or': return atom(f,spec['a'])|atom(f,spec['b'])
    if kind=='and': return atom(f,spec['a'])&atom(f,spec['b'])
    raise ValueError(kind)
def apply(q,on):
    x=q.copy(); mp=on.to_dict(); ok=pd.to_datetime(x.signal_date).map(mp).fillna(False).to_numpy(bool); x.loc[~ok,'rank_test']=np.nan; return x

def subset(q,ph):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[int(ph)::PC]); z=q[q.signal_date.isin(chosen)][[c for c in BASECOLS if c in q.columns]].copy(); z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')
def run(q,ph,cal,members,bm,cash,train=True,cost=1.):
    kw=dict(n=N,entry=E,keep=K,initial_cash=float(cash),cost=float(cost))
    if train: kw.update(start=START,end=TRAIN_END)
    return ma.run_panel(subset(q,ph),cal,members,bm,**kw)
def combine(eqs,initials,start=START):
    start=pd.Timestamp(start); idx={start}; ser=[]
    for e,init in zip(eqs,initials):
        s=e.set_index(pd.to_datetime(e.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]; s=pd.concat([pd.Series({start:float(init)}),s]); s=s[~s.index.duplicated(keep='last')].sort_index(); ser.append(s); idx.update(s.index)
    idx=pd.DatetimeIndex(sorted(idx)); a=[s.reindex(idx).ffill().fillna(float(init)) for s,init in zip(ser,initials)]; t=pd.concat(a,axis=1).sum(axis=1); return pd.DataFrame({'trade_date':idx,'equity':t.to_numpy(float)})
def stats(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); y=max((s.index[-1]-s.index[0]).days/365.25,1e-9); r=s.pct_change().dropna(); return {'total_return':float(s.iloc[-1]/s.iloc[0]-1),'cagr':float((s.iloc[-1]/s.iloc[0])**(1/y)-1),'max_drawdown':float((s/s.cummax()-1).min()),'sharpe':float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else np.nan}
def period(eq,a,b):
    z=eq[(pd.to_datetime(eq.trade_date)>=pd.Timestamp(a))&(pd.to_datetime(eq.trade_date)<=pd.Timestamp(b))]; return float(z.equity.iloc[-1]/z.equity.iloc[0]-1) if len(z)>1 else np.nan

def eval_rule(q,on,cal,members,bm,train=True,cost=1.):
    x=apply(q,on); per=1e6/PC; eqs=[]; cgs=[]; rets=[]; mdds=[]
    for ph in range(PC):
        st,e,tr,tm=run(x,ph,cal,members,bm,per,train=train,cost=cost); eqs.append(e); cgs.append(st['cagr']); rets.append(st['total_return']); mdds.append(st['max_drawdown'])
    eq=combine(eqs,[per]*PC); st=stats(eq); st.update(min_phase_return=float(np.min(rets)),median_phase_return=float(np.median(rets)),min_phase_cagr=float(np.min(cgs)),median_phase_cagr=float(np.median(cgs)),std_phase_cagr=float(np.std(cgs)),worst_phase_mdd=float(np.min(mdds)))
    return st,eq

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=lp.attach_price(p,cal); p=strict.attach_gap_flags(p,cal,'board'); q=lp.rank_signal(p,W,LIQ,FLOOR); f=features(p,bm)
    rows=[]
    for name,spec in RULES:
        on=risk_on(f,spec).fillna(False); trainmask=(f.index>=START)&(f.index<=TRAIN_END); share=float(on[trainmask].mean())
        if share<.55: continue
        print('RULE',name,'share',share,flush=True); st,eq=eval_rule(q,on,cal,members,bm,train=True); score=st['cagr']+.55*st['min_phase_cagr']-.30*st['std_phase_cagr']+.10*st['max_drawdown']
        rows.append({'rule':name,'spec':json.dumps(spec),'train_risk_on_share':share,**st,'train_robust_score':score})
    z=pd.DataFrame(rows).sort_values(['train_robust_score','min_phase_cagr'],ascending=False); z.to_csv(OUT/'train_rules.csv',index=False); win=z.iloc[0]; pd.DataFrame([win]).to_csv(OUT/'train_winner.csv',index=False)
    spec=dict(RULES)[win.rule]; on=risk_on(f,spec).fillna(False); full,eq=eval_rule(q,on,cal,members,bm,train=False); full.update(rule=win.rule,spec=json.dumps(spec),train_selected_score=win.train_robust_score,train_2016_2021_return=period(eq,START,TRAIN_END),pseudo_oos_2022_2026_return=period(eq,PSEUDO,END),full_risk_on_share=float(on.mean()))
    pd.DataFrame([full]).to_csv(OUT/'full_validation.csv',index=False); fa.annual(eq).to_csv(OUT/'annual.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        st,_=eval_rule(q,on,cal,members,bm,train=False,cost=cm); st.update(rule=win.rule,cost_mult_test=cm); costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'costs.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'selection_period':'2016-07-29..2021-12-31','pseudo_oos_not_used_in_selection':1,'stock_signal':'frozen lowprice trend +262pct strict candidate','overlay':'T-only market MA/momentum/drawdown/breadth; risk-off sets rank NaN causing sell-to-cash at sleeve rebalance','candidate_rules':len(rows),'exact_phase_count':PC,'total_cash':1_000_000}]).to_csv(OUT/'audit.csv',index=False)
    print('TRAIN',z.to_string(index=False),flush=True); print('FULL',pd.DataFrame([full]).to_string(index=False),flush=True)
if __name__=='__main__': main()
