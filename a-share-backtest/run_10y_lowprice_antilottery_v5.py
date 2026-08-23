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

OUT=Path('results_lowprice_antilottery_v5'); OUT.mkdir(exist_ok=True)
START=mo.START; TRAIN_END=mo.TRAIN_END; PSEUDO=mo.PSEUDO_START; END=mo.END
LIQ=.55; FLOOR=2.0; HOLD=60; N=4; ENTRY=.15; KEEP=.40; NPHASE=12
# Five predeclared architectures only. No continuous weight tuning.
SPECS={
 'price_heavy': {'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12},
 'price_down': {'price':.30,'iv':.18,'ef':.12,'rmom':.18,'tstat':.10,'down':.12},
 'price_amax': {'price':.30,'iv':.18,'ef':.12,'rmom':.18,'tstat':.10,'amax':.12},
 'price_down_amax': {'price':.28,'iv':.17,'ef':.10,'rmom':.17,'tstat':.08,'down':.10,'amax':.10},
 'price_antilottery': {'price':.26,'iv':.18,'ef':.10,'rmom':.16,'tstat':.08,'down':.08,'amax':.08,'askew':.06},
}
BASECOLS=strict.BASECOLS+['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy']


def subset(q,ph):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[int(ph)::NPHASE])
    cols=[c for c in BASECOLS if c in q.columns]; z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def run_phase(q,ph,cal,members,bm,start,end,cash,cost=1.):
    return ma.run_panel(subset(q,ph),cal,members,bm,n=N,entry=ENTRY,keep=KEEP,initial_cash=float(cash),start=pd.Timestamp(start),end=pd.Timestamp(end),cost=float(cost))
def combine_abs(eqs,initials,start):
    start=pd.Timestamp(start); idx={start}; ss=[]
    for e,init in zip(eqs,initials):
        s=e.set_index(pd.to_datetime(e.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]
        s=pd.concat([pd.Series({start:float(init)}),s]); s=s[~s.index.duplicated(keep='last')].sort_index(); ss.append(s); idx.update(s.index)
    idx=pd.DatetimeIndex(sorted(idx)); arr=[s.reindex(idx).ffill().fillna(float(init)) for s,init in zip(ss,initials)]
    return pd.DataFrame({'trade_date':idx,'equity':pd.concat(arr,axis=1).sum(axis=1).to_numpy(float)})
def eq_cagr(eq,a=None,b=None):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date)
    if a is not None:z=z[z.trade_date>=pd.Timestamp(a)]
    if b is not None:z=z[z.trade_date<=pd.Timestamp(b)]
    if len(z)<2:return np.nan
    y=max((z.trade_date.iloc[-1]-z.trade_date.iloc[0]).days/365.25,1e-9); return float((z.equity.iloc[-1]/z.equity.iloc[0])**(1/y)-1)
def eq_return(eq,a=None,b=None):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date)
    if a is not None:z=z[z.trade_date>=pd.Timestamp(a)]
    if b is not None:z=z[z.trade_date<=pd.Timestamp(b)]
    return float(z.equity.iloc[-1]/z.equity.iloc[0]-1) if len(z)>1 else np.nan
def eq_mdd(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); return float((s/s.cummax()-1).min())

def eval_all(q,cal,members,bm,start,end,total_cash=1e6,cost=1.):
    per=float(total_cash)/NPHASE; eqs=[]; sts=[]; tms=[]
    for ph in range(NPHASE):
        st,e,tr,tm=run_phase(q,ph,cal,members,bm,start,end,per,cost); eqs.append(e); sts.append(st)
        if len(tm):tms.append(tm.assign(phase=ph))
    eq=combine_abs(eqs,[per]*NPHASE,start)
    return eq,sts,tms

def train_row(eq,sts):
    rr=np.array([s['total_return'] for s in sts],float); cg=np.array([s['cagr'] for s in sts],float); md=np.array([s['max_drawdown'] for s in sts],float)
    h1=eq_cagr(eq,START,'2018-12-31'); h2=eq_cagr(eq,'2019-01-01',TRAIN_END); ec=eq_cagr(eq); em=eq_mdd(eq)
    hard=int((rr>0).all() and h1>0 and h2>0 and em>-0.50)
    score=float(ec+.50*np.min(cg)-.25*np.std(cg)+.25*min(h1,h2))
    return {'train_return':eq_return(eq),'train_cagr':ec,'train_mdd':em,'min_phase_return':rr.min(),'median_phase_return':np.median(rr),'min_phase_cagr':cg.min(),'median_phase_cagr':np.median(cg),'std_phase_cagr':cg.std(),'worst_phase_mdd':md.min(),'half1_cagr':h1,'half2_cagr':h2,'all_phases_positive':int((rr>0).all()),'hard_pass':hard,'robust_score':score}

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=lp.attach_price(p,cal); p=strict.attach_gap_flags(p,cal,'board')
    train=[]
    for name,w in SPECS.items():
        print('TRAIN',name,flush=True); q=lp.rank_signal(p,w,LIQ,FLOOR); eq,sts,tms=eval_all(q,cal,members,bm,START,TRAIN_END,1e6,1.); r=train_row(eq,sts); r.update(architecture=name,weights=json.dumps(w,sort_keys=True)); train.append(r)
    t=pd.DataFrame(train); t.to_csv(OUT/'train_architectures.csv',index=False)
    passed=t[t.hard_pass==1].copy(); passed=passed if len(passed) else t.copy()
    win=passed.sort_values(['robust_score','train_cagr','min_phase_cagr'],ascending=False).iloc[0].to_dict(); pd.DataFrame([win]).to_csv(OUT/'train_only_winner.csv',index=False)
    # Freeze before full-period access.
    name=str(win['architecture']); q=lp.rank_signal(p,SPECS[name],LIQ,FLOOR); eq,sts,tms=eval_all(q,cal,members,bm,START,END,1e6,1.)
    full=fa.perf_eq(eq,bm); full['train_2016_2021_return']=fa.period_return(eq,START,TRAIN_END); full['pseudo_oos_2022_2026_return']=fa.period_return(eq,PSEUDO,END); full.update(architecture=name,weights=json.dumps(SPECS[name],sort_keys=True),hold=HOLD,n_hold=N,entry=ENTRY,keep=KEEP,selection_validation_accesses=0)
    pd.DataFrame([full]).to_csv(OUT/'frozen_full_validation.csv',index=False); fa.annual(eq).to_csv(OUT/'annual.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        ceq,_,_=eval_all(q,cal,members,bm,START,END,1e6,cm); cs=fa.perf_eq(ceq,bm); cs['cost_mult']=cm; cs['pseudo_return']=fa.period_return(ceq,PSEUDO,END); costs.append(cs)
    pd.DataFrame(costs).to_csv(OUT/'cost_stress.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'experiment':'LowPrice Anti-Lottery V5','candidate_architectures':'|'.join(SPECS),'candidate_count':len(SPECS),'selection_period':'2016-07-29..2021-12-31','validation_2022_2026_accesses_before_freeze':0,'construction':'fixed from frozen V4: H60 N4 entry15 keep40','selection':'all 12 phases exact RMB1m split; both train halves positive; ensemble MDD>-50%; robust score','objective':'ensemble CAGR + .50 worst phase CAGR - .25 phase dispersion + .25 weaker train-half CAGR','executor':'hard_v3; signal-pure; board-limit blocked; 100-share lots; no replacement'}]).to_csv(OUT/'audit.csv',index=False)
    print('TRAIN',t.to_string(index=False),flush=True); print('WINNER',pd.DataFrame([win]).to_string(index=False),flush=True); print('FULL',pd.DataFrame([full]).to_string(index=False),flush=True)
if __name__=='__main__':main()
