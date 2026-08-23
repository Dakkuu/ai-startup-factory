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

OUT=Path('results_lowprice_concentration_v4'); OUT.mkdir(exist_ok=True)
START=mo.START; TRAIN_END=mo.TRAIN_END; PSEUDO=mo.PSEUDO_START; END=mo.END
WEIGHTS={'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12}
LIQ=.55; FLOOR=2.0
HOLDS=(40,50,60,75,90)
NS=(4,6,8,10)
BUFFERS=((.05,.20),(.10,.30),(.15,.40))
BASECOLS=strict.BASECOLS+['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy']


def phase_count(h): return max(1,round(int(h)/5))
def screen_phases(h):
    n=phase_count(h); return sorted(set(int(round(x))%n for x in np.linspace(0,n-1,4)))
def subset(q,h,ph):
    n=phase_count(h); dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[int(ph)::n])
    cols=[c for c in BASECOLS if c in q.columns]; z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')
def run_one(q,h,ph,n,e,k,cal,members,bm,cash,start,end,cost=1.):
    return ma.run_panel(subset(q,h,ph),cal,members,bm,n=int(n),entry=float(e),keep=float(k),initial_cash=float(cash),start=pd.Timestamp(start),end=pd.Timestamp(end),cost=float(cost))
def combine_abs(eqs,initials,start=START):
    start=pd.Timestamp(start); idx={start}; ss=[]
    for e,init in zip(eqs,initials):
        s=e.set_index(pd.to_datetime(e.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]
        s=pd.concat([pd.Series({start:float(init)}),s]); s=s[~s.index.duplicated(keep='last')].sort_index(); ss.append(s); idx.update(s.index)
    idx=pd.DatetimeIndex(sorted(idx)); total=pd.concat([s.reindex(idx).ffill().fillna(float(init)) for s,init in zip(ss,initials)],axis=1).sum(axis=1)
    return pd.DataFrame({'trade_date':idx,'equity':total.to_numpy(float)})
def eq_return(eq,a=None,b=None):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date)
    if a is not None: z=z[z.trade_date>=pd.Timestamp(a)]
    if b is not None: z=z[z.trade_date<=pd.Timestamp(b)]
    return float(z.equity.iloc[-1]/z.equity.iloc[0]-1) if len(z)>1 else np.nan
def eq_cagr(eq,a=None,b=None):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date)
    if a is not None: z=z[z.trade_date>=pd.Timestamp(a)]
    if b is not None: z=z[z.trade_date<=pd.Timestamp(b)]
    if len(z)<2:return np.nan
    y=max((z.trade_date.iloc[-1]-z.trade_date.iloc[0]).days/365.25,1e-9); r=float(z.equity.iloc[-1]/z.equity.iloc[0]); return float(r**(1/y)-1)
def eq_mdd(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); return float((s/s.cummax()-1).min())


def phase_run(q,h,n,e,k,cal,members,bm,start,end,total_cash=1e6,cost=1.,phases=None):
    pc=phase_count(h); phs=list(range(pc)) if phases is None else list(phases); per=float(total_cash)/len(phs)
    eqs=[]; stats=[]; tms=[]
    for ph in phs:
        st,eq,tr,tm=run_one(q,h,ph,n,e,k,cal,members,bm,per,start,end,cost); eqs.append(eq); stats.append(st)
        if len(tm): tms.append(tm.assign(phase=ph))
    ens=combine_abs(eqs,[per]*len(phs),start)
    return ens,stats,tms


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=lp.attach_price(p,cal); p=strict.attach_gap_flags(p,cal,'board')
    q=lp.rank_signal(p,WEIGHTS,LIQ,FLOOR)

    # Stage 1: coarse, predeclared 4-phase screen. 60 construction configs only.
    rows=[]
    for h in HOLDS:
      phs=screen_phases(h)
      for n in NS:
       for e,k in BUFFERS:
        vals=[]
        for ph in phs:
            st,eq,tr,tm=run_one(q,h,ph,n,e,k,cal,members,bm,1e6,START,TRAIN_END,1.); vals.append(st)
        rr=np.array([x['total_return'] for x in vals],float); cg=np.array([x['cagr'] for x in vals],float); md=np.array([x['max_drawdown'] for x in vals],float)
        score=float(np.median(cg)+.50*np.min(cg)-.25*np.std(cg))
        rows.append({'hold':h,'n_hold':n,'entry':e,'keep':k,'screen_phases':'|'.join(map(str,phs)),'screen_min_return':rr.min(),'screen_median_return':np.median(rr),'screen_min_cagr':cg.min(),'screen_median_cagr':np.median(cg),'screen_std_cagr':cg.std(),'screen_worst_mdd':md.min(),'screen_all_positive':int((rr>0).all()),'screen_score':score})
        print('SCREEN',h,n,e,k,score,flush=True)
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'screen_grid.csv',index=False)
    eligible=grid[(grid.screen_all_positive==1)&(grid.screen_worst_mdd>-0.55)].copy(); eligible=eligible if len(eligible) else grid.copy()
    top=eligible.sort_values(['screen_score','screen_min_cagr'],ascending=False).head(20)

    # Stage 2: exact total RMB1m all-phase train test. Time-block gates are predeclared.
    exact=[]
    for r in top.itertuples(index=False):
        ens,sts,tms=phase_run(q,r.hold,r.n_hold,r.entry,r.keep,cal,members,bm,START,TRAIN_END,1e6,1.)
        pr=np.array([x['total_return'] for x in sts],float); pcg=np.array([x['cagr'] for x in sts],float); pm=np.array([x['max_drawdown'] for x in sts],float)
        h1r=eq_return(ens,START,'2018-12-31'); h2r=eq_return(ens,'2019-01-01',TRAIN_END)
        h1c=eq_cagr(ens,START,'2018-12-31'); h2c=eq_cagr(ens,'2019-01-01',TRAIN_END)
        ec=eq_cagr(ens); er=eq_return(ens); em=eq_mdd(ens)
        hard_pass=int((pr>0).all() and h1r>0 and h2r>0 and em>-0.50)
        exact.append({**r._asdict(),'allphase_count':len(sts),'exact_split_train_return':er,'exact_split_train_cagr':ec,'exact_split_train_mdd':em,'allphase_min_return':pr.min(),'allphase_median_return':np.median(pr),'allphase_min_cagr':pcg.min(),'allphase_median_cagr':np.median(pcg),'allphase_std_cagr':pcg.std(),'allphase_worst_mdd':pm.min(),'half1_2016_2018_return':h1r,'half1_cagr':h1c,'half2_2019_2021_return':h2r,'half2_cagr':h2c,'hard_robust_pass':hard_pass})
        print('EXACT',r.hold,r.n_hold,r.entry,r.keep,'cagr',ec,'minphase',pcg.min(),'halves',h1c,h2c,'pass',hard_pass,flush=True)
    ex=pd.DataFrame(exact); ex.to_csv(OUT/'exact_train_top20.csv',index=False)
    passed=ex[ex.hard_robust_pass==1].copy(); passed=passed if len(passed) else ex.copy()
    # Selection rule deliberately simple: after hard robustness gates, maximize exact train ensemble CAGR.
    win=passed.sort_values(['exact_split_train_cagr','allphase_min_cagr','allphase_std_cagr'],ascending=[False,False,True]).iloc[0].to_dict()
    pd.DataFrame([win]).to_csv(OUT/'train_only_winner.csv',index=False)

    # Freeze before touching 2022-2026.
    h=int(win['hold']); n=int(win['n_hold']); e=float(win['entry']); k=float(win['keep'])
    full_eq,full_sts,full_tms=phase_run(q,h,n,e,k,cal,members,bm,START,END,1e6,1.)
    full=fa.perf_eq(full_eq,bm); full['train_2016_2021_return']=fa.period_return(full_eq,START,TRAIN_END); full['pseudo_oos_2022_2026_return']=fa.period_return(full_eq,PSEUDO,END)
    full.update({'hold':h,'n_hold':n,'entry':e,'keep':k,'weights':json.dumps(WEIGHTS,sort_keys=True),'liq':LIQ,'floor':FLOOR,'selection_validation_accesses':0,'selection_rule':'all phases positive + both train halves positive + ensemble MDD>-50%; then max train exact-split CAGR'})
    pd.DataFrame([full]).to_csv(OUT/'frozen_full_validation.csv',index=False); fa.annual(full_eq).to_csv(OUT/'annual.csv',index=False)

    ph=[]
    for i,st in enumerate(full_sts): ph.append({**st,'phase':i})
    pd.DataFrame(ph).to_csv(OUT/'full_phase_diagnostics.csv',index=False)

    costrows=[]
    for cm in (2.,4.,8.):
        ceq,csts,ctm=phase_run(q,h,n,e,k,cal,members,bm,START,END,1e6,cm); cs=fa.perf_eq(ceq,bm); cs['cost_mult']=cm; cs['pseudo_return']=fa.period_return(ceq,PSEUDO,END); costrows.append(cs)
    pd.DataFrame(costrows).to_csv(OUT/'cost_stress.csv',index=False)

    alltm=pd.concat(full_tms,ignore_index=True) if full_tms else pd.DataFrame(); bad=int((pd.to_datetime(alltm.signal_date)>=pd.to_datetime(alltm.trade_date)).sum()) if len(alltm) else 0
    if bad: raise RuntimeError(f'timing violations {bad}')
    pd.DataFrame([{**ua,'market_factor':market_code,'experiment':'LowPrice Concentration/Horizon V4','weights':json.dumps(WEIGHTS,sort_keys=True),'liq':LIQ,'floor':FLOOR,'holds':'|'.join(map(str,HOLDS)),'n_grid':'|'.join(map(str,NS)),'buffers':'5/20|10/30|15/40','construction_configs':len(grid),'exact_finalists':len(ex),'selection_period':'2016-07-29..2021-12-31','validation_2022_2026_accesses_before_freeze':0,'robust_gates':'all-phase positive; 2016-18 positive; 2019-21 positive; exact-split MDD>-50%','selection_after_gates':'max exact-split train CAGR','executor':'hard_v3; T-close -> later open; 100-share lots; board-limit block; no replacement','timing_violations':bad}]).to_csv(OUT/'audit.csv',index=False)
    print('WINNER TRAIN',pd.DataFrame([win]).to_string(index=False),flush=True); print('FROZEN FULL',pd.DataFrame([full]).to_string(index=False),flush=True); print('COSTS',pd.DataFrame(costrows).to_string(index=False),flush=True)

if __name__=='__main__': main()
