from __future__ import annotations
from pathlib import Path
import json, gc, numpy as np, pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_lowprice_signalpure_v1 as lp
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_max_audit as ma
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_structural_ensemble_v4'); OUT.mkdir(exist_ok=True)
START=mo.START; TRAIN_END=mo.TRAIN_END; PSEUDO=mo.PSEUDO_START; END=mo.END
PH={'price':.35,'iv':.18,'ef':.15,'rmom':.20,'tstat':.12}
MOM={'price':.20,'iv':.15,'ef':.15,'rmom':.32,'tstat':.18}
LIQ=.55; FLOOR=2.0
HOLDS=(60,90,120); NS=(8,10,12); BUFFERS=((.05,.20),(.10,.30))
BASECOLS=strict.BASECOLS+['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy']
SIGNALS=('ph','mom','ph_mom_avg','ph_mom_worst','ph_smooth','ph_mom_avg_smooth','ph_long_avg','triple_avg')


def combine_rank(p, qs, mode='mean'):
    x=p.copy(deep=False); x['rank_test']=np.nan
    A=pd.concat([q.rank_test.rename(str(i)) for i,q in enumerate(qs)],axis=1)
    m=A.notna().all(axis=1)
    if not m.any(): return x
    raw=A.loc[m].mean(axis=1) if mode=='mean' else A.loc[m].max(axis=1)
    x.loc[m,'rank_test']=raw.groupby(x.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return x


def smooth_rank(p,q):
    x=p.copy(deep=False); x['rank_test']=np.nan
    z=q[['signal_date','code','rank_test']].copy().sort_values(['code','signal_date'])
    z['prev']=z.groupby('code',sort=False).rank_test.shift(1)
    m=np.isfinite(z.rank_test)&np.isfinite(z.prev)
    raw=.5*z.loc[m,'rank_test']+.5*z.loc[m,'prev']
    idx=z.index[m]
    x.loc[idx,'rank_test']=raw.groupby(x.loc[idx,'signal_date']).rank(pct=True,method='average',ascending=True)
    return x


def phase_count(h): return max(1,round(int(h)/5))
def screen_phases(h):
    n=phase_count(h); return sorted(set(int(round(v))%n for v in np.linspace(0,n-1,4)))

def subset(q,h,ph):
    pc=phase_count(h); dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[int(ph)::pc])
    cols=[c for c in BASECOLS if c in q.columns]; z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def run_one(q,h,ph,n,e,k,cal,members,bm,cash,start,end,cost=1.):
    return ma.run_panel(subset(q,h,ph),cal,members,bm,n=int(n),entry=float(e),keep=float(k),initial_cash=float(cash),start=pd.Timestamp(start),end=pd.Timestamp(end),cost=float(cost))

def combine_abs(eqs,initials,start):
    start=pd.Timestamp(start); idx={start}; ss=[]
    for eq,init in zip(eqs,initials):
        s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]
        s=pd.concat([pd.Series({start:float(init)}),s]); s=s[~s.index.duplicated(keep='last')].sort_index(); ss.append(s); idx.update(s.index)
    idx=pd.DatetimeIndex(sorted(idx)); arr=[s.reindex(idx).ffill().fillna(float(init)) for s,init in zip(ss,initials)]
    return pd.DataFrame({'trade_date':idx,'equity':pd.concat(arr,axis=1).sum(axis=1).to_numpy(float)})
def eq_return(eq,a=None,b=None):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date)
    if a is not None:z=z[z.trade_date>=pd.Timestamp(a)]
    if b is not None:z=z[z.trade_date<=pd.Timestamp(b)]
    return float(z.equity.iloc[-1]/z.equity.iloc[0]-1) if len(z)>1 else np.nan
def eq_cagr(eq,a=None,b=None):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date)
    if a is not None:z=z[z.trade_date>=pd.Timestamp(a)]
    if b is not None:z=z[z.trade_date<=pd.Timestamp(b)]
    if len(z)<2:return np.nan
    y=max((z.trade_date.iloc[-1]-z.trade_date.iloc[0]).days/365.25,1e-9); return float((z.equity.iloc[-1]/z.equity.iloc[0])**(1/y)-1)
def eq_mdd(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index(); return float((s/s.cummax()-1).min())

def eval_allphases(q,h,n,e,k,cal,members,bm,start,end,total_cash=1e6,cost=1.,phases=None):
    phs=list(range(phase_count(h))) if phases is None else list(phases); per=float(total_cash)/len(phs); eqs=[]; sts=[]; tms=[]
    for ph in phs:
        st,eq,tr,tm=run_one(q,h,ph,n,e,k,cal,members,bm,per,start,end,cost); eqs.append(eq); sts.append(st)
        if len(tm):tms.append(tm.assign(phase=ph))
    return combine_abs(eqs,[per]*len(phs),start),sts,tms

def robust_row(eq,sts):
    rr=np.array([s['total_return'] for s in sts],float); cg=np.array([s['cagr'] for s in sts],float); md=np.array([s['max_drawdown'] for s in sts],float)
    h1=eq_cagr(eq,START,'2018-12-31'); h2=eq_cagr(eq,'2019-01-01',TRAIN_END)
    ec=eq_cagr(eq); em=eq_mdd(eq)
    score=float(ec+.50*np.nanmin(cg)-.25*np.nanstd(cg)+.25*min(h1,h2))
    return {'ensemble_return':eq_return(eq),'ensemble_cagr':ec,'ensemble_mdd':em,'min_phase_return':np.nanmin(rr),'median_phase_return':np.nanmedian(rr),'min_phase_cagr':np.nanmin(cg),'median_phase_cagr':np.nanmedian(cg),'std_phase_cagr':np.nanstd(cg),'worst_phase_mdd':np.nanmin(md),'half1_cagr':h1,'half2_cagr':h2,'all_phases_positive':int((rr>0).all()),'robust_score':score,'hard_pass':int((rr>0).all() and h1>0 and h2>0 and em>-0.50)}


def make_signals(p):
    qph=lp.rank_signal(p,PH,LIQ,FLOOR); qmom=lp.rank_signal(p,MOM,LIQ,FLOOR); qlong=mo.rerank(p,mo.baseline_spec(.54))
    qavg=combine_rank(p,[qph,qmom],'mean'); qworst=combine_rank(p,[qph,qmom],'max')
    out={
      'ph':qph,
      'mom':qmom,
      'ph_mom_avg':qavg,
      'ph_mom_worst':qworst,
      'ph_smooth':smooth_rank(p,qph),
      'ph_mom_avg_smooth':smooth_rank(p,qavg),
      'ph_long_avg':combine_rank(p,[qph,qlong],'mean'),
      'triple_avg':combine_rank(p,[qph,qmom,qlong],'mean'),
    }
    return out,qlong


def dual_portfolio(qph,qlong,ph_share,cal,members,bm,start,end,cost=1.):
    # fixed constructions from prior frozen families; only capital split is varied among two predeclared values
    eq1,st1,tm1=eval_allphases(qph,60,8,.10,.30,cal,members,bm,start,end,1e6*ph_share,cost)
    eq2,st2,tm2=eval_allphases(qlong,120,8,.05,.20,cal,members,bm,start,end,1e6*(1-ph_share),cost)
    eq=combine_abs([eq1,eq2],[1e6*ph_share,1e6*(1-ph_share)],start)
    # phase-level robustness is reported sleeve-by-sleeve; global score uses ensemble plus time blocks
    allsts=st1+st2; row=robust_row(eq,allsts); row['dual_ph_share']=ph_share; return eq,row,tm1+tm2


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=lp.attach_price(p,cal); p=strict.attach_gap_flags(p,cal,'board')
    sigs,qlong=make_signals(p)

    # Stage 1: limited architecture x construction screen, four fixed phases only.
    scr=[]
    for name,q in sigs.items():
      for h in HOLDS:
       phs=screen_phases(h)
       for n in NS:
        for e,k in BUFFERS:
         vals=[]
         for ph in phs:
            st,eq,tr,tm=run_one(q,h,ph,n,e,k,cal,members,bm,1e6,START,TRAIN_END,1.); vals.append(st)
         rr=np.array([x['total_return'] for x in vals],float); cg=np.array([x['cagr'] for x in vals],float); md=np.array([x['max_drawdown'] for x in vals],float)
         score=float(np.median(cg)+.50*np.min(cg)-.25*np.std(cg))
         scr.append({'architecture':name,'hold':h,'n_hold':n,'entry':e,'keep':k,'screen_phases':'|'.join(map(str,phs)),'screen_min_return':rr.min(),'screen_median_return':np.median(rr),'screen_min_cagr':cg.min(),'screen_median_cagr':np.median(cg),'screen_std_cagr':cg.std(),'screen_worst_mdd':md.min(),'screen_all_positive':int((rr>0).all()),'screen_score':score})
         print('SCREEN',name,h,n,e,k,score,flush=True)
    grid=pd.DataFrame(scr); grid.to_csv(OUT/'architecture_screen.csv',index=False)
    g=grid[(grid.screen_all_positive==1)&(grid.screen_worst_mdd>-0.55)].copy(); g=g if len(g) else grid.copy()
    top=g.sort_values(['screen_score','screen_min_cagr'],ascending=False).head(20)

    # Stage 2: all legal phases, exact RMB1m split, plus two disjoint train-time blocks.
    ex=[]
    for r in top.itertuples(index=False):
        q=sigs[r.architecture]; eq,sts,tms=eval_allphases(q,r.hold,r.n_hold,r.entry,r.keep,cal,members,bm,START,TRAIN_END,1e6,1.)
        row=robust_row(eq,sts); ex.append({**r._asdict(),**row,'candidate_type':'rank_arch'})
        print('EXACT',r.architecture,r.hold,r.n_hold,r.entry,r.keep,row['robust_score'],row['ensemble_cagr'],row['min_phase_cagr'],row['half1_cagr'],row['half2_cagr'],flush=True)

    # Two predeclared dual-timescale portfolios. No continuous allocation optimization.
    qph=sigs['ph']
    for sh in (.50,.75):
        eq,row,tms=dual_portfolio(qph,qlong,sh,cal,members,bm,START,TRAIN_END,1.)
        ex.append({'architecture':f'dual_ph{int(sh*100)}_long{int((1-sh)*100)}','hold':-1,'n_hold':-1,'entry':np.nan,'keep':np.nan,**row,'candidate_type':'dual_timescale'})
        print('DUAL TRAIN',sh,row,flush=True)

    exdf=pd.DataFrame(ex); exdf.to_csv(OUT/'exact_train_candidates.csv',index=False)
    passed=exdf[exdf.hard_pass==1].copy(); passed=passed if len(passed) else exdf.copy()
    win=passed.sort_values(['robust_score','ensemble_cagr','min_phase_cagr'],ascending=False).iloc[0].to_dict(); pd.DataFrame([win]).to_csv(OUT/'train_only_winner.csv',index=False)

    # Freeze winner, then and only then open the full period.
    if win['candidate_type']=='dual_timescale':
        sh=float(win['dual_ph_share']); feq,frow,ftms=dual_portfolio(qph,qlong,sh,cal,members,bm,START,END,1.)
        full=fa.perf_eq(feq,bm); full.update({'architecture':win['architecture'],'candidate_type':'dual_timescale','dual_ph_share':sh})
    else:
        q=sigs[str(win['architecture'])]; h=int(win['hold']); n=int(win['n_hold']); e=float(win['entry']); k=float(win['keep'])
        feq,fsts,ftms=eval_allphases(q,h,n,e,k,cal,members,bm,START,END,1e6,1.)
        full=fa.perf_eq(feq,bm); full.update({'architecture':win['architecture'],'candidate_type':'rank_arch','hold':h,'n_hold':n,'entry':e,'keep':k})
    full['train_2016_2021_return']=fa.period_return(feq,START,TRAIN_END); full['pseudo_oos_2022_2026_return']=fa.period_return(feq,PSEUDO,END); full['selection_validation_accesses']=0
    pd.DataFrame([full]).to_csv(OUT/'frozen_full_validation.csv',index=False); fa.annual(feq).to_csv(OUT/'annual.csv',index=False)

    # Cost stress only after freeze; it cannot affect selection.
    costs=[]
    for cm in (2.,4.,8.):
      if win['candidate_type']=='dual_timescale':
        ceq,_,_=dual_portfolio(qph,qlong,float(win['dual_ph_share']),cal,members,bm,START,END,cm)
      else:
        ceq,_,_=eval_allphases(sigs[str(win['architecture'])],int(win['hold']),int(win['n_hold']),float(win['entry']),float(win['keep']),cal,members,bm,START,END,1e6,cm)
      cs=fa.perf_eq(ceq,bm); cs['cost_mult']=cm; cs['pseudo_return']=fa.period_return(ceq,PSEUDO,END); costs.append(cs)
    pd.DataFrame(costs).to_csv(OUT/'cost_stress.csv',index=False)

    pd.DataFrame([{**ua,'market_factor':market_code,'experiment':'Structural Ensemble V4 isolated','architectures':'|'.join(SIGNALS)+'|dual50|dual75','construction_grid':'rank architectures: H60/90/120; N8/10/12; buffers5/20,10/30; dual uses frozen H60 lowprice + H120 long-risk','selection_period':'2016-07-29..2021-12-31','validation_2022_2026_accesses_before_freeze':0,'stage1':'4 fixed phases; no pseudo','stage2':'all legal phases exact RMB1m split + 2016-18 and 2019-21 positive gates','objective':'ensemble CAGR + .50 worst phase CAGR - .25 phase dispersion + .25 weaker train-half CAGR','hard_gates':'all phases positive; both train halves positive; ensemble MDD>-50%','executor':'hard_v3; T close -> later open; 100-share lots; board-limit block; no replacement','candidate_count_screen':len(grid),'candidate_count_exact':len(exdf)}]).to_csv(OUT/'audit.csv',index=False)
    print('TRAIN WINNER'); print(pd.DataFrame([win]).to_string(index=False),flush=True); print('FROZEN FULL'); print(pd.DataFrame([full]).to_string(index=False),flush=True); print('COSTS'); print(pd.DataFrame(costs).to_string(index=False),flush=True)

if __name__=='__main__': main()
