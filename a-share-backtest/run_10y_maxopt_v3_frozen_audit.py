from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_max_audit as ma

OUT=Path('results_maxopt_v3_frozen_audit'); OUT.mkdir(exist_ok=True)
SEED=20260823
CANDIDATES={
 'baseline':dict(w=.60,n=20,hold=60,entry=.10,keep=.30,selection='pre-existing frozen baseline'),
 'short_growth':dict(w=.50,n=8,hold=40,entry=.05,keep=.20,selection='surface-short unique train growth+maximin winner; 2016-2021 only'),
 'long_growth':dict(w=.50,n=10,hold=120,entry=.10,keep=.30,selection='surface-long training growth finalist; 2016-2021 only'),
 'long_robust':dict(w=.50,n=20,hold=120,entry=.10,keep=.30,selection='surface-long training maximin finalist; 2016-2021 only'),
}


def perf_eq(eq,bm=None):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index()
    s=s[~s.index.duplicated(keep='last')]
    r=s.pct_change().dropna(); yrs=max((s.index[-1]-s.index[0]).days/365.25,1e-9)
    total=float(s.iloc[-1]/s.iloc[0]-1); cagr=float((s.iloc[-1]/s.iloc[0])**(1/yrs)-1)
    dd=s/s.cummax()-1; sh=float(r.mean()/r.std()*np.sqrt(252)) if len(r)>2 and r.std()>0 else np.nan
    dn=r[r<0]; sortino=float(r.mean()/dn.std()*np.sqrt(252)) if len(dn)>2 and dn.std()>0 else np.nan
    out=dict(final_asset=float(s.iloc[-1]),total_return=total,cagr=cagr,max_drawdown=float(dd.min()),sharpe=sh,sortino=sortino)
    if bm is not None:
        br=bm.pct_change(fill_method=None).reindex(r.index).dropna(); rr=r.reindex(br.index).dropna(); br=br.reindex(rr.index)
        if len(rr)>20 and br.var()>0:
            beta=float(rr.cov(br)/br.var()); alpha=float((rr.mean()-beta*br.mean())*252); out.update(capm_beta=beta,annualized_capm_alpha=alpha)
    return out


def period_return(eq,a,b):
    z=eq[(pd.to_datetime(eq.trade_date)>=pd.Timestamp(a))&(pd.to_datetime(eq.trade_date)<=pd.Timestamp(b))]
    if len(z)<2:return np.nan
    return float(z.equity.iloc[-1]/z.equity.iloc[0]-1)


def annual(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index()
    s=s[~s.index.duplicated(keep='last')]
    rows=[]
    for y,g in s.groupby(s.index.year):
        before=s[s.index<pd.Timestamp(f'{y}-01-01')]
        start=float(before.iloc[-1]) if len(before) else float(g.iloc[0])
        rows.append({'year':int(y),'return':float(g.iloc[-1]/start-1)})
    return pd.DataFrame(rows)


def phase_ensemble(eqs):
    idx=pd.DatetimeIndex(sorted(set().union(*[set(pd.to_datetime(e.trade_date)) for e in eqs])))
    arr=[]
    for e in eqs:
        s=e.set_index(pd.to_datetime(e.trade_date)).equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]
        x=(s/sim.INITIAL_CASH).reindex(idx).ffill().fillna(1.0); arr.append(x)
    z=pd.concat(arr,axis=1).mean(axis=1)*sim.INITIAL_CASH
    return pd.DataFrame({'trade_date':idx,'equity':z.to_numpy(float)})


def delay_panel(q,hold,phase,delay,cal,members):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(hold/5)); chosen=list(dates[phase::step])
    z=ma.minimal(q[q.signal_date.isin(chosen)]).copy(); chosen_set=set(pd.Timestamp(x) for x in chosen)
    target={}
    for d in chosen_set:
        k=cal.searchsorted(d,side='right')+(int(delay)-1); target[d]=cal[k] if k<len(cal) else pd.NaT
    z['trade_date']=pd.to_datetime(z.signal_date).map(target)
    for c in ['exec_open','exec_high','exec_low','exec_volume','exec_factor']: z[c]=np.nan
    first=members.groupby('code').start.min().to_dict(); last=members.groupby('code').end.max().to_dict()
    for i,(code,idxs) in enumerate(z.groupby('code').groups.items(),1):
        idxs=np.asarray(list(idxs)); ds=pd.DatetimeIndex(z.loc[idxs,'trade_date'])
        for fld,out in [('open','exec_open'),('high','exec_high'),('low','exec_low'),('volume','exec_volume'),('factor','exec_factor')]:
            s=base.qb.read_bin(code,fld,cal)
            if fld=='factor' and s.empty: z.loc[idxs,out]=1.0
            elif not s.empty: z.loc[idxs,out]=s.reindex(ds).to_numpy(float)
        active=(ds>=pd.Timestamp(first.get(code,'1900-01-01')))&(ds<=pd.Timestamp(last.get(code,'2100-01-01')))
        bad=np.array(~active)|pd.isna(ds)
        if bad.any(): z.loc[idxs[bad],['exec_open','exec_high','exec_low','exec_volume']]=np.nan
    z['exec_factor']=z.exec_factor.replace(0,np.nan).fillna(1.0)
    # Critical: retain rows with missing future quote. Ranking universe was fixed on T; execution simply skips untradeable names.
    z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')


def run_delayed(q,cfg,delay,cal,members,bm):
    z=delay_panel(q,cfg['hold'],0,delay,cal,members)
    st,eq,tr,tm=ma.run_panel(z,cal,members,bm,n=cfg['n'],entry=cfg['entry'],keep=cfg['keep'])
    st['delay_sessions']=delay; return st,eq,tr,tm


def noisy_q(q,sigma,rng):
    x=q.copy(); m=np.isfinite(x.rank_test.to_numpy(float)); v=x.rank_test.to_numpy(float)
    v[m]=np.clip(v[m]+rng.normal(0,float(sigma),m.sum()),0,1); x['rank_test']=v
    x.loc[m,'rank_test']=x.loc[m].groupby('signal_date').rank_test.rank(pct=True,method='average')
    return x


def random_q(q,rng):
    x=q.copy(); m=np.isfinite(x.rank_test.to_numpy(float)); v=np.full(len(x),np.nan); v[m]=rng.random(m.sum()); x['rank_test']=v; return x


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    qs={name:mo.rerank(p,mo.baseline_spec(cfg['w'])) for name,cfg in CANDIDATES.items()}
    phase0=[]; phases=[]; ensembles=[]; ensemble_annual=[]; costs=[]; ensemble_costs=[]; capacity=[]; tails=[]; delays=[]; noise=[]; deletes=[]; placebo=[]; equity0={}; ens_eq={}
    rng=np.random.default_rng(SEED); codes=np.array(sorted(p.code.unique()))

    for name,cfg in CANDIDATES.items():
        q=qs[name]; print('CANDIDATE',name,cfg,flush=True)
        st,eq,tr,tm=mo.full_run(q,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm)
        st.update(candidate=name,selection=cfg['selection']); phase0.append(st); equity0[name]=eq
        rr=sim.robustness(eq,tr); rr['candidate']=name; tails.append(rr)
        for cm in (2.,4.,8.):
            x,_,_,_=mo.full_run(q,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm,cost=cm); x.update(candidate=name,cost_mult_test=cm); costs.append(x)
        for cash in (1e7,5e7,1e8):
            for vp in (.01,.05):
                x,_,_,_=mo.full_run(q,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm,initial_cash=cash,vol_part=vp); x.update(candidate=name,cash_test=cash,vp_test=vp); capacity.append(x)
        pe=[]; step=max(1,round(cfg['hold']/5))
        for ph in range(step):
            x,e,_,_=mo.full_run(q,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm,phase=ph); x.update(candidate=name,phase=ph); phases.append(x); pe.append(e)
        ee=phase_ensemble(pe); ens_eq[name]=ee; es=perf_eq(ee,bm); es.update(candidate=name,phase_count=step,train_2016_2021_return=period_return(ee,mo.START,mo.TRAIN_END),pseudo_oos_2022_2026_return=period_return(ee,mo.PSEUDO_START,mo.END)); ensembles.append(es)
        a=annual(ee); a['candidate']=name; ensemble_annual.append(a)
        if name in ('short_growth','long_growth'):
            for cm in (4.,8.):
                ecm=[]
                for ph in range(step):
                    _,e,_,_=mo.full_run(q,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm,phase=ph,cost=cm); ecm.append(e)
                ex=phase_ensemble(ecm); xs=perf_eq(ex,bm); xs.update(candidate=name,cost_mult_test=cm); ensemble_costs.append(xs)
            for d in (1,3,5):
                x,_,_,_=run_delayed(q,cfg,d,cal,members,bm); x['candidate']=name; delays.append(x)
            for sig in (.02,.05,.10):
                for rep in range(10):
                    nq=noisy_q(q,sig,rng); x,_,_,_=mo.full_run(nq,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm); x.update(candidate=name,noise_sigma=sig,rep=rep); noise.append(x)
            for rep in range(15):
                drop=set(rng.choice(codes,size=int(.20*len(codes)),replace=False).tolist()); x,_,_,_=mo.full_run(q[~q.code.isin(drop)],cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm); x.update(candidate=name,deleted_share=.20,rep=rep); deletes.append(x)
            for rep in range(30):
                pq=random_q(q,rng); x,_,_,_=mo.full_run(pq,cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm); x.update(candidate=name,rep=rep); placebo.append(x)

    pd.DataFrame(phase0).to_csv(OUT/'phase0_summary.csv',index=False)
    pd.DataFrame(phases).to_csv(OUT/'phase_offsets.csv',index=False)
    pd.DataFrame(ensembles).to_csv(OUT/'phase_ensemble.csv',index=False)
    pd.concat(ensemble_annual,ignore_index=True).to_csv(OUT/'phase_ensemble_annual.csv',index=False)
    pd.DataFrame(costs).to_csv(OUT/'phase0_costs.csv',index=False)
    pd.DataFrame(ensemble_costs).to_csv(OUT/'phase_ensemble_costs.csv',index=False)
    pd.DataFrame(capacity).to_csv(OUT/'capacity.csv',index=False)
    pd.DataFrame(tails).to_csv(OUT/'tail.csv',index=False)
    pd.DataFrame(delays).to_csv(OUT/'delays.csv',index=False)
    pd.DataFrame(noise).to_csv(OUT/'rank_noise.csv',index=False)
    pd.DataFrame(deletes).to_csv(OUT/'random_delete20.csv',index=False)
    pd.DataFrame(placebo).to_csv(OUT/'placebo.csv',index=False)

    # Equal-capital sleeve mixes, no performance-based timing or reallocation.
    mixes=[]
    for other in ('short_growth','long_growth','long_robust'):
        for source,series in [('phase0',equity0),('phase_ensemble',ens_eq)]:
            e=phase_ensemble([series['baseline'],series[other]]); st=perf_eq(e,bm); st.update(mix=f'baseline50_{other}50',source=source,train_2016_2021_return=period_return(e,mo.START,mo.TRAIN_END),pseudo_oos_2022_2026_return=period_return(e,mo.PSEUDO_START,mo.END)); mixes.append(st)
    pd.DataFrame(mixes).to_csv(OUT/'mixes.csv',index=False)

    ph=pd.DataFrame(phases); en=pd.DataFrame(ensembles); nr=pd.DataFrame(noise); dr=pd.DataFrame(deletes); pl=pd.DataFrame(placebo)
    gates=[]
    for name in ('short_growth','long_growth'):
        b=next(x for x in phase0 if x['candidate']==name); e=en[en.candidate==name].iloc[0]
        psub=ph[ph.candidate==name]
        nsub=nr[nr.candidate==name]; dsub=dr[dr.candidate==name]; rsub=pl[pl.candidate==name]
        obs=float(b['total_return']); pp=float((rsub.total_return>=obs).mean()) if len(rsub) else np.nan
        gates.append({'candidate':name,'phase0_beats_old_baseline':int(obs>1.749407),'all_phases_positive':int((psub.total_return>0).all()),'phase_ensemble_positive':int(e.total_return>0),'phase_ensemble_beats_old_baseline':int(e.total_return>1.749407),'noise10_median_positive':int(nsub[nsub.noise_sigma==.10].total_return.median()>0),'delete20_all_positive':int((dsub.total_return>0).all()),'placebo_p_ge_observed':pp,'placebo_pass_5pct':int(pp<=.05)})
    pd.DataFrame(gates).to_csv(OUT/'gates.csv',index=False)
    alltm=[]
    for name,cfg in CANDIDATES.items():
        _,_,_,tm=mo.full_run(qs[name],cfg['hold'],cfg['n'],cfg['entry'],cfg['keep'],cal,members,bm); 
        if len(tm): alltm.append(tm.assign(candidate=name))
    bad=int((pd.to_datetime(pd.concat(alltm).signal_date)>=pd.to_datetime(pd.concat(alltm).trade_date)).sum()) if alltm else 0
    audit={**ua,'market_factor':market_code,'research_round':'MaxOpt V3 frozen finalist audit','candidates':'|'.join(CANDIDATES),'selection_lock':'candidate formulas and construction fixed before this audit; 2022-2026 not used to retune','signal_universe':'T-only signal-pure; missing later quote retained until execution','volume_unit_shares':100,'timing_violations':bad,'random_seed':SEED}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    print('=== PHASE0 ==='); print(pd.DataFrame(phase0).to_string(index=False),flush=True)
    print('=== ENSEMBLES ==='); print(pd.DataFrame(ensembles).to_string(index=False),flush=True)
    print('=== GATES ==='); print(pd.DataFrame(gates).to_string(index=False),flush=True)
    print('=== MIXES ==='); print(pd.DataFrame(mixes).to_string(index=False),flush=True)
    if bad: raise RuntimeError('timing violation')

if __name__=='__main__': main()
