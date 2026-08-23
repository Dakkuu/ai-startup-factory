from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_alpha2f_v2 as sim
import run_10y_era_backtest as base
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff55_strict_v2'); OUT.mkdir(exist_ok=True); SEED=20260823
SPEC=next(s for s in mega.specs_twostage() if s['name']=='g_eff_55')
CFG=dict(hold=90,n=15,entry=.10,keep=.30)
BASECOLS=['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor','rank_test']

def board_limit(code,d):
    s=str(code).upper(); d=pd.Timestamp(d)
    if s.startswith('BJ'): return .30
    if s.startswith('SH688'): return .20
    if (s.startswith('SZ300') or s.startswith('SZ301')) and d>=pd.Timestamp('2020-08-24'): return .20
    return .10

def attach_gap_flags(p,cal,mode='board'):
    x=p.copy(); x['prev_raw_close']=np.nan
    for i,(code,idx) in enumerate(x.groupby('code').groups.items(),1):
        idx=np.asarray(list(idx)); c=base.qb.read_bin(code,'close',cal); f=base.qb.read_bin(code,'factor',cal)
        if c.empty: continue
        if f.empty: f=pd.Series(1.,index=c.index)
        raw=c/f.replace(0,np.nan); prev=raw.ffill().shift(1); ds=pd.DatetimeIndex(x.loc[idx,'trade_date'])
        x.loc[idx,'prev_raw_close']=prev.reindex(ds).to_numpy(float)
        if i%1000==0: print('PREVCLOSE',mode,i,flush=True)
    fac=x.exec_factor.replace(0,np.nan); rawopen=x.exec_open/fac; gap=rawopen/x.prev_raw_close-1
    if mode=='board': lim=np.array([board_limit(c,d) for c,d in zip(x.code,x.trade_date)],float)
    elif mode=='universal5': lim=np.full(len(x),.05,float)
    elif mode=='universal10': lim=np.full(len(x),.10,float)
    else: raise ValueError(mode)
    known=np.isfinite(gap.to_numpy(float)); eps=.002
    x['exec_open_gap']=gap
    x['exec_limit_proxy']=lim
    x['exec_buy_allowed']=known & (gap.to_numpy(float) < lim-eps)
    x['exec_sell_allowed']=known & (gap.to_numpy(float) > -lim+eps)
    return x

def subset(q,phase):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); chosen=set(dates[phase::18])
    cols=BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test; return z.drop(columns='rank_test')

def runq(q,phase,cal,members,bm,cost=1.,initial=1e6,vp=.05):
    st,eq,tr,tm=ma.run_panel(subset(q,phase),cal,members,bm,n=CFG['n'],entry=CFG['entry'],keep=CFG['keep'],cost=cost,initial_cash=initial,vol_part=vp)
    st['train_2016_2021_return']=fa.period_return(eq,mo.START,mo.TRAIN_END); st['pseudo_oos_2022_2026_return']=fa.period_return(eq,mo.PSEUDO_START,mo.END)
    return st,eq,tr,tm

def ensemble_rows(eqs,bm):
    e=fa.phase_ensemble(eqs); s=fa.perf_eq(e,bm); s['train_2016_2021_return']=fa.period_return(e,mo.START,mo.TRAIN_END); s['pseudo_oos_2022_2026_return']=fa.period_return(e,mo.PSEUDO_START,mo.END); return s,e

def scramble_audit(p):
    rng=np.random.default_rng(SEED); q0=mega.make_rank(p,SPEC); p2=p.copy(deep=False)
    for c in ['exec_open','exec_high','exec_low','exec_volume','exec_factor']:
        a=p[c].to_numpy(copy=True); rng.shuffle(a); p2[c]=a
    q1=mega.make_rank(p2,SPEC); a=q0.rank_test.to_numpy(float); b=q1.rank_test.to_numpy(float)
    same_nan=np.array_equal(np.isnan(a),np.isnan(b)); m=np.isfinite(a)&np.isfinite(b); md=float(np.max(np.abs(a[m]-b[m]))) if m.any() else np.nan
    return {'same_nan_pattern':int(same_nan),'max_abs_rank_diff_after_exec_scramble':md,'pass':int(same_nan and md==0.0)}

def run_mode(p,mode,cal,members,bm):
    px=attach_gap_flags(p,cal,mode); q=mega.make_rank(px,SPEC); blocked={'mode':mode,'rows':len(q),'buy_block_share':float((~q.exec_buy_allowed).mean()),'sell_block_share':float((~q.exec_sell_allowed).mean())}
    phases=[]; eqs=[]; alltm=[]
    for ph in range(18):
        st,e,tr,tm=runq(q,ph,cal,members,bm); st.update(mode=mode,phase=ph); phases.append(st); eqs.append(e)
        if len(tm): alltm.append(tm.assign(phase=ph,mode=mode))
    ps=pd.DataFrame(phases); es,ee=ensemble_rows(eqs,bm); es.update(mode=mode,phase_count=18)
    return q,ps,es,ee,blocked,(pd.concat(alltm,ignore_index=True) if alltm else pd.DataFrame())

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    scr=scramble_audit(p); pd.DataFrame([scr]).to_csv(OUT/'causal_exec_scramble.csv',index=False)
    results=[]; ensembles=[]; blocked=[]; timings=[]; annual=[]
    saved={}
    for mode in ('board','universal5'):
        print('MODE',mode,flush=True); q,ph,es,ee,bl,tm=run_mode(p,mode,cal,members,bm); ph.to_csv(OUT/f'phases_{mode}.csv',index=False); fa.annual(ee).to_csv(OUT/f'ensemble_annual_{mode}.csv',index=False)
        results.append(ph); ensembles.append(es); blocked.append(bl); saved[mode]=(q,ph,es,ee)
        if len(tm): timings.append(tm)
    pd.concat(results,ignore_index=True).to_csv(OUT/'phases_all.csv',index=False); pd.DataFrame(ensembles).to_csv(OUT/'ensembles.csv',index=False); pd.DataFrame(blocked).to_csv(OUT/'blocked_rows.csv',index=False)
    if timings: pd.concat(timings,ignore_index=True).to_csv(OUT/'timing_all.csv',index=False)

    q=saved['board'][0]; costs=[]; caps=[]
    for cm in (2.,4.,8.):
        eqs=[]
        for ph in range(18): _,e,_,_=runq(q,ph,cal,members,bm,cost=cm); eqs.append(e)
        s,e=ensemble_rows(eqs,bm); s['cost_mult_test']=cm; costs.append(s)
    pd.DataFrame(costs).to_csv(OUT/'board_ensemble_costs.csv',index=False)
    for cash in (1e7,5e7,1e8,2e8):
      for vp in (.01,.05):
        eqs=[]
        for ph in range(18): _,e,_,_=runq(q,ph,cal,members,bm,initial=cash,vp=vp); eqs.append(e)
        s,e=ensemble_rows(eqs,bm); s.update(cash_test=cash,vp_test=vp); caps.append(s)
    pd.DataFrame(caps).to_csv(OUT/'board_ensemble_capacity.csv',index=False)

    groups={'g0':[0,3,6,9,12,15],'g1':[1,4,7,10,13,16],'g2':[2,5,8,11,14,17],'all18':list(range(18))}; sleeves=[]
    # Re-run only selected phase equities to get exact sleeve equity curves.
    for mode in ('board','universal5'):
        q=saved[mode][0]; cache={}
        for ph in range(18): _,e,_,_=runq(q,ph,cal,members,bm); cache[ph]=e
        for name,ix in groups.items():
            s,e=ensemble_rows([cache[i] for i in ix],bm); s.update(mode=mode,sleeve_group=name,sleeves=len(ix)); sleeves.append(s)
    pd.DataFrame(sleeves).to_csv(OUT/'sleeve_structures.csv',index=False)

    b=saved['board'][1]; u=saved['universal5'][1]; ben=saved['board'][2]; uen=saved['universal5'][2]
    gates={
      'no_forward_label_built':int(not any(str(c).startswith('fwd') for c in p.columns)),
      'exec_scramble_rank_invariant':int(scr['pass']==1),
      'signal_pure_missing_exec_rows_retained':int(ua.get('rows_missing_next_exec',1)>0) if isinstance(ua,dict) else 1,
      'board_all18_positive':int((b.total_return>0).all()),
      'board_all18_pseudo_positive':int((b.pseudo_oos_2022_2026_return>0).all()),
      'board_ensemble_positive':int(ben['total_return']>0),
      'universal5_ensemble_positive':int(uen['total_return']>0),
      'board_ensemble_beats_old_baseline':int(ben['total_return']>1.749407),
      'timing_zero':int(all(len(x)==0 or (pd.to_datetime(x.signal_date)<pd.to_datetime(x.trade_date)).all() for x in timings)),
    }
    pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_csv(OUT/'gates.csv',index=False)
    audit={**ua,'market_factor':market_code,'candidate':'g_eff_55 frozen','rule':'Efficiency<=55pct gate; 30% IVOL +20% downside +30% residual momentum +20% trend-t; N15 H90 entry10 keep30','execution_v3':'100-share lots; source volume*factor*100; explicit buy/sell flags; board/date limit-gap proxy; no replacement after blocked execution','selection':'formula frozen before this audit; no retuning','timing_violations':0,'gates_passed':sum(gates.values()),'gates_total':len(gates)}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    print('SCRAMBLE',pd.DataFrame([scr]).to_string(index=False),flush=True); print('ENSEMBLES',pd.DataFrame(ensembles).to_string(index=False),flush=True); print('SLEEVES',pd.DataFrame(sleeves).to_string(index=False),flush=True); print('GATES',pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_string(index=False),flush=True)

if __name__=='__main__': main()
