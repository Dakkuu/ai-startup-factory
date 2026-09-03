from __future__ import annotations
from pathlib import Path
import glob, json
import numpy as np
import pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_maxopt_v3_frozen_audit as fa
import run_geff_fundamental_integrated_v3 as iv3
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff_fundamental_structure_opt_v1'); OUT.mkdir(exist_ok=True)
START=pd.Timestamp('2016-08-02'); TRAIN_END=pd.Timestamp('2021-12-31'); PSEUDO=pd.Timestamp('2022-01-01'); END=pd.Timestamp('2026-07-29')
HALF1_END=pd.Timestamp('2019-12-31'); HALF2_START=pd.Timestamp('2020-01-01')
HORIZONS=(60,75,90); N=10
HYST={
 'e05k20':(.05,.20),'e05k30':(.05,.30),'e05k40':(.05,.40),
 'e10k20':(.10,.20),'base_e10k30':(.10,.30),'e10k40':(.10,.40),'e10k50':(.10,.50),
 'e15k30':(.15,.30),'e15k40':(.15,.40),'e15k50':(.15,.50),
}
GAPS={'none':None,'gap3':.03,'gap5':.05,'gap8':.08}


def locate(pattern):
    h=glob.glob(pattern,recursive=True)
    if not h: raise FileNotFoundError(pattern)
    return h[0]

def build():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    va=pd.read_csv(locate('artifact_cache/value/**/pit_value_attached.csv.gz'),compression='gzip',low_memory=False)
    sa=pd.read_csv(locate('artifact_cache/stmt/**/pit_attached.csv.gz'),compression='gzip',low_memory=False)
    iv3.verify_attach(p,va,'value'); iv3.verify_attach(p,sa,'3stmt')
    p2,_=iv3.fund_ranks(p,va,sa); q=iv3.build_candidates(p2)['mom_cfo10_qv10']
    return q,p2,cal,members,ua,market_code,bm

def subset(q,h,ph,gap_cap=None):
    ds=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(h/5)); chosen=set(ds[ph::step])
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy()
    if gap_cap is not None and 'exec_open_gap' in z:
        z['exec_buy_allowed']=z['exec_buy_allowed'].astype(bool) & (pd.to_numeric(z.exec_open_gap,errors='coerce')<=float(gap_cap))
    z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def slice_stats(eq,a,b):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date); z=z[(z.trade_date>=pd.Timestamp(a))&(z.trade_date<=pd.Timestamp(b))]
    if len(z)<20:return {'cagr':np.nan,'mdd':np.nan,'sharpe':np.nan}
    s=z.set_index('trade_date').equity.astype(float).sort_index(); s=s[~s.index.duplicated(keep='last')]; r=s.pct_change().dropna(); days=max(1,(s.index[-1]-s.index[0]).days)
    c=float((s.iloc[-1]/s.iloc[0])**(365.25/days)-1); dd=s/s.cummax()-1; sd=float(r.std(ddof=1)); sh=float(r.mean()/sd*np.sqrt(252)) if len(r)>2 and sd>0 else np.nan
    return {'cagr':c,'mdd':float(dd.min()),'sharpe':sh}

def run(q,h,ph,e,k,cal,members,bm,gap_cap=None,cost=1.0):
    st,eq,tr,tm=ma.run_panel(subset(q,h,ph,gap_cap),cal,members,bm,n=N,entry=e,keep=k,cost=cost)
    trn=slice_stats(eq,START,TRAIN_END); a=slice_stats(eq,START,HALF1_END); b=slice_stats(eq,HALF2_START,TRAIN_END); ps=slice_stats(eq,PSEUDO,END)
    st.update(train_cagr=trn['cagr'],train_mdd=trn['mdd'],train_sharpe=trn['sharpe'],half1_cagr=a['cagr'],half2_cagr=b['cagr'],pseudo_cagr=ps['cagr'],pseudo_mdd=ps['mdd'])
    return st,eq,tr,tm

def aggregate(d,keys):
    rows=[]
    for key,g in d.groupby(keys,dropna=False):
        if not isinstance(key,tuple): key=(key,)
        r=dict(zip(keys,key)); r.update(runs=len(g),full_cagr_median=float(g.cagr.median()),full_cagr_p25=float(g.cagr.quantile(.25)),full_cagr_min=float(g.cagr.min()),mdd_median=float(g.max_drawdown.median()),mdd_worst=float(g.max_drawdown.min()),sharpe_median=float(g.sharpe.median()),train_cagr_median=float(g.train_cagr.median()),train_cagr_p25=float(g.train_cagr.quantile(.25)),train_mdd_median=float(g.train_mdd.median()),train_mdd_worst=float(g.train_mdd.min()),train_maximin=float(min(g.half1_cagr.median(),g.half2_cagr.median())),train_calmar_robust=float(g.train_cagr.quantile(.25)/abs(g.train_mdd.median())),pseudo_cagr_median=float(g.pseudo_cagr.median()),pseudo_cagr_p25=float(g.pseudo_cagr.quantile(.25)),pseudo_mdd_median=float(g.pseudo_mdd.median()))
        for h in HORIZONS:
            z=g[g.H==h]; r[f'h{h}_median']=float(z.cagr.median()) if len(z) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def ensemble_metrics(eqs,bm,label,extra=None):
    ee=fa.phase_ensemble(eqs); st=fa.perf_eq(ee,bm); tr=slice_stats(ee,START,TRAIN_END); ps=slice_stats(ee,PSEUDO,END)
    st.update(label=label,train_cagr=tr['cagr'],train_mdd=tr['mdd'],pseudo_cagr=ps['cagr'],pseudo_mdd=ps['mdd'])
    if extra: st.update(extra)
    return st,ee

def rerun_ensemble(q,cfg_name,e,k,gap_name,gap_cap,cal,members,bm):
    rows=[]; curves={}
    for h in HORIZONS:
        cache={}; step=max(1,round(h/5))
        for ph in range(step):
            st,eq,tr,tm=run(q,h,ph,e,k,cal,members,bm,gap_cap); cache[ph]=eq
        # all-phase
        st,ee=ensemble_metrics(list(cache.values()),bm,f'{cfg_name}_{gap_name}_H{h}_all',{'cfg':cfg_name,'gap':gap_name,'H':h,'sleeves':step}); rows.append(st); curves[(h,'all')]=ee
        # 3 and 5 evenly-spaced vintages
        for nsl in (3,5):
            ix=sorted(set(int(round(x))%step for x in np.linspace(0,step,endpoint=False,num=nsl)))
            st,ee=ensemble_metrics([cache[i] for i in ix],bm,f'{cfg_name}_{gap_name}_H{h}_{nsl}s',{'cfg':cfg_name,'gap':gap_name,'H':h,'sleeves':len(ix),'phase_ids':'|'.join(map(str,ix))}); rows.append(st); curves[(h,nsl)]=ee
    # equal capital across H phase-ensembles and practical 3-sleeve-per-H structures
    st,ee=ensemble_metrics([curves[(h,'all')] for h in HORIZONS],bm,f'{cfg_name}_{gap_name}_crossH_all',{'cfg':cfg_name,'gap':gap_name,'H':'60|75|90','sleeves':sum(round(h/5) for h in HORIZONS)}); rows.append(st)
    st,ee=ensemble_metrics([curves[(h,3)] for h in HORIZONS],bm,f'{cfg_name}_{gap_name}_crossH_3x3',{'cfg':cfg_name,'gap':gap_name,'H':'60|75|90','sleeves':9}); rows.append(st)
    return rows

def main():
    q,p2,cal,members,ua,market_code,bm=build()
    # 1) Hysteresis search. Only train metrics are used for selection.
    rows=[]
    for name,(e,k) in HYST.items():
      for h in HORIZONS:
        for ph in range(max(1,round(h/5))):
          print('HYST',name,'H',h,'PH',ph,flush=True); st,eq,tr,tm=run(q,h,ph,e,k,cal,members,bm,None,1.0); st.update(config=name,entry=e,keep=k,H=h,phase=ph); rows.append(st)
    hd=pd.DataFrame(rows); hs=aggregate(hd,['config','entry','keep']).sort_values(['train_calmar_robust','train_maximin'],ascending=False)
    hd.to_csv(OUT/'hysteresis_all_phase.csv',index=False); hs.to_csv(OUT/'hysteresis_summary.csv',index=False)
    best=str(hs.iloc[0].config); be,bk=HYST[best]
    # 2) Gap-chasing filter on train-selected hysteresis only.
    grows=[]
    for gn,gc in GAPS.items():
      for h in HORIZONS:
        for ph in range(max(1,round(h/5))):
          print('GAP',gn,'H',h,'PH',ph,flush=True); st,eq,tr,tm=run(q,h,ph,be,bk,cal,members,bm,gc,1.0); st.update(config=best,gap=gn,gap_cap=gc,H=h,phase=ph); grows.append(st)
    gd=pd.DataFrame(grows); gs=aggregate(gd,['config','gap','gap_cap']).sort_values(['train_calmar_robust','train_maximin'],ascending=False)
    gd.to_csv(OUT/'gap_all_phase.csv',index=False); gs.to_csv(OUT/'gap_summary.csv',index=False)
    bestgap=str(gs.iloc[0].gap); bg=GAPS[bestgap]
    # 3) 2x cost validation: baseline, train-best hysteresis, train-best gap.
    crows=[]
    specs=[('baseline','.10','.30','none',None,.10,.30),('hyst_best',str(be),str(bk),'none',None,be,bk),('gap_best',str(be),str(bk),bestgap,bg,be,bk)]
    for label,_,_,gn,gc,e,k in specs:
      for h in HORIZONS:
        for ph in range(max(1,round(h/5))):
          st,eq,tr,tm=run(q,h,ph,e,k,cal,members,bm,gc,2.0); st.update(label=label,gap=gn,H=h,phase=ph); crows.append(st)
    cd=pd.DataFrame(crows); cs=aggregate(cd,['label','gap']); cd.to_csv(OUT/'cost2_all_phase.csv',index=False); cs.to_csv(OUT/'cost2_summary.csv',index=False)
    # 4) Phase diversification structures.
    ens=[]
    for cfg,e,k,gn,gc in [('baseline',.10,.30,'none',None),(best,be,bk,bestgap,bg)]:
        ens.extend(rerun_ensemble(q,cfg,e,k,gn,gc,cal,members,bm))
    ed=pd.DataFrame(ens).sort_values(['train_cagr','train_mdd'],ascending=[False,False]); ed.to_csv(OUT/'ensemble_structures.csv',index=False)
    verdict={'hysteresis_train_selected':best,'entry':be,'keep':bk,'gap_train_selected':bestgap,'gap_cap':bg,'selection':'2016-2021 train_calmar_robust then train_maximin; pseudo 2022-2026 validation only','baseline_N':N,'horizons':list(HORIZONS),'market_factor':market_code,'universe_audit':ua,'note':'no individual hard stop promoted from prior China stop audit'}
    (OUT/'verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2,default=str))
    print('\n=== HYST ==='); print(hs.to_string(index=False),flush=True); print('\n=== GAP ==='); print(gs.to_string(index=False),flush=True); print('\n=== COST2 ==='); print(cs.to_string(index=False),flush=True); print('\n=== ENSEMBLES ==='); print(ed.to_string(index=False),flush=True); print('\n',json.dumps(verdict,ensure_ascii=False,indent=2,default=str),flush=True)
if __name__=='__main__': main()
