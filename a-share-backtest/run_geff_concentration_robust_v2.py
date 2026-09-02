from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

import run_geff_concentration_robust_v1 as v1
import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff_concentration_robust_v2'); OUT.mkdir(exist_ok=True)
HORIZONS=v1.HORIZONS; NS=v1.NS; ENTRY=v1.ENTRY; KEEP=v1.KEEP; HALF1=v1.HALF1; HALF2=v1.HALF2; PSEUDO=v1.PSEUDO; W=v1.W

def subset_exact(q,h,phase=0):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    step=max(1,round(h/5)); chosen=set(dates[phase::step])
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def run_one(q,h,ph,n,cal,members,bm,cost=1.0):
    st,eq,tr,tm=ma.run_panel(subset_exact(q,h,ph),cal,members,bm,n=int(n),entry=ENTRY,keep=KEEP,cost=float(cost))
    st['half1_cagr']=v1.period_cagr(eq,*HALF1); st['half2_cagr']=v1.period_cagr(eq,*HALF2); st['pseudo_cagr']=v1.period_cagr(eq,*PSEUDO)
    return st,eq,tr,tm

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    q=mega.make_rank(p,{'name':'geff55_mom','kind':'gate','g':{'ef':.55},'w':W}); rows=[]
    for n in NS:
      for h in HORIZONS:
        step=max(1,round(h/5))
        for ph in range(step):
          print('RUN N',n,'H',h,'phase',ph,flush=True); st,eq,tr,tm=run_one(q,h,ph,n,cal,members,bm,1.0); st.update(N=n,H=h,phase=ph,phase_count=step); rows.append(st)
    d=pd.DataFrame(rows); agg=[]
    for n,g in d.groupby('N'):
      x={'N':int(n),'runs':len(g),'full_cagr_median':float(g.cagr.median()),'full_cagr_p25':float(g.cagr.quantile(.25)),'full_cagr_min':float(g.cagr.min()),'mdd_median':float(g.max_drawdown.median()),'mdd_worst':float(g.max_drawdown.min()),'sharpe_median':float(g.sharpe.median()),'half1_cagr_median':float(g.half1_cagr.median()),'half2_cagr_median':float(g.half2_cagr.median()),'train_maximin':float(min(g.half1_cagr.median(),g.half2_cagr.median())),'pseudo_cagr_median':float(g.pseudo_cagr.median()),'pseudo_cagr_p25':float(g.pseudo_cagr.quantile(.25))}
      for h in HORIZONS:
        z=g[g.H==h]; x[f'h{h}_cagr_median']=float(z.cagr.median()); x[f'h{h}_cagr_p25']=float(z.cagr.quantile(.25))
      z=g[(g.H==90)&(g.phase==0)]; x['h90_phase0_cagr']=float(z.cagr.iloc[0]); agg.append(x)
    a=pd.DataFrame(agg); base=a[a.N==10].iloc[0]
    for h in HORIZONS:a[f'beat_base_h{h}']=(a[f'h{h}_cagr_median']>base[f'h{h}_cagr_median']).astype(int)
    a['horizons_beaten']=a[[f'beat_base_h{h}' for h in HORIZONS]].sum(axis=1)
    a['promotion_pass']=((a.N!=10)&(a.train_maximin>=base.train_maximin+.015)&(a.pseudo_cagr_median>=base.pseudo_cagr_median)&(a.full_cagr_median>=base.full_cagr_median+.01)&(a.full_cagr_p25>=base.full_cagr_p25)&(a.mdd_median>=base.mdd_median-.05)&(a.horizons_beaten>=2)).astype(int)
    eligible=a[a.N!=10].sort_values(['train_maximin','full_cagr_p25'],ascending=False); train_selected=int(eligible.iloc[0].N); passes=a[a.promotion_pass==1].sort_values(['train_maximin','full_cagr_median'],ascending=False); promoted=int(passes.iloc[0].N) if len(passes) else 0
    a['train_selected']=(a.N==train_selected).astype(int); a['promoted']=(a.N==promoted).astype(int) if promoted else 0
    stress=[]
    for n in NS:
      for h in HORIZONS:
        for ph in range(max(1,round(h/5))):
          st,eq,tr,tm=run_one(q,h,ph,n,cal,members,bm,2.0); stress.append({'N':n,'H':h,'phase':ph,'cagr':st['cagr'],'max_drawdown':st['max_drawdown'],'pseudo_cagr':v1.period_cagr(eq,*PSEUDO)})
    s=pd.DataFrame(stress); sa=s.groupby('N').agg(cost2_cagr_median=('cagr','median'),cost2_cagr_p25=('cagr',lambda x:x.quantile(.25)),cost2_mdd_median=('max_drawdown','median'),cost2_pseudo_median=('pseudo_cagr','median')).reset_index()
    d.to_csv(OUT/'all_phase_detail.csv',index=False); a.to_csv(OUT/'summary.csv',index=False); s.to_csv(OUT/'cost2_detail.csv',index=False); sa.to_csv(OUT/'cost2_summary.csv',index=False)
    control=d[(d.N==10)&(d.H==90)&(d.phase==0)].iloc[0]; control_pass=bool(abs(float(control.cagr)-0.2229800205653422)<1e-10)
    verdict={'status':('PROMOTED_SHADOW' if promoted else 'NO_CONCENTRATION_UPGRADE_PASSED') if control_pass else 'INVALID_CONTROL_MISMATCH','control_h90_phase0_cagr':float(control.cagr),'control_pass':control_pass,'train_selected_N':train_selected,'promoted_N':promoted,'implementation':'source-identical statistics_v2 subset + run_panel','selection':'2016-2021 phase-robust maximin; 2022-2026 validation only','horizons':list(HORIZONS),'Ns':list(NS),'market_factor':market_code,'universe_audit':ua}
    (OUT/'verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2,default=str)); (OUT/'PREREGISTRATION.json').write_text(json.dumps({'same_search_as_v1':True,'implementation_correction_only':True,'HORIZONS':HORIZONS,'NS':NS,'ENTRY':ENTRY,'KEEP':KEEP,'weights':W},ensure_ascii=False,indent=2))
    print(a.to_string(index=False),flush=True); print(sa.to_string(index=False),flush=True); print(json.dumps(verdict,ensure_ascii=False,indent=2,default=str),flush=True)
if __name__=='__main__': main()
