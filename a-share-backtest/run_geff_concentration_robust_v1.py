from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff_concentration_robust_v1'); OUT.mkdir(exist_ok=True)
HORIZONS=(60,75,90); NS=(5,7,10); ENTRY=.10; KEEP=.30
HALF1=('2016-08-02','2019-12-31'); HALF2=('2020-01-01','2021-12-31'); PSEUDO=('2022-01-01','2026-07-29')
W={'iv':.25,'down':.15,'rmom':.35,'tstat':.25}

def period_cagr(eq,a,b):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date); z=z[(z.trade_date>=pd.Timestamp(a))&(z.trade_date<=pd.Timestamp(b))]
    if len(z)<2:return np.nan
    v0=float(z.equity.iloc[0]); v1=float(z.equity.iloc[-1]); days=max(1,(z.trade_date.iloc[-1]-z.trade_date.iloc[0]).days)
    return float((v1/v0)**(365.25/days)-1) if v0>0 and v1>0 else np.nan

def run_one(q,h,ph,n,cal,members,bm,cost=1.0):
    st,eq,tr,tm=ma.run_q(q,int(h),int(ph),cal,members,bm,n=int(n),entry=ENTRY,keep=KEEP,cost=float(cost))
    st['half1_cagr']=period_cagr(eq,*HALF1); st['half2_cagr']=period_cagr(eq,*HALF2); st['pseudo_cagr']=period_cagr(eq,*PSEUDO)
    return st,eq,tr,tm

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    q=mega.make_rank(p,{'name':'geff55_mom','kind':'gate','g':{'ef':.55},'w':W})
    rows=[]
    for n in NS:
      for h in HORIZONS:
        step=max(1,round(h/5))
        for ph in range(step):
          print('RUN N',n,'H',h,'phase',ph,flush=True)
          st,eq,tr,tm=run_one(q,h,ph,n,cal,members,bm,1.0); st.update(N=n,H=h,phase=ph,phase_count=step); rows.append(st)
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
    eligible=a[a.N!=10].sort_values(['train_maximin','full_cagr_p25'],ascending=False); train_selected=int(eligible.iloc[0].N)
    passes=a[a.promotion_pass==1].sort_values(['train_maximin','full_cagr_median'],ascending=False); promoted=int(passes.iloc[0].N) if len(passes) else 0
    a['train_selected']=(a.N==train_selected).astype(int); a['promoted']=(a.N==promoted).astype(int) if promoted else 0
    # 2x cost stress for all three predeclared N values.
    stress=[]
    for n in NS:
      for h in HORIZONS:
        step=max(1,round(h/5))
        for ph in range(step):
          st,eq,tr,tm=run_one(q,h,ph,n,cal,members,bm,2.0); stress.append({'N':n,'H':h,'phase':ph,'cagr':st['cagr'],'max_drawdown':st['max_drawdown'],'pseudo_cagr':period_cagr(eq,*PSEUDO)})
    s=pd.DataFrame(stress); sa=s.groupby('N').agg(cost2_cagr_median=('cagr','median'),cost2_cagr_p25=('cagr',lambda x:x.quantile(.25)),cost2_mdd_median=('max_drawdown','median'),cost2_pseudo_median=('pseudo_cagr','median')).reset_index()
    d.to_csv(OUT/'all_phase_detail.csv',index=False); a.to_csv(OUT/'summary.csv',index=False); s.to_csv(OUT/'cost2_detail.csv',index=False); sa.to_csv(OUT/'cost2_summary.csv',index=False)
    verdict={'status':'PROMOTED_SHADOW' if promoted else 'NO_CONCENTRATION_UPGRADE_PASSED','train_selected_N':train_selected,'promoted_N':promoted,'selection':'2016-2021 phase-robust maximin; 2022-2026 validation only','horizons':list(HORIZONS),'Ns':list(NS),'market_factor':market_code,'universe_audit':ua}
    (OUT/'verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2,default=str)); (OUT/'PREREGISTRATION.json').write_text(json.dumps({'HORIZONS':HORIZONS,'NS':NS,'ENTRY':ENTRY,'KEEP':KEEP,'weights':W},ensure_ascii=False,indent=2))
    print(a.to_string(index=False),flush=True); print(sa.to_string(index=False),flush=True); print(json.dumps(verdict,ensure_ascii=False,indent=2,default=str),flush=True)
if __name__=='__main__': main()
