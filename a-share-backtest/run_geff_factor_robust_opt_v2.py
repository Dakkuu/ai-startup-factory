from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

import run_geff_factor_robust_opt_v1 as v1
import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff_factor_robust_opt_v2'); OUT.mkdir(exist_ok=True)
HORIZONS=v1.HORIZONS; N=v1.N; ENTRY=v1.ENTRY; KEEP=v1.KEEP
HALF1=v1.HALF1; HALF2=v1.HALF2; PSEUDO=v1.PSEUDO
CANDIDATES=v1.CANDIDATES

# IMPORTANT: exact same subset construction as run_10y_geff55_statistics_v2.py.
def subset_exact(q,h,phase=0):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    step=max(1,round(h/5)); chosen=set(dates[phase::step])
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')

def run_one(q,h,ph,cal,members,bm,cost=1.0):
    st,eq,tr,tm=ma.run_panel(subset_exact(q,h,ph),cal,members,bm,n=N,entry=ENTRY,keep=KEEP,cost=float(cost))
    st['half1_cagr']=v1.period_cagr(eq,*HALF1); st['half2_cagr']=v1.period_cagr(eq,*HALF2); st['pseudo_cagr']=v1.period_cagr(eq,*PSEUDO)
    return st,eq,tr,tm

def make_q(p,name,w): return mega.make_rank(p,{'name':name,'kind':'gate','g':{'ef':.55},'w':w})

def coverage(q):
    x=q[np.isfinite(q.rank_test)].groupby('signal_date').code.nunique()
    return {'eligible_min':int(x.min()),'eligible_median':float(x.median()),'eligible_max':int(x.max())}

def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    rows=[]; cov=[]; phase0={}; qc={}
    for i,(name,w) in enumerate(CANDIDATES.items(),1):
        print('CANDIDATE',i,'/',len(CANDIDATES),name,flush=True); q=make_q(p,name,w); qc[name]=q
        c=coverage(q); c.update(candidate=name,weights=json.dumps(w,sort_keys=True)); cov.append(c)
        for h in HORIZONS:
            step=max(1,round(h/5))
            for ph in range(step):
                print('RUN',name,'H',h,'phase',ph,flush=True)
                st,eq,tr,tm=run_one(q,h,ph,cal,members,bm,1.0); st.update(candidate=name,H=h,phase=ph,phase_count=step,weights=json.dumps(w,sort_keys=True)); rows.append(st)
                if h==90 and ph==0: phase0[name]=v1.eqret(eq).rename(name)
    detail,summary=v1.aggregate(rows); base=summary[summary.candidate=='base_mom'].iloc[0]
    for h in HORIZONS: summary[f'beat_base_h{h}']=(summary[f'h{h}_cagr_median']>float(base[f'h{h}_cagr_median'])).astype(int)
    summary['horizons_beaten']=summary[[f'beat_base_h{h}' for h in HORIZONS]].sum(axis=1)
    summary['train_delta_pp']=(summary.train_maximin-base.train_maximin)*100; summary['pseudo_delta_pp']=(summary.pseudo_cagr_median-base.pseudo_cagr_median)*100; summary['full_delta_pp']=(summary.full_cagr_median-base.full_cagr_median)*100; summary['p25_delta_pp']=(summary.full_cagr_p25-base.full_cagr_p25)*100
    summary['promotion_pass']=((summary.candidate!='base_mom')&(summary.train_maximin>=base.train_maximin+.015)&(summary.pseudo_cagr_median>=base.pseudo_cagr_median)&(summary.full_cagr_median>=base.full_cagr_median+.010)&(summary.full_cagr_p25>=base.full_cagr_p25)&(summary.mdd_median>=base.mdd_median-.03)&(summary.horizons_beaten>=2)).astype(int)
    eligible=summary[summary.candidate!='base_mom'].sort_values(['train_maximin','full_cagr_p25'],ascending=False); train_winner=str(eligible.iloc[0].candidate)
    passes=summary[summary.promotion_pass==1].sort_values(['train_maximin','full_cagr_median'],ascending=False); promoted=str(passes.iloc[0].candidate) if len(passes) else ''
    summary['train_selected']=(summary.candidate==train_winner).astype(int); summary['promoted']=(summary.candidate==promoted).astype(int) if promoted else 0
    summary=summary.sort_values(['promotion_pass','train_maximin','full_cagr_median'],ascending=[False,False,False])
    top3=[str(x) for x in eligible.head(3).candidate]; stress=[]
    for name in ['base_mom']+top3:
        q=qc[name]
        for h in HORIZONS:
            for ph in range(max(1,round(h/5))):
                st,eq,tr,tm=run_one(q,h,ph,cal,members,bm,2.0); stress.append({'candidate':name,'H':h,'phase':ph,'cagr':st['cagr'],'max_drawdown':st['max_drawdown'],'sharpe':st['sharpe'],'pseudo_cagr':v1.period_cagr(eq,*PSEUDO)})
    stressdf=pd.DataFrame(stress); stressagg=stressdf.groupby('candidate').agg(cost2_cagr_median=('cagr','median'),cost2_cagr_p25=('cagr',lambda s:s.quantile(.25)),cost2_mdd_median=('max_drawdown','median'),cost2_pseudo_median=('pseudo_cagr','median')).reset_index()
    detail.to_csv(OUT/'all_phase_detail.csv',index=False); summary.to_csv(OUT/'candidate_robust_summary.csv',index=False); pd.DataFrame(cov).to_csv(OUT/'candidate_coverage.csv',index=False); stressdf.to_csv(OUT/'cost2_phase_detail.csv',index=False); stressagg.to_csv(OUT/'cost2_summary.csv',index=False)
    R=pd.concat(phase0.values(),axis=1,join='inner').sort_index(); R.index.name='trade_date'; R.to_csv(OUT/'h90_phase0_daily_returns.csv')
    # Exact control sanity: base H90 phase0 should reproduce the frozen original naked core (~22.298% CAGR).
    control=detail[(detail.candidate=='base_mom')&(detail.H==90)&(detail.phase==0)].iloc[0]
    control_pass=bool(abs(float(control.cagr)-0.2229800205653422)<1e-10)
    verdict={'status':('PROMOTED_SHADOW' if promoted else 'NO_FACTOR_UPGRADE_PASSED') if control_pass else 'INVALID_CONTROL_MISMATCH','control_h90_phase0_cagr':float(control.cagr),'control_pass':control_pass,'train_selected':train_winner,'promoted':promoted,'candidate_count':len(CANDIDATES),'horizons':list(HORIZONS),'implementation':'source-identical statistics_v2 subset + run_panel','selection_rule':'training-only phase-median maximin; 2022-2026 validation only','market_factor':market_code,'universe_audit':ua}
    (OUT/'verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2,default=str)); (OUT/'PREREGISTRATION.json').write_text(json.dumps({'same_candidate_set_as_v1':True,'implementation_correction_only':True,'candidates':CANDIDATES,'HORIZONS':HORIZONS,'N':N,'ENTRY':ENTRY,'KEEP':KEEP},ensure_ascii=False,indent=2))
    print(summary.to_string(index=False),flush=True); print(stressagg.to_string(index=False),flush=True); print(json.dumps(verdict,ensure_ascii=False,indent=2,default=str),flush=True)
if __name__=='__main__': main()
