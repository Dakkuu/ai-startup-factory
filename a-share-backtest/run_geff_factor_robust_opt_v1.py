from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
import pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()

OUT=Path('results_geff_factor_robust_opt_v1'); OUT.mkdir(exist_ok=True)
HORIZONS=(60,75,90)
N=10; ENTRY=.10; KEEP=.30
HALF1=('2016-08-02','2019-12-31')
HALF2=('2020-01-01','2021-12-31')
PSEUDO=('2022-01-01','2026-07-29')

BASE={'iv':.25,'down':.15,'rmom':.35,'tstat':.25}

def norm(w):
    s=sum(w.values()); return {k:float(v)/s for k,v in w.items()}

def add10(k):
    return {**{x:v*.90 for x,v in BASE.items()}, k:.10}

# PRE-REGISTERED candidate set. Do not expand after seeing outcomes.
CANDIDATES={
 'base_mom': BASE,
 'abl_no_down': norm({'iv':.25,'rmom':.35,'tstat':.25}),
 'abl_no_iv': norm({'down':.15,'rmom':.35,'tstat':.25}),
 'abl_no_rmom': norm({'iv':.25,'down':.15,'tstat':.25}),
 'abl_no_tstat': norm({'iv':.25,'down':.15,'rmom':.35}),
 'tilt_rmom45': {'iv':.20,'down':.10,'rmom':.45,'tstat':.25},
 'tilt_trend35': {'iv':.20,'down':.10,'rmom':.35,'tstat':.35},
 'tilt_risklight': {'iv':.15,'down':.10,'rmom':.45,'tstat':.30},
 'add_eff10': add10('ef'),
 'add_lowbeta10': add10('beta'),
 'add_capture10': add10('capture'),
 'add_antilottery10': add10('amax'),
 'add_lowskew10': add10('askew'),
 'add_drawdown10': add10('dd'),
 'add_mom120_10': add10('mom'),
 'add_lowvolshock10': add10('volshock'),
}


def period_cagr(eq,a,b):
    z=eq.copy(); z['trade_date']=pd.to_datetime(z.trade_date)
    z=z[(z.trade_date>=pd.Timestamp(a))&(z.trade_date<=pd.Timestamp(b))]
    if len(z)<2:return np.nan
    v0=float(z.equity.iloc[0]); v1=float(z.equity.iloc[-1])
    days=max(1,(z.trade_date.iloc[-1]-z.trade_date.iloc[0]).days)
    if v0<=0 or v1<=0:return np.nan
    return float((v1/v0)**(365.25/days)-1)


def eqret(eq):
    s=eq.set_index(pd.to_datetime(eq.trade_date)).equity.astype(float).sort_index()
    s=s[~s.index.duplicated(keep='last')]
    return s.pct_change().dropna()


def make_q(p,name,w):
    sp={'name':name,'kind':'gate','g':{'ef':.55},'w':w}
    return mega.make_rank(p,sp)


def candidate_coverage(q):
    x=q[np.isfinite(q.rank_test)].groupby('signal_date').code.nunique()
    return {'eligible_min':int(x.min()) if len(x) else 0,
            'eligible_median':float(x.median()) if len(x) else 0.0,
            'eligible_max':int(x.max()) if len(x) else 0}


def run_one(q,h,ph,cal,members,bm,cost=1.0):
    st,eq,tr,tm=ma.run_q(q,int(h),int(ph),cal,members,bm,n=N,entry=ENTRY,keep=KEEP,cost=float(cost))
    st['half1_cagr']=period_cagr(eq,*HALF1)
    st['half2_cagr']=period_cagr(eq,*HALF2)
    st['pseudo_cagr']=period_cagr(eq,*PSEUDO)
    return st,eq,tr,tm


def aggregate(rows):
    df=pd.DataFrame(rows)
    agg=[]
    for name,g in df.groupby('candidate',sort=False):
        d={
          'candidate':name,
          'runs':len(g),
          'full_cagr_median':float(g.cagr.median()),
          'full_cagr_p25':float(g.cagr.quantile(.25)),
          'full_cagr_min':float(g.cagr.min()),
          'full_cagr_max':float(g.cagr.max()),
          'sharpe_median':float(g.sharpe.median()),
          'mdd_median':float(g.max_drawdown.median()),
          'mdd_worst':float(g.max_drawdown.min()),
          'half1_cagr_median':float(g.half1_cagr.median()),
          'half2_cagr_median':float(g.half2_cagr.median()),
          'train_maximin':float(min(g.half1_cagr.median(),g.half2_cagr.median())),
          'pseudo_cagr_median':float(g.pseudo_cagr.median()),
          'pseudo_cagr_p25':float(g.pseudo_cagr.quantile(.25)),
        }
        for h in HORIZONS:
            z=g[g.H==h]
            d[f'h{h}_cagr_median']=float(z.cagr.median())
            d[f'h{h}_cagr_p25']=float(z.cagr.quantile(.25))
        h90p0=g[(g.H==90)&(g.phase==0)]
        d['h90_phase0_cagr']=float(h90p0.cagr.iloc[0]) if len(h90p0) else np.nan
        agg.append(d)
    return df,pd.DataFrame(agg)


def main():
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False)
    p=strict.attach_gap_flags(p,cal,'board')
    all_rows=[]; coverage=[]; phase0_returns={}; qc={}
    for i,(name,w) in enumerate(CANDIDATES.items(),1):
        print('CANDIDATE',i,'/',len(CANDIDATES),name,w,flush=True)
        q=make_q(p,name,w); qc[name]=q
        cv=candidate_coverage(q); cv.update(candidate=name,weights=json.dumps(w,sort_keys=True)); coverage.append(cv)
        for h in HORIZONS:
            step=max(1,round(h/5))
            for ph in range(step):
                print('RUN',name,'H',h,'phase',ph,'/',step-1,flush=True)
                st,eq,tr,tm=run_one(q,h,ph,cal,members,bm,1.0)
                st.update(candidate=name,H=h,phase=ph,phase_count=step,weights=json.dumps(w,sort_keys=True))
                all_rows.append(st)
                if h==90 and ph==0: phase0_returns[name]=eqret(eq).rename(name)
    detail,summary=aggregate(all_rows)
    base=summary[summary.candidate=='base_mom'].iloc[0]
    # Promotion rules are frozen before outcomes and emphasize phase robustness, not phase0.
    for h in HORIZONS:
        summary[f'beat_base_h{h}']=(summary[f'h{h}_cagr_median'] > float(base[f'h{h}_cagr_median'])).astype(int)
    summary['horizons_beaten']=summary[[f'beat_base_h{h}' for h in HORIZONS]].sum(axis=1)
    summary['train_delta_pp']=(summary.train_maximin-base.train_maximin)*100
    summary['pseudo_delta_pp']=(summary.pseudo_cagr_median-base.pseudo_cagr_median)*100
    summary['full_delta_pp']=(summary.full_cagr_median-base.full_cagr_median)*100
    summary['p25_delta_pp']=(summary.full_cagr_p25-base.full_cagr_p25)*100
    summary['promotion_pass']=(
        (summary.candidate!='base_mom') &
        (summary.train_maximin >= base.train_maximin + .015) &
        (summary.pseudo_cagr_median >= base.pseudo_cagr_median) &
        (summary.full_cagr_median >= base.full_cagr_median + .010) &
        (summary.full_cagr_p25 >= base.full_cagr_p25) &
        (summary.mdd_median >= base.mdd_median - .03) &
        (summary.horizons_beaten >= 2)
    ).astype(int)
    # Winner selection uses training-only maximin; pseudo/full are validation gates only.
    eligible=summary[summary.candidate!='base_mom'].sort_values(['train_maximin','full_cagr_p25'],ascending=False)
    train_winner=str(eligible.iloc[0].candidate)
    passes=summary[summary.promotion_pass==1].sort_values(['train_maximin','full_cagr_median'],ascending=False)
    promoted=str(passes.iloc[0].candidate) if len(passes) else ''
    summary['train_selected']=(summary.candidate==train_winner).astype(int)
    summary['promoted']=(summary.candidate==promoted).astype(int) if promoted else 0
    summary=summary.sort_values(['promotion_pass','train_maximin','full_cagr_median'],ascending=[False,False,False])

    # Predeclared 2x cost stress for baseline + top three training candidates.
    top3=[str(x) for x in eligible.head(3).candidate]
    stress_names=['base_mom']+top3
    stress=[]
    for name in stress_names:
        q=qc[name]
        for h in HORIZONS:
            step=max(1,round(h/5))
            for ph in range(step):
                st,eq,tr,tm=run_one(q,h,ph,cal,members,bm,2.0)
                stress.append({'candidate':name,'H':h,'phase':ph,'cagr':st['cagr'],'max_drawdown':st['max_drawdown'],'sharpe':st['sharpe'],'pseudo_cagr':period_cagr(eq,*PSEUDO)})
    stressdf=pd.DataFrame(stress)
    stressagg=stressdf.groupby('candidate').agg(cost2_cagr_median=('cagr','median'),cost2_cagr_p25=('cagr',lambda s:s.quantile(.25)),cost2_mdd_median=('max_drawdown','median'),cost2_pseudo_median=('pseudo_cagr','median')).reset_index()

    detail.to_csv(OUT/'all_phase_detail.csv',index=False)
    summary.to_csv(OUT/'candidate_robust_summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'candidate_coverage.csv',index=False)
    stressdf.to_csv(OUT/'cost2_phase_detail.csv',index=False)
    stressagg.to_csv(OUT/'cost2_summary.csv',index=False)
    R=pd.concat(phase0_returns.values(),axis=1,join='inner').sort_index(); R.index.name='trade_date'; R.to_csv(OUT/'h90_phase0_daily_returns.csv')
    verdict={
      'status':'PROMOTED_SHADOW' if promoted else 'NO_FACTOR_UPGRADE_PASSED',
      'train_selected':train_winner,
      'promoted':promoted,
      'candidate_count':len(CANDIDATES),
      'horizons':list(HORIZONS),
      'selection_rule':'training-only maximin of phase-median CAGR over 2016-2019 and 2020-2021; 2022-2026 is validation only',
      'promotion_rule':'train maximin +1.5pp; pseudo median non-inferior; full median +1pp; full p25 non-inferior; median MDD <=3pp worse; beat baseline median CAGR on >=2/3 horizons',
      'market_factor':market_code,
      'universe_audit':ua,
    }
    (OUT/'verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2,default=str))
    (OUT/'PREREGISTRATION.json').write_text(json.dumps({'candidates':CANDIDATES,'HORIZONS':HORIZONS,'N':N,'ENTRY':ENTRY,'KEEP':KEEP,'HALF1':HALF1,'HALF2':HALF2,'PSEUDO':PSEUDO},ensure_ascii=False,indent=2))
    print('\n=== ROBUST SUMMARY ==='); print(summary.to_string(index=False),flush=True)
    print('\n=== COST2 ==='); print(stressagg.to_string(index=False),flush=True)
    print('\n=== VERDICT ==='); print(json.dumps(verdict,ensure_ascii=False,indent=2,default=str),flush=True)

if __name__=='__main__': main()
