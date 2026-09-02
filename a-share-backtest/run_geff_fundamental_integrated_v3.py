# Integrated GEff PIT fundamental backtest v3
from __future__ import annotations
from pathlib import Path
import json, glob
import numpy as np
import pandas as pd

import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
import run_geff_fundamental_fastpit_v1 as fp
hv3.patch()

OUT=Path('results_geff_fundamental_integrated_v3'); OUT.mkdir(exist_ok=True)
START=pd.Timestamp('2016-08-02'); TRAIN_END=pd.Timestamp('2021-12-31')
PSEUDO=pd.Timestamp('2022-01-01'); END=pd.Timestamp('2026-07-29')
HORIZONS=(60,75,90); N=10; ENTRY=.10; KEEP=.30
BASE_W={'iv':.25,'down':.15,'rmom':.35,'tstat':.25}
RISK_W={'iv':.30,'down':.25,'amax':.30,'volshock':.15}


def locate(pattern):
    hits=glob.glob(pattern,recursive=True)
    if not hits: raise FileNotFoundError(pattern)
    return hits[0]


def verify_attach(p,a,label):
    if '_row' not in a: raise RuntimeError(f'{label}: missing _row')
    if a._row.duplicated().any(): raise RuntimeError(f'{label}: duplicate _row')
    if a._row.min()!=0 or a._row.max()!=len(p)-1 or len(a)!=len(p):
        raise RuntimeError(f'{label}: row identity failure {len(a)} {a._row.min()} {a._row.max()} vs {len(p)}')
    chk=a.sort_values('_row')
    sd=pd.to_datetime(chk.signal_date).reset_index(drop=True)
    pc=pd.to_datetime(p.signal_date).reset_index(drop=True)
    cc=chk.code.astype(str).reset_index(drop=True)
    pcode=p.code.astype(str).reset_index(drop=True)
    if not sd.equals(pc) or not cc.equals(pcode): raise RuntimeError(f'{label}: signal/code identity failure')


def fund_ranks(p,va,sa):
    z=pd.DataFrame({'_row':np.arange(len(p)), 'signal_date':pd.to_datetime(p.signal_date).to_numpy()})
    vv=va.sort_values('_row').reset_index(drop=True)
    ss=sa.sort_values('_row').reset_index(drop=True)
    for c in ['earnings_yield','book_yield','cashflow_yield','roe','gross_margin']:
        z[c]=pd.to_numeric(vv.get(c),errors='coerce')
    for c in ['cfo_assets','accrual_quality','cash_conversion']:
        z[c]=pd.to_numeric(ss.get(c),errors='coerce')
    for c in ['earnings_yield','book_yield','cashflow_yield','roe','gross_margin','cfo_assets','accrual_quality','cash_conversion']:
        z[c+'_z']=z.groupby('signal_date')[c].transform(fp.robust_z)
    z['value3_raw']=z[['earnings_yield_z','book_yield_z','cashflow_yield_z']].mean(axis=1,skipna=True)
    z['quality_value_raw']=z[['earnings_yield_z','book_yield_z','cashflow_yield_z','roe_z','gross_margin_z']].mean(axis=1,skipna=True)
    rawmap={'book_yield':'book_yield_z','value3':'value3_raw','quality_value':'quality_value_raw','cfo_assets':'cfo_assets_z','accrual_quality':'accrual_quality_z','cash_conversion':'cash_conversion_z'}
    for k,c in rawmap.items():
        z[k+'_rank']=z.groupby('signal_date')[c].rank(pct=True,method='average',ascending=False)
    cols=['_row']+[k+'_rank' for k in rawmap]
    x=p.reset_index(drop=True).copy(); x['_row']=np.arange(len(x)); x=x.merge(z[cols],on='_row',how='left',validate='one_to_one').drop(columns='_row')
    return x,z


def old_mom(p):
    return mega.make_rank(p,{'name':'old_mom','kind':'gate','g':{'ef':.55},'w':BASE_W})


def risk_core(p):
    return mega.make_rank(p,{'name':'riskcore','kind':'gate','g':{'ef':.55},'w':RISK_W})


def blend(q, weights):
    x=q.copy(); m=np.isfinite(x.rank_test)
    raw=pd.Series(0.0,index=x.index,dtype=float); sw=0.0
    for key,w in weights.items():
        w=float(w); sw+=w
        if key=='tech': v=x.rank_test
        else:
            if key not in x.columns: raise KeyError(key)
            v=x[key].fillna(.5)
        raw=raw+w*v
    raw=raw/sw
    x.loc[m,'rank_test']=raw.loc[m].groupby(x.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return x


def build_candidates(p):
    om=old_mom(p); rc=risk_core(p)
    return {
      'base_mom':om,
      'riskcore_tech':rc,
      'mom_qv10':blend(om,{'tech':.90,'quality_value_rank':.10}),
      'mom_cfo10':blend(om,{'tech':.90,'cfo_assets_rank':.10}),
      'mom_cfo20':blend(om,{'tech':.80,'cfo_assets_rank':.20}),
      'mom_cfo10_value10':blend(om,{'tech':.80,'cfo_assets_rank':.10,'value3_rank':.10}),
      'mom_cfo10_qv10':blend(om,{'tech':.80,'cfo_assets_rank':.10,'quality_value_rank':.10}),
      'mom_accr10_value10':blend(om,{'tech':.80,'accrual_quality_rank':.10,'value3_rank':.10}),
      'risk_cfo10_value10':blend(rc,{'tech':.80,'cfo_assets_rank':.10,'value3_rank':.10}),
      'risk_cfo10_qv10':blend(rc,{'tech':.80,'cfo_assets_rank':.10,'quality_value_rank':.10}),
      'risk_accr10_value10':blend(rc,{'tech':.80,'accrual_quality_rank':.10,'value3_rank':.10}),
      'risk_cfo15_value15':blend(rc,{'tech':.70,'cfo_assets_rank':.15,'value3_rank':.15}),
      'risk_cfo10_accr10_value10':blend(rc,{'tech':.70,'cfo_assets_rank':.10,'accrual_quality_rank':.10,'value3_rank':.10}),
    }


def subset(q,h,ph):
    ds=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique())))
    chosen=set(ds[ph::max(1,round(h/5))])
    cols=strict.BASECOLS+[c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in q.columns]
    z=q[q.signal_date.isin(chosen)][cols].copy(); z['ivol60_pct']=z.rank_test
    return z.drop(columns='rank_test')


def pc(eq,a,b): return fp.period_cagr(eq,a,b)


def run_all(cands,cal,members,bm,cost=1.0,only=None):
    rows=[]; names=list(cands) if only is None else list(only)
    for ci,nm in enumerate(names,1):
        print('CANDIDATE',ci,'/',len(names),nm,'cost',cost,flush=True); q=cands[nm]
        for h in HORIZONS:
            step=max(1,round(h/5))
            for ph in range(step):
                st,eq,trd,tm=ma.run_panel(subset(q,h,ph),cal,members,bm,n=N,entry=ENTRY,keep=KEEP,cost=cost)
                st.update(candidate=nm,H=h,phase=ph,cost_mult=cost,half1_cagr=pc(eq,START,pd.Timestamp('2019-12-31')),half2_cagr=pc(eq,pd.Timestamp('2020-01-01'),TRAIN_END),pseudo_cagr=pc(eq,PSEUDO,END))
                rows.append(st)
    return pd.DataFrame(rows)


def summarize(d):
    rows=[]
    for c,g in d.groupby('candidate'):
        rows.append({'candidate':c,'runs':len(g),'full_cagr_median':g.cagr.median(),'full_cagr_p25':g.cagr.quantile(.25),'full_cagr_min':g.cagr.min(),'mdd_median':g.max_drawdown.median(),'mdd_worst':g.max_drawdown.min(),'sharpe_median':g.sharpe.median(),'train_maximin':min(g.half1_cagr.median(),g.half2_cagr.median()),'pseudo_cagr_median':g.pseudo_cagr.median(),'pseudo_cagr_p25':g.pseudo_cagr.quantile(.25),'h60_median':g[g.H==60].cagr.median(),'h75_median':g[g.H==75].cagr.median(),'h90_median':g[g.H==90].cagr.median()})
    return pd.DataFrame(rows).sort_values(['full_cagr_p25','train_maximin','full_cagr_median'],ascending=False)


def main():
    print('BUILD BASE PANEL',flush=True)
    p,cal,members,ua,market_code,bm=mo.build_panel(OUT,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    vpath=locate('artifact_cache/value/**/pit_value_attached.csv.gz'); spath=locate('artifact_cache/stmt/**/pit_attached.csv.gz')
    va=pd.read_csv(vpath,compression='gzip',low_memory=False); sa=pd.read_csv(spath,compression='gzip',low_memory=False)
    verify_attach(p,va,'value'); verify_attach(p,sa,'3stmt')
    p2,z=fund_ranks(p,va,sa)
    cov={c:float(p2[c].notna().mean()) for c in ['book_yield_rank','value3_rank','quality_value_rank','cfo_assets_rank','accrual_quality_rank','cash_conversion_rank']}
    cands=build_candidates(p2)
    d=run_all(cands,cal,members,bm,1.0); d.to_csv(OUT/'all_phase.csv',index=False)
    s=summarize(d); s.to_csv(OUT/'summary.csv',index=False)
    ranked=[x for x in s.candidate if x not in ('base_mom','riskcore_tech')]
    finalists=['base_mom','riskcore_tech']+ranked[:3]
    d2=run_all(cands,cal,members,bm,2.0,finalists); d2.to_csv(OUT/'cost2x_all_phase.csv',index=False)
    s2=summarize(d2); s2.to_csv(OUT/'cost2x_summary.csv',index=False)
    control=d[(d.candidate=='base_mom')&(d.H==90)&(d.phase==0)].iloc[0]
    meta={'status':'NEW_STOCK_LEVEL_PIT_INTEGRATED_BACKTEST_NOT_ORIGINAL_EXACT','fundamental_inputs':'period-corrected V2 artifacts from run 33634216408','artifact_row_identity_pass':True,'pit_rule':'next exchange day after announcement; stale >550d neutral/missing','horizons':list(HORIZONS),'N':N,'entry':ENTRY,'keep':KEEP,'old_mom_weights':BASE_W,'riskcore_weights':RISK_W,'control_h90_phase0_cagr':float(control.cagr),'original_exact_h90_phase0_cagr':0.30697333,'source_identity_pass':False,'market_factor':market_code,'fundamental_rank_coverage':cov,'universe_audit':ua,'cost2x_finalists':finalists}
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    print('=== SUMMARY ==='); print(s.to_string(index=False),flush=True)
    print('=== COST 2X ==='); print(s2.to_string(index=False),flush=True)
    print('=== META ==='); print(json.dumps(meta,ensure_ascii=False,indent=2,default=str),flush=True)

if __name__=='__main__': main()
