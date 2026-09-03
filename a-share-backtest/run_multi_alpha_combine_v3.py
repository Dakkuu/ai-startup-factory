from __future__ import annotations
from pathlib import Path
import glob,json
import numpy as np
import pandas as pd
import run_multi_alpha_system_v1 as v1
from run_multi_alpha_shard_common_v2 import normalize_equity
ROOT=Path('multi_alpha_v3_inputs'); OUT=Path('results_multi_alpha_system_v3'); OUT.mkdir(exist_ok=True)
FAMS=('pullback60','pullback120','pullback_lowiv','quiet_pullback','market_relative_pullback')
ALLOC={'B1_10_60_30':(.10,.60,.30),'B2_15_55_30':(.15,.55,.30),'B3_15_60_25':(.15,.60,.25),'B4_20_50_30':(.20,.50,.30),'B5_10_70_20':(.10,.70,.20)}

def one(pat):
    h=glob.glob(str(ROOT/pat),recursive=True)
    if len(h)!=1: raise RuntimeError(f'{pat}: {h}')
    return h[0]
def eq(p):
    x=pd.read_csv(p); x['trade_date']=pd.to_datetime(x.trade_date); return normalize_equity(x)
def metr(e):
    s=v1.series_from_eq(e); f=v1.perf_series(s); tr=v1.perf_series(s,v1.START,v1.TRAIN_END); ps=v1.perf_series(s,v1.PSEUDO,v1.END); return {**f,'train_cagr':tr['cagr'],'train_mdd':tr['max_drawdown'],'train_sharpe':tr['sharpe'],'train_calmar':tr['calmar'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe']}
def main():
    sw=[]
    for fam in FAMS:
        x=pd.read_csv(one(f'**/short_v3_{fam}_winner_metrics.csv')); r=x[np.isclose(x.cost_mult,1)].iloc[0].copy(); r['artifact_family']=fam; sw.append(r)
    sw=pd.DataFrame(sw); sw.to_csv(OUT/'short_family_winners.csv',index=False); eligible=sw[(sw.train_cagr>0)&(sw.train_mdd>-0.45)].copy(); short_valid=bool(len(eligible))
    if short_valid:
        srow=eligible.sort_values(['train_calmar','train_sharpe','trade_count_mean'],ascending=[False,False,True]).iloc[0]; sfam=str(srow.artifact_family); skey=str(srow.key); seq={cm:eq(one(f'**/short_v3_{sfam}_equity_cost{cm}.csv')) for cm in (1,2,4)}
    else:
        srow=sw.sort_values(['train_calmar','train_sharpe'],ascending=False).iloc[0]; sfam=str(srow.artifact_family); skey=str(srow.key); seq={cm:eq(one(f'**/short_v3_{sfam}_equity_cost{cm}.csv')) for cm in (1,2,4)}
    lmeta=json.load(open(one('**/long_v3_meta.json'))); lsel=str(lmeta['selected_train_only']); leq={cm:eq(one(f'**/long_v3_selected_equity_cost{cm}.csv')) for cm in (1,2,4)}
    meq={cm:eq(one(f'**/medium_equity_cost{cm}.csv')) for cm in (1,2,4)}
    b=pd.read_csv(one('**/benchmark.csv')); b['trade_date']=pd.to_datetime(b.trade_date); bm=b.set_index('trade_date').benchmark_close.astype(float).sort_index()
    base={'short':seq[1],'medium':meq[1],'long':leq[1]}; stand=pd.DataFrame([{'sleeve':n,**metr(e)} for n,e in base.items()]); stand.to_csv(OUT/'selected_sleeves.csv',index=False)
    R,ctr,cp,cf,cd,roll=v1.correlations(base,bm); R.to_csv(OUT/'selected_daily_returns.csv'); ctr.to_csv(OUT/'correlation_train.csv'); cp.to_csv(OUT/'correlation_pseudo.csv'); cf.to_csv(OUT/'correlation_full.csv'); cd.to_csv(OUT/'correlation_down_days.csv'); roll.to_csv(OUT/'rolling_corr_252.csv',index=False)
    arows=[]; ac={}
    if short_valid:
      for name,w in ALLOC.items():
        pe=normalize_equity(v1.fixed_mix([seq[1],meq[1],leq[1]],list(w))); st=metr(pe); arows.append({'allocation':name,'w_short':w[0],'w_medium':w[1],'w_long':w[2],**st}); ac[name]=pe
      ag=pd.DataFrame(arows); win=str(ag.sort_values(['train_calmar','train_sharpe'],ascending=False).iloc[0].allocation); weights=ALLOC[win]; portfolio=ac[win]
    else:
      # Diagnostic only; no forced short allocation. 70/30 medium/long is fixed fallback, not promoted as 3-alpha system.
      weights=(0.0,.70,.30); win='NO_VALID_SHORT__DIAG_0_70_30'; portfolio=normalize_equity(v1.fixed_mix([meq[1],leq[1]],[.70,.30])); ag=pd.DataFrame([{'allocation':win,'w_short':0.0,'w_medium':.70,'w_long':.30,**metr(portfolio)}])
    ag.to_csv(OUT/'allocation_grid.csv',index=False); portfolio.to_csv(OUT/'equity_portfolio_cost1.csv',index=False)
    names=['short','medium','long']; marginal=[]; pmet=metr(portfolio)
    if short_valid:
      for i,nm in enumerate(names):
        keep=[j for j in range(3) if j!=i]; ws=np.array([weights[j] for j in keep],float); ws=ws/ws.sum(); kn=[names[j] for j in keep]; ee=normalize_equity(v1.fixed_mix([base[n] for n in kn],ws.tolist())); st=metr(ee); marginal.append({'removed':nm,'portfolio_cagr':pmet['cagr'],'without_cagr':st['cagr'],'delta_cagr':pmet['cagr']-st['cagr'],'portfolio_sharpe':pmet['sharpe'],'without_sharpe':st['sharpe'],'delta_sharpe':pmet['sharpe']-st['sharpe'],'portfolio_mdd':pmet['max_drawdown'],'without_mdd':st['max_drawdown']})
    pd.DataFrame(marginal).to_csv(OUT/'marginal_contribution.csv',index=False)
    costs=[]
    for cm in (1,2,4):
      if short_valid: pe=normalize_equity(v1.fixed_mix([seq[cm],meq[cm],leq[cm]],list(weights)))
      else: pe=normalize_equity(v1.fixed_mix([meq[cm],leq[cm]],[.70,.30]))
      pe.to_csv(OUT/f'equity_portfolio_cost{cm}.csv',index=False); costs.append({'cost_mult':cm,**metr(pe)})
    costs=pd.DataFrame(costs); costs.to_csv(OUT/'allocation_cost_stress.csv',index=False)
    v1.annual_table({**base,'portfolio':portfolio}).to_csv(OUT/'annual_selected.csv',index=False)
    med=stand[stand.sleeve=='medium'].iloc[0]; ps=metr(portfolio); mincorr=min(float(cf.loc['short','medium']),float(cf.loc['long','medium']))
    if short_valid:
      md=pd.DataFrame(marginal); marginal_gate=int((md[md.removed.isin(['short','long'])].without_sharpe<=md.portfolio_sharpe+.05).all())
    else: marginal_gate=0
    gates={'short_train_positive':int(short_valid),'all_sleeves_pseudo_positive':int((stand.pseudo_cagr>0).all()),'portfolio_pseudo_positive':int(ps['pseudo_cagr']>0),'portfolio_train_calmar_ge_medium':int(ps['train_calmar']>=med.train_calmar),'portfolio_full_sharpe_ge_medium_minus_003':int(ps['sharpe']>=med.sharpe-.03),'one_nonmedium_corr_le_060':int(mincorr<=.60),'short_long_marginal_sharpe_gate':marginal_gate,'cost2_positive_cagr':int(float(costs[costs.cost_mult==2].cagr.iloc[0])>0)}
    gd=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); gd.to_csv(OUT/'gates.csv',index=False); promoted=all(gates.values())
    strategy={'status':'NEW_STOCK_LEVEL_PIT_MULTI_ALPHA_V3_POST_DIAGNOSTIC_NOT_CLEAN_OOS','promotion_status':'PROMOTED_SHADOW_MULTI_ALPHA_V3' if promoted else 'RESEARCH_ONLY_GATES_NOT_ALL_PASSED','short':{'selected':skey,'train_gate_passed':short_valid},'medium':'GEff-F10QV10 H60 ranktilt 75%N10+25%N5 phases0/4/8','long':{'selected_ensemble':lsel},'allocation':{'selected_train_only':win,'weights':{'short':weights[0],'medium':weights[1],'long':weights[2]}},'gates_passed':sum(gates.values()),'gates_total':len(gates),'selection_window':'2016-08-02..2021-12-31','pseudo_oos':'2022-01-01..2026-07-29 diagnostic only'}; (OUT/'strategy.json').write_text(json.dumps(strategy,ensure_ascii=False,indent=2))
    print(json.dumps(strategy,ensure_ascii=False,indent=2),flush=True); print('SLEEVES'); print(stand.to_string(index=False),flush=True); print('CORR'); print(cf.to_string(),flush=True); print('ALLOC'); print(ag.sort_values(['train_calmar','train_sharpe'],ascending=False).to_string(index=False),flush=True); print('COST'); print(costs.to_string(index=False),flush=True); print('GATES'); print(gd.to_string(index=False),flush=True)
if __name__=='__main__': main()
