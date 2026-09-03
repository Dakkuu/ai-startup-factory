from __future__ import annotations
from pathlib import Path
import glob,json
import numpy as np
import pandas as pd
import run_multi_alpha_system_v1 as v1
from run_multi_alpha_shard_common_v2 import normalize_equity

ROOT=Path('multi_alpha_inputs'); OUT=Path('results_multi_alpha_system_v2'); OUT.mkdir(exist_ok=True)

def one(pat):
    h=glob.glob(str(ROOT/pat),recursive=True)
    if len(h)!=1: raise RuntimeError(f'{pat}: expected 1, got {h}')
    return h[0]

def read_eq(path):
    x=pd.read_csv(path); x['trade_date']=pd.to_datetime(x.trade_date); return normalize_equity(x)

def metrics(eq):
    s=v1.series_from_eq(eq); f=v1.perf_series(s); tr=v1.perf_series(s,v1.START,v1.TRAIN_END); ps=v1.perf_series(s,v1.PSEUDO,v1.END)
    return {**f,'train_cagr':tr['cagr'],'train_mdd':tr['max_drawdown'],'train_sharpe':tr['sharpe'],'train_calmar':tr['calmar'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe']}

def main():
    # Global short winner = best among family winners under identical preregistered objective.
    sw=[]
    for fam in v1.SHORT_FAMILIES:
        x=pd.read_csv(one(f'**/short_{fam}_winner_metrics.csv')); z=x[np.isclose(x.cost_mult,1.0)].iloc[0].copy(); z['family_artifact']=fam; sw.append(z)
    sw=pd.DataFrame(sw); short_win=sw.sort_values(['train_calmar','train_sharpe','trade_count_mean'],ascending=[False,False,True]).iloc[0]; sfam=str(short_win.family_artifact)
    short_key=str(short_win.key)
    short_eq={cm:read_eq(one(f'**/short_{sfam}_equity_cost{cm}.csv')) for cm in (1,2,4)}

    lm=pd.read_csv(one('**/long_winner_metrics.csv')); long_win=lm[np.isclose(lm.cost_mult,1.0)].iloc[0]; long_key=str(long_win.key)
    long_eq={cm:read_eq(one(f'**/long_equity_cost{cm}.csv')) for cm in (1,2,4)}
    mm=pd.read_csv(one('**/medium_metrics.csv')); medium_eq={cm:read_eq(one(f'**/medium_equity_cost{cm}.csv')) for cm in (1,2,4)}
    bm0=pd.read_csv(one('**/benchmark.csv')); bm0['trade_date']=pd.to_datetime(bm0.trade_date); bm=bm0.set_index('trade_date').benchmark_close.astype(float).sort_index()

    eqs={'short':short_eq[1],'medium':medium_eq[1],'long':long_eq[1]}
    standalone=[]; standref={}
    for name,e in eqs.items():
        st=metrics(e); standalone.append({'sleeve':name,**st}); standref[name]={'train_cagr':st['train_cagr']}
    standalone=pd.DataFrame(standalone); standalone.to_csv(OUT/'selected_sleeves.csv',index=False)

    R,ctr,cp,cf,cd,roll=v1.correlations(eqs,bm); R.to_csv(OUT/'selected_daily_returns.csv'); ctr.to_csv(OUT/'correlation_train.csv'); cp.to_csv(OUT/'correlation_pseudo.csv'); cf.to_csv(OUT/'correlation_full.csv'); cd.to_csv(OUT/'correlation_down_days.csv'); roll.to_csv(OUT/'rolling_corr_252.csv',index=False)

    ag,acache,alloc_key=v1.allocation_grid(eqs,standref); ag.to_csv(OUT/'allocation_grid.csv',index=False); portfolio=normalize_equity(acache[alloc_key]); portfolio.to_csv(OUT/'equity_portfolio_cost1.csv',index=False); weights=v1.ALLOCATIONS[alloc_key]

    # Marginal contribution under selected allocation.
    basep=metrics(portfolio); names=['short','medium','long']; marg=[]
    for i,name in enumerate(names):
        keep=[j for j in range(3) if j!=i]; ws=np.array([weights[j] for j in keep],float); ws=ws/ws.sum(); kn=[names[j] for j in keep]
        e=v1.fixed_mix([eqs[n] for n in kn],ws.tolist()); st=metrics(e)
        marg.append({'removed':name,'portfolio_cagr':basep['cagr'],'without_cagr':st['cagr'],'delta_cagr':basep['cagr']-st['cagr'],'portfolio_sharpe':basep['sharpe'],'without_sharpe':st['sharpe'],'delta_sharpe':basep['sharpe']-st['sharpe'],'portfolio_mdd':basep['max_drawdown'],'without_mdd':st['max_drawdown']})
    marginal=pd.DataFrame(marg); marginal.to_csv(OUT/'marginal_contribution.csv',index=False)

    # Costs: weights fixed from train-only cost1 allocation selection.
    costrows=[]
    for cm in (1,2,4):
        pe=v1.fixed_mix([short_eq[cm],medium_eq[cm],long_eq[cm]],list(weights)); pe=normalize_equity(pe); pe.to_csv(OUT/f'equity_portfolio_cost{cm}.csv',index=False); st=metrics(pe); costrows.append({'cost_mult':float(cm),**st})
    costs=pd.DataFrame(costrows); costs.to_csv(OUT/'allocation_cost_stress.csv',index=False)

    # Annual returns for selected sleeves and portfolio cost1.
    annual=v1.annual_table({**eqs,'portfolio':portfolio}); annual.to_csv(OUT/'annual_selected.csv',index=False)
    for n,e in eqs.items(): normalize_equity(e).to_csv(OUT/f'equity_{n}_cost1.csv',index=False)

    # Merge grids for audit.
    sg=[]
    for fam in v1.SHORT_FAMILIES:
        x=pd.read_csv(one(f'**/short_{fam}_grid.csv')); sg.append(x)
    pd.concat(sg,ignore_index=True).to_csv(OUT/'short_grid_all.csv',index=False)
    pd.read_csv(one('**/long_grid.csv')).to_csv(OUT/'long_grid.csv',index=False)
    sw.to_csv(OUT/'short_family_winners.csv',index=False)

    med=standalone[standalone.sleeve=='medium'].iloc[0]; sel=ag[ag.allocation==alloc_key].iloc[0]
    min_corr=min(float(cf.loc['short','medium']),float(cf.loc['long','medium']))
    gates={
      'all_sleeves_pseudo_positive':int((standalone.pseudo_cagr>0).all()),
      'portfolio_pseudo_positive':int(sel.pseudo_cagr>0),
      'portfolio_train_calmar_gt_medium':int(sel.train_calmar>med.train_calmar),
      'portfolio_mdd_not_gt_medium_by_5pp':int(sel.max_drawdown>=med.max_drawdown-.05),
      'one_nonmedium_corr_le_060':int(min_corr<=.60),
      'no_sleeve_removal_sharpe_improves_gt_010':int((marginal.without_sharpe<=marginal.portfolio_sharpe+.10).all()),
      'cost2_positive_cagr':int(float(costs[costs.cost_mult==2].cagr.iloc[0])>0),
    }
    gd=pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]); gd.to_csv(OUT/'gates.csv',index=False)
    status='PROMOTED_SHADOW_MULTI_ALPHA' if all(gates.values()) else 'RESEARCH_ONLY_GATES_NOT_ALL_PASSED'
    strategy={
      'status':'NEW_STOCK_LEVEL_PIT_MULTI_ALPHA_RESEARCH_NOT_ORIGINAL_EXACT',
      'promotion_status':status,
      'preregistration':'MULTI_ALPHA_PREREG_2026-09-03.md committed before v1/v2 runs',
      'short':{'selected':short_key,'alpha_source':'behavioral price-volume; no fundamentals/GEff'},
      'medium':{'selected':'mom_cfo10_qv10 H60; rank tilt 75% N10 + 25% N5; stagger phases 0/4/8','alpha_source':'GEff technical + PIT CFO/assets + quality-value'},
      'long':{'selected':long_key,'alpha_source':'PIT value/quality/cash-flow; no momentum'},
      'allocation':{'selected_train_only':alloc_key,'short':weights[0],'medium':weights[1],'long':weights[2]},
      'selection_window':'2016-08-02..2021-12-31 only',
      'pseudo_oos':'2022-01-01..2026-07-29; diagnostic only, research-contaminated',
      'gates_passed':int(sum(gates.values())),'gates_total':len(gates)
    }
    (OUT/'strategy.json').write_text(json.dumps(strategy,ensure_ascii=False,indent=2,default=str))
    print('=== STRATEGY ==='); print(json.dumps(strategy,ensure_ascii=False,indent=2),flush=True)
    print('=== SLEEVES ==='); print(standalone.to_string(index=False),flush=True)
    print('=== CORR FULL ==='); print(cf.to_string(),flush=True)
    print('=== CORR DOWN ==='); print(cd.to_string(),flush=True)
    print('=== ALLOCATIONS ==='); print(ag.sort_values(['train_calmar','train_sharpe'],ascending=False).to_string(index=False),flush=True)
    print('=== COST ==='); print(costs.to_string(index=False),flush=True)
    print('=== MARGINAL ==='); print(marginal.to_string(index=False),flush=True)
    print('=== GATES ==='); print(gd.to_string(index=False),flush=True)
if __name__=='__main__': main()
