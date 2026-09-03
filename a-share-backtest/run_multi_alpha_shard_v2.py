from __future__ import annotations
from pathlib import Path
import glob, json, sys
import numpy as np
import pandas as pd

import run_multi_alpha_system_v1 as v1
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_max_audit as ma
import run_geff_fundamental_ranktilt_v1 as rt
import run_10y_hard_executor_v3 as hv3
hv3.patch()


def find_one(pat):
    x=glob.glob(pat,recursive=True)
    if not x: raise FileNotFoundError(pat)
    return x[0]


def load_runtime():
    # Same Qlib release / membership / daily closes as the panel build. The signal artifact
    # only avoids rebuilding factors; daily MTM remains stock-level, not rebalance-point approximation.
    base.START=v1.START; base.END=v1.END; base.WARM=pd.Timestamp('2014-01-01'); base.OUT=Path('results_multi_alpha_runtime')
    sim.START=v1.START; sim.END=v1.END; sim.WARM=pd.Timestamp('2014-01-01')
    cal,members,ua=base.load_base()
    bm=base.qb.read_bin('SH000300','close',cal).loc[v1.START:v1.END].dropna()
    sig=pd.read_parquet(find_one('panel_artifact/**/signals.parquet'))
    sig['signal_date']=pd.to_datetime(sig.signal_date); sig['trade_date']=pd.to_datetime(sig.trade_date)
    return sig,cal,members,ua,bm


def make_q(sig,rankcol):
    cols=['signal_date','trade_date','code','liq20','exec_open','exec_high','exec_low','exec_volume','exec_factor']
    cols += [c for c in ['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy'] if c in sig.columns]
    q=sig[cols].copy(); q['rank_test']=pd.to_numeric(sig[rankcol],errors='coerce').astype(float)
    return q


def metric_row(eq):
    s=v1.series_from_eq(eq); full=v1.perf_series(s); tr=v1.perf_series(s,v1.START,v1.TRAIN_END); ps=v1.perf_series(s,v1.PSEUDO,v1.END)
    return {**full,'train_cagr':tr['cagr'],'train_mdd':tr['max_drawdown'],'train_sharpe':tr['sharpe'],'train_calmar':tr['calmar'],'pseudo_cagr':ps['cagr'],'pseudo_mdd':ps['max_drawdown'],'pseudo_sharpe':ps['sharpe']}


def save_eq(eq,path):
    eq[['trade_date','equity']].to_csv(path,index=False)


def run_short(fam,sig,cal,members,bm,out):
    q=make_q(sig,f'rank_short_{fam}'); rows=[]; cache={}
    for h in v1.SHORT_H:
      for n in v1.SHORT_N:
       for e,k in v1.SHORT_BUF:
        print('SHORT',fam,h,n,e,k,flush=True)
        eq,st=v1.run_candidate(q,h,n,e,k,cal,members,bm,1.0,long_mode=False)
        key=f'{fam}|h{h}|n{n}|e{e}|k{k}'; rows.append({**st,'family':fam,'H':h,'N':n,'entry':e,'keep':k,'key':key}); cache[key]=(eq,h,n,e,k)
    d=pd.DataFrame(rows); d.to_csv(out/'grid.csv',index=False)
    ok=d[(d.train_cagr>0)&(d.train_mdd>-0.45)].copy()
    if len(ok)==0: ok=d.copy()
    win=ok.sort_values(['train_calmar','train_sharpe'],ascending=[False,False],kind='stable').iloc[0]
    key=str(win.key); eq,h,n,e,k=cache[key]
    pd.DataFrame([win]).to_csv(out/'winner.csv',index=False); save_eq(eq,out/'winner_equity_cost1.csv')
    for cm in (2.0,4.0):
        ceq,cst=v1.run_candidate(q,int(h),int(n),float(e),float(k),cal,members,bm,cm,long_mode=False)
        save_eq(ceq,out/f'winner_equity_cost{int(cm)}.csv'); pd.DataFrame([{**cst,'cost_mult':cm,'key':key}]).to_csv(out/f'winner_metrics_cost{int(cm)}.csv',index=False)
    return key


def run_long(fam,sig,cal,members,bm,out):
    q=make_q(sig,f'rank_long_{fam}'); rows=[]; cache={}
    for h in v1.LONG_H:
      for n in v1.LONG_N:
        e,k=v1.LONG_BUF
        print('LONG',fam,h,n,flush=True)
        eq,st=v1.run_candidate(q,h,n,e,k,cal,members,bm,1.0,long_mode=True)
        key=f'{fam}|h{h}|n{n}|e{e}|k{k}'; rows.append({**st,'family':fam,'H':h,'N':n,'entry':e,'keep':k,'key':key}); cache[key]=(eq,h,n,e,k)
    d=pd.DataFrame(rows); d.to_csv(out/'grid.csv',index=False)
    ok=d[(d.train_cagr>0)&(d.train_mdd>-0.45)].copy()
    if len(ok)==0: ok=d.copy()
    win=ok.sort_values(['train_calmar','train_sharpe'],ascending=[False,False],kind='stable').iloc[0]
    key=str(win.key); eq,h,n,e,k=cache[key]
    pd.DataFrame([win]).to_csv(out/'winner.csv',index=False); save_eq(eq,out/'winner_equity_cost1.csv')
    for cm in (2.0,4.0):
        ceq,cst=v1.run_candidate(q,int(h),int(n),float(e),float(k),cal,members,bm,cm,long_mode=True)
        save_eq(ceq,out/f'winner_equity_cost{int(cm)}.csv'); pd.DataFrame([{**cst,'cost_mult':cm,'key':key}]).to_csv(out/f'winner_metrics_cost{int(cm)}.csv',index=False)
    return key


def medium_eq(q,cal,members,bm,cost):
    phase_eq=[]
    for ph in (0,4,8):
        z=v1.subset(q,60,ph)
        _,e5,_,_=ma.run_panel(z,cal,members,bm,n=5,entry=.10,keep=.30,cost=cost)
        _,e10,_,_=ma.run_panel(z,cal,members,bm,n=10,entry=.10,keep=.30,cost=cost)
        phase_eq.append(rt.weighted_mix([e5,e10],[.25,.75]))
    return rt.weighted_mix(phase_eq,[1/3,1/3,1/3])


def run_medium(sig,cal,members,bm,out):
    q=make_q(sig,'rank_medium'); rows=[]
    for cm in (1.0,2.0,4.0):
        print('MEDIUM COST',cm,flush=True); eq=medium_eq(q,cal,members,bm,cm); st=metric_row(eq); st.update(cost_mult=cm,key='mom_cfo10_qv10|H60|RT25|ph0_4_8'); rows.append(st); save_eq(eq,out/f'equity_cost{int(cm)}.csv')
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    return rows[0]['key']


def main():
    if len(sys.argv)<2: raise SystemExit('usage: run_multi_alpha_shard_v2.py short FAMILY | long FAMILY | medium')
    kind=sys.argv[1]; fam=sys.argv[2] if len(sys.argv)>2 else ''
    tag=f'{kind}_{fam}' if fam else kind; out=Path(f'results_multi_alpha_shard_v2_{tag}'); out.mkdir(exist_ok=True)
    sig,cal,members,ua,bm=load_runtime()
    if kind=='short': key=run_short(fam,sig,cal,members,bm,out)
    elif kind=='long': key=run_long(fam,sig,cal,members,bm,out)
    elif kind=='medium': key=run_medium(sig,cal,members,bm,out)
    else: raise ValueError(kind)
    meta={'kind':kind,'family':fam or None,'selected_with_2016_2021_only':key,'release_tag':'2026-07-29','market_factor':'SH000300','universe_audit':ua,'status':'NEW_STOCK_LEVEL_MULTI_ALPHA_RESEARCH_NOT_ORIGINAL_EXACT'}
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str)); print(json.dumps(meta,ensure_ascii=False,indent=2,default=str),flush=True)

if __name__=='__main__': main()
