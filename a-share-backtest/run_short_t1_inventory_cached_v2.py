from __future__ import annotations

import argparse, glob, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_short_t1_inventory_v1 as t1

OUT=Path('results_short_t1_inventory_cached_v2')
PREP=OUT/'prepare'; EXEC=OUT/'exec'; GRID=OUT/'grid'; FINAL=OUT/'final'
for p in (PREP,EXEC,GRID,FINAL): p.mkdir(parents=True,exist_ok=True)
SHARDS=12
MAX_WINDOW=max(t1.MEMORIES)+t1.EXIT_RETRY


def setup():
    base.START=t1.START; base.WARM=t1.WARM; base.END=t1.END
    sim.START=t1.START; sim.WARM=t1.WARM; sim.END=t1.END
    base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base()
    market_code,market_close,mc=v4.pick_market(cal)
    bm=market_close.loc[t1.START:t1.END].dropna()
    tc,_=t1.next_map(cal)
    return cal,members,ua,market_code,bm,tc


def one(pattern):
    h=glob.glob(pattern,recursive=True)
    if not h: raise FileNotFoundError(pattern)
    return h[0]


def parse_dates(x):
    for c in ('signal_date','trade_date'):
        if c in x: x[c]=pd.to_datetime(x[c])
    return x


def build_liq_only(cal,members,tc):
    codes=sorted(members.code.unique())
    liq=np.full((len(codes),len(tc)),np.nan,dtype=np.float32)
    for i,code in enumerate(codes):
        cl=base.qb.read_bin(code,'close',cal).loc[t1.WARM:t1.END]
        vo=base.qb.read_bin(code,'volume',cal).loc[t1.WARM:t1.END]
        fa=base.qb.read_bin(code,'factor',cal).loc[t1.WARM:t1.END]
        if cl.empty or vo.empty: continue
        z=pd.concat({'close':cl,'volume':vo,'factor':fa},axis=1)
        if fa.empty: z['factor']=1.0
        z['factor']=z.factor.replace(0,np.nan).ffill().fillna(1.0)
        raw_close=z.close/z.factor
        rawvol=z.volume.abs()*z.factor.abs()*100.0
        liq20=(raw_close.abs()*rawvol.abs()).rolling(20,min_periods=15).mean()
        liq[i,:]=liq20.reindex(tc).to_numpy(dtype=np.float32)
        if (i+1)%1000==0: print('LIQ',i+1,'/',len(codes),flush=True)
    return liq,codes


def prepare():
    cal,members,ua,market_code,bm,tc=setup()
    C=parse_dates(pd.read_csv(one('cache_input/**/broad_flush_candidates.csv.gz'),compression='gzip'))
    E=parse_dates(pd.read_csv(one('cache_input/**/broad_event_candidates.csv.gz'),compression='gzip'))
    liq,codes=build_liq_only(cal,members,tc)
    origins=[]; rows=[]
    for name,v in t1.VARIANTS.items():
        cfg=t1.cfg_from(name,v,5,10)
        s=t1.filter_signals(C,cfg,liq,tc,False)
        e=t1.filter_signals(E,cfg,liq,tc,True)
        s.to_csv(PREP/f'signals_{name}.csv.gz',index=False,compression='gzip')
        e.to_csv(PREP/f'eventonly_{name}.csv.gz',index=False,compression='gzip')
        origins += [s[['signal_date','code']],e[['signal_date','code']]]
        rows.append({'variant':name,'flush_signals':len(s),'flush_days':s.signal_date.nunique(),'flush_codes':s.code.nunique(),'eventonly_signals':len(e),'eventonly_days':e.signal_date.nunique(),'eventonly_codes':e.code.nunique()})
    O=pd.concat(origins,ignore_index=True).drop_duplicates(['signal_date','code']).sort_values(['code','signal_date'])
    O.to_csv(PREP/'origins.csv.gz',index=False,compression='gzip')
    pd.DataFrame(rows).to_csv(PREP/'prepare_counts.csv',index=False)
    meta={'source_run':33820425120,'source_artifact':'short-t1-inventory-v1-results','broad_flush_rows':len(C),'broad_event_rows':len(E),'filtered_origin_rows':len(O),'filtered_origin_codes':O.code.nunique(),'market_factor':market_code,'universe_audit':ua,'note':'same preregistered T1-IE rules; cache/sharding changes computation scheduling only'}
    (PREP/'prepare_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('ORIGINS',len(O),O.code.nunique(),flush=True)


def exec_shard(shard,shards=SHARDS):
    cal,members,ua,market_code,bm,tc=setup()
    O=parse_dates(pd.read_csv(one('prepare_input/**/origins.csv.gz'),compression='gzip'))
    codes=sorted(O.code.unique()); take=set(codes[int(shard)::int(shards)])
    z=O[O.code.isin(take)].copy()
    print('EXEC SHARD',shard,'codes',len(take),'origins',len(z),flush=True)
    X=t1.build_exec_rows(cal,members,z,tc,MAX_WINDOW)
    X.to_pickle(EXEC/f'execution_rows_{int(shard):02d}.pkl.gz',compression='gzip')
    pd.DataFrame([{'shard':int(shard),'shards':int(shards),'codes':len(take),'origins':len(z),'execution_rows':len(X)}]).to_csv(EXEC/f'audit_{int(shard):02d}.csv',index=False)
    print('EXEC DONE',shard,len(X),flush=True)


def load_exec():
    files=glob.glob('exec_input/**/execution_rows_*.pkl.gz',recursive=True)
    if not files: raise FileNotFoundError('execution shards')
    xs=[pd.read_pickle(f,compression='gzip') for f in sorted(files)]
    x=pd.concat(xs,ignore_index=True).drop_duplicates(['signal_date','code'],keep='last')
    return parse_dates(x)


def load_signal(name,event=False):
    stem='eventonly' if event else 'signals'
    return parse_dates(pd.read_csv(one(f'prepare_input/**/{stem}_{name}.csv.gz'),compression='gzip'))


def grid_variant(name):
    cal,members,ua,market_code,bm,tc=setup(); X=load_exec(); sig=load_signal(name,False)
    rows=[]
    for mem in t1.MEMORIES:
        for n in t1.NS:
            cfg=t1.cfg_from(name,t1.VARIANTS[name],mem,n)
            print('GRID',name,mem,n,flush=True)
            p=t1.panel_from(sig,cfg,X,tc,False)
            st,eq,tr,tm=t1.run_panel(p,cfg,cal,members,bm,1.0)
            if st is None: continue
            key=f'{name}|m{mem}|n{n}'
            rows.append({**st,'key':key,'variant':name,'memory_sessions':mem,'n_hold':n,'signal_count':len(sig),'signal_days':sig.signal_date.nunique()})
    d=pd.DataFrame(rows); d.to_csv(GRID/f'grid_{name}.csv',index=False)
    print(d.to_string(index=False),flush=True)


def select_grid():
    fs=glob.glob('grid_input/**/grid_*.csv',recursive=True)
    if not fs: raise FileNotFoundError('grid files')
    G=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
    good=G[(G.train_cagr>0)&(G.train_mdd>-.45)&(G.half1_cagr>0)&(G.half2_cagr>0)].copy()
    if good.empty: good=G[(G.train_cagr>0)&(G.train_mdd>-.45)].copy()
    if good.empty: good=G.copy()
    win=good.sort_values(['train_calmar','train_sharpe','turnover'],ascending=[False,False,True]).iloc[0]
    return G,win


def final():
    cal,members,ua,market_code,bm,tc=setup(); X=load_exec(); G,win=select_grid()
    G.to_csv(FINAL/'grid.csv',index=False)
    name=str(win.variant); mem=int(win.memory_sessions); n=int(win.n_hold)
    cfg=t1.cfg_from(name,t1.VARIANTS[name],mem,n); sig=load_signal(name,False)
    p=t1.panel_from(sig,cfg,X,tc,False); st,eq,tr,tm=t1.run_panel(p,cfg,cal,members,bm,1.0)
    pd.DataFrame([{**st,'key':f'{name}|m{mem}|n{n}','variant':name,'memory_sessions':mem,'n_hold':n,'signal_count':len(sig),'signal_days':sig.signal_date.nunique()}]).to_csv(FINAL/'selected_metrics.csv',index=False)
    sig.to_csv(FINAL/'selected_signals.csv.gz',index=False,compression='gzip'); eq.to_csv(FINAL/'selected_equity.csv',index=False); tr.to_csv(FINAL/'selected_trades.csv',index=False); tm.to_csv(FINAL/'selected_timing.csv',index=False); t1.annual(eq).to_csv(FINAL/'selected_annual.csv',index=False)
    sig.assign(year=pd.to_datetime(sig.signal_date).dt.year).groupby('year').size().rename('signals').reset_index().to_csv(FINAL/'selected_signal_counts_year.csv',index=False)
    stress=[]
    for cm in (2.0,4.0):
        st2,e2,t2,tm2=t1.run_panel(p,cfg,cal,members,bm,cm); stress.append({'cost_mult':cm,**st2}); e2.to_csv(FINAL/f'selected_equity_cost{int(cm)}.csv',index=False)
    pd.DataFrame(stress).to_csv(FINAL/'cost_stress.csv',index=False)
    pr=t1.panel_from(sig,cfg,X,tc,True); sr,er,trr,tmr=t1.run_panel(pr,cfg,cal,members,bm,1.0)
    pd.DataFrame([{'control':'reverse_score',**sr}]).to_csv(FINAL/'reverse_control.csv',index=False)
    esig=load_signal(name,True); pe=t1.panel_from(esig,cfg,X,tc,False); se,ee,te,tme=t1.run_panel(pe,cfg,cal,members,bm,1.0)
    pd.DataFrame([{'control':'event_only_no_flush_confirmation',**se,'signal_count':len(esig)}]).to_csv(FINAL/'event_only_ablation.csv',index=False)
    timing_bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    stx=pd.DataFrame(stress).set_index('cost_mult')
    gates={
      'train_cagr_positive':int(st['train_cagr']>0),
      'pseudo_cagr_positive':int(st['pseudo_cagr']>0),
      'train_sharpe_positive':int(st['train_sharpe']>0),
      'pseudo_sharpe_positive':int(st['pseudo_sharpe']>0),
      'full_mdd_better_than_minus45':int(st['max_drawdown']>-.45),
      'cost2_cagr_positive':int(float(stx.loc[2.0,'cagr'])>0),
      'absorption_train_calmar_gt_event_only':int(st['train_calmar']>se['train_calmar'] if se is not None and np.isfinite(se['train_calmar']) else 0),
      'timing_zero':int(timing_bad==0),
    }
    pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_csv(FINAL/'gates.csv',index=False)
    engine_hash=hashlib.sha256(Path(t1.__file__).read_bytes()).hexdigest(); wrapper_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec={'label':'NEW_STOCK_LEVEL_CAUSAL_SHORT_ALPHA_RESEARCH_NOT_ORIGINAL_EXACT','alpha':'T1 Inventory Exhaustion (T1-IE)','selected_key':f'{name}|m{mem}|n{n}','selected_config':cfg,'market_factor':market_code,'selection_uses':'2016-08-02..2021-12-31 only','pseudo':'2022-01-01..2026-07-29 research diagnostic, not clean OOS','prereg':'T1_INVENTORY_EXHAUSTION_PREREG_2026-09-04.md','source_engine_sha256':engine_hash,'cached_wrapper_sha256':wrapper_hash,'gates_passed':sum(gates.values()),'gates_total':len(gates),'universe_audit':ua,'computation_note':'broad candidates reused from run 33820425120; execution rows sharded only; strategy rules unchanged'}
    (FINAL/'strategy_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2,default=str))
    pd.DataFrame([{'execution_rows':len(X),'execution_codes':X.code.nunique(),'market_factor':market_code,'timing_violations':timing_bad,'engine_sha256':engine_hash,'wrapper_sha256':wrapper_hash}]).to_csv(FINAL/'audit.csv',index=False)
    print('=== SELECTED ==='); print(pd.DataFrame([st]).assign(variant=name,memory_sessions=mem,n_hold=n).to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(stress).to_string(index=False),flush=True)
    print('=== EVENT ONLY ==='); print(pd.DataFrame([{'control':'event_only_no_flush_confirmation',**se,'signal_count':len(esig)}]).to_string(index=False),flush=True)
    print('=== REVERSE ==='); print(pd.DataFrame([{'control':'reverse_score',**sr}]).to_string(index=False),flush=True)
    print('=== GATES ==='); print(pd.DataFrame([{'gate':k,'pass':v} for k,v in gates.items()]).to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['prepare','exec','grid','final']); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--shards',type=int,default=SHARDS); ap.add_argument('--variant',choices=list(t1.VARIANTS))
    a=ap.parse_args()
    if a.mode=='prepare': prepare()
    elif a.mode=='exec': exec_shard(a.shard,a.shards)
    elif a.mode=='grid': grid_variant(a.variant)
    else: final()

if __name__=='__main__': main()
