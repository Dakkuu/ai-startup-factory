from __future__ import annotations
from pathlib import Path
import argparse, json, numpy as np, pandas as pd
import run_10y_baseline_maxopt_v3 as mo
import run_10y_megaopt_v1 as mega
import run_10y_max_audit as ma
import run_10y_maxopt_v3_frozen_audit as fa
import run_10y_geff55_strict_audit_v2 as strict
import run_10y_hard_executor_v3 as hv3
hv3.patch()

GATES=(.50,.55,.60,.65); HOLDS=(60,90,120); NS=(8,10,15); BUFFERS=((.05,.20),(.10,.30))
SHARDS={
 'a':[('orig',{'iv':.30,'down':.20,'rmom':.30,'tstat':.20}),('def',{'iv':.35,'down':.25,'rmom':.25,'tstat':.15})],
 'b':[('trend',{'iv':.30,'down':.20,'rmom':.25,'tstat':.25}),('lowiv',{'iv':.40,'down':.20,'rmom':.25,'tstat':.15})],
 'c':[('mom',{'iv':.25,'down':.15,'rmom':.35,'tstat':.25}),('down',{'iv':.30,'down':.30,'rmom':.25,'tstat':.15})],
}
BASECOLS=strict.BASECOLS+['exec_buy_allowed','exec_sell_allowed','exec_open_gap','exec_limit_proxy']

def spec(g,nm,w): return {'name':f'geff{int(g*100)}_{nm}','kind':'gate','g':{'ef':g},'w':w}

def phase_panels(q,h):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(q.signal_date.unique()))); step=max(1,round(h/5)); out={}
    for ph in range(step):
        chosen=set(dates[ph::step]); z=q[q.signal_date.isin(chosen)][[c for c in BASECOLS if c in q.columns]].copy(); z['ivol60_pct']=z.rank_test; out[ph]=z.drop(columns='rank_test')
    return out,step

def run_train(z,cal,members,bm,n,e,k):
    return ma.run_panel(z,cal,members,bm,n=n,entry=e,keep=k,start=mo.START,end=mo.TRAIN_END)

def group_offsets(step):
    # Six equally spaced sleeves; step is 12/18/24 for H60/H90/H120.
    spacing=step//6; return [[off+j*spacing for j in range(6)] for off in range(spacing)]

def metrics(eqs,stats,step,bm):
    rets=np.array([s['total_return'] for s in stats],float); cagr=np.array([s['cagr'] for s in stats],float); mdd=np.array([s['max_drawdown'] for s in stats],float)
    groups=[]
    for ix in group_offsets(step):
        ee=fa.phase_ensemble([eqs[i] for i in ix]); es=fa.perf_eq(ee,bm.loc[mo.START:mo.TRAIN_END]); groups.append(es)
    gc=np.array([g['cagr'] for g in groups],float); gr=np.array([g['total_return'] for g in groups],float); gm=np.array([g['max_drawdown'] for g in groups],float)
    all_e=fa.phase_ensemble(eqs); ae=fa.perf_eq(all_e,bm.loc[mo.START:mo.TRAIN_END])
    # Pre-registered objective: reward worst six-sleeve CAGR and median CAGR, penalize anchor dispersion.
    score=float(gc.min()+.50*np.median(gc)-.50*np.std(gc))
    return {'train_phase_count':step,'train_phase_min':float(rets.min()),'train_phase_median':float(np.median(rets)),'train_phase_mean':float(rets.mean()),'train_phase_all_positive':int((rets>0).all()),'worst_phase_mdd':float(mdd.min()),'group_count':len(groups),'group_return_min':float(gr.min()),'group_return_median':float(np.median(gr)),'group_cagr_min':float(gc.min()),'group_cagr_median':float(np.median(gc)),'group_cagr_std':float(np.std(gc)),'group_mdd_worst':float(gm.min()),'all_group_positive':int((gr>0).all()),'allphase_ensemble_return':float(ae['total_return']),'allphase_ensemble_cagr':float(ae['cagr']),'allphase_ensemble_mdd':float(ae['max_drawdown']),'robust_score':score}

def main(shard):
    out=Path(f'results_geff_phase_robust_v2_{shard}'); out.mkdir(exist_ok=True)
    p,cal,members,ua,market_code,bm=mo.build_panel(out,need_fwd=False); p=strict.attach_gap_flags(p,cal,'board')
    rows=[]
    for nm,w in SHARDS[shard]:
      for g in GATES:
        sp=spec(g,nm,w); print('SIGNAL',sp['name'],flush=True); q=mega.make_rank(p,sp)
        for h in HOLDS:
            panels,step=phase_panels(q,h); print('HOLD',h,'PHASES',step,flush=True)
            for n in NS:
              for e,k in BUFFERS:
                eqs=[]; sts=[]
                for ph in range(step):
                    st,eq,tr,tm=run_train(panels[ph],cal,members,bm,n,e,k); eqs.append(eq); sts.append(st)
                z=metrics(eqs,sts,step,bm); z.update({'shard':shard,'signal':sp['name'],'gate':g,'weights_name':nm,'weights':json.dumps(w,sort_keys=True),'hold':h,'n_hold':n,'entry_pct':e,'keep_pct':k,'key':f'{sp["name"]}|h{h}|n{n}|e{e}|k{k}'}); rows.append(z)
            del panels
        del q
    df=pd.DataFrame(rows); df.to_csv(out/'train_only_grid.csv',index=False)
    eligible=df[(df.train_phase_all_positive==1)&(df.all_group_positive==1)&(df.worst_phase_mdd>-0.55)&(df.group_mdd_worst>-0.45)].copy()
    if len(eligible)==0: eligible=df.copy()
    sel=eligible.sort_values(['robust_score','group_cagr_min','group_cagr_median'],ascending=[False,False,False]).head(10)
    sel.to_csv(out/'train_only_selected.csv',index=False)
    pd.DataFrame([{**ua,'market_factor':market_code,'shard':shard,'configs':len(df),'exact_train_runs':int(sum(int(x) for x in df.train_phase_count)),'selection_period':'2016-07-29..2021-12-31 ONLY','validation_2022_2026_accessed':0,'execution':'hard_v3 board-limit proxy, signal-pure, volume unit 100 shares','objective':'min six-sleeve CAGR + .5 median six-sleeve CAGR - .5 cross-anchor CAGR std','rule':'no phase0 objective; all phases used in train robustness'}]).to_csv(out/'audit.csv',index=False)
    print('TOP TRAIN ONLY'); print(sel.to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('shard',choices=tuple(SHARDS)); a=ap.parse_args(); main(a.shard)
