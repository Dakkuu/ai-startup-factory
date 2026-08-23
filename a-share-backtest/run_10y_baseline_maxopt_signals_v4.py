from __future__ import annotations
from pathlib import Path
import argparse, gc, json
import numpy as np
import pandas as pd

import run_10y_baseline_maxopt_v3 as v3
import run_10y_alpha2f_v2 as sim


def run_lane(lane:str):
    out=Path(f'results_baseline_maxopt_{lane}_v4'); out.mkdir(exist_ok=True)
    p,cal,members,ua,market_code,bm=v3.build_panel(out,need_fwd=True)
    specs={'linear':v3.linear_specs,'augmented':v3.augmented_specs,'nonlinear':v3.nonlinear_specs}[lane]()
    specmap={s['name']:s for s in specs}

    screens=[]
    for s in specs:
        print('SCREEN',s['name'],flush=True)
        q=v3.rerank(p,s); d=v3.signal_diag(q); d.update({'signal':s['name'],'spec':json.dumps(s,sort_keys=True)}); screens.append(d)
        del q; gc.collect()
    sc=pd.DataFrame(screens)
    sc['pass_gate']=(sc.n_dates>=200)&(sc.mean_ic>0)&(sc.ic_t>=2)&(sc.top_bottom_spread>0)&(sc.positive_years>=4)
    sc=sc.sort_values(['pass_gate','ic_t','mean_ic'],ascending=[False,False,False]); sc.to_csv(out/'signal_screen.csv',index=False)
    shortlist=sc[sc.pass_gate].head(8)
    if len(shortlist)<4: shortlist=sc.head(8)

    anchors=[]
    for r in shortlist.itertuples(index=False):
        name=str(r.signal); q=v3.rerank(p,specmap[name])
        st,eq,tr,tm=v3.train_run(q,60,20,.10,.30,cal,members,bm)
        st.update({'signal':name,'key':name,'ic_t':float(r.ic_t),'mean_ic':float(r.mean_ic)}); anchors.append(st)
        del q,eq,tr,tm; gc.collect()
    adf=pd.DataFrame(anchors); adf.to_csv(out/'anchor_train.csv',index=False)
    seed=v3.choose_two(adf); seed.to_csv(out/'seed_train.csv',index=False)

    grid=[]
    for rr in seed.itertuples(index=False):
        name=str(rr.signal); q=v3.rerank(p,specmap[name])
        for n in (10,15,20,30):
          for h in (40,60,80,120):
           for e,k in ((.05,.20),(.10,.30)):
            print('TRAIN GRID',lane,name,n,h,e,k,flush=True)
            st,eq,tr,tm=v3.train_run(q,h,n,e,k,cal,members,bm); key=f'{name}|n{n}|h{h}|e{e}|k{k}'
            st.update({'signal':name,'n_hold':n,'hold_days':h,'entry_pct':e,'keep_pct':k,'key':key}); grid.append(st)
            del eq,tr,tm
        del q; gc.collect()
    g=pd.DataFrame(grid); g.to_csv(out/'construction_train.csv',index=False)
    finalists=v3.choose_two(g); finalists.to_csv(out/'train_selected.csv',index=False)

    full=[]
    for r in finalists.itertuples(index=False):
        q=v3.rerank(p,specmap[str(r.signal)])
        st,eq,tr,tm=v3.full_run(q,r.hold_days,r.n_hold,r.entry_pct,r.keep_pct,cal,members,bm)
        st.update({'signal':str(r.signal),'key':str(r.key)}); full.append(st)
        del q,eq,tr,tm; gc.collect()
    f=pd.DataFrame(full); f.to_csv(out/'finalists_full.csv',index=False)

    wr=finalists.sort_values(['min_half_return','total_return'],ascending=[False,False]).iloc[0]
    q=v3.rerank(p,specmap[str(wr.signal)])
    st,eq,tr,tm=v3.full_run(q,wr.hold_days,wr.n_hold,wr.entry_pct,wr.keep_pct,cal,members,bm)
    sim.annual_returns(eq).to_csv(out/'winner_annual.csv',index=False)
    pd.DataFrame([sim.robustness(eq,tr)]).to_csv(out/'winner_tail.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        x,_,_,_=v3.full_run(q,wr.hold_days,wr.n_hold,wr.entry_pct,wr.keep_pct,cal,members,bm,cost=cm); x['cost_mult_test']=cm; costs.append(x)
    pd.DataFrame(costs).to_csv(out/'winner_costs.csv',index=False)
    phases=[]; step=max(1,round(float(wr.hold_days)/5))
    for ph in range(step):
        x,_,_,_=v3.full_run(q,wr.hold_days,wr.n_hold,wr.entry_pct,wr.keep_pct,cal,members,bm,phase=ph); x['phase']=ph; phases.append(x)
    pd.DataFrame(phases).to_csv(out/'winner_phases.csv',index=False)
    bad=int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum()) if len(tm) else 0
    audit={**ua,'market_factor':market_code,'lane':lane,'signal_candidates':len(sc),'construction_points':len(g),'memory_fix':'sequential rerank; no panel-copy cache','selection':'IC screen + construction 2016-2021 only; robust maximin 2016-2019 vs 2020-2021; 2022-2026 pseudo-OOS only','signal_universe':'T-only signal-pure','volume_unit_shares':100,'timing_violations':bad}
    pd.DataFrame([audit]).to_csv(out/'audit.csv',index=False)
    print('=== SCREEN TOP ==='); print(sc.head(25).to_string(index=False),flush=True)
    print('=== TRAIN FINALISTS ==='); print(finalists.to_string(index=False),flush=True)
    print('=== FULL FINALISTS ==='); print(f.to_string(index=False),flush=True)
    print('=== PHASES ==='); print(pd.DataFrame(phases).to_string(index=False),flush=True)
    if bad: raise RuntimeError('timing violation')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('lane',choices=('linear','augmented','nonlinear')); args=ap.parse_args(); run_lane(args.lane)
