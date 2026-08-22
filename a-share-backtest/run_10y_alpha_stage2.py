from __future__ import annotations
from pathlib import Path
import argparse, math
import numpy as np
import pandas as pd

import run_10y_alpha_discovery_qv as qv
import run_10y_alpha_composites_qv as comp
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_factor_mine2 as mine2
import run_10y_max_audit as ma


def build_panel(out):
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=out; v4.OUT=out; qv.sp.OUT=out
    cal,members,ua=base.load_base(); market_code,market_close,_=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=mine2.add_extra(p,cal,market_close); p=qv.add_qv_fields(p,cal); p=qv.attach_oriented_existing(p)
    bm=market_close.loc[sim.START:sim.END].dropna()
    return p,cal,members,ua,market_code,bm


def percentile(p,col,mask):
    return p.loc[mask].groupby('signal_date')[col].rank(pct=True,method='average',ascending=True)


def add_equal_rank_score(p,name,cols):
    q=p.copy(); q[name]=np.nan; m=q.liq_rank_pct<=qv.LIQ_KEEP
    for c in cols: m &= np.isfinite(q[c])
    if m.any():
        parts=[percentile(q,c,m).rename(c) for c in cols]
        q.loc[m,name]=pd.concat(parts,axis=1).mean(axis=1)
    return q


def add_sweetspot_score(p,name,cols,center):
    q=p.copy(); q[name]=np.nan; m=q.liq_rank_pct<=qv.LIQ_KEEP
    for c in cols: m &= np.isfinite(q[c])
    if m.any():
        parts=[]
        for c in cols:
            r=percentile(q,c,m)
            parts.append((1.0-(r-center).abs()).rename(c))
        q.loc[m,name]=pd.concat(parts,axis=1).mean(axis=1)
    return q


def ranked(p,name):
    q=p.copy(); q['rank_test']=np.nan; m=(q.liq_rank_pct<=qv.LIQ_KEEP)&np.isfinite(q[name])
    q.loc[m,'rank_test']=q.loc[m].groupby('signal_date')[name].rank(pct=True,method='average',ascending=False)
    return q


def exact(rq,hold,n,e,k,cal,members,bm,cost=1.0):
    st,eq,tr,tm=ma.run_q(rq,hold,0,cal,members,bm,n=n,entry=e,keep=k,cost=cost)
    st['train_2016_2021_return']=sim.period_return(eq,'2016-07-29','2021-12-31')
    st['pseudo_oos_2022_2026_return']=sim.period_return(eq,'2022-01-01','2026-07-29')
    return st,eq,tr,tm


def finalize(out,rows,cache,meta_key,ua,market_code,round_name):
    grid=pd.DataFrame(rows)
    if len(grid): grid=grid.sort_values(['train_2016_2021_return','max_drawdown'],ascending=[False,False])
    grid.to_csv(out/'grid.csv',index=False)
    costs=[]; annual=[]; robust=[]
    if len(grid):
        w=grid.iloc[0]; key=w['cache_key']; rq,eq,tr,tm=cache[key]
        pd.DataFrame([w.drop(labels=['cache_key'])]).to_csv(out/'winner_train_selected.csv',index=False)
        for cm in (2.,4.,8.):
            st,_,_,_=exact(rq,int(w.hold_days),int(w.n_hold),float(w.entry_pct),float(w.keep_pct),cal=GLOBAL['cal'],members=GLOBAL['members'],bm=GLOBAL['bm'],cost=cm)
            st[meta_key]=w[meta_key]; costs.append(st)
        pd.DataFrame(costs).to_csv(out/'winner_costs.csv',index=False)
        a=sim.annual_returns(eq); a[meta_key]=w[meta_key]; a.to_csv(out/'winner_annual.csv',index=False)
        rr=sim.robustness(eq,tr); rr[meta_key]=w[meta_key]; pd.DataFrame([rr]).to_csv(out/'winner_robust.csv',index=False)
    allt=pd.concat([x[3].assign(test_key=str(k)) for k,x in cache.items() if len(x[3])],ignore_index=True) if cache else pd.DataFrame()
    bad=int((pd.to_datetime(allt.signal_date)>=pd.to_datetime(allt.trade_date)).sum()) if len(allt) else 0
    hits=int((grid.total_return>=5.0).sum()) if len(grid) else 0
    audit={**ua,'market_factor':market_code,'research_round':round_name,'points':len(grid),'target_500_hits':hits,'selection':'2016-2021 only; 2022-2026 exploratory/pseudo-OOS','signal_universe':'signal-pure T-only','volume_source_unit_shares':100,'timing_violations':bad}
    pd.DataFrame([audit]).to_csv(out/'audit.csv',index=False)
    print('=== TOP GRID ==='); print(grid.drop(columns=['cache_key']).head(60).to_string(index=False) if len(grid) else 'NONE',flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
    if bad: raise RuntimeError('timing violation')


def lane_discovery_b():
    out=Path('results_alpha_stage2_discovery_b'); out.mkdir(exist_ok=True)
    p,cal,members,ua,market_code,bm=build_panel(out); GLOBAL.update(cal=cal,members=members,bm=bm)
    # These direction reversals are POST-SCREEN hypotheses derived solely from the 2016-2021 IC screen.
    p['low_relative_volume']=-p.relative_volume
    p['gap_continuation']=-p.gap_reversal
    p['high_liquidity_amihud']=-p.amihud_illiquidity
    specs=[
      ('low_relative_volume',60,('low_relative_volume',),'inverse of training-period relative-volume direction'),
      ('gap_continuation',60,('gap_continuation',),'inverse of training-period gap-reversal direction'),
      ('high_liquidity_amihud',60,('high_liquidity_amihud',),'inverse of training-period Amihud direction'),
      ('lowrisk_quiet',60,('low_ivol','low_downside','anti_max','low_relative_volume'),'anti-lottery plus low attention'),
      ('lowrisk_quiet_mom',60,('low_ivol','low_downside','anti_max','low_relative_volume','residual_momentum','efficiency'),'low-risk quiet trend'),
      ('liquid_momentum',60,('high_liquidity_amihud','low_relative_volume','residual_momentum','efficiency'),'liquid low-attention momentum'),
      ('gap_trend',60,('gap_continuation','residual_momentum','efficiency','near_52w_high'),'gap continuation plus medium trend'),
    ]
    for name,h,cols,why in specs:
        if len(cols)>1: p=add_equal_rank_score(p,name,cols)
    for center in (.80,.85,.90):
        name=f'lowrisk_sweet_{int(center*100)}'; p=add_sweetspot_score(p,name,('low_ivol','low_downside','anti_max'),center); specs.append((name,60,('low_ivol','low_downside','anti_max'),f'training deciles show upper-middle sweet spot center {center:.2f}'))
    pd.DataFrame([{'candidate':n,'horizon':h,'components':'|'.join(cols),'status':'post-screen exploratory','rationale':why} for n,h,cols,why in specs]).to_csv(out/'manifest.csv',index=False)
    screens=[]
    for name,h,cols,why in specs:
        r,_,_=qv.factor_diag(p,name,h); r['candidate']=name; screens.append(r)
    screen=pd.DataFrame(screens).sort_values(['screen_pass','ic_t'],ascending=[False,False]); screen.to_csv(out/'screen.csv',index=False)
    rows=[]; cache={}
    finalists=screen[(screen.mean_ic>0)&np.isfinite(screen.ic_t)].head(8)
    for r in finalists.itertuples(index=False):
        name=str(r.candidate); rq=ranked(p,name)
        for n in (10,20):
            for e,k in ((.05,.20),(.10,.30)):
                print('DISCOVERY_B',name,n,e,k,flush=True); st,eq,tr,tm=exact(rq,60,n,e,k,cal,members,bm); key=(name,n,e,k); st.update({'candidate':name,'hold_days':60,'n_hold':n,'entry_pct':e,'keep_pct':k,'cache_key':key,'screen_pass':int(r.screen_pass),'ic_t_train':float(r.ic_t),'mean_ic_train':float(r.mean_ic)}); rows.append(st); cache[key]=(rq,eq,tr,tm)
    finalize(out,rows,cache,'candidate',ua,market_code,'post-screen direction and sweet-spot discovery B')


def lane_concentration():
    out=Path('results_alpha_stage2_concentration'); out.mkdir(exist_ok=True)
    p,cal,members,ua,market_code,bm=build_panel(out); GLOBAL.update(cal=cal,members=members,bm=bm)
    p=comp.add_composite_scores(p)
    signals=('anti_lottery_momentum','defensive_lottery','lowrisk_capture','low_downside')
    rows=[]; cache={}
    for name in signals:
        rq=comp.ranked(p,name) if name in [x[0] for x in comp.COMPOSITES] else ranked(p,name)
        for n in (5,8,10,15,20):
            for h in (40,60,80,120):
                for e,k in ((.02,.10),(.05,.20),(.10,.30)):
                    print('CONC',name,'N',n,'H',h,e,k,flush=True); st,eq,tr,tm=exact(rq,h,n,e,k,cal,members,bm); key=(name,n,h,e,k); st.update({'signal':name,'hold_days':h,'n_hold':n,'entry_pct':e,'keep_pct':k,'cache_key':key}); rows.append(st); cache[key]=(rq,eq,tr,tm)
    finalize(out,rows,cache,'signal',ua,market_code,'training-only concentration/holding/buffer surface')


def training_ic(p,factors):
    vals=[]
    for f in factors:
        r,_,_=qv.factor_diag(p,f,60); vals.append(r)
    return pd.DataFrame(vals)


def weighted_score(p,name,weights):
    q=p.copy(); q[name]=np.nan; m=q.liq_rank_pct<=qv.LIQ_KEEP
    for c in weights: m &= np.isfinite(q[c])
    if m.any():
        z=[]; ws=[]
        for c,w in weights.items(): z.append(percentile(q,c,m).rename(c)); ws.append(float(w))
        M=pd.concat(z,axis=1); wv=np.asarray(ws,float); wv=wv/wv.sum(); q.loc[m,name]=(M.to_numpy(float)*wv).sum(axis=1)
    return q


def lane_ensemble():
    out=Path('results_alpha_stage2_ensemble'); out.mkdir(exist_ok=True)
    p,cal,members,ua,market_code,bm=build_panel(out); GLOBAL.update(cal=cal,members=members,bm=bm)
    factors=('low_ivol','anti_max','low_downside','near_52w_high','skip_momentum_126_20','residual_momentum')
    d=training_ic(p,factors); d.to_csv(out/'training_factor_stats.csv',index=False)
    meanw={r.factor:max(float(r.mean_ic),0.0) for r in d.itertuples(index=False)}
    tw={r.factor:max(float(r.ic_t),0.0) for r in d.itertuples(index=False)}
    specs={
      'screen6_equal':{f:1.0 for f in factors},
      'screen6_meanIC':meanw,
      'screen6_tstat':tw,
      'top3_meanIC':{f:meanw[f] for f in sorted(factors,key=lambda x:meanw[x],reverse=True)[:3]},
      'top4_meanIC':{f:meanw[f] for f in sorted(factors,key=lambda x:meanw[x],reverse=True)[:4]},
      'role3_equal':{'low_ivol':1.0,'anti_max':1.0,'residual_momentum':1.0},
    }
    for name,w in specs.items(): p=weighted_score(p,name,w)
    pd.DataFrame([{'ensemble':n,'weights':'|'.join(f'{k}:{v:.6g}' for k,v in w.items()),'weight_source':'2016-2021 factor IC only'} for n,w in specs.items()]).to_csv(out/'manifest.csv',index=False)
    rows=[]; cache={}
    for name in specs:
        rq=ranked(p,name)
        for n in (5,10,20):
            for e,k in ((.05,.20),(.10,.30)):
                print('ENS',name,n,e,k,flush=True); st,eq,tr,tm=exact(rq,60,n,e,k,cal,members,bm); key=(name,n,e,k); st.update({'ensemble':name,'hold_days':60,'n_hold':n,'entry_pct':e,'keep_pct':k,'cache_key':key}); rows.append(st); cache[key]=(rq,eq,tr,tm)
    finalize(out,rows,cache,'ensemble',ua,market_code,'static training-IC ensemble')

GLOBAL={}
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('lane',choices=('discovery_b','concentration','ensemble')); args=ap.parse_args()
    {'discovery_b':lane_discovery_b,'concentration':lane_concentration,'ensemble':lane_ensemble}[args.lane]()
