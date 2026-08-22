from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_skewfilter_hard as hard
import run_10y_grand_opt as grand

OUT=Path('results_balanced_exact'); OUT.mkdir(exist_ok=True)
BASE_CANDIDATES=[
    ('anchor','all80'),
    ('anchor','liq60'),
    ('anchor','liq70'),
    ('tstat','quality_pool'),
    ('tri_dd','liq60'),
    ('tri_rmom','ex_star_bj'),
]
WEIGHTS=(.55,.60,.65)
NS=(15,20,25)
HOLDS=(50,60,70)
BUFFERS=((.05,.20),(.10,.30),(.15,.40))


def anchor_weighted(p,pool,w):
    q=p.copy(); q['rank_test']=np.nan; m=grand.pool_mask(q,pool)
    if not m.any(): return q
    iv=grand.pct_rank(q,m,'ivol60',True); ef=grand.pct_rank(q,m,'eff120',False)
    raw=w*iv+(1-w)*ef
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q


def exact(q,h,n,e,k,cal,members,bm,cost=1.0):
    st,eq,tr,tm=grand.run(q,h,n,e,k,cal,members,bm,cost,fast=False)
    return st,eq,tr,tm


def main():
    base.START=sim.START;base.WARM=sim.WARM;base.END=sim.END;base.OUT=OUT;v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members)
    keepcols=['signal_date','trade_date','code','liq20','liq_rank_pct','exec_open','exec_high','exec_low','exec_volume','exec_factor','ivol40','ivol60','ivol80','eff120','skew40','rmom126','tstat120','posratio120','dd120','mom120raw','volshock','age_days','board']
    p=p[keepcols].copy(); bm=market_close.loc[sim.START:sim.END].dropna()

    base_rows=[]; base_cache={}
    for sig,pool in BASE_CANDIDATES:
        q=grand.rerank(p,sig,pool)
        print('BASE EXACT',sig,pool,flush=True)
        st,eq,tr,tm=exact(q,60,20,.10,.30,cal,members,bm); st.update({'signal':sig,'pool':pool,'ivol_weight':.60}); base_rows.append(st); base_cache[(sig,pool)]=(eq,tr,tm)
        del q
    base_df=pd.DataFrame(base_rows); base_df.to_csv(OUT/'base_exact.csv',index=False)

    surf=[]; surf_cache={}
    for w in WEIGHTS:
        q=anchor_weighted(p,'liq70',w)
        for n in NS:
            for h in HOLDS:
                for e,k in BUFFERS:
                    print('SURFACE',w,n,h,e,k,flush=True)
                    st,eq,tr,tm=exact(q,h,n,e,k,cal,members,bm); st.update({'signal':'anchor','pool':'liq70','ivol_weight':w}); surf.append(st)
                    if (w==.60 and n==20 and h==60 and e==.10 and k==.30): surf_cache[('central',)]=(eq,tr,tm)
        del q
    s=pd.DataFrame(surf); s.to_csv(OUT/'anchor_liq70_surface.csv',index=False)

    eligible=s[(s.train_return>0)&(s.train_mdd>-0.30)&(s.positions_max<=s.n_hold)].copy()
    eligible['train_score']=eligible.train_cagr - .25*eligible.train_mdd.abs()
    ranked=eligible.sort_values(['train_score','train_cagr'],ascending=[False,False])
    best=ranked.iloc[0]
    bw=float(best.ivol_weight); bn=int(best.n_hold); bh=int(best.hold_days); be=float(best.entry_pct); bk=float(best.keep_pct)
    qb=anchor_weighted(p,'liq70',bw); bst,beq,btr,btm=exact(qb,bh,bn,be,bk,cal,members,bm); bst.update({'signal':'anchor','pool':'liq70','ivol_weight':bw,'train_score':float(best.train_score)})
    pd.DataFrame([bst]).to_csv(OUT/'train_selected_winner.csv',index=False)

    qc=anchor_weighted(p,'liq70',.60); cst,ceq,ctr,ctm=exact(qc,60,20,.10,.30,cal,members,bm); cst.update({'signal':'anchor','pool':'liq70','ivol_weight':.60})
    pd.DataFrame([cst]).to_csv(OUT/'central_summary.csv',index=False)

    by_weight=s.groupby('ivol_weight').agg(points=('total_return','size'),median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),median_train_cagr=('train_cagr','median'),min_train_cagr=('train_cagr','min'),median_validation_return=('validation_return','median'),min_validation_return=('validation_return','min'),median_mdd=('max_drawdown','median'),worst_mdd=('max_drawdown','min')).reset_index()
    by_weight.to_csv(OUT/'surface_by_weight.csv',index=False)
    by_hold=s.groupby('hold_days').agg(points=('total_return','size'),median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),median_train_cagr=('train_cagr','median'),median_validation_return=('validation_return','median'),median_mdd=('max_drawdown','median')).reset_index(); by_hold.to_csv(OUT/'surface_by_hold.csv',index=False)
    by_n=s.groupby('n_hold').agg(points=('total_return','size'),median_return=('total_return','median'),min_return=('total_return','min'),max_return=('total_return','max'),median_train_cagr=('train_cagr','median'),median_validation_return=('validation_return','median'),median_mdd=('max_drawdown','median')).reset_index(); by_n.to_csv(OUT/'surface_by_n.csv',index=False)

    ann=sim.annual_returns(ceq); ann.to_csv(OUT/'central_annual.csv',index=False)
    rob=sim.robustness(ceq,ctr); pd.DataFrame([rob]).to_csv(OUT/'central_robust.csv',index=False)
    blocks=[]
    for label,a,b in [('2016_2018','2016-07-29','2018-12-31'),('2019_2021','2019-01-01','2021-12-31'),('2022_2024','2022-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-07-29')]:
        z=grand.period_metrics(ceq,a,b); z.update({'block':label}); blocks.append(z)
    pd.DataFrame(blocks).to_csv(OUT/'central_blocks.csv',index=False)
    costs=[]
    for cm in (2.,4.,8.):
        st,_,_,_=exact(qc,60,20,.10,.30,cal,members,bm,cm); st.update({'signal':'anchor','pool':'liq70','ivol_weight':.60}); costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'central_cost.csv',index=False)

    mixes=[]
    def mix_stats(name,eq1,eq2):
        a=eq1.set_index(pd.to_datetime(eq1.trade_date)).equity.astype(float).sort_index()
        b=eq2.set_index(pd.to_datetime(eq2.trade_date)).equity.astype(float).sort_index()
        idx=a.index.union(b.index).sort_values(); a=a.reindex(idx).ffill(); b=b.reindex(idx).ffill()
        x=.5*(a/sim.INITIAL_CASH)+.5*(b/sim.INITIAL_CASH); r=x.pct_change().dropna(); dd=x/x.cummax()-1; yrs=max((x.index[-1]-x.index[0]).days/365.25,1e-9)
        return {'mix':name,'total_return':float(x.iloc[-1]-1),'cagr':float(x.iloc[-1]**(1/yrs)-1),'max_drawdown':float(dd.min()),'sharpe':float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else np.nan}
    ctrl_eq=base_cache[('anchor','all80')][0]
    for sig,pool in [('tstat','quality_pool'),('tri_dd','liq60'),('tri_rmom','ex_star_bj')]: mixes.append(mix_stats(f'central50_{sig}_{pool}50',ceq,base_cache[(sig,pool)][0]))
    mixes.append(mix_stats('central50_old_anchor50',ceq,ctrl_eq)); pd.DataFrame(mixes).to_csv(OUT/'mixes.csv',index=False)

    bad=int((pd.to_datetime(ctm.signal_date)>=pd.to_datetime(ctm.trade_date)).sum()) if len(ctm) else 0
    central_within=int(cst['positions_max']<=cst['n_hold'])
    audit={**ua,'market_factor':market_code,'base_candidates':'|'.join(f'{a}:{b}' for a,b in BASE_CANDIDATES),'surface_points':len(s),'surface':'liq70 anchor only; ivol weights .55/.60/.65; N15/20/25; hold50/60/70; buffers 5/20,10/30,15/40','selection':'train-selected row uses 2016-2021 only; central candidate fixed from prior structural discovery; 2022-2026 is reused validation NOT untouched OOS','execution':'daily MTM deterministic hard execution; trapped positions occupy slots','timing_violations':bad,'central_positions_within_target':central_within}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad or not audit['central_positions_within_target']: raise RuntimeError('balanced exact audit failed')

    print('=== BASE EXACT ==='); print(base_df.sort_values('total_return',ascending=False).to_string(index=False),flush=True)
    print('=== CENTRAL ==='); print(pd.DataFrame([cst]).to_string(index=False),flush=True)
    print('=== TRAIN SELECTED ==='); print(pd.DataFrame([bst]).to_string(index=False),flush=True)
    print('=== SURFACE TOP TRAIN ==='); print(ranked.head(20).to_string(index=False),flush=True)
    print('=== BY WEIGHT ==='); print(by_weight.to_string(index=False),flush=True)
    print('=== BY HOLD ==='); print(by_hold.to_string(index=False),flush=True)
    print('=== BY N ==='); print(by_n.to_string(index=False),flush=True)
    print('=== BLOCKS ==='); print(pd.DataFrame(blocks).to_string(index=False),flush=True)
    print('=== ANNUAL ==='); print(ann.to_string(index=False),flush=True)
    print('=== COST ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== ROBUST ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)
    print('=== MIXES ==='); print(pd.DataFrame(mixes).to_string(index=False),flush=True)
    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)

if __name__=='__main__': main()
