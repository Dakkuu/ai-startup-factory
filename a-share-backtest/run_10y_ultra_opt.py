from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd

import run_10y_era_backtest as base
import run_10y_alpha2f_v2 as sim
import run_10y_alpha2f_v4 as v4
import run_10y_factor_quality as fq
import run_10y_skewfilter_surface as sf
import run_10y_skewfilter_hard as hard
import run_10y_grand_opt as grand

OUT=Path('results_ultra_opt'); OUT.mkdir(exist_ok=True)
SEED=20260822

# Frozen before this search is observed. No fine decimals.
LIQ_LEVELS=(.60,.70,.80)
SKEW_KEEP=(.70,.80,.90)
BASE_IVOL_WEIGHTS=(.55,.60,.65)
EFF_HORIZONS=(60,120,180)
MODIFIERS=('none','downivol60','tail60','maxres60','gapvol60','range60','downcap126','capture_asym126','resid_pos120')
MOD_WEIGHTS=(.10,.20)
TOP_EXACT=15
TOP_CONSTRUCT=3
NS=(15,20,25)
HOLDS=(50,60,70)
BUFFERS=((.05,.20),(.10,.30),(.15,.40))


def add_ultra_fields(p,cal,market_close):
    q=p.copy()
    cols=['eff60','eff180','downivol60','tail60','maxres60','gapvol60','range60','downcap126','capture_asym126','resid_pos120']
    for c in cols: q[c]=np.nan
    bm=market_close.reindex(cal[(cal>=sim.WARM)&(cal<=sim.END)]).pct_change(fill_method=None)
    bm_mu=bm.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).mean().shift(1)
    bm_var=bm.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).var().shift(1)
    groups=q.groupby('code').groups
    for i,(code,idx) in enumerate(groups.items(),1):
        c=base.qb.read_bin(code,'close',cal).loc[sim.WARM:sim.END]
        o=base.qb.read_bin(code,'open',cal).loc[sim.WARM:sim.END]
        h=base.qb.read_bin(code,'high',cal).loc[sim.WARM:sim.END]
        l=base.qb.read_bin(code,'low',cal).loc[sim.WARM:sim.END]
        if c.empty: continue
        r=c.pct_change(fill_method=None); m=bm.reindex(c.index)
        smu=r.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).mean().shift(1)
        cov=r.rolling(sim.BETA_LOOKBACK,min_periods=sim.MIN_BETA_OBS).cov(m).shift(1)
        beta=cov/bm_var.reindex(c.index); alpha=smu-beta*bm_mu.reindex(c.index)
        resid=r-alpha-beta*m
        # Trend efficiency on multiple economically coarse horizons.
        def eff(w):
            return (c/c.shift(w)-1)/r.abs().rolling(w,min_periods=max(40,int(.8*w))).sum().replace(0,np.nan)
        e60=eff(60); e180=eff(180)
        # Downside / tail / lottery risks.
        downiv=np.sqrt(resid.clip(upper=0).pow(2).rolling(60,min_periods=48).mean())
        tail=-resid.rolling(60,min_periods=48).quantile(.10)  # lower is better
        mx=resid.rolling(60,min_periods=48).max()             # lower is less lottery-like
        gap=(o/c.shift(1)-1).replace([np.inf,-np.inf],np.nan) if not o.empty else pd.Series(np.nan,index=c.index)
        gapv=gap.rolling(60,min_periods=48).std()
        rng=(h/l-1).replace([np.inf,-np.inf],np.nan) if (not h.empty and not l.empty) else pd.Series(np.nan,index=c.index)
        rangev=rng.rolling(60,min_periods=48).mean()
        # Market down/up capture. Ratios use only days of the relevant market sign.
        dmask=m<0; umask=m>0
        dnum=r.where(dmask).rolling(126,min_periods=70).sum(); dden=m.where(dmask).rolling(126,min_periods=70).sum()
        unum=r.where(umask).rolling(126,min_periods=70).sum(); uden=m.where(umask).rolling(126,min_periods=70).sum()
        dcap=dnum/dden.replace(0,np.nan); ucap=unum/uden.replace(0,np.nan)
        asym=ucap-dcap
        pos=(resid>0).where(resid.notna()).astype(float).rolling(120,min_periods=96).mean()
        ds=pd.DatetimeIndex(q.loc[idx,'signal_date'])
        vals={'eff60':e60,'eff180':e180,'downivol60':downiv,'tail60':tail,'maxres60':mx,'gapvol60':gapv,'range60':rangev,'downcap126':dcap,'capture_asym126':asym,'resid_pos120':pos}
        for k,s in vals.items(): q.loc[idx,k]=s.reindex(ds).to_numpy(float)
        if i%1000==0: print('ULTRA FIELDS',i,'/',len(groups),flush=True)
    return q


def pct_rank(q,m,col,better_high=False):
    return q.loc[m].groupby('signal_date')[col].rank(pct=True,method='average',ascending=not better_high)


def make_score(p,liq,skkeep,wiv,effh,modifier='none',modw=.10):
    q=p.copy(); q['rank_test']=np.nan
    effcol=f'eff{effh}'
    m=np.isfinite(q.ivol60)&np.isfinite(q[effcol])&np.isfinite(q.skew40)&np.isfinite(q.liq_rank_pct)&(q.liq_rank_pct<=liq)
    # Causal cross-sectional anti-lottery filter.
    sk=q.loc[m].groupby('signal_date').skew40.rank(pct=True,method='average',ascending=True)
    ok=pd.Series(False,index=q.index); ok.loc[sk.index]=sk<=skkeep; m &= ok
    if modifier!='none': m &= np.isfinite(q[modifier])
    if not m.any(): return q
    iv=pct_rank(q,m,'ivol60',False); ef=pct_rank(q,m,effcol,True)
    if modifier=='none': raw=wiv*iv+(1-wiv)*ef
    else:
        high=modifier in ('capture_asym126','resid_pos120')
        mr=pct_rank(q,m,modifier,high)
        raw=(1-modw)*(wiv*iv+(1-wiv)*ef)+modw*mr
    q.loc[m,'rank_test']=raw.groupby(q.loc[m,'signal_date']).rank(pct=True,method='average',ascending=True)
    return q


def period(eq,a,b):
    return grand.period_metrics(eq,a,b)


def train_objective(st,eq):
    # Only pre-2022 data. Reward CAGR, penalize drawdown, and require both sub-blocks to work.
    a=period(eq,'2016-07-29','2018-12-31'); b=period(eq,'2019-01-01','2021-12-31')
    if not np.isfinite(a['cagr']) or not np.isfinite(b['cagr']): return -999.,a,b
    worst=min(a['cagr'],b['cagr'])
    score=float(st['train_cagr'] - .25*abs(st['train_mdd']) + .50*worst)
    if a['return']<=0 or b['return']<=0: score-=1.0
    return score,a,b


def main():
    base.START=sim.START; base.WARM=sim.WARM; base.END=sim.END; base.OUT=OUT; v4.OUT=OUT
    cal,members,ua=base.load_base(); market_code,market_close,mc=v4.pick_market(cal)
    p=v4.build_panel(cal,members,market_close); p=fq.add_factors(p,cal); p=sf.add_skews(p,cal,market_close); p=grand.add_grand_fields(p,cal,members); p=add_ultra_fields(p,cal,market_close)
    keep=['signal_date','trade_date','code','liq20','liq_rank_pct','exec_open','exec_high','exec_low','exec_volume','exec_factor','ivol60','eff60','eff120','eff180','skew40','downivol60','tail60','maxres60','gapvol60','range60','downcap126','capture_asym126','resid_pos120']
    p=p[keep].copy(); bm=market_close.loc[sim.START:sim.END].dropna()

    # Stage A: structural discovery. Fixed construction, sparse MTM only for screening.
    rows=[]
    configs=[]
    for liq in LIQ_LEVELS:
      for sk in SKEW_KEEP:
       for wiv in BASE_IVOL_WEIGHTS:
        for eh in EFF_HORIZONS:
         configs.append((liq,sk,wiv,eh,'none',0.0))
         for mod in MODIFIERS:
          if mod=='none': continue
          for mw in MOD_WEIGHTS: configs.append((liq,sk,wiv,eh,mod,mw))
    print('DISCOVERY CONFIGS',len(configs),flush=True)
    for j,(liq,sk,wiv,eh,mod,mw) in enumerate(configs,1):
        q=make_score(p,liq,sk,wiv,eh,mod,mw)
        st,eq,tr,tm=grand.run(q,60,20,.10,.30,cal,members,bm,fast=True)
        obj,a,b=train_objective(st,eq)
        st.update({'liq_keep':liq,'skew_keep':sk,'ivol_weight':wiv,'eff_horizon':eh,'modifier':mod,'modifier_weight':mw,'train_objective':obj,'train_block1_return':a['return'],'train_block1_cagr':a['cagr'],'train_block2_return':b['return'],'train_block2_cagr':b['cagr']})
        rows.append(st)
        if j%100==0: print('DISCOVERY',j,'/',len(configs),flush=True)
        del q,eq,tr,tm
    disc=pd.DataFrame(rows); disc.to_csv(OUT/'discovery.csv',index=False)
    eligible=disc[(disc.train_return>0)&(disc.train_mdd>-0.35)&(disc.train_block1_return>0)&(disc.train_block2_return>0)&(disc.positions_max<=20)].copy()
    eligible=eligible.sort_values(['train_objective','train_cagr'],ascending=False)
    top=eligible.head(TOP_EXACT).copy(); top.to_csv(OUT/'discovery_top15.csv',index=False)
    if len(top)<5: raise RuntimeError('too few eligible candidates')

    # Stage B: exact daily MTM for training-selected structural finalists.
    exact_rows=[]; exact_cache={}
    for rank,(_,r) in enumerate(top.iterrows(),1):
        cfg=(float(r.liq_keep),float(r.skew_keep),float(r.ivol_weight),int(r.eff_horizon),str(r.modifier),float(r.modifier_weight))
        print('EXACT',rank,cfg,flush=True)
        q=make_score(p,*cfg)
        st,eq,tr,tm=grand.run(q,60,20,.10,.30,cal,members,bm,fast=False); obj,a,b=train_objective(st,eq)
        st.update({'candidate_rank':rank,'liq_keep':cfg[0],'skew_keep':cfg[1],'ivol_weight':cfg[2],'eff_horizon':cfg[3],'modifier':cfg[4],'modifier_weight':cfg[5],'train_objective':obj,'train_block1_return':a['return'],'train_block1_cagr':a['cagr'],'train_block2_return':b['return'],'train_block2_cagr':b['cagr']})
        exact_rows.append(st); exact_cache[cfg]=(q,eq,tr,tm)
    exact=pd.DataFrame(exact_rows).sort_values(['train_objective','train_cagr'],ascending=False); exact.to_csv(OUT/'exact_top15.csv',index=False)

    # Stage C: construction search only around top 3 exact structural signals; selection still train-only.
    construct=[]; ccache={}
    structs=[]
    for _,r in exact.head(TOP_CONSTRUCT).iterrows():
        structs.append((float(r.liq_keep),float(r.skew_keep),float(r.ivol_weight),int(r.eff_horizon),str(r.modifier),float(r.modifier_weight)))
    for si,cfg in enumerate(structs,1):
        q=exact_cache[cfg][0]
        for n in NS:
         for h in HOLDS:
          for e,k in BUFFERS:
            st,eq,tr,tm=grand.run(q,h,n,e,k,cal,members,bm,fast=False); obj,a,b=train_objective(st,eq)
            st.update({'structure_id':si,'liq_keep':cfg[0],'skew_keep':cfg[1],'ivol_weight':cfg[2],'eff_horizon':cfg[3],'modifier':cfg[4],'modifier_weight':cfg[5],'train_objective':obj,'train_block1_return':a['return'],'train_block2_return':b['return']})
            construct.append(st); ccache[(cfg,n,h,e,k)]=(eq,tr,tm)
    cons=pd.DataFrame(construct); cons.to_csv(OUT/'construction_exact.csv',index=False)
    ok=cons[(cons.train_return>0)&(cons.train_mdd>-0.35)&(cons.train_block1_return>0)&(cons.train_block2_return>0)&(cons.positions_max<=cons.n_hold)].sort_values(['train_objective','train_cagr'],ascending=False)
    winner=ok.iloc[0]
    wcfg=(float(winner.liq_keep),float(winner.skew_keep),float(winner.ivol_weight),int(winner.eff_horizon),str(winner.modifier),float(winner.modifier_weight))
    key=(wcfg,int(winner.n_hold),int(winner.hold_days),float(winner.entry_pct),float(winner.keep_pct))
    weq,wtr,wtm=ccache[key]
    pd.DataFrame([winner]).to_csv(OUT/'train_selected_winner.csv',index=False)
    sim.annual_returns(weq).to_csv(OUT/'winner_annual.csv',index=False)

    # Baseline exact current Qlib rule for apples-to-apples comparison (PIT-ST safe external benchmark is reported separately).
    bq=make_score(p,.70,.80,.60,120,'none',0.0); bst,beq,btr,btm=grand.run(bq,60,20,.10,.30,cal,members,bm,fast=False); pd.DataFrame([bst]).to_csv(OUT/'baseline_qlib.csv',index=False)

    # Winner stress: costs, hold neighborhood, phase offsets, tail dependence.
    costs=[]
    wq=exact_cache[wcfg][0]
    for cm in (2.,4.,8.):
        st,_,_,_=grand.run(wq,int(winner.hold_days),int(winner.n_hold),float(winner.entry_pct),float(winner.keep_pct),cal,members,bm,cost=cm,fast=False); st['cost_mult_test']=cm; costs.append(st)
    pd.DataFrame(costs).to_csv(OUT/'winner_costs.csv',index=False)
    rob=sim.robustness(weq,wtr); pd.DataFrame([rob]).to_csv(OUT/'winner_tail_robustness.csv',index=False)
    blocks=[]
    for label,a,b in [('2016_2018','2016-07-29','2018-12-31'),('2019_2021','2019-01-01','2021-12-31'),('2022_2024','2022-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-07-29')]:
        z=period(weq,a,b); z['block']=label; blocks.append(z)
    pd.DataFrame(blocks).to_csv(OUT/'winner_blocks.csv',index=False)

    # Deterministic audit.
    alltm=[x[2] for x in ccache.values() if len(x[2])]
    bad=0
    for tm in alltm:
        bad += int((pd.to_datetime(tm.signal_date)>=pd.to_datetime(tm.trade_date)).sum())
    audit={**ua,'market_factor':market_code,'discovery_configs':len(configs),'exact_structural_finalists':len(exact),'construction_exact_points':len(cons),'selection':'all choices use 2016-2021 only; 2022-2026 is reused validation, not untouched OOS','timing_violations':bad,'baseline_total':bst['total_return'],'winner_total':float(winner.total_return),'winner_train':float(winner.train_return),'winner_validation':float(winner.validation_return),'winner_rule':f'liq{wcfg[0]}; skew_keep{wcfg[1]}; ivol{wcfg[2]} + eff{wcfg[3]}; mod={wcfg[4]}@{wcfg[5]}; N{int(winner.n_hold)} H{int(winner.hold_days)} E{winner.entry_pct} K{winner.keep_pct}'}
    pd.DataFrame([audit]).to_csv(OUT/'audit.csv',index=False)
    if bad: raise RuntimeError('timing audit failed')

    print('=== AUDIT ==='); print(pd.DataFrame([audit]).to_string(index=False),flush=True)
    print('=== BASELINE ==='); print(pd.DataFrame([bst]).to_string(index=False),flush=True)
    print('=== EXACT TOP ==='); print(exact.head(15).to_string(index=False),flush=True)
    print('=== CONSTRUCTION TOP TRAIN ==='); print(ok.head(20).to_string(index=False),flush=True)
    print('=== WINNER ==='); print(pd.DataFrame([winner]).to_string(index=False),flush=True)
    print('=== ANNUAL ==='); print(sim.annual_returns(weq).to_string(index=False),flush=True)
    print('=== BLOCKS ==='); print(pd.DataFrame(blocks).to_string(index=False),flush=True)
    print('=== COSTS ==='); print(pd.DataFrame(costs).to_string(index=False),flush=True)
    print('=== TAIL ==='); print(pd.DataFrame([rob]).to_string(index=False),flush=True)

if __name__=='__main__': main()
